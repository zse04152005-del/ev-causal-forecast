"""分时电价情景仿真（论文 4.9 节；设计文档 8.3–8.4；大纲 9.1）。不需要重新训练，也不需要 GPU。

做法（推论 1）：反事实预测 = 事实预测 × exp(β (Δℓ' − Δℓ))，再截到 1。
- 事实预测：主模型（v2_main_A0，5 个种子的分位数取平均）的日前预测（第 24 步）中位数，测试期，只看严格分时小区
- 情景：flat 取消分时（每天取该小区当天平均价）；spread2 峰谷价差放大到 2 倍；evening 18:00–22:00 加价 20%
- 需求口径：预测的充电时长 = 利用率 × 桩数（桩·小时）
- 弹性：锚定值 β̂ = −0.213；论文所用不确定性为按小区整群自助法 se 0.313（--se 0.3131），默认读锚定文件里的聚类 se 0.202；截断正态（β ≤ 0）抽样；另算 −0.45、−0.76、−1.58 作对照
- 稳态弹性（定理 T4，平均充电时长 60 分钟）：--beta -0.3119 --se 0.4582，输出放 results/scenarios/steady_state_mu60
  （−0.45：PIAST 原代码的参考弹性；−0.76：Kuang et al. 2024 SCS；−1.58：本文数据上的混合 OLS 斜率）

用法（在仓库根目录下）：
    python scripts/simulate_scenarios.py --runs <runs 目录> --out results/scenarios [paths.data_dir=...]
输出：scenario_summary.csv（每个情景 × 弹性一行）、scenario_profiles.csv（每个情景 × 一天中的小时一行）。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.simulate.scenarios import draw_betas, period_surcharge, widen
from src.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--group", default="v2_main_A0")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--beta", type=float, default=None, help="override the anchor (e.g. the steady-state elasticity)")
ap.add_argument("--se", type=float, default=None, help="override its standard error (e.g. the zone-bootstrap s.e.)")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(_bootstrap.DEFAULT_CFG, a.overrides)
P = prepare(cfg)
HERE = os.path.dirname(os.path.abspath(__file__))
anc = json.load(open(os.path.join(HERE, "..", "configs", "anchor", "fivemin_main.json"), encoding="utf-8"))["own"][0]
B, SE = float(anc["beta"]), float(anc["se"])
if a.beta is not None:
    B = a.beta
if a.se is not None:
    SE = a.se
ALT = {"prior_0.45": -0.45, "kuang_0.76": -0.76, "naive_1.58": -1.58}

# ---- 事实预测：第 24 步中位数，种子平均
acc, n = None, 0
for s in range(a.seeds):
    f = os.path.join(a.runs, f"{a.group}_s{s}", "predictions_test.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    k = int(np.flatnonzero(z["steps"] == 24)[0])
    qi = int(np.argmin(np.abs(z["quantiles"] - 0.5)))
    med = z["yq"][:, k, :, qi].astype(np.float64)
    acc = med if acc is None else acc + med
    origins = z["origins"]
    n += 1
yhat = acc / n                                       # [起点, 小区]
t = origins + 24                                     # 目标时刻下标
tou = P.tou_mask()
cap = P.capacity.astype(np.float64)
time = P.time[t]
hour = np.asarray(time.hour)
full_day = pd.Series(1, index=time.normalize()).groupby(level=0).sum()
keep = np.isin(time.normalize(), full_day.index[full_day == 24])      # 只用预测覆盖了全部 24 小时的日子
print(f"种子 {n} 个；目标时刻 {len(t)} 个，其中完整日 {int(keep.sum()) // 24} 天；分时小区 {int(tou.sum())} 个；β̂ = {B:.4f}（se {SE:.4f}）")

# ---- 情景价格（在全样本的价格矩阵上生成，再取目标时刻）
price = np.exp(P.lp)
SCEN = {"flat": widen(price, P.time, tou, factor=0.0),
        "spread2": widen(price, P.time, tou, factor=2.0),
        "evening": period_surcharge(price, P.time, tou, hours=(18, 19, 20, 21), pct=0.20)}
dl_f = P.lp[t][:, tou]
y0 = np.minimum(yhat[:, tou], 1.0) * cap[tou][None, :]                 # 事实预测，桩·小时
y0, dl_f, hour_k, time_k = y0[keep], dl_f[keep], hour[keep], time[keep]
ndays = len(time_k) // 24
betas = draw_betas(np.array(B), np.array(SE), S=a.draws, seed=0)       # [S]


def cf(beta, d):
    return np.minimum(yhat[keep][:, tou] * np.exp(beta * d), 1.0) * cap[tou][None, :]


def stats(y1, d):
    up, dn = d > 1e-6, d < -1e-6
    prof0 = np.array([y0[hour_k == h].sum() for h in range(24)]) / ndays
    prof1 = np.array([y1[hour_k == h].sum() for h in range(24)]) / ndays
    r = {"total_pct": (y1.sum() / y0.sum() - 1) * 100,
         "up_pct": (y1[up].sum() / y0[up].sum() - 1) * 100 if up.any() else np.nan,
         "down_pct": (y1[dn].sum() / y0[dn].sum() - 1) * 100 if dn.any() else np.nan,
         "total_pilehours_per_day": (y1.sum() - y0.sum()) / ndays,
         "up_pilehours_per_day": (y1[up].sum() - y0[up].sum()) / ndays if up.any() else np.nan,
         "down_pilehours_per_day": (y1[dn].sum() - y0[dn].sum()) / ndays if dn.any() else np.nan,
         "peak_pct": (prof1.max() / prof0.max() - 1) * 100,
         "peak_to_mean_change": prof1.max() / prof1.mean() - prof0.max() / prof0.mean()}
    return r, prof1


rows, profs = [], []
for nm, pa in SCEN.items():
    d = np.log(pa)[t][:, tou][keep] - dl_f
    up, dn = d > 1e-6, d < -1e-6
    base = {"scenario": nm, "share_up": up.mean(), "share_down": dn.mean(), "mean_abs_dlog": np.abs(d[up | dn]).mean(),
            "mean_dlog_up": d[up].mean() if up.any() else np.nan, "mean_dlog_down": d[dn].mean() if dn.any() else np.nan,
            "fact_pilehours_per_day": y0.sum() / ndays, "fact_share_in_up_hours": y0[up].sum() / y0.sum(),
            "n_days": ndays, "n_zones": int(tou.sum())}
    r, prof1 = stats(cf(B, d), d)
    dr = [stats(cf(b, d), d) for b in betas]
    ci = {k: (np.nanpercentile([x[0][k] for x in dr], 5), np.nanpercentile([x[0][k] for x in dr], 95)) for k in r}
    row = {**base, "elasticity": "anchor", "beta": B, **r}
    row.update({f"{k}_lo": v[0] for k, v in ci.items()})
    row.update({f"{k}_hi": v[1] for k, v in ci.items()})
    rows.append(row)
    P1 = np.array([x[1] for x in dr])
    prof0 = np.array([y0[hour_k == h].sum() for h in range(24)]) / ndays
    alt_prof = {}
    for an, ab in ALT.items():
        ra, pa1 = stats(cf(ab, d), d)
        rows.append({**base, "elasticity": an, "beta": ab, **ra})
        alt_prof[an] = pa1
    w0 = y0 / y0.sum()
    for h in range(24):
        m = hour_k == h
        profs.append({"scenario": nm, "hour": h, "dlog_fact": float((dl_f[m] - P.ref_lp[tou][None, :]).mean()),
                      "dlog_change": float(d[m].mean()), "fact": prof0[h], "cf": prof1[h],
                      "cf_lo": np.percentile(P1[:, h], 5), "cf_hi": np.percentile(P1[:, h], 95),
                      **{f"cf_{an}": alt_prof[an][h] for an in ALT}})
os.makedirs(a.out, exist_ok=True)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(a.out, "scenario_summary.csv"), index=False)
pd.DataFrame(profs).to_csv(os.path.join(a.out, "scenario_profiles.csv"), index=False)
dq = np.percentile(betas, [5, 50, 95])
json.dump({"beta": B, "se": SE, "draws": a.draws, "beta_draws_p5_p50_p95": dq.tolist(), "n_seeds": n, "n_days": ndays, "n_tou_zones": int(tou.sum()), "step": 24,
           "share_capped_fact": float((yhat[keep][:, tou] >= 1.0).mean())},
          open(os.path.join(a.out, "scenario_meta.json"), "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(S.round(3).to_string())
print("β 抽样的 5%、50%、95% 分位：", np.round(dq, 3))
