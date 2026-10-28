"""分时电价情景的电量与负荷换算（论文 4.7 节）。只用 CPU，不需要预测文件。

在假设 1 下，电价改变使每个小区·小时的充电按 exp(β (Δℓ' − Δℓ)) 成比例变化（推论 1），这个比例与基线无关；
把它乘到测试期记录的充电电量上，就得到少充或多充的电量（MWh），某一小时的电量变化也就是该小时的平均负荷变化（MW）。
UrbanEV 的电量由充电时长与桩的额定功率推算，所以电量变化是充电时长变化按各小区功率加权的结果。
- 目标时刻与 scripts/simulate_scenarios.py 相同：测试期第 24 步的目标时刻，只取完整日，只看严格分时小区
- 弹性：窗口估计 β̂ = −0.213（按小区整群自助法 se 0.313）；命题 1 的到达弹性 β_a = β̂ / Ḡ_W（W = 180 分钟，
  指数分布的平均会话时长 30、60、120 分钟，se 同比例放大）；自适应锚定的 −0.343；另算 −0.76、−1.58 作对照
- 不确定性：截断正态（β ≤ 0）抽样，与式 (draws) 相同；区间靠近零的一端由截断决定，论文只报另一端
- 充电时长的百分比（*_pct_time）与 simulate_scenarios.py 用预测算出的百分比只差 0.01 个百分点左右，用作核对

用法（在仓库根目录下）：
    python scripts/scenario_energy.py --out results/scenarios/energy [paths.data_dir=...]
输出：energy_summary.csv（每个情景 × 弹性一行）、energy_profiles.csv（每个情景 × 弹性 × 小时一行）、energy_meta.json。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.data.windows import WindowDataset
from src.simulate.scenarios import draw_betas, period_surcharge, widen
from src.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--window", type=float, default=180.0, help="half-window W of the anchor (min)")
ap.add_argument("--sessions", type=float, nargs="+", default=[30.0, 60.0, 120.0], help="mean session lengths (min)")
ap.add_argument("--adaptive", type=float, default=-0.343, help="adaptive-anchoring elasticity (Section 4.5)")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(_bootstrap.DEFAULT_CFG, a.overrides)
P = prepare(cfg)
HERE = os.path.dirname(os.path.abspath(__file__))
anc = json.load(open(os.path.join(HERE, "..", "configs", "anchor", "fivemin_main.json"), encoding="utf-8"))["own"][0]
boot = json.load(open(os.path.join(HERE, "..", "results", "causal", "round2", "summary.json"), encoding="utf-8"))["bootstrap"]["D_exp"]
B, SE = float(anc["beta"]), float(boot["se"])


def gbar(mu, W):
    """命题 1：指数分布的会话时长（均值 mu）下，半窗口 W 内存量中新到达者的平均占比 Ḡ_W。"""
    return 1.0 - (mu / W) * (1.0 - np.exp(-W / mu))


EL = [("anchor", B, SE)] + [(f"arrival_mu{mu:g}", B / gbar(mu, a.window), SE / gbar(mu, a.window)) for mu in a.sessions]
EL.append(("adaptive", a.adaptive, SE))
REF = {"kuang_0.76": -0.76, "naive_1.58": -1.58}

# ---- 目标时刻：测试期第 24 步，完整日，严格分时小区
W_te = WindowDataset(P, "test", L=int(cfg.data.L), H=int(cfg.data.H), stride=int(cfg.train.eval_stride))
t = W_te.origins + 24
time = P.time[t]
full_day = pd.Series(1, index=time.normalize()).groupby(level=0).sum()
t = t[np.isin(time.normalize(), full_day.index[full_day == 24])]
tou = P.tou_mask()
hour = np.asarray(P.time[t].hour)
nd = len(t) // 24
E = P.raw["volume"][t][:, tou].astype(np.float64) / 1000.0           # 记录的充电电量，MWh / 小区·小时
D = P.raw["duration"][t][:, tou].astype(np.float64)                  # 充电时长，桩·小时
EVE = (18, 19, 20, 21)
fact = np.array([E[hour == h].sum() for h in range(24)]) / nd       # 平均日负荷曲线，MW
peak_h = int(np.argmax(fact))
print(f"测试期完整日 {nd} 天，分时小区 {int(tou.sum())} 个（桩 {int(P.capacity[tou].sum())} / {int(P.capacity.sum())}）；"
      f"电量 {E.sum() / nd:.1f} MWh/天，每桩·小时 {E.sum() / D.sum() * 1000:.2f} kWh；负荷高峰 {peak_h}:00 {fact[peak_h]:.1f} MW")

price = np.exp(P.lp)
SCEN = {"flat": widen(price, P.time, tou, factor=0.0),
        "spread2": widen(price, P.time, tou, factor=2.0),
        "evening": period_surcharge(price, P.time, tou, hours=EVE, pct=0.20)}
dl_f = P.lp[t][:, tou]
hidx = [np.flatnonzero(hour == h) for h in range(24)]


def stats(betas, d):
    """betas [S] → 各统计量 [S] 与负荷变化曲线 [S, 24]（按 S 分块，避免占用过多内存）。"""
    up, dn = d > 1e-6, d < -1e-6
    out, prof = [], []
    for c in np.array_split(np.asarray(betas, dtype=np.float64), max(1, len(betas) // 200)):
        f = np.expm1(c[:, None, None] * d[None])                     # [s, n, Z]
        dE, dD = E[None] * f, D[None] * f
        p = np.stack([dE[:, i].sum(axis=(1, 2)) for i in hidx], axis=1) / nd
        r = {"dE_total": dE.sum(axis=(1, 2)) / nd,
             "dE_up": dE[:, up].sum(axis=1) / nd,
             "dE_down": dE[:, dn].sum(axis=1) / nd if dn.any() else np.full(len(c), np.nan),
             "up_pct_time": dD[:, up].sum(axis=1) / D[up].sum() * 100,
             "down_pct_time": dD[:, dn].sum(axis=1) / D[dn].sum() * 100 if dn.any() else np.full(len(c), np.nan),
             "up_pct_energy": dE[:, up].sum(axis=1) / E[up].sum() * 100,
             "down_pct_energy": dE[:, dn].sum(axis=1) / E[dn].sum() * 100 if dn.any() else np.full(len(c), np.nan),
             "dMW_peak": (fact[None] + p).max(axis=1) - fact.max(),
             "dMW_evening": p[:, list(EVE)].mean(axis=1)}
        out.append(r)
        prof.append(p)
    return {k: np.concatenate([r[k] for r in out]) for k in out[0]}, np.concatenate(prof)


rows, profs = [], []
for nm, pa in SCEN.items():
    d = np.log(pa)[t][:, tou] - dl_f
    up, dn = d > 1e-6, d < -1e-6
    base = {"scenario": nm, "n_days": nd, "n_zones": int(tou.sum()), "E_day": E.sum() / nd, "E_up_day": E[up].sum() / nd,
            "E_down_day": E[dn].sum() / nd, "peak_hour": peak_h, "peak_MW": fact[peak_h], "evening_MW": fact[list(EVE)].mean()}
    for en, b, se in EL:
        r, p = stats(np.array([b]), d)
        dr, dp = stats(draw_betas(np.array(b), np.array(se), S=a.draws, seed=0), d)
        row = {**base, "elasticity": en, "beta": b, "se": se, **{k: v[0] for k, v in r.items()}}
        row.update({f"{k}_lo": np.percentile(v, 5) if np.isfinite(v).all() else np.nan for k, v in dr.items()})
        row.update({f"{k}_hi": np.percentile(v, 95) if np.isfinite(v).all() else np.nan for k, v in dr.items()})
        rows.append(row)
        for h in range(24):
            profs.append({"scenario": nm, "elasticity": en, "hour": h, "fact_MW": fact[h], "d_MW": p[0, h],
                          "d_MW_lo": np.percentile(dp[:, h], 5), "d_MW_hi": np.percentile(dp[:, h], 95)})
    for en, b in REF.items():
        r, p = stats(np.array([b]), d)
        rows.append({**base, "elasticity": en, "beta": b, **{k: v[0] for k, v in r.items()}})
        profs += [{"scenario": nm, "elasticity": en, "hour": h, "fact_MW": fact[h], "d_MW": p[0, h]} for h in range(24)]

os.makedirs(a.out, exist_ok=True)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(a.out, "energy_summary.csv"), index=False)
pd.DataFrame(profs).to_csv(os.path.join(a.out, "energy_profiles.csv"), index=False)
json.dump({"beta": B, "se_bootstrap": SE, "window_min": a.window, "draws": a.draws, "truncation": "beta <= 0",
           "gbar": {f"{mu:g}": gbar(mu, a.window) for mu in a.sessions},
           "test_days": [str(P.time[t[0]].date()), str(P.time[t[-1]].date())], "n_days": nd, "n_tou_zones": int(tou.sum()),
           "piles_tou": int(P.capacity[tou].sum()), "piles_all": int(P.capacity.sum()),
           "energy_MWh_per_day": E.sum() / nd, "energy_MWh_per_day_evening": E[np.isin(hour, EVE)].sum() / nd,
           "kWh_per_pile_hour": E.sum() / D.sum() * 1000, "peak_hour": peak_h, "peak_MW": fact[peak_h],
           "evening_MW": fact[list(EVE)].mean()},
          open(os.path.join(a.out, "energy_meta.json"), "w"), indent=1)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(S[["scenario", "elasticity", "beta", "dE_total", "dE_up", "dE_down", "up_pct_time", "down_pct_time", "dMW_peak", "dMW_evening",
         "dE_up_lo", "dE_down_hi", "dMW_peak_lo", "dMW_peak_hi", "dMW_evening_lo"]].round(3).to_string())
