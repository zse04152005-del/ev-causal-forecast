"""阶段 6 补充：5 分钟站点级半合成验证——设计 D 的小估计是"真实弹性小"还是"估计量把效应压小了"？

做法（真实价格、真实切换时刻、真实无效点与样本，只把分时站点的需求换成已知弹性的合成需求）：
1. 到达率 λ_s(t) = λ0_s(t) · 日因子 · 小时因子 · exp(β (ℓ_s(t) − ℓ̄_s))
   - λ0：该站点训练期"工作日/休息日 × 5 分钟时刻"的平均充电时长（桩·小时）换算成会话数，再除以平均会话时长
   - 日因子、小时因子：对数正态，均值 1（sd 分别为 --sd-day、--sd-hour），模拟真实数据的日间波动
2. 到达数 ~ Poisson(λ)；每次会话时长 ~ 对数正态（均值 m 分钟，对数 sd 0.6），同时在充的会话数不超过桩数
3. 合成"时长"（桩·小时 / 5 分钟）= 在充会话数 / 12，与真实 duration 同单位
4. 用与正式估计完全相同的面板（build_panel_5min 选出的站点、时刻、窗口）和 PPML 估计设计 C、D

三种结果变量：
- stock：合成时长（与真实估计同口径）
- flow：到达数（没有"存量稀释"，估计量无偏时应还原 β）
- truth：无噪声的期望存量（λ 不含随机因子、不抽样、不截断）→ 窗口口径下的"真值" β*_W

判断：stock 的估计 ≈ truth → 估计量本身没有明显衰减，小窗口下估计偏小来自"存量稀释"这一口径特征；
stock 明显比 truth 更接近 0 → 噪声/伪计数导致衰减。再把真实数据在各窗口上的估计与合成结果对照，反推与之相符的到达弹性。

用法：python scripts/semisynthetic_5min.py --data-dir <fiveMin> --d1 <UrbanEV/data> --out <结果目录> [--seeds 2]
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

import _bootstrap  # noqa: F401
from src.causal import switch_5min as s5
from src.causal.switch_did import DESIGNS
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--data-dir", default=None)
ap.add_argument("--d1", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--betas", default="0,-0.5,-1.0")
ap.add_argument("--means", default="30,60,120", help="平均会话时长（分钟）")
ap.add_argument("--windows", default="6,12,24,36", help="半窗口（5 分钟步数）")
ap.add_argument("--sd-day", type=float, default=0.25)
ap.add_argument("--sd-hour", type=float, default=0.25)
ap.add_argument("--no-cap", action="store_true", help="不按桩数截断在充会话数")
ap.add_argument("--tag", default="base")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides) + ([f"paths.data_dir={a.d1}"] if a.d1 else []))
root = project_root(cfg.paths.get("root"))
data_dir = a.data_dir or os.path.join(root, "02_数据", "interim", "fiveMin")
out_dir = ensure_dir(a.out or os.path.join(root, "05_实验结果", "因果估计", "5min"))
out_file = os.path.join(out_dir, f"semisynth_5min_{a.tag}.csv")
betas = [float(b) for b in a.betas.split(",")]
means = [int(m) for m in a.means.split(",")]
windows = [int(w) for w in a.windows.split(",")]

t0 = time.time()
P = prepare(cfg)
fm = s5.load_fivemin(data_dir)
zpos = {int(z): i for i, z in enumerate(P.zones)}
S_zone = s5.zone_ring_exposure(fm.zone_P, P.ring_members, P.ref_lp)
zone_index = np.array([zpos[int(z)] for z in fm.st_zone])
inv = s5.invalid_mask(fm, P.frozen, zone_index)
T_HI = (P.split.train_end + 1) * 12 - 1
T = T_HI + 1
gz = {int(z): int(g) for z, g in zip(P.zones, P.groups)}
cc = cfg.causal
types = s5.station_types(fm.P, fm.time, T_HI)
tou = np.flatnonzero(types == 2)

# 只保留分时站点与训练期，省内存
R = s5.FiveMin(time=fm.time[:T], st_ids=fm.st_ids[tou], st_zone=fm.st_zone[tou], st_piles=fm.st_piles[tou],
               P=fm.P[:T, tou], dur=fm.dur[:T, tou].astype(np.float64), occ=fm.occ[:T, tou], vol=fm.vol[:T, tou],
               zone_ids=fm.zone_ids, zone_P=fm.zone_P[:T])
invR = inv[:T][:, tou]
S_zoneR = S_zone[:T]
del fm, inv, S_zone
gc.collect()
Ns = len(tou)
piles = R.st_piles.astype(np.float64)
print(f"载入完成 {time.time() - t0:.0f}s；分时站点 {Ns}，训练期 {T:,} 个 5 分钟点", flush=True)

# ---------------------------------------------------------------- 面板（与正式估计同一样本）：每个窗口建一次
panels = {}
for W in windows:
    spec = s5.Spec5(window=W, donut=0, outcome="duration", jump_threshold=cc.jump_threshold,
                    exclude_holiday_pm1=cc.exclude_holiday_pm1)
    pn, _ = s5.build_panel_5min(R, gz, zpos, S_zoneR, invR, T_HI, spec, cfg.data.context.day_start,
                                cfg.data.context.night_start)
    panels[W] = pn[["station", "zone", "t", "x", "dh", "zh", "Apre", "Apost"]].copy()
    print(f"[面板] W={W * 5}min 行数 {len(pn):,}  事件 {(pn.x != 0).sum():,}  {time.time() - t0:.0f}s", flush=True)
    del pn
    gc.collect()


def estimate_all(Y: np.ndarray, label: str, extra: dict) -> list:
    """Y [T, Ns]（桩·小时 / 5 分钟）。按各窗口面板重算前后窗口计数，估计设计 C、D。"""
    Yc = np.where(invR, 0.0, np.clip(Y, 0, None))
    cy = np.vstack([np.zeros((1, Ns)), np.cumsum(Yc, axis=0)])
    rows = []
    for W, pn in panels.items():
        j, t = pn.station.to_numpy(), pn.t.to_numpy()
        d = pn.assign(Apre=cy[t, j] - cy[t - W, j], Apost=cy[t + W, j] - cy[t, j])
        for dsg in "CD":
            r = s5.ppml_pair(d, ["x"], DESIGNS[dsg]["fe"])
            rows.append({**extra, "outcome": label, "W_min": W * 5, "design": dsg, "beta_hat": r["coef"]["x"],
                         "se": r["se"]["x"], "n": r["n"]})
        z = d[d.x == 0]
        jr = np.log((z.Apost + 0.1) / (z.Apre + 0.1))
        rows.append({**extra, "outcome": label, "W_min": W * 5, "design": "sdJ_nonevent", "beta_hat": float(jr.std()),
                     "se": np.nan, "n": len(z)})
    return rows


rows = []
# 真实数据（核对：应与 spec_curve.csv 的 C、D 一致）
rows += estimate_all(R.dur, "real", {"beta_true": np.nan, "mean_min": np.nan, "seed": -1})
pd.DataFrame(rows).to_csv(out_file, index=False)
print(pd.DataFrame(rows).pivot_table(index="W_min", columns="design", values="beta_hat").round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 合成需求
tix = pd.DatetimeIndex(R.time)
slot = ((tix.hour * 60 + tix.minute) // 5).to_numpy()
wkend = np.asarray(tix.dayofweek >= 5)
dayi = np.asarray((tix.normalize() - tix.normalize()[0]).days)
hour_i = dayi * 24 + tix.hour.to_numpy()
lp = np.log(R.P)
lp_c = lp - lp.mean(axis=0, keepdims=True)
# 基线存量曲线：站点 × 周末 × 时刻的有效点平均（桩·小时 / 5 分钟）→ 在充会话数
valid = ~invR
S0 = np.zeros((T, Ns))
for w in (False, True):
    m_t = wkend == w
    Yw = np.where(valid[m_t], R.dur[m_t], np.nan)
    prof = pd.DataFrame(Yw).groupby(slot[m_t]).mean().reindex(range(288)).to_numpy()
    prof = np.nan_to_num(prof, nan=np.nanmean(prof))
    S0[m_t] = prof[slot[m_t]] * 12.0
S0 = np.clip(S0, 0.02, None)
n_day, n_hour = dayi.max() + 1, hour_i.max() + 1


def session_survival(mean_min: float, sd_log: float = 0.6) -> np.ndarray:
    mu = np.log(mean_min / 5.0) - sd_log ** 2 / 2               # 以 5 分钟为单位
    k = np.arange(0, int(np.ceil(np.exp(mu + 4 * sd_log))) + 1)
    from scipy.stats import lognorm
    surv = lognorm.sf(k, s=sd_log, scale=np.exp(mu))           # P(D > k)：到达当步 k=0 计入
    return surv


def stock_from_arrivals(A: np.ndarray, surv: np.ndarray, cap: np.ndarray | None) -> np.ndarray:
    S = fftconvolve(A, surv[:, None], axes=0)[: A.shape[0]]
    S = np.clip(S, 0, None)
    return np.minimum(S, cap[None, :]) if cap is not None else S


for m in means:
    surv = session_survival(m)
    lam0 = S0 / surv.sum()                                     # 稳态：存量 = 到达率 × 平均时长（步）
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 * m + seed)
        fd = np.exp(rng.normal(0, a.sd_day, (n_day, Ns)) - a.sd_day ** 2 / 2)
        fh = np.exp(rng.normal(0, a.sd_hour, (n_hour, Ns)) - a.sd_hour ** 2 / 2)
        noise = fd[dayi] * fh[hour_i]
        for b in betas:
            if (m, seed, b) in {(r["mean_min"], r["seed"], r["beta_true"]) for r in rows if r["outcome"] == "stock"}:
                continue
            resp = np.exp(b * lp_c)
            lam = lam0 * noise * resp
            A = rng.poisson(lam).astype(np.float64)
            stock = stock_from_arrivals(A, surv, None if a.no_cap else piles)
            ex = {"beta_true": b, "mean_min": m, "seed": seed}
            rows += estimate_all(stock / 12.0, "stock", ex)
            rows += estimate_all(A * surv.sum() / 12.0, "flow", ex)          # 到达数换算到同量级（只影响伪计数的相对大小）
            if seed == 0:
                truth = stock_from_arrivals(lam0 * resp, surv, None)
                rows += estimate_all(truth / 12.0, "truth", ex)
            pd.DataFrame(rows).to_csv(out_file, index=False)
            print(f"[合成] m={m}min seed={seed} β={b:+.1f}  {time.time() - t0:.0f}s", flush=True)
            del A, stock, lam, resp
            gc.collect()

df = pd.DataFrame(rows)
df.to_csv(out_file, index=False)
summ = df[df.design.isin(["C", "D"])].groupby(["outcome", "mean_min", "beta_true", "design", "W_min"]).beta_hat.mean()
print(summ.unstack("W_min").round(3).to_string())
print(f"全部完成 {time.time() - t0:.0f}s")
