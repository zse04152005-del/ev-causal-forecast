"""阶段 5.2 探索性分析（EDA）：周期性、节假日、价格结构、价格与需求的朴素相关、切换前后曲线、天气、空间自相关。

只做描述与朴素统计（不是因果估计）；所有"相关"都在训练期上算，避免测试期信息进入模型选择。
输出：05_实验结果/EDA/ 下的 CSV 与 eda_stats.json（绘图脚本 04_图表/绘图脚本/fig_eda.py 读取）

用法：python scripts/eda.py [--d1 <UrbanEV/data>] [--perm 999]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.causal.switch_did import fe_ols
from src.data import calendar as cal
from src.data.dataset import prepare
from src.data.prices import PRICING_CODES
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--d1", default=None)
ap.add_argument("--perm", type=int, default=999, help="Moran's I 置换检验次数")
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides) + ([f"paths.data_dir={a.d1}"] if a.d1 else []))
root = project_root(cfg.paths.get("root"))
out = ensure_dir(os.path.join(resolve(cfg.paths.results_dir, root), "EDA"))
t0 = time.time()
P = prepare(cfg)
T, N = P.y.shape
tr = P.split.train_end
time_idx = pd.DatetimeIndex(P.time)
hour = time_idx.hour.to_numpy()
date = time_idx.normalize()
Y = np.where(P.valid, P.y.astype(np.float64), np.nan)                 # 利用率，冻结点记为缺失
hol = cal.is_holiday(time_idx)
rest = cal.is_restday(time_idx)
daytype = np.where(hol, "holiday", np.where(rest, "weekend", "weekday"))
tou = P.pricing == PRICING_CODES["TOU"]
fixed = P.pricing == PRICING_CODES["fixed"]
price = np.exp(P.lp)                                                   # 总价（元 / kWh）
stats = {"n_zones": int(N), "n_hours": int(T), "n_tou": int(tou.sum()), "n_fixed": int(fixed.sum()),
         "valid_share": float(P.valid.mean()), "train_end": str(time_idx[tr]),
         "val_end": str(time_idx[P.split.val_end])}
print(f"载入 {time.time() - t0:.0f}s；{N} 个小区 × {T} 小时；有效点 {P.valid.mean():.3f}", flush=True)

# ---------------------------------------------------------------- 1. 日内与周内周期
city = np.nanmean(Y, axis=1)                                           # 每小时全市平均利用率（有效小区）
df_h = pd.DataFrame({"date": date, "hour": hour, "daytype": daytype, "u": city})
d = df_h.groupby(["daytype", "hour"]).u.agg(mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                                             n="count").reset_index()
d.to_csv(os.path.join(out, "diurnal.csv"), index=False)
dow = df_h.assign(dow=time_idx.dayofweek).query("daytype != 'holiday'").groupby(["dow", "hour"]).u.mean().unstack()
dow.to_csv(os.path.join(out, "weekly_profile.csv"))
peak = d[d.daytype == "weekday"].sort_values("mean")
stats["diurnal"] = {"weekday_peak_hour": int(peak.hour.iloc[-1]), "weekday_peak": float(peak["mean"].iloc[-1]),
                    "weekday_trough_hour": int(peak.hour.iloc[0]), "weekday_trough": float(peak["mean"].iloc[0]),
                    "weekend_over_weekday": float(d[d.daytype == "weekend"]["mean"].mean() / d[d.daytype == "weekday"]["mean"].mean()),
                    "holiday_over_weekday": float(d[d.daytype == "holiday"]["mean"].mean() / d[d.daytype == "weekday"]["mean"].mean())}
# 自相关：24、168 小时
cs = pd.Series(city)
stats["acf"] = {f"lag{k}": float(cs.autocorr(k)) for k in (1, 24, 168)}

# ---------------------------------------------------------------- 2. 逐日序列（节假日、切分）
day = pd.DataFrame({"date": date, "u": city, "frozen": P.frozen.mean(axis=1), "hol": hol, "rest": rest}).groupby("date").agg(
    u=("u", "mean"), frozen=("frozen", "mean"), holiday=("hol", "max"), restday=("rest", "max")).reset_index()
day.to_csv(os.path.join(out, "daily.csv"), index=False)
sf = day[(day.date >= "2023-01-21") & (day.date <= "2023-01-27")].u.mean()
pre = day[(day.date >= "2023-01-07") & (day.date <= "2023-01-13")].u.mean()
stats["spring_festival"] = {"mean_util": float(sf), "two_weeks_before": float(pre), "ratio": float(sf / pre)}

# ---------------------------------------------------------------- 3. 天气（去掉 时刻 × 日类型 的平均后）
w = pd.read_csv(os.path.join(resolve(cfg.paths.data_dir, root), "weather_central.csv"))
w["time"] = pd.to_datetime(w["time"])
w = w.set_index("time").reindex(time_idx)
key = pd.Series(list(zip(hour, daytype)))
resid = city - pd.Series(city).groupby(key).transform("mean").to_numpy()
wx = pd.DataFrame({"T": w["T"].to_numpy(), "rain": w["nRAIN"].to_numpy(), "U": w["U"].to_numpy(), "resid": resid,
                   "u": city, "train": np.arange(T) <= tr})
wt = wx[wx.train].dropna()
bins = np.arange(np.floor(wt["T"].min() / 2) * 2, wt["T"].max() + 2, 2)
wt = wt.assign(Tbin=pd.cut(wt["T"], bins, right=False))
tb = wt.groupby("Tbin", observed=True).resid.agg(["mean", "std", "count"]).reset_index()
tb["T_mid"] = [iv.left + 1 for iv in tb.Tbin]
tb["se"] = tb["std"] / np.sqrt(tb["count"])
tb[["T_mid", "mean", "se", "count"]].to_csv(os.path.join(out, "weather_temperature.csv"), index=False)
rain = wt.rain > 0
X = np.column_stack([np.ones(len(wt)), wt["T"], wt["T"] ** 2, rain.astype(float), wt["U"]])
beta_w = np.linalg.lstsq(X, wt.resid.to_numpy(), rcond=None)[0]
stats["weather"] = {"rain_hours_share": float(rain.mean()),
                    "rain_minus_dry_resid": float(wt.resid[rain].mean() - wt.resid[~rain].mean()),
                    "corr_T_resid": float(np.corrcoef(wt["T"], wt.resid)[0, 1]),
                    "ols_resid_on_T_T2_rain_RH": dict(zip(["const", "T", "T2", "rain", "RH"], map(float, beta_w))),
                    "resid_sd": float(wt.resid.std()), "note": "残差 = 全市小时利用率 − 同一 时刻×日类型 的平均"}

# ---------------------------------------------------------------- 4. 价格结构
rows = []
for name, m in (("TOU", tou), ("fixed", fixed)):
    hh = np.repeat(hour[: tr + 1], m.sum())
    s = pd.DataFrame({"hour": hh, "p": price[: tr + 1][:, m].reshape(-1)})
    g = s.groupby("hour").p.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    g.columns = ["p10", "p25", "p50", "p75", "p90"]
    g["type"] = name
    rows.append(g.reset_index())
pbh = pd.concat(rows)
pbh.to_csv(os.path.join(out, "price_by_hour.csv"), index=False)
ep, sp = P.raw["e_price"], P.raw["s_price"]
comp = pd.DataFrame({"hour": hour[: tr + 1], "e": np.nanmean(ep[: tr + 1][:, tou], axis=1),
                     "s": np.nanmean(sp[: tr + 1][:, tou], axis=1)}).groupby("hour").mean()
comp.to_csv(os.path.join(out, "price_components_tou.csv"))
# 日内结构：各分时小区对数价格减去"该小区当天的平均"（消掉小区之间的水平差异）
dev = {}
for nm, arr in (("total", price), ("electricity", ep), ("service", sp)):
    lv = np.log(np.clip(arr[: tr + 1][:, tou].astype(np.float64), 1e-6, None))
    dm = pd.DataFrame(lv).groupby(date[: tr + 1]).transform("mean").to_numpy()
    dev[nm] = pd.DataFrame({"hour": np.repeat(hour[: tr + 1], tou.sum()), "d": (lv - dm).reshape(-1)})
dv = dev["total"].groupby("hour").d.quantile([0.25, 0.5, 0.75]).unstack()
dv.columns = ["q25", "median", "q75"]
dv["electricity_mean"] = dev["electricity"].groupby("hour").d.mean()
dv["service_mean"] = dev["service"].groupby("hour").d.mean()
dv["total_mean"] = dev["total"].groupby("hour").d.mean()
dv.to_csv(os.path.join(out, "price_dev_by_hour.csv"))
tou_p = price[: tr + 1][:, tou]
dayp = pd.DataFrame(tou_p).groupby(date[: tr + 1]).agg(lambda s: np.log(s.max() / s.min()))
stats["price"] = {"tou_mean": float(tou_p.mean()), "fixed_mean": float(price[: tr + 1][:, fixed].mean()),
                  "tou_within_day_log_range_median": float(np.nanmedian(dayp.to_numpy())),
                  "electricity_hourly_range": [float(comp.e.min()), float(comp.e.max())],
                  "service_hourly_range": [float(comp.s.min()), float(comp.s.max())],
                  "corr_e_s_hourly_tou": float(np.corrcoef(comp.e, comp.s)[0, 1])}

# ---------------------------------------------------------------- 5. 价格与需求的朴素相关（训练期，分时 + 固定小区）
use = tou | fixed
tt, zz = np.nonzero(P.valid[: tr + 1][:, use])
zi = np.flatnonzero(use)[zz]
panel = pd.DataFrame({"zone": zi, "t": tt, "y": np.log(P.y[tt, zi].astype(np.float64) + 0.01), "x": P.lp[tt, zi],
                      "hour": hour[tt], "day": (date[tt] - date[0]).days})
panel["zh"] = panel.zone * 24 + panel.hour
panel["dh"] = panel.day * 24 + panel.hour
specs = [("pooled", []), ("hour", ["hour"]), ("zone", ["zone"]), ("zone+hour", ["zone", "hour"]),
         ("zone×hour + date×hour", ["zh", "dh"])]
naive = []
for nm, fe in specs:
    r = fe_ols(panel, "y", ["x"], fe) if fe else fe_ols(panel.assign(_c=0), "y", ["x"], ["_c"])
    naive.append({"spec": nm, "beta": r["coef"]["x"], "se": r["se"]["x"], "n": r["n"],
                  "price_var_left": r["resid_var_share_x0"]})
    print(f"[朴素] {nm:<24} β = {r['coef']['x']:+.3f} ({r['se']['x']:.3f})  剩余价格变动 {r['resid_var_share_x0']:.3f}", flush=True)
nv = pd.DataFrame(naive)
nv.to_csv(os.path.join(out, "naive_price_demand.csv"), index=False)
zm = panel.groupby("zone").agg(y=("y", "mean"), x=("x", "mean"))
stats["naive"] = {"cross_zone_spearman_meanlogprice_meanlogutil": float(zm.x.rank().corr(zm.y.rank())),
                  "note": "log(利用率 + 0.01) 对 log(总价) 的 OLS；按小区聚类；只是相关，不是因果"}

# ---------------------------------------------------------------- 6. 切换前后曲线（原始数据，分时小区，训练期）
K = 6
lp = P.lp
jump = np.zeros_like(lp)
jump[1:] = lp[1:] - lp[:-1]
flat = np.zeros_like(lp, dtype=bool)
flat[1:] = np.abs(lp[1:] - lp[:-1]) < 1e-4
lr = np.log(np.where(P.valid, P.y.astype(np.float64), np.nan) + 0.01)
fix_lr = np.nanmean(lr[:, fixed], axis=1)                                # 同一时刻固定电价小区的平均（对照）
curves = []
for zj in np.flatnonzero(tou):
    for t in np.flatnonzero(np.abs(jump[: tr + 1, zj]) > 0.01):
        if t - K < 0 or t + K >= tr + 1:
            continue
        if not (flat[t - K + 1: t, zj].all() and flat[t + 1: t + K, zj].all()):
            continue
        seg = lr[t - K: t + K, zj]
        if np.isnan(seg).any():
            continue
        ctrl = fix_lr[t - K: t + K]
        curves.append({"zone": zj, "t": t, "dir": "up" if jump[t, zj] > 0 else "down", "x": jump[t, zj],
                       **{f"k{k}": seg[k + K] - seg[K - 1] for k in range(-K, K)},
                       **{f"c{k}": ctrl[k + K] - ctrl[K - 1] for k in range(-K, K)}})
cv = pd.DataFrame(curves)
rows = []
for dr, g in cv.groupby("dir"):
    for k in range(-K, K):
        a_, c_ = g[f"k{k}"], g[f"c{k}"]
        dd = a_ - c_
        rows.append({"dir": dr, "k": k, "tou": a_.mean(), "tou_se": a_.std() / np.sqrt(len(g)), "fixed": c_.mean(),
                     "diff": dd.mean(), "diff_se": dd.std() / np.sqrt(len(g)), "n_events": len(g),
                     "mean_x": g.x.mean()})
sc = pd.DataFrame(rows)
sc.to_csv(os.path.join(out, "switch_curves.csv"), index=False)
stats["switch_curves"] = {dr: {"n_events": int((cv.dir == dr).sum()), "mean_log_price_jump": float(cv[cv.dir == dr].x.mean()),
                               "diff_k0_to_k2_mean": float(sc[(sc.dir == dr) & (sc.k.between(0, 2))]["diff"].mean())}
                          for dr in ("up", "down")}
stats["switch_curves"]["note"] = "对数利用率相对切换前 1 小时的变化；diff = 分时小区 − 同一时刻固定小区平均；干净窗口 ±6 小时"

# ---------------------------------------------------------------- 7. 空间自相关（Moran's I、LISA）
rng = np.random.default_rng(0)


def row_std(W):
    s = W.sum(1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def moran(z, W):
    z = z - z.mean()
    return float(len(z) / W.sum() * (z @ W @ z) / (z @ z))


def moran_perm(v, Wr, n_perm):
    I0 = moran(v, Wr)
    sims = np.array([moran(rng.permutation(v), Wr) for _ in range(n_perm)])
    return I0, float((np.sum(sims >= I0) + 1) / (n_perm + 1)), float(sims.mean())


zmean = np.nanmean(Y[: tr + 1], axis=0)
ok = np.isfinite(zmean)
A = (P.adj > 0).astype(np.float64)
Wd = ((P.dist_km > 0) & (P.dist_km <= 5)).astype(np.float64)          # 5 km 距离带
moran_rows = []
for wname, W0 in (("adjacency", A), ("distance_5km", Wd)):
    m = ok & (W0.sum(1) > 0)
    Wr = row_std(W0[np.ix_(m, m)])
    I0, p, e = moran_perm(zmean[m], Wr, a.perm)
    moran_rows.append({"weights": wname, "variable": "zone_mean_util_train", "I": I0, "p_perm": p, "E_I_perm": e,
                       "n": int(m.sum())})
    # 逐小时（训练期）
    Is = []
    for t in range(0, tr + 1):
        v = Y[t]
        mm = m & np.isfinite(v)
        if mm.sum() < 30:
            continue
        Wt = row_std(W0[np.ix_(mm, mm)])
        if Wt.sum() == 0 or np.nanstd(v[mm]) == 0:
            continue
        Is.append((t, moran(v[mm], Wt)))
    Is = pd.DataFrame(Is, columns=["t", "I"]).assign(hour=lambda x: hour[x.t])
    moran_rows.append({"weights": wname, "variable": "hourly_util_train", "I": float(Is.I.mean()),
                       "I_q10": float(Is.I.quantile(0.1)), "I_q90": float(Is.I.quantile(0.9)), "n": int(len(Is))})
    Is.groupby("hour").I.mean().rename(f"I_{wname}").to_csv(os.path.join(out, f"moran_by_hour_{wname}.csv"))
mo = pd.DataFrame(moran_rows)
mo.to_csv(os.path.join(out, "moran.csv"), index=False)
# LISA（邻接权重，训练期小区平均利用率）
m = ok & (A.sum(1) > 0)
idx = np.flatnonzero(m)
Wr = row_std(A[np.ix_(m, m)])
z = (zmean[m] - zmean[m].mean()) / zmean[m].std()
lag = Wr @ z
Ii = z * lag
p_loc = np.ones(len(z))
for i in range(len(z)):
    nb = Wr[i] > 0
    k = int(nb.sum())
    others = np.delete(z, i)
    sims = np.array([z[i] * rng.choice(others, k, replace=False).mean() for _ in range(a.perm)])
    p_loc[i] = (np.sum(np.abs(sims) >= abs(Ii[i])) + 1) / (a.perm + 1)
quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
lisa = pd.DataFrame({"zone": P.zones[idx], "mean_util": zmean[m], "z": z, "lag": lag, "I_local": Ii, "p": p_loc,
                     "cluster": np.where(p_loc < 0.05, quad, "ns")})
lisa.to_csv(os.path.join(out, "lisa_adjacency.csv"), index=False)
stats["moran"] = mo.to_dict(orient="records")
stats["lisa_counts"] = lisa.cluster.value_counts().to_dict()
with open(os.path.join(out, "eda_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: stats[k] for k in ("diurnal", "acf", "spring_festival", "weather", "price")}, ensure_ascii=False,
                 indent=1, default=float))
print(mo.round(3).to_string(), "\n", stats["lisa_counts"], "\n", sc[sc.k.isin([-3, -1, 0, 1, 2])].round(3).to_string())
print(f"全部完成 {time.time() - t0:.0f}s；输出 {out}")
