import numpy as np, pandas as pd
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); return df.set_index("time")
occ, dur, vol, v11, pe, ps = [load(n) for n in ["occupancy","duration","volume","volume-11kW","e_price","s_price"]]
print("shapes", occ.shape, dur.shape, pe.shape, ps.shape, "same cols:", all((x.columns == occ.columns).all() for x in [dur,vol,v11,pe,ps]))
print("time", occ.index.min(), occ.index.max(), "freq ok:", (occ.index.to_series().diff().dropna() == pd.Timedelta("1h")).all())
print("NaN counts", {n: int(x.isna().sum().sum()) for n, x in zip(["occ","dur","vol","v11","pe","ps"], [occ,dur,vol,v11,pe,ps])})
# ---- 9.4 target distributions
for n, x in [("occupancy", occ), ("duration", dur), ("volume", vol), ("volume-11kW", v11)]:
    a = x.values
    print(f"{n:12s} min {a.min():.3f} max {a.max():.3f} mean {a.mean():.3f} zero {np.mean(a==0):.3f}", end="")
    if n == "occupancy": print(f" >0.9 {np.mean(a>0.9):.4f} >0.95 {np.mean(a>0.95):.4f} ==1 {np.mean(a>=0.999):.4f}")
    else: print()
print("zones with >20% zero occupancy:", int(((occ==0).mean() > 0.2).sum()), " >20% zero duration:", int(((dur==0).mean() > 0.2).sum()))
# ---- 9.7 timing alignment: duration of hour t vs occupancy snapshot at t, t+1, t-1
def mean_corr(a, b):
    cs = [np.corrcoef(a[c], b[c])[0,1] for c in a.columns if a[c].std() > 0 and b[c].std() > 0]
    return np.nanmean(cs)
print("corr(dur_t, occ_t) %.3f  corr(dur_t, occ_t+1) %.3f  corr(dur_t, occ_t-1) %.3f" % (
    mean_corr(dur.iloc[1:-1], occ.iloc[1:-1]), mean_corr(dur.iloc[1:-1], occ.shift(-1).iloc[1:-1]), mean_corr(dur.iloc[1:-1], occ.shift(1).iloc[1:-1])))
print("corr(dur,vol) %.3f corr(vol,v11) %.3f" % (mean_corr(dur, vol), mean_corr(vol, v11)))
# ---- prices
p = pe + ps; lp = np.log(p)
day = occ.index.normalize(); hour = occ.index.hour
wkend = occ.index.dayofweek >= 5
def within_day_var(x): return (x.groupby(day).nunique() > 1)
tou_e = within_day_var(pe.round(4)).any(); tou_s = within_day_var(ps.round(4)).any(); tou_p = within_day_var(p.round(4)).any()
print("zones with within-day variation: e_price %d, s_price %d, total %d; e-only %d, s-only %d, both %d" % (
    tou_e.sum(), tou_s.sum(), tou_p.sum(), (tou_e & ~tou_s).sum(), (~tou_e & tou_s).sum(), (tou_e & tou_s).sum()))
share_days = within_day_var(p.round(4)).mean()
print("TOU zones: share of days with within-day variation — quantiles", np.round(share_days[tou_p].quantile([0,.1,.25,.5,.75,.9,1]).values, 2))
tou = tou_p[tou_p].index; fixed = tou_p[~tou_p].index
# fixed zones: do their levels change across days?
lvl_change = (p[fixed].round(4).nunique() > 1)
print("fixed zones whose (constant-within-day) level changes across days:", int(lvl_change.sum()), "of", len(fixed))
# R2 of log price on hour dummies and hour x daytype within TOU zones
def r2(y, groups):
    fit = y.groupby(groups).transform("mean"); return 1 - ((y-fit)**2).sum() / ((y-y.mean())**2).sum()
r2h, r2hd, r2hm = [], [], []
month = occ.index.month
for z in tou:
    y = lp[z]
    if y.std() == 0: continue
    r2h.append(r2(y, hour)); r2hd.append(r2(y, [hour, wkend])); r2hm.append(r2(y, [hour, month]))
print("TOU zones R2(log p ~ hour): median %.3f, p10 %.3f, p90 %.3f" % (np.median(r2h), np.quantile(r2h,.1), np.quantile(r2h,.9)))
print("TOU zones R2(log p ~ hour x weekend): median %.3f ; R2(~ hour x month): median %.3f" % (np.median(r2hd), np.median(r2hm)))
print("share of TOU zones with R2(hour) > 0.9: %.2f, > 0.95: %.2f" % (np.mean(np.array(r2h)>0.9), np.mean(np.array(r2h)>0.95)))
# modal daily profile stability
stab = []
for z in tou:
    prof = p[z].round(3).groupby(day).apply(lambda s: tuple(s.values))
    stab.append(prof.value_counts(normalize=True).iloc[0])
print("TOU zones: share of days equal to the modal 24h profile — median %.2f, p10 %.2f, p90 %.2f" % (np.median(stab), np.quantile(stab,.1), np.quantile(stab,.9)))
# switch events
dl = lp[tou].diff()
ev = dl.stack(); ev = ev[ev.abs() > 0.01]   # 修正：pandas 3 的 stack() 保留 NaN，必须先 stack 再筛选
print("switch events |dlog p|>1%%: %d total, %.1f per TOU zone-day; up %d, down %d" % (len(ev), len(ev)/ (len(tou)*181), (ev>0).sum(), (ev<0).sum()))
print("|dlog p| quantiles:", np.round(ev.abs().quantile([.1,.25,.5,.75,.9]).values, 3))
big = ev[ev.abs() > 0.05]
print("events with |dlog p|>5%%: %d (%.0f%%)" % (len(big), 100*len(big)/len(ev)))
hrs = pd.Series(big.index.get_level_values(0).hour).value_counts().sort_index()
print("big events by clock hour:", dict(hrs))
sign_by_hour = big.groupby(big.index.get_level_values(0).hour).apply(lambda s: f"{(s>0).sum()}up/{(s<0).sum()}dn")
print("direction by hour:", dict(sign_by_hour))
# typical schedule: mean log price deviation by hour across TOU zones
dev = (lp[tou] - lp[tou].mean()).groupby(hour).mean().mean(axis=1)
print("mean dlog p by hour (TOU zones):", " ".join(f"{h}:{v:+.2f}" for h, v in dev.items()))
# s_price sync
dls = np.log(ps[tou].clip(lower=1e-6)).diff(); dle = np.log(pe[tou]).diff()
both = ((dle.abs() > 0.01) & (dls.abs() > 0.01)).sum().sum(); eo = ((dle.abs() > 0.01) & (dls.abs() <= 0.01)).sum().sum(); so = ((dle.abs() <= 0.01) & (dls.abs() > 0.01)).sum().sum()
print("switch composition in TOU zones: e only %d, s only %d, both %d" % (eo, so, both))
# price level summary
print("total price mean %.3f (TOU %.3f, fixed %.3f); e %.3f s %.3f" % (p.values.mean(), p[tou].values.mean(), p[fixed].values.mean(), pe.values.mean(), ps.values.mean()))
pd.Series(tou, name="zone").to_csv("tou_zones.csv", index=False)
pd.Series(fixed, name="zone").to_csv("fixed_zones.csv", index=False)
