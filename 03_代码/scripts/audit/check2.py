import numpy as np, pandas as pd, json
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
OUT = {}
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
occ, dur, vol, pe, ps = [load(n) for n in ["occupancy","duration","volume","e_price","s_price"]]
Z = list(occ.columns)
inf = pd.read_csv(f"{D}/inf.csv"); raw = pd.read_csv(f"{D}/inf_raw.csv")

print("=== B. capacity & units ===")
cap = inf.groupby("TAZID").charge_count.sum().reindex(Z); capr = raw.groupby("TAZID").charge_count.sum().reindex(Z)
nst = inf.groupby("TAZID").size().reindex(Z)
print("piles total inf", cap.sum(), "raw", capr.sum(), "| stations/zone median", nst.median(), "min", nst.min(), "max", nst.max())
print("capacity/zone: median", cap.median(), "p10", cap.quantile(.1), "p90", cap.quantile(.9), "min", cap.min(), "max", cap.max())
r_occ = occ.max() / cap; r_dur = dur.max() / cap
print("max(occ)/cap: median %.2f p90 %.2f max %.2f  share>1: %.3f" % (r_occ.median(), r_occ.quantile(.9), r_occ.max(), (r_occ > 1.0001).mean()))
print("max(dur)/cap: median %.2f p90 %.2f max %.2f  share>1: %.3f" % (r_dur.median(), r_dur.quantile(.9), r_dur.max(), (r_dur > 1.0001).mean()))
print("max(occ)/cap_raw share>1: %.3f ; max(occ)<=100 share: %.3f" % (((occ.max()/capr) > 1.0001).mean(), (occ.max() <= 100).mean()))
rate = occ / cap; urate = dur / cap
print("occ rate (occ/cap): mean %.3f median %.3f p99 %.3f" % (rate.stack().mean(), rate.stack().median(), rate.stack().quantile(.99)))
print("util rate (dur/cap): mean %.3f median %.3f p99 %.3f" % (urate.stack().mean(), urate.stack().median(), urate.stack().quantile(.99)))
print("dur <= occ share: %.3f" % ((dur <= occ + 1e-9).stack().mean()))
kw = (vol.sum() / dur.sum()); print("implied mean power kW/pile-hour: median %.1f p10 %.1f p90 %.1f" % (kw.median(), kw.quantile(.1), kw.quantile(.9)))
OUT["capacity"] = {"piles_inf": int(cap.sum()), "piles_raw": int(capr.sum()), "cap_median": float(cap.median()),
                   "occ_over_cap_share": float((r_occ > 1.0001).mean()), "dur_over_cap_share": float((r_dur > 1.0001).mean()),
                   "occ_rate_mean": float(rate.stack().mean()), "util_rate_mean": float(urate.stack().mean())}
cap.rename("capacity").to_csv("zone_capacity.csv")

print("\n=== A. TOU definition & switch events (fixed count) ===")
p = pe + ps; lp = np.log(p); day = occ.index.normalize()
var_day = (p.round(4).groupby(day).nunique() > 1)          # days x zones
frac = var_day.mean()
anyt = frac[frac > 0].index
print("zones with any within-day variation:", len(anyt))
print("share-of-days distribution among them:", np.round(np.quantile(frac[anyt], [0, .1, .25, .5, .75, .9, 1]), 3).tolist())
for thr in [0.05, 0.25, 0.5, 0.75, 0.9]:
    print(f"  TOU if share>={thr}: {(frac >= thr).sum()}")
strict = frac[frac >= 0.5].index; weak = frac[(frac > 0) & (frac < 0.5)].index; fixed = frac[frac == 0].index
print("strict TOU", len(strict), "weak", len(weak), "fixed", len(fixed))
# between-day level changes for fixed zones (daily mean price)
dm = p.groupby(day).mean()
print("fixed zones with any day-to-day level change > 0.1%:", int((dm[fixed].pct_change().abs() > 1e-3).any().sum()))
dl = lp.diff()
s = dl[strict].stack(); s = s[s.abs() > 0.01]
nzd = len(strict) * day.nunique()
print("events |dlog p|>1%% in strict TOU: %d (up %d, down %d) = %.2f per zone-day" % (len(s), (s > 0).sum(), (s < 0).sum(), len(s) / nzd))
big = s[s.abs() > 0.05]; print("big events >5%%: %d (%.0f%% of events)" % (len(big), 100 * len(big) / len(s)))
print("abs dlog p quantiles of events:", np.round(s.abs().quantile([.1, .25, .5, .75, .9]).values, 3).tolist())
hrs = big.index.get_level_values(0).hour
tab = pd.crosstab(hrs, np.sign(big.values)); tab.columns = ["down", "up"]
print("big events by clock hour:\n", tab.T.to_string())
# clean windows: price constant for >=2h before and >=2h after the switch
clean = 0; clean3 = 0
for z in strict:
    x = lp[z].values; d = np.diff(x)
    for t in np.where(np.abs(d) > 0.05)[0] + 1:   # switch between t-1 and t
        if t >= 3 and t + 2 < len(x):
            pre = np.abs(np.diff(x[t-3:t])).max() < 1e-4; post = np.abs(np.diff(x[t:t+3])).max() < 1e-4
            clean += pre and post
        if t >= 4 and t + 3 < len(x):
            clean3 += (np.abs(np.diff(x[t-4:t])).max() < 1e-4) and (np.abs(np.diff(x[t:t+4])).max() < 1e-4)
print("big events with >=3h flat price each side:", clean, "| >=4h:", clean3)
OUT["tou"] = {"any": int(len(anyt)), "strict_ge50": int(len(strict)), "weak": int(len(weak)), "fixed": int(len(fixed)),
              "events_1pct": int(len(s)), "events_per_zone_day": float(len(s) / nzd), "big_5pct": int(len(big)),
              "big_clean_3h": int(clean), "big_clean_4h": int(clean3)}
pd.Series(frac, name="share_days_tou").to_csv("zone_tou_share.csv")

print("\n=== C. distances & rings ===")
dist = pd.read_csv(f"{D}/distance.csv"); dist.columns = dist.columns.astype(int); dist.index = dist.columns
adj = pd.read_csv(f"{D}/adj.csv"); adj.columns = adj.columns.astype(int); adj.index = adj.columns
assert list(dist.columns) == Z and list(adj.columns) == Z
Dm = dist.values / 1000.0; np.fill_diagonal(Dm, np.inf)
nn = Dm.min(1); print("nearest-neighbour km: median %.2f p10 %.2f p90 %.2f max %.2f" % (np.median(nn), np.quantile(nn, .1), np.quantile(nn, .9), nn.max()))
A = adj.values.copy(); np.fill_diagonal(A, 0); deg = A.sum(1)
print("adjacency: symmetric", bool((A == A.T).all()), "degree median", np.median(deg), "min", deg.min(), "max", deg.max(), "isolated", int((deg == 0).sum()))
nd = Dm[A == 1]; print("1-hop neighbour distance km: median %.2f p90 %.2f max %.2f" % (np.median(nd), np.quantile(nd, .9), nd.max()))
rings = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 10)]
for lo, hi in rings:
    c = ((Dm >= lo) & (Dm < hi)).sum(1)
    print(f"  zones in ring [{lo},{hi}) km: median {np.median(c):.0f} p10 {np.quantile(c,.1):.0f} p90 {np.quantile(c,.9):.0f}; share with 0: {(c==0).mean():.2f}")
idx = {z: i for i, z in enumerate(Z)}
si = [idx[z] for z in strict]; fi = [idx[z] for z in fixed]
d_f2t = Dm[np.ix_(fi, si)].min(1)
print("fixed -> nearest strict-TOU km: median %.2f p10 %.2f p90 %.2f" % (np.median(d_f2t), np.quantile(d_f2t, .1), np.quantile(d_f2t, .9)))
for k in [1, 2, 4, 6]:
    print(f"  fixed zones with a strict-TOU zone within {k} km: {(d_f2t < k).sum()} / {len(fi)}")
d_t2t = Dm[np.ix_(si, si)].min(1); print("strict-TOU -> nearest strict-TOU km: median %.2f" % np.median(d_t2t))
d_t2f = Dm[np.ix_(si, fi)].min(1); print("strict-TOU -> nearest fixed km: median %.2f p90 %.2f" % (np.median(d_t2f), np.quantile(d_t2f, .9)))
OUT["dist"] = {"nn_median_km": float(np.median(nn)), "deg_median": float(np.median(deg)),
               "fixed_within_2km_of_tou": int((d_f2t < 2).sum()), "fixed_beyond_4km": int((d_f2t >= 4).sum()),
               "fixed_beyond_6km": int((d_f2t >= 6).sum())}
pd.DataFrame({"zone": [Z[i] for i in fi], "km_to_nearest_tou": d_f2t}).to_csv("fixed_to_tou_km.csv", index=False)

print("\n=== D. split, holidays, dips ===")
days = pd.Series(sorted(day.unique())); nd_ = len(days)
ntr = int(round(nd_ * 0.7)); nva = int(round(nd_ * 0.1)); nte = nd_ - ntr - nva
print("days", nd_, "-> train", ntr, days[0].date(), "~", days[ntr-1].date(), "| val", nva, days[ntr].date(), "~", days[ntr+nva-1].date(), "| test", nte, days[ntr+nva].date(), "~", days[nd_-1].date())
hol = {"中秋": ("2022-09-10", "2022-09-12"), "国庆": ("2022-10-01", "2022-10-07"), "元旦": ("2022-12-31", "2023-01-02"), "春节": ("2023-01-21", "2023-01-27")}
def seg(d):
    d = pd.Timestamp(d); return "train" if d <= days[ntr-1] else ("val" if d <= days[ntr+nva-1] else "test")
for k, (a, b) in hol.items(): print(f"  {k} {a}~{b}: {seg(a)} ~ {seg(b)}")
dd = dur.sum(1).groupby(day).sum(); do = occ.sum(1).groupby(day).sum()
wk = pd.Series(pd.to_datetime(dd.index).dayofweek, index=dd.index)
base = dd.rolling(15, center=True, min_periods=5).median(); rel = dd / base - 1
print("daily total duration: mean %.0f; weekend/weekday ratio %.3f" % (dd.mean(), dd[wk >= 5].mean() / dd[wk < 5].mean()))
low = rel[rel < -0.12]
print("days >12% below 15-day rolling median (duration):"); print("  ", [(str(i.date()), round(v, 3)) for i, v in low.items()])
mo = dd.groupby(pd.to_datetime(dd.index).to_period("M")).mean(); print("monthly mean daily duration:", {str(k): int(v) for k, v in mo.items()})
wkly = dd.resample("W").mean(); print("weekly mean daily duration (first 10 and around Dec-Jan):")
print("  ", {str(k.date()): int(v) for k, v in wkly.items()})
OUT["split"] = {"train": [str(days[0].date()), str(days[ntr-1].date())], "val": [str(days[ntr].date()), str(days[ntr+nva-1].date())],
                "test": [str(days[ntr+nva].date()), str(days[nd_-1].date())]}
dd.rename("daily_duration").to_csv("daily_duration.csv")
json.dump(OUT, open("check2_summary.json", "w"), ensure_ascii=False, indent=1)
