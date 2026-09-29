import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
occ, dur, pe, ps = [load(n) for n in ["occupancy","duration","e_price","s_price"]]
p = pe + ps
cap = pd.read_csv("zone_capacity.csv", index_col=0).iloc[:, 0]; cap.index = cap.index.astype(int)
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
strict = frac[frac >= .5].index
pre = dur["2022-10-12":"2022-10-25"].mean(); mid = dur["2022-10-27":"2022-11-01"].mean(); post = dur["2022-11-05":"2022-11-18"].mean()
r = (mid / ((pre + post) / 2)).replace([np.inf], np.nan)
print("per-zone ratio (10-27..11-01) / avg(pre,post): quantiles", r.quantile([0, .05, .1, .25, .5, .75, .9, 1]).round(2).to_dict())
print("zones with ratio < 0.2:", int((r < 0.2).sum()), "| 0.2-0.8:", int(((r >= .2) & (r < .8)).sum()), "| >=0.8:", int((r >= .8).sum()))
lowz = r[r < 0.2].index
print("share of city duration (pre) in collapsed zones: %.2f" % (pre[lowz].sum() / pre.sum()))
print("collapsed zones that are strict TOU: %d of %d" % (len(set(lowz) & set(strict)), len(lowz)))
z = lowz[0]; print("example zone", z, "cap", cap[z]); print(dur.loc["2022-10-25":"2022-10-27", z].round(2).values.reshape(-1, 24)[:, ::2].tolist())
print("occupancy same zone:", occ.loc["2022-10-25":"2022-10-27", z].values.reshape(-1, 24)[:, ::2].tolist())
print("price same zone 10-25..10-27:", p.loc["2022-10-25":"2022-10-27", z].round(3).values.reshape(-1, 24)[:, ::3].tolist())
# fractional-value runs (real data rarely repeat exact fractional values >= 6h)
def frun(a):
    best = cur = 1
    for i in range(1, len(a)):
        same = a[i] == a[i-1] and a[i] != 0 and abs(a[i] - round(a[i])) > 1e-6
        cur = cur + 1 if same else 1; best = max(best, cur)
    return best
days = dur.index.normalize().unique()
fr = np.array([[frun(dur[z].values[k*24:(k+1)*24]) for z in dur.columns] for k in range(len(days))])
share = (fr >= 6).mean(1); s = pd.Series(share, index=days)
print("zone-days with >=6h run of identical FRACTIONAL duration: overall %.3f" % (fr >= 6).mean())
print("top dates:", {str(k.date()): round(v, 2) for k, v in s.sort_values(ascending=False).head(12).items()})
# hour-level outage mask: city total below 60% of same-hour median of +-7 days
tot = dur.sum(axis=1); hh = tot.index.hour
med = pd.Series(index=tot.index, dtype=float)
for h in range(24):
    x = tot[hh == h]; med[x.index] = x.rolling(15, center=True, min_periods=5).median()
bad = tot < 0.6 * med
print("hours flagged (city total < 60% of 15-day same-hour median):", int(bad.sum()))
bd = bad[bad].index; print("flagged dates:", pd.Series(bd.normalize()).value_counts().sort_index().to_dict())
bad.rename("outage_flag").astype(int).to_csv("hour_outage_flag.csv")
