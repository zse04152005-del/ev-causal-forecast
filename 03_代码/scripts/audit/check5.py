# are the day-to-day price changes real or imputation artifacts? detect constant runs (ffill/bfill) in demand and price
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
occ, dur, vol, pe, ps = [load(n) for n in ["occupancy","duration","volume","e_price","s_price"]]
p = (pe + ps).round(4)
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
weak = frac[(frac > 0) & (frac < .5)].index; strict = frac[frac >= .5].index
for z in list(weak[:3]):
    for dstr in ["2022-10-21", "2022-10-22", "2022-10-23", "2022-11-03"]:
        v = p.loc[dstr, z].values
        if np.ptp(v) > 1e-4: print(z, dstr, v.tolist())
# constant-run detector on hourly duration: longest run of identical nonzero values per zone-day
def maxrun(a):
    best = cur = 1
    for i in range(1, len(a)):
        cur = cur + 1 if (a[i] == a[i-1] and a[i] != 0) else 1; best = max(best, cur)
    return best
day = dur.index.normalize(); days = day.unique()
runs = pd.DataFrame({d: [maxrun(dur.loc[str(d.date()), z].values) for z in dur.columns] for d in days}, index=dur.columns).T
share6 = (runs >= 6).mean(1)
print("share of zones with a >=6h run of identical non-zero duration, by date (top 15):")
print({str(k.date()): round(v, 2) for k, v in share6.sort_values(ascending=False).head(15).items()})
print("overall share of zone-days with such runs: %.3f" % (runs >= 6).values.mean())
# same for occupancy
runs_o = pd.DataFrame({d: [maxrun(occ.loc[str(d.date()), z].values) for z in occ.columns] for d in days}, index=occ.columns).T
s6o = (runs_o >= 6).mean(1); print("occupancy >=6h identical runs, top dates:", {str(k.date()): round(v, 2) for k, v in s6o.sort_values(ascending=False).head(10).items()})
print("occupancy overall share: %.3f" % (runs_o >= 6).values.mean())
# citywide hourly total duration for 2022-10-20..11-06 to see outage shape
tot = dur.sum(axis=1)
sub = tot["2022-10-19":"2022-11-06"].values.reshape(-1, 24).round(0)
for d, row in zip(pd.date_range("2022-10-19", "2022-11-06"), sub): print(d.date(), row[::2].astype(int).tolist())
