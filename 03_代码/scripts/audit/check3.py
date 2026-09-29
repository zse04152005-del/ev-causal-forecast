import numpy as np, pandas as pd
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
occ, dur, pe, ps = [load(n) for n in ["occupancy","duration","e_price","s_price"]]
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
strict = frac[frac >= .5].index; fixed = frac[frac == 0].index
p = pe + ps; h = p.index.hour
print("mean total price by clock hour (strict TOU):"); print((p[strict].groupby(h).mean().mean(1)).round(3).to_string())
print("mean e_price by hour (strict):", pe[strict].groupby(h).mean().mean(1).round(3).tolist())
print("mean s_price by hour (strict):", ps[strict].groupby(h).mean().mean(1).round(3).tolist())
# a typical weekday for 3 zones
day = "2022-11-16"
for z in list(strict[:3]):
    print(z, "total:", p.loc[day, z].round(3).tolist())
    print(z, "e    :", pe.loc[day, z].round(3).tolist())
# modal schedule clusters: pattern of hourly price rank on median day
prof = p[strict].groupby(h).median(); norm = prof / prof.mean()
lowh = norm.idxmin(); highh = norm.idxmax()
print("cheapest clock hour per zone:", lowh.value_counts().to_dict()); print("dearest clock hour per zone:", highh.value_counts().to_dict())
# demand alignment: hour-to-hour change in log(1+dur) TOU minus fixed, by clock hour, vs dlog p
ld = np.log1p(dur); dld = ld.diff(); dlp = np.log(p).diff()
res = pd.DataFrame({"dlogp_TOU": dlp[strict].groupby(h).mean().mean(1),
                    "dld_TOU": dld[strict].groupby(h).mean().mean(1), "dld_FIX": dld[fixed].groupby(h).mean().mean(1)})
res["DiD"] = res.dld_TOU - res.dld_FIX
res["dld_TOU_next"] = res.DiD.shift(-1)
print(res.round(4).to_string())
print("corr(DiD_h, dlogp_h) = %.3f ; corr(DiD_{h+1}, dlogp_h) = %.3f ; corr(DiD_{h-1}, dlogp_h) = %.3f" % (
    np.corrcoef(res.DiD, res.dlogp_TOU)[0,1], np.corrcoef(res.DiD.shift(-1).fillna(0), res.dlogp_TOU)[0,1], np.corrcoef(res.DiD.shift(1).fillna(0), res.dlogp_TOU)[0,1]))
# zone-level event study around big down-switches (price drop >5%), duration log, relative to fixed-zone same-time mean
lp = np.log(p); ev = []
fixmean = ld[fixed].mean(1)
for z in strict:
    d = lp[z].diff().values
    for t in np.where(d < -0.05)[0]:
        if 6 <= t < len(d) - 6:
            ev.append((ld[z].values[t-6:t+6] - fixmean.values[t-6:t+6]) - (ld[z].values[t-1] - fixmean.values[t-1]))
ev = np.array(ev); print("price-DROP events:", len(ev), "mean path k=-6..+5 (k=0 is first hour at new lower price):", np.round(ev.mean(0), 3).tolist())
ev = []
for z in strict:
    d = lp[z].diff().values
    for t in np.where(d > 0.05)[0]:
        if 6 <= t < len(d) - 6:
            ev.append((ld[z].values[t-6:t+6] - fixmean.values[t-6:t+6]) - (ld[z].values[t-1] - fixmean.values[t-1]))
ev = np.array(ev); print("price-RISE events:", len(ev), "mean path:", np.round(ev.mean(0), 3).tolist())
