import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
pe, ps = load("e_price"), load("s_price"); lp = np.log(pe + ps)
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
strict = list(frac[frac >= .5].index)
tr = lp.loc[:"2023-01-05 23:00", strict]
long = tr.stack().rename("lp").reset_index(); long.columns = ["time", "zone", "lp"]
long["h"] = long.time.dt.hour; long["we"] = (long.time.dt.dayofweek >= 5).astype(int); long["date"] = long.time.dt.date.astype(str)
long["dl"] = long.lp - long.groupby("zone").lp.transform("mean")
tot = (long.dl ** 2).sum()
def r2(fes, iters=30):
    v = long.dl.copy()
    for _ in range(iters):
        for f in fes: v = v - v.groupby(long[f]).transform("mean")
    return 1 - (v ** 2).sum() / tot
long["zhw"] = long.zone.astype(str) + "_" + long.h.astype(str) + "_" + long.we.astype(str)
long["dh"] = long.date + "_" + long.h.astype(str)
long["zhm"] = long.zone.astype(str) + "_" + long.h.astype(str) + "_" + long.time.dt.month.astype(str)
out = {"R2_zone_x_hour_x_daytype": r2(["zhw"]), "R2_plus_date_x_hour": r2(["zhw", "dh"]), "R2_zone_x_hour_x_month": r2(["zhm"]), "R2_zhm_plus_dh": r2(["zhm", "dh"])}
print({k: round(v, 4) for k, v in out.items()})
fz = pd.read_csv("frozen_mask.csv", index_col=0, parse_dates=True)
te = fz.loc["2023-01-24":]; print("test frozen share by week:", {str(k.date()): round(v, 3) for k, v in te.mean(axis=1).resample("W").mean().items()})
print("test frozen share excluding 01-21..01-29: %.3f" % fz.loc["2023-01-30":].values.mean())
print("zones >50%% frozen in test: %d" % (te.mean() > .5).sum())
json.dump(out, open("check11_assumptionA.json", "w"), indent=1)
