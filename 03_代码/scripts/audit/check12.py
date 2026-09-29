import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time"); df.columns = df.columns.astype(int); return df
pe, ps = load("e_price"), load("s_price"); lp = np.log(pe + ps)
st = pd.read_csv("zone_static.csv", index_col=0); fz = pd.read_csv("frozen_mask.csv", index_col=0, parse_dates=True); fz.columns = fz.columns.astype(int)
fixed = st.index[st.pricing == "fixed"]; tou = st.index[st.pricing == "TOU"]
ov = fz.mean()
print("fixed zones with overall frozen share <0.2: %d / %d ; <0.1: %d" % ((ov[fixed] < .2).sum(), len(fixed), (ov[fixed] < .1).sum()))
print("TOU zones with overall frozen share <0.2: %d / %d" % ((ov[tou] < .2).sum(), len(tou)))
# clean big switch events (|dlog p|>1%, flat 3h both sides, unfrozen) in TRAIN by group x context
L = lp.values; F = fz.values.astype(bool); idx = {z: j for j, z in enumerate(lp.columns)}
tr_end = np.where(lp.index <= pd.Timestamp("2023-01-05 23:00"))[0][-1]
rows = []
for z in tou:
    j = idx[z]; x = L[:, j]; d = np.diff(x)
    for t in np.where(np.abs(d) > 0.01)[0] + 1:
        if 3 <= t <= tr_end - 3 and np.ptp(x[t-3:t]) < 1e-4 and np.ptp(x[t:t+3]) < 1e-4 and not F[t-3:t+3, j].any():
            ts = lp.index[t]; rows.append((z, st.loc[z, "group"], "weekend" if ts.dayofweek >= 5 else "weekday", "day" if 7 <= ts.hour < 21 else "night", np.sign(d[t-1])))
ev = pd.DataFrame(rows, columns=["zone", "group", "daytype", "dn", "sign"])
print("clean TOU switch events in train:", len(ev))
print(pd.crosstab([ev.group], [ev.daytype, ev.dn]))
print("zones contributing per group:", ev.groupby("group").zone.nunique().to_dict())
print("up/down by context:", pd.crosstab([ev.daytype, ev.dn], ev.sign).to_dict())
