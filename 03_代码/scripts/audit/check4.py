# schedule structure: switch timing across TOU zones, within-zone schedule changes across days, weak-TOU days
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
pe, ps = load("e_price"), load("s_price"); p = (pe + ps).round(4)
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
strict = frac[frac >= .5].index; weak = frac[(frac > 0) & (frac < .5)].index
days = p.index.normalize().unique()
def daymat(z):  # days x 24
    return p[z].values.reshape(len(days), 24)
# 1) switch timing heterogeneity at each clock hour on the modal schedule
up = np.zeros(24, int); dn = np.zeros(24, int); none = np.zeros(24, int)
nsched = {}; chg_dates = []
for z in strict:
    M = daymat(z); lp = np.log(M)
    rows = [tuple(r) for r in M]; vc = pd.Series(rows).value_counts()
    nsched[z] = (vc >= 3).sum()                       # distinct schedules used on >=3 days
    modal = np.array(vc.index[0]); d = np.diff(np.log(np.r_[modal[-1], modal]))  # jump into hour h (h=0 uses previous day's 23h)
    up += d > 0.05; dn += d < -0.05; none += np.abs(d) <= 0.01
    # schedule change days: daily vector differs from previous day's
    for k in range(1, len(days)):
        if np.abs(lp[k] - lp[k-1]).max() > 0.01: chg_dates.append((z, days[k]))
print("modal-schedule switches among %d strict TOU zones, by clock hour (up / down / no change):" % len(strict))
print(pd.DataFrame({"up": up, "down": dn, "none": none}).T.to_string())
ns = pd.Series(nsched); print("distinct daily schedules (used >=3 days) per strict zone:", ns.value_counts().sort_index().to_dict())
cd = pd.DataFrame(chg_dates, columns=["zone", "date"])
print("zone-days with a schedule change vs previous day:", len(cd), "| zones with >=1 change:", cd.zone.nunique())
top = cd.date.value_counts().head(15); print("dates with most zones changing schedule:", {str(k.date()): int(v) for k, v in top.items()})
per_zone = cd.groupby("zone").size(); print("changes per zone quantiles:", per_zone.quantile([.1, .25, .5, .75, .9]).to_dict())
# persistent changes: modal schedule of month m differs from month m-1
mon = pd.Series(days.to_period("M"))
pers = []
for z in strict:
    M = daymat(z); prev = None
    for m in mon.unique():
        sub = M[(mon == m).values]; vc = pd.Series([tuple(r) for r in sub]).value_counts(); mod = np.array(vc.index[0])
        if prev is not None and np.abs(np.log(mod) - np.log(prev)).max() > 0.01: pers.append((z, str(m)))
        prev = mod
pdf = pd.DataFrame(pers, columns=["zone", "month"]); print("persistent monthly modal-schedule changes:", len(pdf), "zones:", pdf.zone.nunique(), pdf.month.value_counts().sort_index().to_dict())
# 2) weak TOU zones: which days are TOU?
wd = []
for z in weak:
    M = daymat(z); tdays = days[(np.ptp(M, axis=1) > 1e-4)]
    wd += list(tdays)
wd = pd.Series(wd); print("weak zones:", len(weak), "| TOU zone-days:", len(wd))
print("  by month:", wd.dt.to_period("M").value_counts().sort_index().to_dict())
print("  top dates:", {str(k.date()): int(v) for k, v in wd.value_counts().head(12).items()})
