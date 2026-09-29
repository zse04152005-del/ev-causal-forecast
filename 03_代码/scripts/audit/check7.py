# (1) data-quality mask for frozen (imputed) segments; (2) feasibility of within-zone identification (training period only)
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings("ignore")
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    df.columns = df.columns.astype(int); return df
occ, dur, pe, ps = [load(n) for n in ["occupancy","duration","e_price","s_price"]]
Z = list(dur.columns); T = len(dur)
p = pe + ps; lp = np.log(p)
# ---- (1) frozen mask: run of >=6h with (occ,dur) unchanged and (dur fractional or run >= 24h) ----
O = occ.values; U = dur.values
same = np.vstack([np.zeros((1, len(Z)), bool), (O[1:] == O[:-1]) & (U[1:] == U[:-1])])
frozen = np.zeros_like(same)
for j in range(len(Z)):
    t = 0
    while t < T:
        if same[t, j]:
            s = t - 1
            while t < T and same[t, j]: t += 1
            L = t - s; frac_ = abs(U[s, j] - round(U[s, j])) > 1e-6
            if (L >= 6 and frac_ and U[s, j] > 0) or L >= 24: frozen[s:t, j] = True
        else: t += 1
fz = pd.DataFrame(frozen, index=dur.index, columns=Z)
print("frozen share of zone-hours: %.3f" % frozen.mean())
day = dur.index.normalize()
bysplit = {"train": fz[:"2023-01-05"].values.mean(), "val": fz["2023-01-06":"2023-01-23"].values.mean(), "test": fz["2023-01-24":].values.mean()}
print("frozen share by split:", {k: round(v, 3) for k, v in bysplit.items()})
dshare = fz.groupby(day).mean().mean(axis=1); print("top frozen dates:", {str(k.date()): round(v, 2) for k, v in dshare.sort_values(ascending=False).head(12).items()})
zshare = fz.mean(); print("per-zone frozen share quantiles:", zshare.quantile([.5, .75, .9, .95, 1]).round(3).to_dict(), "| zones >30%:", int((zshare > .3).sum()))
fz.astype(np.int8).to_csv("frozen_mask.csv")
# ---- (2) feasibility regression on training period ----
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
strict = set(frac[frac >= .5].index); fixed = set(frac[frac == 0].index)
dist = pd.read_csv(f"{D}/distance.csv").values / 1000.0; np.fill_diagonal(dist, np.inf)
Y = np.log1p(dur.values); LP = lp.values; FR = frozen
tr_end = np.where(dur.index <= pd.Timestamp("2023-01-05 23:00"))[0][-1]
isT = np.array([z in strict for z in Z]); isF = np.array([z in fixed for z in Z])
rows = []
for t in range(4, tr_end - 3):
    h = dur.index[t].hour
    dlp = LP[t] - LP[t-1]
    flat_pre = np.abs(LP[t-3:t].max(0) - LP[t-3:t].min(0)) < 1e-4
    flat_post = np.abs(LP[t:t+3].max(0) - LP[t:t+3].min(0)) < 1e-4
    ok = flat_pre & flat_post & ~FR[t-3:t+3].any(0) & (isT | isF)
    J = Y[t:t+3].mean(0) - Y[t-3:t].mean(0)
    x = np.where(np.abs(dlp) > 0.01, dlp, 0.0)          # ignore sub-1% averaging noise
    xT = np.where(isT, x, 0.0)
    S1 = np.array([xT[(dist[i] >= 0) & (dist[i] < 2)].sum() for i in range(len(Z))])
    S2 = np.array([xT[(dist[i] >= 2) & (dist[i] < 4)].sum() for i in range(len(Z))])
    for i in np.where(ok)[0]:
        rows.append((i, t, h, str(dur.index[t].date()), J[i], x[i], S1[i], S2[i], isT[i]))
df = pd.DataFrame(rows, columns=["zone", "t", "h", "date", "J", "x", "S1", "S2", "tou"])
print("\nfeasibility sample (train, clean 3h windows, unfrozen): %d zone-hours; with own switch: %d" % (len(df), int((df.x != 0).sum())))
def twfe(d, cols, fes, iters=50):
    d = d.copy(); V = d[["J"] + cols].astype(float)
    for _ in range(iters):
        for f in fes: V = V - V.groupby(d[f]).transform("mean")
    y = V["J"].values; X = V[cols].values
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ y); u = y - X @ b
    meat = np.zeros((len(cols), len(cols)))
    for g, idx in d.groupby("zone").indices.items():
        s = X[idx].T @ u[idx]; meat += np.outer(s, s)
    Vb = np.linalg.inv(XtX) @ meat @ np.linalg.inv(XtX); se = np.sqrt(np.diag(Vb))
    resid_var_x = (X ** 2).sum(0)
    return dict(zip(cols, zip(np.round(b, 3), np.round(se, 3)))), resid_var_x
df["zh"] = df.zone.astype(str) + "_" + df.h.astype(str); df["dh"] = df.date + "_" + df.h.astype(str)
res = {}
for name, sub, fes in [("A naive diff-in-disc (date x hour FE), TOU+fixed", df, ["dh"]),
                       ("B + zone x hour FE (within-zone, across days)", df, ["dh", "zh"]),
                       ("C TOU zones only, date x hour FE (switch timing across TOU zones)", df[df.tou], ["dh"]),
                       ("D TOU only, + zone x hour FE", df[df.tou], ["dh", "zh"])]:
    b, rv = twfe(sub, ["x", "S1", "S2"], fes)
    tot_var = (sub.x - sub.x.mean()).pow(2).sum()
    print(f"{name}: beta_own={b['x']}, delta_0-2km={b['S1']}, delta_2-4km={b['S2']} | share of own-price variation left after FE: {rv[0]/tot_var:.3f}")
    res[name] = {k: [float(v[0]), float(v[1])] for k, v in b.items()}
json.dump(res, open("check7_feasibility.json", "w"), ensure_ascii=False, indent=1)
