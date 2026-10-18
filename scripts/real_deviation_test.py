"""Out-of-sample check on real price deviations (Section 4.6, "Real price deviations"; Supplementary Material, Table S6).

In the test period the log price of many TOU zone-hours departs from the zone's training schedule (holiday tariffs of the
Spring Festival and later departures). The schedule is the median log-price deviation of the zone by hour of day and
day type over the training period. A forecaster with elasticity beta and a price-blind baseline does not see these
departures in its baseline, so by Corollary 2 its log forecast error on such hours is (beta0 - beta) * departure.
For every run group the script regresses log(y) - log(median forecast) on the departure, with zone-by-date and
zone-by-hour fixed effects (optionally also date-by-hour), with standard errors clustered by zone, and reports
slope, p-value and the implied elasticity beta + slope, which should be the same for every model.

Usage (repository root; needs predictions_test.npz of the runs and the UrbanEV data):
    python scripts/real_deviation_test.py --runs-dir results/runs --out results/comparisons/realdev_slopes.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data.calendar import is_holiday  # noqa: E402
from src.data.dataset import prepare  # noqa: E402
from src.utils.config import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default=os.path.join(ROOT, "results", "runs"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "comparisons", "realdev_slopes.csv"))
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()

# group name -> (elasticity of the price term, run directories); beta of the anchored model from configs/anchor/fivemin_main.json
GROUPS = {"price-blind": (0.0, [f"v2_gwn_blind_s{s}" for s in range(5)]),
          "anchored": (-0.2131, [f"v2_main_A0_s{s}" for s in range(5)])}
for b in ["-0.05", "-0.1", "-0.2", "-0.4", "-0.8", "-1.2", "-1.6", "-2.0", "-3.0"]:
    GROUPS[f"fixed {b}"] = (float(b), [f"v2_eid_profile{b}_s0"])
for b in ["-0.45", "-0.76", "-1.48"]:
    GROUPS[f"fixed {b} (3 seeds)"] = (float(b), [f"v2_abl_A3_prior{b}_s{s}" for s in range(3)])

P = prepare(load_config(a.config, a.overrides))
T = P.T
tou = np.where(P.pricing == 2)[0]
hour = P.tod
daytype = ((P.dow >= 5) | is_holiday(P.time)).astype(int)
dates = P.time.normalize()
train = np.arange(T) <= P.split.train_end
regular = np.full_like(P.dl, np.nan)
for dt in (0, 1):
    for h in range(24):
        m = train & (hour == h) & (daytype == dt)
        regular[(hour == h) & (daytype == dt)] = np.nanmedian(np.where(P.valid[m], P.dl[m], np.nan), axis=0)
dev = P.dl - regular
for part, (lo, hi) in (("train", (0, P.split.train_end)), ("test", (P.split.val_end + 1, T - 1))):
    d = dev[lo:hi + 1][:, tou]
    print(f"{part}: share of TOU zone-hours with |departure| > 0.02: {np.nanmean(np.abs(d) > 0.02):.3f}")

rows = []
for name, (beta, runs) in GROUPS.items():
    files = [os.path.join(a.runs_dir, r, "predictions_test.npz") for r in runs]
    if not all(os.path.exists(f) for f in files):
        print("skipped (predictions missing):", name)
        continue
    zs = [np.load(f) for f in files]
    o, steps = zs[0]["origins"], zs[0]["steps"]
    med = np.mean([z["yq"][..., 3].astype(np.float32) for z in zs], axis=0)       # median quantile, mean over seeds
    for j, s in enumerate(steps):
        t = o + s
        Y, V, F, D = P.y[t][:, tou], P.valid[t][:, tou], med[:, j, :][:, tou], dev[t][:, tou]
        ok = V & (Y > 0) & (F > 0) & np.isfinite(D)
        tt = np.broadcast_to(t[:, None], Y.shape)[ok]
        rows.append(pd.DataFrame({"group": name, "beta": beta, "step": int(s), "t": tt,
                                  "z": np.broadcast_to(np.arange(len(tou))[None, :], Y.shape)[ok],
                                  "r": np.log(Y[ok]) - np.log(F[ok]), "dev": D[ok]}))
df = pd.concat(rows, ignore_index=True)
df["date"] = dates[df.t].values
df["h"] = hour[df.t]
df["zd"] = df.z.astype(str) + "_" + df.date.astype(str)
df["zh"] = df.z * 24 + df.h
df["dh"] = df.date.astype(str) + "_" + df.h.astype(str)


def demean(x, d, keys, it=40):
    r = x.copy()
    for _ in range(it):
        for k in keys:
            r = r - r.groupby(d[k]).transform("mean")
    return r


def slope(d, keys):
    yr, xr = demean(d.r, d, keys), demean(d.dev, d, keys)
    b = (xr * yr).sum() / (xr * xr).sum()
    se = np.sqrt(((xr * (yr - b * xr)).groupby(d.z).sum() ** 2).sum()) / (xr * xr).sum()
    return b, se


out = []
for spec, keys in (("zone-date + zone-hour", ["zd", "zh"]), ("+ date-hour", ["zd", "zh", "dh"])):
    for period, sel in (("test period", df.date > "2000"), ("after Spring Festival", df.date > "2023-01-29")):
        d0 = df[(df.step == 24) & sel]
        for name, d in d0.groupby("group", sort=False):
            b, se = slope(d, keys)
            beta = d.beta.iloc[0]
            out.append({"spec": spec, "period": period, "group": name, "beta": beta, "slope": b, "se": se,
                        "p": 2 * stats.norm.sf(abs(b / se)), "implied": beta + b, "n": len(d)})
res = pd.DataFrame(out)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
res.to_csv(a.out, index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
