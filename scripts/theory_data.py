"""Theorem 1 on the UrbanEV data (Supplementary Material, Table S1): TOU zones, training period, valid zone-hours.

Function classes G (what a forecaster can represent without prices):
  G1 = zone x hour of day x day type   (a deterministic schedule)
  G2 = zone x hour of day + date x hour (adds common daily shocks)
For each G: R^2 of the log-price deviation, sd of the residual price variation V = M_G D, sd of the residual log
demand M_G Y, the amplification sd(M Y) / sd(V) that multiplies any correlation between the two in the bias of a
loss-based elasticity, and the loss-minimising elasticity itself.

Usage (repository root): python scripts/theory_data.py [--out results/theory/t1_data_numbers.json] [overrides ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data.dataset import prepare  # noqa: E402
from src.utils.config import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "theory", "t1_data_numbers.json"))
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
P = prepare(load_config(a.config, a.overrides))
tr = np.zeros(P.T, bool)
tr[: P.split.train_end + 1] = True
tou = P.pricing == 2
y = P.y[:, tou]
D = P.dl[:, tou]
ok = P.valid[:, tou] & (y > 0)
T, N = y.shape
mask = ok & tr[:, None]
day = (P.time.normalize() - P.time.normalize()[0]).days.values
rows = np.argwhere(mask)
df = pd.DataFrame({"t": rows[:, 0], "i": rows[:, 1]})
df["Y"] = np.log(y[mask])
df["D"] = D[mask]
df["h"] = P.tod[df.t]
df["day"] = day[df.t]
df["dt"] = P.dow[df.t] >= 5


def demean(s, keys, iters=30):
    r = s.copy()
    for _ in range(iters):
        for k in keys:
            r = r - r.groupby(df[k]).transform("mean")
    return r


out = {"n_obs": int(len(df)), "n_zones": int(N), "var_D": float(df.D.var())}
df["zh"] = df.i * 24 + df.h
df["zhd"] = df.zh * 2 + df.dt.astype(int)
df["dh"] = df.day * 24 + df.h
for name, keys in {"G1_zone_hour_daytype": ["zhd"], "G2_zonehour_plus_datehour": ["zh", "dh"]}.items():
    V = demean(df.D, keys)
    U = demean(df.Y, keys)
    out[name] = {"R2_price": float(1 - V.var() / df.D.var()), "sd_V": float(V.std()), "sd_MY": float(U.std()),
                 "amplification_sdMY_over_sdV": float(U.std() / V.std()),
                 "corr_V_MY": float(np.corrcoef(V, U)[0, 1]),
                 "naive_loss_based_beta": float((V * U).sum() / (V * V).sum())}
print(json.dumps(out, indent=1))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
