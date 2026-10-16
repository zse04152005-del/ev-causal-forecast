"""Elasticity implied by a trained price term with group-by-context elasticities (modes free, none, soft, proj).

The paper reads the elasticity of a price-aware model as the response of its forecasts to a 5% rise of the prices in
the TOU zones over the test period. Below the cap, this response of CPA-STGNN is the average of beta[group, context] over
the TOU zone-hours of the test period, so each cell is weighted by its number of TOU zones and by the test-period hours
in its context. This script computes that weighted elasticity (and, for comparison, the plain mean over the cells) for
every run directory given.
Usage: python scripts/learned_elasticity.py --out results/comparisons/learned_elasticity.csv <run dir> [<run dir> ...]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
P = prepare(load_config(_bootstrap.DEFAULT_CFG))
tou = P.tou_mask()
test = np.arange(P.split.val_end + 1, P.T)
n_g = np.bincount(P.groups[tou], minlength=int(P.groups.max()) + 1).astype(float)
n_c = np.bincount(P.ctx[test], minlength=int(P.ctx.max()) + 1).astype(float)
W = np.outer(n_g, n_c)
W /= W.sum()
rows = []
for d in a.runs:
    p = os.path.join(d, "price_params.json")
    if not os.path.exists(p):
        continue
    B = np.asarray(json.load(open(p, encoding="utf-8"))["beta_tou_by_group_ctx"], dtype=float)
    name = os.path.basename(os.path.normpath(d))
    rows.append({"run": name, "group": re.sub(r"_s\d+$", "", name), "beta_weighted": float((W * B).sum()),
                 "beta_cell_mean": float(np.nanmean(B))})
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
df.to_csv(a.out, index=False)
print("weights (group x context):\n", np.round(W, 3))
print(df.groupby("group")[["beta_weighted", "beta_cell_mean"]].agg(["mean", "std"]).round(3).to_string())
