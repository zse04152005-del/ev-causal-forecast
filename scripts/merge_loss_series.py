"""Merge loss series of the same model computed on different seeds (round 1: seeds 0-4; round 2: seeds 5-9).

Every key of a loss file is an average over seeds of per-seed losses, so the average over all ten seeds is the
seed-count-weighted mean of the two parts. Writes, for each group given:
    <group>      average over all seeds of both files
    <group>@5    the seeds of the first file (seeds 0-4), so that five-seed models are compared on the same seeds
Usage: python scripts/merge_loss_series.py --first results/comparisons/dm/loss_a.npz \
           --second results/comparisons/dm/loss_r2.npz --out results/comparisons/dm/loss_r2m.npz v2_main_A0 v2_gwn_blind
The output name sorts after loss_r2.npz, so scripts/significance_tests.py uses the merged keys.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--first", required=True)
ap.add_argument("--second", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("groups", nargs="+")
a = ap.parse_args()
f1, f2 = np.load(a.first), np.load(a.second)
assert np.array_equal(f1["steps"], f2["steps"]), "the two files use different steps"
out = {"steps": f1["steps"]}
for g in a.groups:
    n1, n2 = len(f1[g + "|seed_mae"]), len(f2[g + "|seed_mae"])
    for k in ("mae_o", "mae_z", "crps_o", "crps_z"):
        x1, x2 = f1[f"{g}|{k}"], f2[f"{g}|{k}"]
        assert x1.shape == x2.shape, (g, k, x1.shape, x2.shape)
        out[f"{g}|{k}"] = (n1 * x1 + n2 * x2) / (n1 + n2)
        out[f"{g}@{n1}|{k}"] = x1
    out[g + "|seed_mae"] = np.concatenate([f1[g + "|seed_mae"], f2[g + "|seed_mae"]])
    out[f"{g}@{n1}|seed_mae"] = f1[g + "|seed_mae"]
    print(g, n1, "+", n2, "seeds; first", f1[g + "|seed_mae"].round(5), "second", f2[g + "|seed_mae"].round(5))
np.savez_compressed(a.out, **out)
