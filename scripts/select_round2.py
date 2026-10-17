"""Pick the best hyperparameters of each baseline from the round-2 tuning runs (validation pinball loss, seed 0).

Usage (repository root): python scripts/select_round2.py [--runs-dir results/runs]
Writes configs/experiment/round2_best.json: {model: [override, ...]} used by scripts/run_round2_baselines.ps1 -Phase seeds,
and results/comparisons/round2_tuning.csv with every tuning run.
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils.paths import ensure_dir, project_root

PAT = re.compile(r"^r2_tune_(?P<model>[a-z]+)_(?P<hp>.+)_s0$")
KEYS = {"D": "baseline.D", "lr": "train.lr", "do": "baseline.dropout"}
KEYS_STAE = {"lr": "train.lr", "do": "model.staeformer.dropout"}


def parse_hp(model: str, hp: str) -> list:
    out = []
    for part in hp.split("_"):
        m = re.match(r"^(D|lr|do)(.+)$", part)
        if m:
            key = (KEYS_STAE if model == "staeformer" else KEYS)[m.group(1)]
            out.append(f"{key}={m.group(2)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args()
    root = project_root(None)
    runs = a.runs_dir or os.path.join(root, "results", "runs")
    rows = []
    for d in glob.glob(os.path.join(runs, "r2_tune_*_s0")):
        m = PAT.match(os.path.basename(d))
        p = os.path.join(d, "summary.json")
        if not m or not os.path.exists(p):
            continue
        s = json.load(open(p, encoding="utf-8"))
        # CPA-STGNN runs store best_val_pinball, the baseline runs best_val_loss (also the pinball loss)
        val = s.get("best_val_pinball", s.get("best_val_loss"))
        if val is None:
            raise SystemExit(f"{d}: summary.json has no validation loss")
        rows.append({"model": m.group("model"), "hp": m.group("hp"), "val_pinball": val,
                     "test_MAE": s.get("test_MAE"), "overrides": " ".join(parse_hp(m.group("model"), m.group("hp")))})
    if not rows:
        raise SystemExit("没有找到 r2_tune_* 的运行结果")
    df = pd.DataFrame(rows).sort_values(["model", "val_pinball"])
    df.to_csv(os.path.join(ensure_dir(os.path.join(root, "results", "comparisons")), "round2_tuning.csv"), index=False)
    best = {m: g.iloc[0].overrides.split() for m, g in df.groupby("model")}
    path = os.path.join(root, "configs", "experiment", "round2_best.json")
    json.dump(best, open(path, "w", encoding="utf-8"), indent=1)
    print(df.groupby("model").head(1).to_string(index=False))
    print("→", path)


if __name__ == "__main__":
    main()
