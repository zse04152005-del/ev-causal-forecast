"""Summarise the semi-synthetic benchmark v3 (scripts/run_ess_v3.ps1) and add the post-hoc estimators of Theorem T2.

Usage (repository root):
    python scripts/summarize_ess_v3.py [--runs-dir results/runs] [--out results/comparisons] [--scenario flat]

Per run it reads ess_report.json (implied elasticities, effect moments, T3 intervals) and ess_anchor.json (anchor and its
standard error, re-estimated on the synthetic training period with design D). Trained arms:
    A0pooled  cut feedback with the pooled anchor        Bnone0  unconstrained elasticity, initialised at 0
    Bproj0    sign-constrained elasticity (projection)   A5blind price-blind      A6input  price as an ordinary input
    A0oracle  anchor at the true value (upper bound, one seed)
Post-hoc arms (no training; computed from the anchor and the learned elasticity of the same cell and seed):
    Zero      the price-blind effect, beta = 0
    Shrink    kappa * anchor with kappa = max(0, 1 - s^2 / anchor^2)        (Theorem T2 e, plug-in)
    T2combo   w * anchor + (1 - w) * learned (Bproj0), w = e_L / (e_L + s^2) with e_L = max((learned - anchor)^2 - s^2, 0);
              the anchor enters truncated at 0 as in the cut model, and the combination is truncated at 0 (sign restriction)
              (Theorem T2 b with b_A = 0; (learned - anchor)^2 - s^2 is unbiased for b_L^2 + v_L when the anchor is unbiased)
Metrics per cell (setting x true beta x rho [x endo]) over seeds:
    bias and RMSE of the elasticity, share with the wrong sign (beta > 0),
    effect RMSE of the multiplicative effect beta * d against the true effect (from the stored moments, x100),
    for trained arms also the directly measured effect MAE (x100) and the factual MAE,
    coverage of the T3 counterfactual interval (A0pooled).
Outputs: ess_v3_runs.csv (one row per run and arm) and ess_v3_summary.csv (one row per cell and arm).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils.paths import ensure_dir, project_root

NAME = re.compile(r"^v3_ess_(?P<setting>rich|scarce)_b(?P<beta>-?[0-9.]+)_rho(?P<rho>-?[0-9.]+)(?P<tag>_endo1)?_"
                  r"(?P<arm>[A-Za-z0-9]+)_s(?P<seed>\d+)$")


def anchor_from(run_dir: str):
    p = os.path.join(run_dir, "ess_anchor.json")
    if not os.path.exists(p):
        return np.nan, np.nan
    js = json.load(open(p, encoding="utf-8"))
    city = [r for r in js.get("own_levels", []) if r.get("level") == "city"]
    if city:                                                   # untruncated pooled estimate
        return float(city[0]["beta"]), float(city[0]["se"])
    own = js.get("own", [])
    return (float(own[0]["beta"]), float(own[0]["se"])) if own else (np.nan, np.nan)


def effect_rmse(b: float, m: dict) -> float:
    if not np.isfinite(b) or not m or m["n"] == 0:
        return np.nan
    return float(np.sqrt(max(b * b * m["Sdd"] - 2 * b * m["Sdr"] + m["Srr"], 0.0) / m["n"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", default="flat")
    a = ap.parse_args()
    root = project_root(None)
    runs_dir = a.runs_dir or os.path.join(root, "results", "runs")
    out = ensure_dir(a.out or os.path.join(root, "results", "comparisons"))
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "v3_ess_*"))):
        m = NAME.match(os.path.basename(d))
        rep_p = os.path.join(d, "ess_report.json")
        if not m or not os.path.exists(rep_p):
            continue
        rep = json.load(open(rep_p, encoding="utf-8"))
        sc = rep["scenarios"].get(a.scenario)
        if sc is None:
            continue
        b_a, se_a = anchor_from(d)
        rows.append({**{k: m.group(k) for k in ("setting", "arm")}, "beta_true": float(m.group("beta")),
                     "rho": float(m.group("rho")), "endo": int(bool(m.group("tag"))), "seed": int(m.group("seed")),
                     "true_implied": sc["true_implied_beta"], "implied": sc["model"]["implied_beta"],
                     "effect_MAE_measured": 100 * sc["model"]["effect_MAE"], "fact_MAE": rep["fact_MAE_treated"],
                     "anchor": b_a, "anchor_se": se_a, "moments": sc.get("moments"),
                     "T3_coverage": (sc.get("T3_interval") or {}).get("coverage_cf", np.nan),
                     "T3_fact_coverage": (sc.get("T3_interval") or {}).get("coverage_fact", np.nan)})
    if not rows:
        print("没有找到 v3_ess_* 的运行结果")
        return
    df = pd.DataFrame(rows)
    keys = ["setting", "beta_true", "rho", "endo", "seed"]
    # post-hoc arms from the anchor (taken from the A0pooled run) and the learned elasticity (Bproj0)
    extra = []
    for k, g in df.groupby(keys):
        base = g[g.arm == "A0pooled"]
        learn = g[g.arm == "Bproj0"]
        if base.empty:
            continue
        r0 = base.iloc[0]
        b_a, s = r0.anchor, r0.anchor_se
        common = dict(zip(keys, k), true_implied=r0.true_implied, moments=r0.moments, effect_MAE_measured=np.nan,
                      fact_MAE=np.nan, anchor=b_a, anchor_se=s, T3_coverage=np.nan, T3_fact_coverage=np.nan)
        kappa = max(0.0, 1.0 - s * s / (b_a * b_a)) if b_a != 0 else 0.0
        extra.append(dict(common, arm="Zero", implied=0.0))
        extra.append(dict(common, arm="Shrink", implied=kappa * b_a))
        if not learn.empty:
            b_l = learn.iloc[0].implied
            b_at = min(b_a, 0.0)                                   # the anchor as the model uses it (sign-restricted)
            e_l = max((b_l - b_at) ** 2 - s * s, 0.0)
            w = e_l / (e_l + s * s) if (e_l + s * s) > 0 else 1.0
            extra.append(dict(common, arm="T2combo", implied=min(w * b_at + (1 - w) * b_l, 0.0), weight_anchor=w))
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df["error"] = df.implied - df.true_implied
    df["wrong_sign"] = (df.implied > 0).astype(float)
    df["effect_RMSE"] = [100 * effect_rmse(b, m) for b, m in zip(df.implied, df.moments)]
    df.drop(columns=["moments"]).to_csv(os.path.join(out, "ess_v3_runs.csv"), index=False)
    g = df.groupby(["setting", "beta_true", "rho", "endo", "arm"])
    summ = g.agg(n=("seed", "size"), true_implied=("true_implied", "mean"), implied_mean=("implied", "mean"),
                 bias=("error", "mean"), rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
                 wrong_sign=("wrong_sign", "mean"), effect_RMSE=("effect_RMSE", "mean"),
                 effect_MAE_measured=("effect_MAE_measured", "mean"), fact_MAE=("fact_MAE", "mean"),
                 T3_coverage=("T3_coverage", "mean"), anchor_se=("anchor_se", "mean")).reset_index()
    summ.to_csv(os.path.join(out, "ess_v3_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summ.round(3).to_string(index=False))
    print("→", out)


if __name__ == "__main__":
    main()
