"""汇总全部运行（两轮）→ results/comparisons/：groups_mean.csv（各组跨种子均值与标准差）、all_runs_flat.csv、
eps_summary.csv（先验敏感性）、eid_summary.csv（可识别性）。E-SS 另见 summarize_ess.py。

用法：python scripts/summarize_all.py [--runs-dir results/runs]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default=None)
a = ap.parse_args()
cfg = load_config(_bootstrap.DEFAULT_CFG)
res = resolve(cfg.paths.results_dir, project_root(cfg.paths.get("root")))
runs = a.runs_dir or os.path.join(res, "runs")
out = ensure_dir(os.path.join(res, "comparisons"))


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


rows = []
for d in sorted(glob.glob(os.path.join(runs, "*"))):
    name = os.path.basename(d)
    s = jload(os.path.join(d, "summary.json"))
    if not s or name.startswith(("smoke", "tune")) or "ess_" in name:
        continue
    m = pd.read_csv(os.path.join(d, "metrics_test.csv")).set_index("step")
    m.index = m.index.astype(str)
    r = {"run": name, "group": re.sub(r"_s\d+$", "", name), "MAE": s["test_MAE"], "RMSE": s["test_RMSE"],
         "WAPE": s["test_WAPE"], "CRPS": s["test_CRPS_q"], "PICP90_raw": s["test_PICP90_raw"],
         "PICP90_aci": s["test_PICP90_aci_mean"], "val": s.get("best_val_pinball") or s.get("best_val_loss"),
         "epochs": s.get("epochs_run"), "minutes": (s.get("seconds") or 0) / 60, "n_params": s.get("n_params")}
    for st in ("1", "3", "6", "12", "24"):
        if st in m.index:
            r[f"MAE_{st}"] = m.loc[st, "MAE"]
    p = jload(os.path.join(d, "price_params.json"))
    b = p.get("beta_tou_by_group_ctx")
    if b:
        r["beta_model"] = float(np.nanmean([x for row in b for x in row if x is not None]))
    for sub in ("all", "novel"):
        sc = p.get(f"switch_consistency_{sub}") or {}
        r[f"switch_{sub}_model"], r[f"switch_{sub}_obs"] = sc.get("beta_model"), sc.get("beta_obs")
    arc = p.get("arc_elasticity_tou") or {}
    r["arc_mean"], r["arc_median"], r["arc_share_neg"] = arc.get("mean"), arc.get("median"), arc.get("share_negative")
    for k in ("prior", "lambda_mean_tou", "lambda_mean_all", "clamp"):
        if k in p:
            r[f"piast_{k}"] = str(p[k]) if k == "clamp" else p[k]
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "all_runs_flat.csv"), index=False)
num = [c for c in df.columns if c not in ("run", "group", "piast_clamp") and pd.api.types.is_numeric_dtype(df[c])]
g = df.groupby("group")[num].agg(["mean", "std"])
g.columns = [f"{c}_{k}" if k == "std" else c for c, k in g.columns]
g.insert(0, "n", df.groupby("group").size())
g.to_csv(os.path.join(out, "groups_mean.csv"))
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
show = ["n", "MAE", "MAE_std", "CRPS", "CRPS_std", "val", "PICP90_raw", "PICP90_aci", "MAE_1", "MAE_24", "beta_model",
        "switch_all_model", "minutes"]
print(g[[c for c in show if c in g]].round(4).to_string())

eps = df[df.group.str.startswith("eps_")].copy()
if len(eps):
    eps["method"] = eps.group.str.extract(r"eps_(pag_paper|pag_released|piast_clamp|piast_noclamp)")[0]
    eps["prior_injected"] = eps.group.str.extract(r"(-[\d.]+)$")[0].astype(float)
    cols = ["method", "prior_injected", "arc_mean", "arc_median", "arc_share_neg", "piast_lambda_mean_tou", "MAE",
            "switch_all_model"]
    e = eps[[c for c in cols if c in eps]].sort_values(["method", "prior_injected"], ascending=[True, False])
    e.to_csv(os.path.join(out, "eps_summary.csv"), index=False)
    print("\n先验敏感性（E-PS）\n" + e.round(4).to_string(index=False))

eid = df[df.group.str.contains("eid_")].copy()
if len(eid):
    eid["round"] = np.where(eid.group.str.startswith("v2_"), "v2", "v1")
    eid["kind"] = np.where(eid.group.str.contains("profile"), "profile", "init")
    eid["beta_set"] = eid.group.str.extract(r"(-[\d.]+)$")[0].astype(float)
    e = eid.groupby(["round", "kind", "beta_set"])[["beta_model", "val", "MAE", "CRPS"]].agg(["mean", "std"])
    e.columns = [f"{c}_{k}" if k == "std" else c for c, k in e.columns]
    e.to_csv(os.path.join(out, "eid_summary.csv"))
    print("\n可识别性（E-ID）\n" + e.round(5).to_string())
print(f"\n{len(df)} 次运行 → {out}")
