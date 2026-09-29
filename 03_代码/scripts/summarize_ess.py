"""汇总 E-SS 运行（runs/ess_*/ess_report.json + summary.json）→ 05_实验结果/comparisons/ess_summary.csv 与按设置平均的表。

用法：python scripts/summarize_ess.py [--runs-dir 05_实验结果/runs]
"""
import argparse
import glob
import json
import os
import re

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
rows = []
for f in sorted(glob.glob(os.path.join(runs, "ess_*", "ess_report.json"))):
    name = os.path.basename(os.path.dirname(f))
    m = re.match(r"ess_rho([\d.]+)_(\w+?)_s(\d+)$", name)
    rep = json.load(open(f, encoding="utf-8"))
    sm_f = os.path.join(os.path.dirname(f), "summary.json")
    sm = json.load(open(sm_f, encoding="utf-8")) if os.path.exists(sm_f) else {}
    base = {"run": name, "rho": float(m.group(1)) if m else rep.get("rho"), "variant": m.group(2) if m else "",
            "seed": int(m.group(3)) if m else -1, "fact_MAE_treated": rep["fact_MAE_treated"],
            "test_CRPS": sm.get("test_CRPS_q"), "elasticity_MAE": rep.get("elasticity", {}).get("MAE_treated"),
            "sign_correct": rep.get("elasticity", {}).get("sign_correct_treated")}
    for sc, v in rep["scenarios"].items():
        for k in ("model", "plugin_P0"):
            rows.append({**base, "scenario": sc, "predictor": k, "true_implied_beta": v["true_implied_beta"],
                         **{m_: v[k][m_] for m_ in ("CF_MAE", "effect_MAE", "implied_beta", "mean_effect")}})
df = pd.DataFrame(rows)
out = ensure_dir(os.path.join(res, "comparisons"))
df.to_csv(os.path.join(out, "ess_summary.csv"), index=False)
if len(df):
    g = df.groupby(["scenario", "rho", "variant", "predictor"])[
        ["fact_MAE_treated", "CF_MAE", "effect_MAE", "implied_beta", "true_implied_beta", "elasticity_MAE"]].mean()
    g.to_csv(os.path.join(out, "ess_summary_mean.csv"))
    pd.set_option("display.width", 200)
    print(g.round(4).to_string())
print(f"{df.run.nunique() if len(df) else 0} 次运行 → {out}")
