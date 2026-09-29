"""把一批运行的 summary.json（和 price_params.json 的关键数）汇成一张表，用于调参选型与结果总表。

用法：
    python scripts/summarize_runs.py --prefix tune_ --sort best_val_pinball      # 调参：按验证集 pinball 排序
    python scripts/summarize_runs.py --prefix main_A0_ --prefix base_             # 主结果与基线
输出：05_实验结果/comparisons/runs_<第一个前缀>.csv（并打印）
"""
import argparse
import glob
import json
import os

import pandas as pd

import _bootstrap  # noqa: F401
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", action="append", default=[])
ap.add_argument("--runs-dir", default=None)
ap.add_argument("--sort", default=None)
a = ap.parse_args()
cfg = load_config(_bootstrap.DEFAULT_CFG)
res = resolve(cfg.paths.results_dir, project_root(cfg.paths.get("root")))
runs = a.runs_dir or os.path.join(res, "runs")
prefixes = a.prefix or [""]
KEYS = ["test_MAE", "test_RMSE", "test_WAPE", "test_CRPS_q", "test_PICP90_raw", "test_PICP90_aci_mean", "val_MAE",
        "best_val_pinball", "best_val_loss", "epochs_run", "seconds", "n_params"]
rows = []
for pf in prefixes:
    for d in sorted(glob.glob(os.path.join(runs, pf + "*"))):
        f = os.path.join(d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f, encoding="utf-8"))
        row = {"run": os.path.basename(d), **{k: s.get(k) for k in KEYS}}
        pp_f = os.path.join(d, "price_params.json")
        if os.path.exists(pp_f):
            pp = json.load(open(pp_f, encoding="utf-8"))
            sc = pp.get("switch_consistency_all") or {}
            row.update({"switch_beta_obs": sc.get("beta_obs"), "switch_beta_model": sc.get("beta_model")})
        cfg_f = os.path.join(d, "config.yaml")
        if os.path.exists(cfg_f):
            c = load_config(cfg_f)
            row.update({"D": c.model.get("D"), "lr": c.train.get("lr"), "dropout": c.model.get("dropout"),
                        "seed": c.train.get("seed")})
        rows.append(row)
df = pd.DataFrame(rows)
if a.sort and a.sort in df:
    df = df.sort_values(a.sort)
out = ensure_dir(os.path.join(res, "comparisons"))
name = (prefixes[0] or "all").rstrip("_")
df.to_csv(os.path.join(out, f"runs_{name}.csv"), index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
