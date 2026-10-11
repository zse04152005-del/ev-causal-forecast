"""比较多次运行：指标表、Diebold–Mariano 检验（逐预测起点）、按小区 Wilcoxon 检验。

用法：python scripts/evaluate.py --runs 运行名1 运行名2 ... [--ref 运行名1] [--name 比较名] [--paper]
--paper：生成论文用表格时，拒绝任何使用占位锚定的运行（设计文档 6.7）。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.eval.stats import dm_test, wilcoxon_zones
from src.utils.config import load_config
from src.utils.paths import project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ref", default=None)
ap.add_argument("--name", default="comparison")
ap.add_argument("--paper", action="store_true")
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
a = ap.parse_args()
cfg0 = load_config(a.config)
root = project_root(cfg0.paths.get("root"))
runs_dir = os.path.join(resolve(cfg0.paths.results_dir, root), "runs")
ref = a.ref or a.runs[0]

info, preds = {}, {}
for r in a.runs:
    d = os.path.join(runs_dir, r)
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    if a.paper and s.get("anchor_placeholder"):
        raise SystemExit(f"{r} 使用了占位锚定，不能进入论文表格（去掉 --paper 可以查看）")
    info[r] = s
    preds[r] = np.load(os.path.join(d, "predictions_test.npz"))

cfg = load_config(os.path.join(runs_dir, ref, "config.yaml"))
P = prepare(cfg)
p0 = preds[ref]
origins, steps, q = p0["origins"], list(p0["steps"]), list(np.round(p0["quantiles"], 4))
for r in a.runs:
    if not np.array_equal(preds[r]["origins"], origins) or list(preds[r]["steps"]) != steps:
        raise SystemExit(f"{r} 的预测起点或步长与参照运行 {ref} 不一致，不能逐点比较")
f = origins[:, None] + np.array(steps)[None, :]
y, v = P.y[f], P.valid[f]                                   # [n, S, N]
mi = q.index(0.5)

rows, dm_rows = [], []
err = {r: np.abs(preds[r]["yq"][..., mi].astype(np.float32) - y) for r in a.runs}
for r in a.runs:
    e = err[r]
    rows.append({"run": r, **{f"MAE_h{s}": float((e[:, k] * v[:, k]).sum() / v[:, k].sum()) for k, s in enumerate(steps)},
                 "test_MAE_all_h": info[r].get("test_MAE"), "CRPS_q": info[r].get("test_CRPS_q"),
                 "PICP90_aci": info[r].get("test_PICP90_aci_mean"), "anchor": info[r].get("anchor_source", "-")})
    if r == ref:
        continue
    for k, s in enumerate(steps):
        l1 = (err[r][:, k] * v[:, k]).sum(1) / np.maximum(v[:, k].sum(1), 1)
        l0 = (err[ref][:, k] * v[:, k]).sum(1) / np.maximum(v[:, k].sum(1), 1)
        z1 = (err[r][:, k] * v[:, k]).sum(0) / np.maximum(v[:, k].sum(0), 1)
        z0 = (err[ref][:, k] * v[:, k]).sum(0) / np.maximum(v[:, k].sum(0), 1)
        dm_rows.append({"run": r, "ref": ref, "step": s, **{f"dm_{kk}": vv for kk, vv in dm_test(l1, l0, h=s).items()},
                        **{f"wilcoxon_{kk}": vv for kk, vv in wilcoxon_zones(z1, z0).items()}})
out = os.path.join(resolve(cfg0.paths.results_dir, root), "comparisons", a.name)
os.makedirs(out, exist_ok=True)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(out, "table.csv"), index=False)
pd.DataFrame(dm_rows).to_csv(os.path.join(out, "dm_wilcoxon.csv"), index=False)
print(tab.round(4).to_string(index=False))
if dm_rows:
    print(pd.DataFrame(dm_rows)[["run", "step", "dm_dm", "dm_p", "wilcoxon_p", "wilcoxon_share_better"]].round(4).to_string(index=False))
print(f"已写出 {out}")
