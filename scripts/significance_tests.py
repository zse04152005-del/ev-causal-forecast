"""显著性检验：主模型对各基线的 Diebold–Mariano（按预测起点）与 Wilcoxon（按小区）。

输入：comparisons/dm/loss_*.npz（由 scripts/loss_series.py 从 predictions_test.npz 压缩而来；
      键 `<组>|mae_o`[起点, 步长]、`|mae_z`[小区]、`|crps_o`、`|crps_z`，都是先算每个种子的损失再对种子取平均）
输出：comparisons/dm_wilcoxon.csv（长表）与 dm_wilcoxon_overall.csv（"全部步长"一行一个基线，含 Holm 校正）

用法：python scripts/significance_tests.py --dir <comparisons 目录> [--ref v2_main_A0]
说明：DM 的长期方差用 h−1 阶自协方差（h = 预测步长；"全部步长"取 24），HLN 小样本修正。
      差值 = 参照模型 − 对比模型，负值表示参照模型更好。
      损失只在保存了逐点预测的 9 个步长（1–6、9、12、24）上计算，"全部步长"是这 9 个步长的等权平均，
      与汇总表里 1–24 步平均的口径不同。
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.eval.stats import dm_test, wilcoxon_zones

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--ref", default="v2_main_A0")
a = ap.parse_args()

L = {}
for f in sorted(glob.glob(os.path.join(a.dir, "dm", "loss_*.npz"))):
    z = np.load(f)
    for k in z.files:
        L[k] = z[k]
steps = [int(s) for s in L["steps"]]
groups = sorted({k.split("|")[0] for k in L if "|" in k})
assert a.ref in groups, groups
rows = []
for g in groups:
    if g == a.ref or "@" in g:
        continue
    ns = len(L[f"{g}|seed_mae"])
    # 对比组只有 3 个种子（消融）时，参照模型也只用种子 0–2 的平均（键名带 @3），避免种子构成不同带来的差异
    ref = f"{a.ref}@{ns}" if f"{a.ref}@{ns}|mae_o" in L else a.ref
    for met in ("mae", "crps"):
        r_o, g_o = L[f"{ref}|{met}_o"], L[f"{g}|{met}_o"]
        w = wilcoxon_zones(L[f"{ref}|{met}_z"], L[f"{g}|{met}_z"])
        for j, s in list(enumerate(steps)) + [(None, "all")]:
            x, y = (r_o[:, j], g_o[:, j]) if j is not None else (np.nanmean(r_o, 1), np.nanmean(g_o, 1))
            d = dm_test(x, y, h=s if j is not None else 24)
            rows.append({"ref": ref, "vs": g, "seeds": ns, "metric": met, "step": s, "ref_loss": float(np.nanmean(x)),
                         "vs_loss": float(np.nanmean(y)), "rel_diff_pct": 100 * (np.nanmean(x) / np.nanmean(y) - 1),
                         "dm": d["dm"], "dm_p": d["p"], "n_origins": d["n"],
                         "share_origins_better": float(np.mean(x < y)),
                         **({"wilcoxon_p": w["p"], "share_zones_better": w["share_better"], "n_zones": w["n"]}
                            if j is None else {})})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.dir, "dm_wilcoxon.csv"), index=False)
ov = df[df.step == "all"].copy()
for met in ("mae", "crps"):                      # Holm 校正：每个指标内对全部基线
    for col in ("dm_p", "wilcoxon_p"):
        m = ov.metric == met
        p = ov.loc[m, col].values
        o = np.argsort(p)
        adj = np.empty(len(p))
        run = 0.0
        for rank, i in enumerate(o):
            run = max(run, min(1.0, (len(p) - rank) * p[i]))
            adj[i] = run
        ov.loc[m, col + "_holm"] = adj
ov.to_csv(os.path.join(a.dir, "dm_wilcoxon_overall.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ov.drop(columns=["ref", "step", "n_origins", "n_zones"]).round(4).to_string(index=False))
print("\n逐步长 MAE 相对差（%），* 表示 DM p<0.05")
t = df[(df.metric == "mae") & (df.step != "all")]
print(t.assign(v=[f"{r:+.1f}{'*' if p < .05 else ''}" for r, p in zip(t.rel_diff_pct, t.dm_p)]).pivot(index="vs", columns="step", values="v")[steps].to_string())
t = df[(df.metric == "crps") & (df.step != "all")]
print("\n逐步长 CRPS 相对差（%）")
print(t.assign(v=[f"{r:+.1f}{'*' if p < .05 else ''}" for r, p in zip(t.rel_diff_pct, t.dm_p)]).pivot(index="vs", columns="step", values="v")[steps].to_string())
