"""显著性检验：主模型对各基线的 Diebold–Mariano（按预测起点）与 Wilcoxon（按小区）。

输入：comparisons/dm/loss_*.npz（由 scripts/loss_series.py 从 predictions_test.npz 压缩而来；
      键 `<组>|mae_o`[起点, 步长]、`|mae_z`[小区]、`|crps_o`、`|crps_z`，都是先算每个种子的损失再对种子取平均）
输出：comparisons/dm_wilcoxon.csv（长表）与 dm_wilcoxon_overall.csv（"全部步长"一行一个基线，含 Holm 校正）
Holm 校正的范围：论文每张表报告的比较各成一族（表 2 的 15 个基线、表 3 的 13 个变体），每个指标分开校正；
      列 dm_p_holm_table2 / dm_p_holm_table3（不属于该表的行为空）。其余比较不进入任何一族。

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
# Holm families: the comparisons reported in each table of the paper (rows of make_tables.py and make_exp_tables.py)
FAMILIES = {
    "table2": ["r2_fcnn", "r2_lstm", "r2_stid", "r2_dlinear", "r2_gbdt", "r2_gcn", "r2_gcnlstm", "r2_stgcn", "r2_astgcn",
               "r2_agcrn", "v2_gwn_blind", "r2_staeformer", "base_stgcn_price", "base_agcrn_price", "r2_pag_util"],
    "table3": ["v2_gwn_blind", "v2_abl_A6_priceinput", "v2_abl_A2_none", "v2_abl_A1_free", "r2_proj0", "v2_abl_A13_soft1",
               "v2_abl_A3_prior-0.45", "v2_abl_A3_prior-0.76", "v2_abl_A3_prior-1.48", "r2_nograph", "main_A0",
               "abl_A11_predefined", "v2_abl_A10_staeformer"],
}


def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


for fam, members in FAMILIES.items():
    missing = sorted(set(members) - set(ov.vs))
    if missing:
        print(f"注意：{fam} 中没有损失序列的比较（不参与校正）：{missing}")
    for met in ("mae", "crps"):
        m = (ov.metric == met) & ov.vs.isin(members)
        for col in ("dm_p", "wilcoxon_p"):
            ov.loc[m, f"{col}_holm_{fam}"] = holm(ov.loc[m, col].values)
ov.to_csv(os.path.join(a.dir, "dm_wilcoxon_overall.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(ov.drop(columns=["ref", "step", "n_origins", "n_zones"]).round(4).to_string(index=False))
print("\n逐步长 MAE 相对差（%），* 表示 DM p<0.05")
t = df[(df.metric == "mae") & (df.step != "all")]
print(t.assign(v=[f"{r:+.1f}{'*' if p < .05 else ''}" for r, p in zip(t.rel_diff_pct, t.dm_p)]).pivot(index="vs", columns="step", values="v")[steps].to_string())
t = df[(df.metric == "crps") & (df.step != "all")]
print("\n逐步长 CRPS 相对差（%）")
print(t.assign(v=[f"{r:+.1f}{'*' if p < .05 else ''}" for r, p in zip(t.rel_diff_pct, t.dm_p)]).pivot(index="vs", columns="step", values="v")[steps].to_string())
