"""生成论文主结果表（表 3：事实预测精度，含 DM 检验标记）的 LaTeX 与 CSV。

输入：comparisons/groups_mean.csv、dm_wilcoxon_overall.csv（先跑 summarize_all.py 与 significance_tests.py）
输出：<out>/tab_main_results.tex（booktabs + threeparttable，可直接 \\input）、tab_main_results.csv

用法：python scripts/make_tables.py --dir <comparisons 目录> [--out <输出目录>]
口径：测试期、利用率、只算质量掩码有效的点；MAE / RMSE / CRPS 为 1–24 步平均，表中数值 × 100；± 为跨种子标准差。
      显著性：DM 检验（9 个保存步长的等权平均，h = 24，HLN 修正），Holm 校正后 p < 0.05。
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--ref", default="v2_main_A0")
a = ap.parse_args()
out = a.out or a.dir
os.makedirs(out, exist_ok=True)
g = pd.read_csv(os.path.join(a.dir, "groups_mean.csv")).set_index("group")
dm = pd.read_csv(os.path.join(a.dir, "dm_wilcoxon_overall.csv"))
dm = dm[dm.metric == "mae"].set_index("vs")

ROWS = [("Reference methods", [("baseline_naive_last_utilization", "Last value", "--"),
                               ("baseline_naive_seasonal_utilization", "Seasonal naive", "--"),
                               ("baseline_profile_utilization", "Historical profile", "--")]),
        ("Networks without a graph", [("base_fcnn", "FCNN", "--"), ("base_lstm", "LSTM", "--")]),
        ("Spatio-temporal graph networks", [("base_gcn", "GCN", "--"), ("base_gcnlstm", "GCN-LSTM", "--"),
                                            ("base_stgcn", "STGCN", "--"), ("base_astgcn", "ASTGCN", "--"),
                                            ("base_agcrn", "AGCRN", "--"), ("base_gwn_blind", "Graph WaveNet", "--"),
                                            ("v2_staeformer_blind", "STAEformer", "--")]),
        ("Price as an ordinary input", [("base_stgcn_price", "STGCN + price", "input"),
                                        ("base_agcrn_price", "AGCRN + price", "input")]),
        ("Proposed", [(a.ref, "\\model{}", "anchored")])]
COLS = [("MAE", True), ("RMSE", True), ("CRPS", True), ("MAE_1", False), ("MAE_6", False), ("MAE_24", False)]
keys = [k for _, rs in ROWS for k, _, _ in rs]
rank = {c: np.argsort(np.argsort(g.loc[keys, c].values)) for c, _ in COLS}       # 0 = 最好


def cell(k, c, with_sd):
    v, sd = g.loc[k, c] * 100, g.loc[k].get(c + "_std", np.nan) * 100
    s = f"{v:.2f}"
    r = rank[c][keys.index(k)]
    s = f"\\textbf{{{s}}}" if r == 0 else f"\\underline{{{s}}}" if r == 1 else s
    if with_sd and np.isfinite(sd):
        s += f"{{\\tiny\\,$\\pm${sd:.2f}}}"
    return s


def mark(k):
    if k == a.ref or k not in dm.index:
        return ""
    r = dm.loc[k]
    if r.dm_p_holm >= 0.05:
        return ""
    return "$^{\\ast}$" if r.rel_diff_pct < 0 else "$^{\\circ}$"


SDS = (g.loc[[k for k in keys if np.isfinite(g.loc[k].get("MAE_std", np.nan))], "MAE_std"] * 100)
HEAD = ["\\begin{table}", "\\tblcaption{tab:main}{Forecast accuracy on the test period (errors $\\times100$, averaged over horizons of 1 to 24~h).}",
        "{\\tblfont\\setlength{\\tabcolsep}{1pt}", "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccccc@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{Average over 1--24 h} & \\multicolumn{3}{c}{MAE at horizon} \\\\",
        "\\cmidrule{2-4}\\cmidrule{5-7}",
        "Model & MAE & RMSE & CRPS & 1 h & 6 h & 24 h \\\\", "\\midrule"]
L = list(HEAD)
flat = []
for gi, (head, rs) in enumerate(ROWS):
    if gi:
        L.append("\\addlinespace[2pt]")
    L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{head}}}}} \\\\")
    for k, name, price in rs:
        cells = [cell(k, c, False) for c, sd in COLS]
        cells[0] += mark(k)
        L.append(f"\\hspace{{0.6em}}{name} & " + " & ".join(cells) + " \\\\")
        flat.append({"model": name, "group": k, "price": price, "seeds": int(g.loc[k, "n"]),
                     **{c: g.loc[k, c] * 100 for c, _ in COLS},
                     **{c + "_sd": g.loc[k].get(c + "_std", np.nan) * 100 for c, sd in COLS if sd},
                     "dm_rel_diff_pct": dm.rel_diff_pct.get(k, np.nan), "dm_p": dm.dm_p.get(k, np.nan),
                     "dm_p_holm": dm.dm_p_holm.get(k, np.nan), "wilcoxon_p_holm": dm.wilcoxon_p_holm.get(k, np.nan),
                     "share_zones_ref_better": dm.share_zones_better.get(k, np.nan)})
L += ["\\bottomrule", "\\end{tabular*}\\par}",
      "\\tblnote{Bold: best; underlined: second best. $^{\\ast}$ Significantly less accurate than \\model{} "
      "(Diebold--Mariano test, Holm-adjusted $p<0.05$).}",
      "\\end{table}"]
open(os.path.join(out, "tab_main_results.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
# quoted in Section 4.3 (no longer in the table note)
print(f"seed sd of MAE x100: {SDS.min():.2f} to {SDS.max():.2f}; reference {SDS[a.ref]:.2f}")
pd.DataFrame(flat).to_csv(os.path.join(out, "tab_main_results.csv"), index=False)
print(pd.DataFrame(flat).round(3).to_string(index=False))
