"""论文第 4 节的消融表（tab_ablation）；情景结果另存 tab_scenario.csv 供核对（不输出 tex）。数字全部来自结果文件，不要手改 .tex。

输入：comparisons/all_runs_flat.csv、dm_wilcoxon_overall.csv（scripts/summarize_all.py、significance_tests.py）、
      learned_elasticity.csv（scripts/learned_elasticity.py，可训练弹性按 5% 涨价读出的口径）；
      scenarios/scenario_summary.csv（scripts/simulate_scenarios.py）
用法（在仓库根目录下）：
    python scripts/make_exp_tables.py --cmp results/comparisons --scen results/scenarios --out results/tables
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--cmp", required=True)
ap.add_argument("--scen", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def num(x, d=2):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return ("$-$" + s[1:]) if s.startswith("-") else s


# ------------------------------------------------------------------ 消融表
runs = pd.read_csv(os.path.join(a.cmp, "all_runs_flat.csv"))
runs["seed"] = runs.run.str.extract(r"_s(\d+)$").astype(float)
dm = pd.read_csv(os.path.join(a.cmp, "dm_wilcoxon_overall.csv"))
dm = dm[dm.metric == "mae"].set_index(["ref", "vs"])
REF = "v2_main_A0"
# elasticity implied by a 5% price rise in the TOU zones (scripts/learned_elasticity.py), used for trained elasticities
le_f = os.path.join(a.cmp, "learned_elasticity.csv")
LE = pd.read_csv(le_f).groupby("group").beta_weighted.mean() if os.path.exists(le_f) else pd.Series(dtype=float)
ROWS = [
    ("row", "\\model{} (reference)", REF, 10, "$-0.21$, fixed"),
    ("head", "Treatment of the price response", None, None, None),
    ("row", "Price-blind", "v2_gwn_blind", 10, "--"),
    ("row", "Price as input", "v2_abl_A6_priceinput", 3, "implicit"),
    ("row", "Trainable, no constraint", "v2_abl_A2_none", 3, "learned"),
    ("row", "Trainable, sign constraint", "v2_abl_A1_free", 3, "learned"),
    ("row", "From zero, sign constraint", "r2_proj0", 10, "learned"),
    ("row", "Soft anchor", "v2_abl_A13_soft1", 3, "learned"),
    ("row", "Fixed at $-0.45$", "v2_abl_A3_prior-0.45", 3, "$-0.45$, fixed"),
    ("row", "Fixed at $-0.76$", "v2_abl_A3_prior-0.76", 3, "$-0.76$, fixed"),
    ("row", "Fixed at $-1.48$", "v2_abl_A3_prior-1.48", 3, "$-1.48$, fixed"),
    ("head", "Graph and backbone", None, None, None),
    ("row", "No graph", "r2_nograph", 10, "$-0.21$, fixed"),
    ("row", "Predefined + adaptive", "main_A0", 5, "$-0.21$, fixed"),
    ("row", "Predefined only", "abl_A11_predefined", 3, "$-0.21$, fixed"),
    ("row", "STAEformer", "v2_abl_A10_staeformer", 3, "$-0.21$, fixed"),
]
L = ["\\begin{table}", "\\tblcaption{tab:ablation}{Results of the ablation experiment (errors $\\times100$).}",
     "{\\tblfont\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccr@{}}", "\\toprule",
     "Variant & $\\beta$ & MAE & CRPS & $\\Delta$MAE (\\%) \\\\", "\\midrule"]
out_rows = []
for kind, lab, g, ns, el in ROWS:
    if kind == "head":
        L.append("\\addlinespace[2pt]")
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        continue
    d = runs[(runs.group == g) & (runs.seed < ns)]
    r = runs[(runs.group == REF) & (runs.seed < ns)]
    assert len(d) == ns and len(r) == ns, (g, len(d), len(r))
    mae, sd, crps, csd = d.MAE.mean() * 100, d.MAE.std(ddof=1) * 100, d.CRPS.mean() * 100, d.CRPS.std(ddof=1) * 100
    if el == "learned":
        el = f"{num(LE[g] if g in LE.index else d.beta_model.mean())}, learned"
    el = el.replace(", fixed", "").replace(", learned", "$^{\\dagger}$")
    if g == REF:
        delta, mark, p = "", "", np.nan
    else:
        rel = (d.MAE.mean() / r.MAE.mean() - 1) * 100
        key = (REF if ns == 10 else f"{REF}@{ns}", g)            # reference averaged over the same seeds as the variant
        t = dm.loc[key]
        p = float(t.dm_p_holm_table3)
        mark = ""
        if p < 0.05:
            mark = "$^{\\ast}$" if t.rel_diff_pct < 0 else "$^{\\circ}$"       # rel_diff_pct < 0：参照模型（主模型）更准
        delta = ("$+$" if rel >= 0 else "$-$") + f"{abs(rel):.1f}" + mark
    pad = "" if g == REF else "\\hspace{0.6em}"
    L.append(f"{pad}{lab} & {el} & {mae:.2f} & {crps:.2f} & {delta} \\\\")
    out_rows.append({"variant": lab, "group": g, "seeds": ns, "MAE_x100": mae, "MAE_sd": sd, "CRPS_x100": crps, "CRPS_sd": csd,
                     "beta_model": d.beta_model.mean(), "ref_MAE_x100_same_seeds": r.MAE.mean() * 100,
                     "dMAE_pct": np.nan if g == REF else (d.MAE.mean() / r.MAE.mean() - 1) * 100,
                     "dm_rel_diff_pct_9steps": np.nan if g == REF else float(t.rel_diff_pct), "dm_p_holm": p,
                     "PICP90_raw": d.PICP90_raw.mean(), "PICP90_aci": d.PICP90_aci.mean(), "n_params": d.n_params.mean(),
                     "minutes": d.minutes.mean()})
L += ["\\bottomrule", "\\end{tabular*}\\par}",
      # one-line note, as in the template paper; seeds and the meaning of Delta MAE are explained in Section 4.4
      "\\tblnote{$^{\\dagger}$ Trained value. $^{\\ast}$ Reference significantly more accurate; "
      "$^{\\circ}$ variant significantly more accurate.}",
      "\\end{table}"]
open(os.path.join(a.out, "tab_ablation.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
pd.DataFrame(out_rows).to_csv(os.path.join(a.out, "tab_ablation.csv"), index=False)

# ------------------------------------------------------------------ 情景表
ss = pd.read_csv(os.path.join(a.scen, "scenario_summary.csv"))
NAMES = {"flat": "No time-of-use tariff", "spread2": "Peak--valley spread doubled", "evening": "Evening surcharge of 20\\%"}


def ci(r, k, d=1):
    if not np.isfinite(r[k]):
        return "--"
    lo, hi = sorted((r[k + "_lo"], r[k + "_hi"]))
    return f"{num(r[k], d)} [{num(lo, d)}, {num(hi, d)}]"


def pr(share, dlog):
    return "-- & --" if not np.isfinite(dlog) or share == 0 else f"{share * 100:.0f} & {num(dlog * 100, 1)}"


# 2026-10-07：情景结果只在正文和图里给出，不再输出 tab_scenario.tex（精简篇幅）；csv 仍保留供核对
ss.to_csv(os.path.join(a.out, "tab_scenario.csv"), index=False)
print("→", a.out)
