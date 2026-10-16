"""Semi-synthetic benchmark v3: elasticity implied by each method against the alignment rho of the schedule with demand.

Input: results/comparisons/ess_v3_runs.csv (scripts/summarize_ess_v3.py). Panels: scarce / rich price variation (rows)
x true elasticity -0.2 / -1.0 (columns). Points are means over five seeds, bars one standard deviation across seeds; the
dashed line is the true implied elasticity. Endogenous-shift runs are excluded (they are reported in the text).
Usage: python figures/scripts/fig_ess_v3.py [runs.csv] [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import FS, INK, INK2, LW, MM, PAL, RULE, plt, setup_rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "comparisons", "ess_v3_runs.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "results")
os.makedirs(out, exist_ok=True)
setup_rc()
d = pd.read_csv(src)
d = d[d.endo == 0]
ARMS = [  # (arm, label, colour, marker, x offset)
    ("A6input", "Price as input", PAL["price"]["stroke"], "s", -0.075),
    ("Bproj0", "Learned, sign-constrained", PAL["cov"]["stroke"], "^", -0.025),
    ("A0pooled", "Anchored (cut)", PAL["causal"]["stroke"], "D", 0.025),
    ("T2combo", "Adaptive anchoring", PAL["demand"]["stroke"], "o", 0.075),
]
PANELS = [("scarce", -0.2), ("scarce", -1.0), ("rich", -0.2), ("rich", -1.0)]
TITLES = {"scarce": "scarce variation", "rich": "rich variation"}
fig, axes = plt.subplots(2, 2, figsize=(190 * MM, 112 * MM), sharex=True)
for k, (ax, (st, bt)) in enumerate(zip(axes.ravel(), PANELS)):
    x = d[(d.setting == st) & (d.beta_true == bt)]
    truth = float(x.true_implied.iloc[0])
    ax.axhline(0, color=RULE, lw=LW["thin"], zorder=1)
    ax.axhline(truth, color=INK2, lw=LW["flow"], ls=(0, (4, 2)), zorder=1)
    for arm, lab, col, mk, dx in ARMS:
        g = x[x.arm == arm].groupby("rho").implied.agg(["mean", "std"]).reset_index()
        ax.errorbar(g.rho + dx, g["mean"], yerr=g["std"], color=col, lw=LW["flow"], marker=mk, ms=3.6, mew=0,
                    elinewidth=LW["thin"], capsize=0, zorder=3, label=lab)
    ax.set_title(f"({'abcd'[k]}) {TITLES[st].capitalize()}, true elasticity \u2212{abs(bt):.1f}", fontsize=FS["label"], color=INK,
                 loc="left")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(-1.2, 1.2)
    ax.tick_params(labelsize=FS["body"], colors=INK2, width=LW["hair"], length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(LW["hair"])
        ax.spines[s].set_color(RULE)
for ax in axes[1]:
    ax.set_xlabel(r"Alignment of the schedule with demand, $\rho$", fontsize=FS["body"], color=INK)
for ax in axes[:, 0]:
    ax.set_ylabel("Implied elasticity", fontsize=FS["body"], color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D  # noqa: E402
h.append(Line2D([], [], color=INK2, lw=LW["flow"], ls=(0, (4, 2))))
l.append("True value")
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=FS["body"], bbox_to_anchor=(0.5, 1.0),
           handletextpad=0.4, columnspacing=1.3)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2, w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"Fig_R_ess_v3.{ext}"), dpi=600 if ext == "png" else None)
print("saved", out)
