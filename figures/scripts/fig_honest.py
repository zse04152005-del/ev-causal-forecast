"""Sensitivity of the event-study response to violations of parallel trends (results/causal/round3/honest_sd.csv).

Rambachan and Roth (2023), smoothness restriction Delta^SD(M): 95% fixed-length intervals for the average response over
the twelve 15-min bins after the switch (base: the last 15 min before it) as the bound M on the second difference of
the violation grows; M = 0 allows linear trends through the base. The marker at the left is the interval under
parallel trends, the dashed line is the pooled slope and the dashed line marks the pooled comparison slope.
Usage: python figures/scripts/fig_honest.py [round3_dir] [out_dir]
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import FS, INK, INK2, LW, MM, PAL, RULE, plt, setup_rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "causal", "round3")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "results")
os.makedirs(out, exist_ok=True)
setup_rc()
h = pd.read_csv(os.path.join(rd, "honest_sd.csv"))
h = h[h.M <= 0.05]                                                    # display range; the full numerical grid is retained in the CSV
sm = json.load(open(os.path.join(rd, "summary.json"), encoding="utf-8"))["honest"]
C = PAL["causal"]["stroke"]
fig, ax = plt.subplots(figsize=(110 * MM, 70 * MM))
ax.fill_between(h.M, h.lo, h.hi, color=PAL["causal"]["mid"], lw=0, zorder=1)
ax.plot(h.M, h.theta_hat, color=C, lw=LW["flow"], zorder=3)
lo_pt, hi_pt = sm["ci_parallel_trends"]
x0 = -0.012
ax.vlines(x0, lo_pt, hi_pt, color=INK2, lw=LW["thin"] * 2, zorder=3)
ax.scatter([x0], [sm["theta_hat_parallel_trends"]], s=9, color=INK2, zorder=4)
ax.text(x0, hi_pt + 0.08, "parallel\ntrends", ha="center", va="bottom", fontsize=FS["body"] - 1, color=INK2)
ax.axhline(0, color=RULE, lw=LW["thin"], zorder=0)
ax.axhline(sm["pooled"], color=INK2, lw=LW["thin"], ls=(0, (3, 2)), zorder=0)
ax.text(h.M.max(), sm["pooled"] - 0.06, "pooled slope", ha="right", va="top", fontsize=FS["body"], color=INK2)
ax.set_xlim(x0 - 0.012, h.M.max() * 1.02)
ax.set_xlabel("Smoothness bound $M$ (15-min bins)", fontsize=FS["body"], color=INK)
ax.set_ylabel("Mean response, 0–180 min (95% interval)", fontsize=FS["body"], color=INK)
ax.tick_params(labelsize=FS["body"], colors=INK2, width=LW["hair"], length=2.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(LW["hair"])
    ax.spines[s].set_color(RULE)
fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(out, f"Fig_S_honest.{ext}"), dpi=600 if ext == "png" else None)
print("saved", out)
