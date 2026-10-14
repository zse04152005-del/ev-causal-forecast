"""Event study of design D by direction of the price change (results/causal/round2/event_study_D_direction.csv).

Each point is the PPML slope of log(mu_k / mu_pre) on the switch size x for a 15-min bin k, where mu_pre is the mean
over the 180-min pre window; bars are 95% intervals with zone-clustered standard errors. A negative slope means that
charging rises after a price decrease (x < 0) and falls after an increase (x > 0).
Usage: python figures/scripts/fig_event_direction.py [csv] [out_dir]
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import FS, INK, INK2, LW, MM, PAL, RULE, plt, setup_rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "causal", "round2", "event_study_D_direction.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "results")
os.makedirs(out, exist_ok=True)
setup_rc()
es = pd.read_csv(csv)
C = PAL["causal"]["stroke"]
panels = (("all", "(a) All switches"), ("up", "(b) Price increases"), ("down", "(c) Price decreases"))
fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 58 * MM), sharey=True)
for ax, (key, title) in zip(axes, panels):
    d = es[es.direction == key].sort_values("minutes")
    m = d.minutes.to_numpy()
    lo, hi = d.beta - 1.96 * d.se, d.beta + 1.96 * d.se
    ax.fill_between(m, lo, hi, color=PAL["causal"]["mid"], lw=0, zorder=1)
    ax.plot(m, d.beta, color=C, lw=LW["flow"] * 1.2, ls="-", zorder=3)
    ax.scatter(m, d.beta, s=5, color=C, linewidth=0, zorder=4)
    ax.axhline(0, color=RULE, lw=LW["thin"], zorder=2)
    ax.axvline(0, color=INK2, lw=LW["thin"], ls=(0, (3, 2)), zorder=2)
    ax.set_title(title, fontsize=FS["label"], color=INK, loc="left")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_xlim(-175, 175)
    ax.set_xlabel("Minutes from price switch", fontsize=FS["body"], color=INK)
    ax.tick_params(labelsize=FS["body"], colors=INK2, width=LW["hair"], length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(LW["hair"])
        ax.spines[s].set_color(RULE)
axes[0].set_ylabel("Slope on switch size (semi-elasticity)", fontsize=FS["body"], color=INK)
fig.tight_layout(w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"Fig_S_event_direction.{ext}"), dpi=600 if ext == "png" else None)
print("saved", out)
