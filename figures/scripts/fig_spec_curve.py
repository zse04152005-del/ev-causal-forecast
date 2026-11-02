"""Specification curve of design D (results/causal/5min/spec_curve.csv, from estimate_causal_5min.py --stages spec).

Top: the 36 estimates of design D (PPML and log-ratio OLS; half-window 30-180 min; with and without a 10-min donut;
duration, occupancy and charged energy as the outcome; with and without exposure terms), sorted, with 95% intervals
from zone-clustered standard errors; the anchor (PPML, 180 min, duration, exposure terms) is filled, and the dashed
line is the pooled slope. Bottom: which choice each estimate uses.
Usage: python figures/scripts/fig_spec_curve.py [csv] [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import FS, INK, INK2, LW, MM, PAL, RULE, plt, setup_rc  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "causal", "5min", "spec_curve.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "results")
os.makedirs(out, exist_ok=True)
POOLED = -1.58
setup_rc()
d = pd.read_csv(csv)
d = d[d.design == "D"].sort_values("beta").reset_index(drop=True)
anchor = ((d.est == "PPML") & (d.W_min == 180) & (d.donut_min == 0) & (d.outcome == "duration") & d.exposures).to_numpy()
C = PAL["causal"]["stroke"]
x = np.arange(len(d))
fig, (ax, bx) = plt.subplots(2, 1, figsize=(190 * MM, 112 * MM), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.25]})
ax.vlines(x, d.beta - 1.96 * d.se, d.beta + 1.96 * d.se, color=PAL["causal"]["mid"], lw=LW["thin"] * 2.2, zorder=1)
ax.scatter(x[~anchor], d.beta[~anchor], s=9, facecolor="white", edgecolor=C, linewidth=LW["thin"], zorder=3)
ax.scatter(x[anchor], d.beta[anchor], s=14, color=C, zorder=4)
ax.axhline(0, color=RULE, lw=LW["thin"], zorder=0)
ax.axhline(POOLED, color=INK2, lw=LW["thin"], ls=(0, (3, 2)), zorder=0)
ax.text(len(d) - 0.5, POOLED + 0.06, "pooled slope", ha="right", va="bottom", fontsize=FS["body"], color=INK2)
ax.set_ylabel("Elasticity (95% interval)", fontsize=FS["body"], color=INK)
rows = [("Estimator", "est", [("PPML", "PPML"), ("OLS_log_eps0.01", "log-ratio OLS")]),
        ("Half-window", "W_min", [(30, "30 min"), (60, "60 min"), (120, "120 min"), (180, "180 min")]),
        ("Donut", "donut_min", [(0, "none"), (10, "10 min")]),
        ("Outcome", "outcome", [("duration", "duration"), ("occupancy", "occupancy"), ("volume", "energy")]),
        ("Exposure terms", "exposures", [(False, "no"), (True, "yes")])]
ticks, labels, y = [], [], 0
for title, col, opts in rows:
    for val, lab in opts:
        m = (d[col] == val).to_numpy()
        bx.scatter(x[m], np.full(m.sum(), -y), s=5, marker="s", color=np.where(anchor[m], C, INK2), linewidth=0)
        ticks.append(-y)
        labels.append(f"{title}: {lab}" if val == opts[0][0] else lab)
        y += 1
    y += 0.9
bx.set_yticks(ticks)
bx.set_yticklabels(labels)
bx.set_ylim(-y + 0.4, 0.8)
bx.set_xlabel("Specifications of design D, sorted by the estimate", fontsize=FS["body"], color=INK)
bx.set_xticks([])
for a_ in (ax, bx):
    a_.tick_params(labelsize=FS["body"], colors=INK2, width=LW["hair"], length=2.5)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a_.spines[s].set_linewidth(LW["hair"])
        a_.spines[s].set_color(RULE)
bx.spines["bottom"].set_visible(False)
fig.tight_layout(h_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"Fig_S_spec_curve.{ext}"), dpi=600 if ext == "png" else None)
neg = int((d.beta < 0).sum())
excl = int(((d.beta - 1.96 * d.se) > POOLED).sum())
print(f"saved {out}: {len(d)} specifications, {neg} negative, {excl} with the 95% interval above the pooled slope")
