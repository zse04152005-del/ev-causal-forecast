"""阶段 6 诊断图（图 S：5 分钟站点级切换点估计的事件研究、安慰剂与规格曲线）。

输入：05_实验结果/因果估计/5min/{event_study_C,event_study_D,placebo,spec_curve}.csv
输出：Fig_S_causal_5min.{pdf,png,svg}（190 × 62 mm）

运行：python fig_causal_5min.py [结果目录] [输出目录]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import FS, INK, INK2, PAL, RULE, fix_svg_fonts, setup_rc  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results_cloud", "因果估计", "5min")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
COL_D, COL_C = PAL["causal"]["stroke"], PAL["gray"]["stroke"]
FILL_D, FILL_C = PAL["causal"]["mid"], PAL["gray"]["mid"]

setup_rc()
fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 72 * MM), gridspec_kw={"wspace": 0.45, "left": 0.075, "right": 0.995,
                                                                       "top": 0.87, "bottom": 0.36})


def panel_label(ax, s, title):
    ax.text(0.0, 1.13, s, transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold", va="bottom", ha="left")
    ax.text(0.13, 1.13, title, transform=ax.transAxes, fontsize=FS["label"], fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)


# ---------------------------------------------------------------- (a) 事件研究
ax = axs[0]
for dsg, col, fc, lab in (("C", COL_C, FILL_C, "Design C (across stations)"), ("D", COL_D, FILL_D, "Design D (within station)")):
    f = os.path.join(RES, f"event_study_{dsg}.csv")
    if not os.path.exists(f):
        continue
    es = pd.read_csv(f)
    ax.fill_between(es.minutes, es.beta - 1.96 * es.se, es.beta + 1.96 * es.se, color=fc, alpha=0.6, lw=0)
    ax.plot(es.minutes, es.beta, color=col, lw=1.0, marker="o", ms=2.2, label=lab)
ax.axhline(0, color=RULE, lw=0.5)
ax.axvline(0, color=INK2, lw=0.5, ls=(0, (2, 2)))
ax.set_xlabel("Minutes since price switch", fontsize=7)
ax.set_ylabel("Log demand change per\nunit log price change", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, handlelength=1.4)
clean(ax)
panel_label(ax, "(a)", "Event study")

# ---------------------------------------------------------------- (b) 安慰剂
ax = axs[1]
pl = os.path.join(RES, "placebo.csv")
sp = os.path.join(RES, "spec_curve.csv")
real = None
if os.path.exists(sp):
    sc = pd.read_csv(sp)
    real = sc[(sc.est == "PPML") & (sc.W_min == 60) & (sc.donut_min == 0) & (sc.outcome == "duration") & (~sc.exposures)]
if os.path.exists(pl):
    p = pd.read_csv(pl)
    for dsg, col, off in (("C", COL_C, -4), ("D", COL_D, 4)):
        q = p[p.design == dsg]
        xs = list(q.shift_min + off)
        ys = list(q.beta)
        es_ = list(q.se)
        if real is not None and len(real[real.design == dsg]):
            r = real[real.design == dsg].iloc[0]
            xs.append(0 + off)
            ys.append(r.beta)
            es_.append(r.se)
        xs, ys, es_ = map(np.array, (xs, ys, es_))
        real_mask = np.isclose(xs, off)
        ax.errorbar(xs[~real_mask], ys[~real_mask], yerr=1.96 * es_[~real_mask], fmt="o", ms=2.6, color=col, mfc="white",
                    mec=col, lw=0.7, capsize=1.2, label=f"Placebo, design {dsg}")
        if real_mask.any():
            ax.errorbar(xs[real_mask], ys[real_mask], yerr=1.96 * es_[real_mask], fmt="o", ms=3.4, color=col, lw=0.9,
                        capsize=1.5, label=f"Actual switch, design {dsg}")
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xlabel("Shift of switch time (minutes)", fontsize=7)
ax.set_ylabel(r"Estimate ($\beta$, 1-hour window)", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1, handletextpad=0.3, bbox_to_anchor=(0.5, -0.28),
          labelspacing=0.15)
clean(ax)
panel_label(ax, "(b)", "Placebo shifts")

# ---------------------------------------------------------------- (c) 规格曲线
ax = axs[2]
if os.path.exists(sp):
    sc = pd.read_csv(sp)
    base = sc[(sc.est == "PPML") & (sc.outcome == "duration") & (sc.donut_min == 0) & (~sc.exposures)]
    for dsg, col, mk in (("A", "#B0B0B0", "s"), ("B", "#8C8C8C", "s"), ("C", COL_C, "o"), ("D", COL_D, "o")):
        q = base[base.design == dsg].sort_values("W_min")
        if len(q) == 0:
            continue
        off = {"A": -3.6, "B": -1.2, "C": 1.2, "D": 3.6}[dsg]
        ax.errorbar(q.W_min + off, q.beta, yerr=1.96 * q.se, fmt=mk, ms=2.6, color=col, lw=0.7, capsize=1.2,
                    label=f"Design {dsg}")
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xlabel("Half-window around switch (minutes)", fontsize=7)
ax.set_ylabel(r"Estimate ($\beta$)", fontsize=7)
ax.set_xticks([30, 60, 120, 180])
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.8, handletextpad=0.3)
clean(ax)
panel_label(ax, "(c)", "Designs and windows")

base = os.path.join(OUT, "Fig_S_causal_5min")
fig.savefig(base + ".pdf")
fig.savefig(base + ".png", dpi=600)
fig.savefig(base + ".svg")
fix_svg_fonts(base + ".svg")
print("saved", base)
