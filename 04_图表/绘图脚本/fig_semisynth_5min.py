"""阶段 6 半合成验证图（图 S：5 分钟站点级数据上设计 C、D 的已知真值检验）。

输入：05_实验结果/因果估计/5min/semisynth_5min_summary.csv、semisynth_5min_implied.csv（summarize_semisynth_5min.py 生成）
输出：Fig_S_semisynth_5min.{pdf,png,svg}（190 × 70 mm）

运行：python fig_semisynth_5min.py [结果目录] [输出目录]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import FS, INK2, PAL, RULE, fix_svg_fonts, setup_rc  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results_cloud", "因果估计", "5min")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
summ = pd.read_csv(os.path.join(RES, "semisynth_5min_summary.csv"))
imp = pd.read_csv(os.path.join(RES, "semisynth_5min_implied.csv"))
MEANS = sorted(summ.mean_min.dropna().unique())
# 平均会话时长：同一色相由浅到深（顺序量）
RAMP = {30: "#B28AD3", 60: PAL["causal"]["stroke"], 120: "#3A1454"}
COL_C = PAL["gray"]["stroke"]
COL_D = PAL["causal"]["stroke"]

setup_rc()
fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 70 * MM),
                        gridspec_kw={"wspace": 0.5, "left": 0.07, "right": 0.99, "top": 0.86, "bottom": 0.34})


def panel_label(ax, s, title):
    ax.text(0.0, 1.12, s, transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold", va="bottom", ha="left")
    ax.text(0.13, 1.12, title, transform=ax.transAxes, fontsize=FS["label"], fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


WINS = [30, 60, 120, 180]

# ---------------------------------------------------------------- (a) 存量稀释：设计 D 的估计 / 到达弹性
ax = axs[0]
d = summ[summ.design == "D"]
for m in MEANS:
    q = d[d.mean_min == m].sort_values("W_min")
    col = RAMP.get(int(m), COL_D)
    ax.plot(q.W_min, q.truth_slope, color=col, lw=1.0, label=f"{int(m)} min")
    ax.plot(q.W_min, q.stock_slope, ls="none", marker="o", ms=3.0, mfc="white", mec=col, mew=0.8)
ax.axhline(1, color=RULE, lw=0.5, ls=(0, (2, 2)))
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xticks(WINS)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("Half-window around switch (minutes)", fontsize=7)
ax.set_ylabel("Estimate / arrival elasticity", fontsize=7)
h1 = ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3,
               title="Mean session length", title_fontsize=6.5, handlelength=1.4, columnspacing=0.8)
ax.add_artist(h1)
from matplotlib.lines import Line2D  # noqa: E402

ax.legend([Line2D([], [], color=INK2, lw=1.0), Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=INK2)],
          ["Noise-free", "With noise"], frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4)
clean(ax)
panel_label(ax, "(a)", "Stock dilution, design D")

# ---------------------------------------------------------------- (b) β = 0 时的估计：只保留站点平均日内曲线（到达口径，
# 曲线形状与真实时长相同），设计 C 就能复现真实估计；设计 D 为 0
ax = axs[1]
mm = 60 if 60 in MEANS else MEANS[len(MEANS) // 2]
for dsg, col, off in (("C", COL_C, -7), ("D", COL_D, 7)):
    q = summ[(summ.design == dsg) & (summ.mean_min == mm)].sort_values("W_min")
    sd = q.get("flow_b+0.0_sd", pd.Series(np.zeros(len(q)), index=q.index)).fillna(0)
    ax.errorbar(q.W_min + off - 2.5, q["flow_b+0.0"], yerr=1.96 * sd, fmt="o", ms=3.0, color=col, mfc="white", mec=col,
                lw=0.7, capsize=1.2, label=f"Daily profile only, no price effect ({dsg})")
    ax.plot(q.W_min + off + 2.5, q.real_beta, ls="none", marker="o", ms=3.4, color=col, label=f"Real data ({dsg})")
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xticks(WINS)
ax.set_xlabel("Half-window around switch (minutes)", fontsize=7)
ax.set_ylabel(r"Estimate ($\beta$)", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1, columnspacing=0.8,
          handletextpad=0.3, labelspacing=0.15)
clean(ax)
panel_label(ax, "(b)", "No-price-effect check")

# ---------------------------------------------------------------- (c) 与真实估计相符的到达弹性
ax = axs[2]
for i, m in enumerate(MEANS):
    col = RAMP.get(int(m), COL_D)
    q = imp[(imp.mean_min == m) & (imp.W_min != "pooled")].copy()
    q["W"] = q.W_min.astype(float)
    xs = i + np.linspace(-0.27, 0.27, len(q))
    ax.errorbar(xs, q.implied_arrival_beta, yerr=1.96 * q.se, fmt="o", ms=2.4, color=col, mfc="white", mec=col, lw=0.6,
                capsize=1.0)
    p = imp[(imp.mean_min == m) & (imp.W_min == "pooled")].iloc[0]
    ax.errorbar([i], [p.implied_arrival_beta], yerr=[1.96 * p.se], fmt="D", ms=4.0, color=col, lw=1.2, capsize=2.0,
                zorder=3)
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xticks(range(len(MEANS)))
ax.set_xticklabels([f"{int(m)} min" for m in MEANS])
ax.set_xlabel("Assumed mean session length", fontsize=7)
ax.set_ylabel("Implied arrival elasticity", fontsize=7)
ax.legend([Line2D([], [], ls="none", marker="o", ms=2.4, mfc="white", mec=INK2, color=INK2),
           Line2D([], [], ls="none", marker="D", ms=4, color=INK2)],
          ["Single window (30/60/120/180 min)", "Pooled over windows"], frameon=False, fontsize=6.5,
          loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1, handletextpad=0.3)
clean(ax)
panel_label(ax, "(c)", "Back-out elasticity")

base = os.path.join(OUT, "Fig_S_semisynth_5min")
fig.savefig(base + ".pdf")
fig.savefig(base + ".png", dpi=600)
fig.savefig(base + ".svg")
fix_svg_fonts(base + ".svg")
print("saved", base)
