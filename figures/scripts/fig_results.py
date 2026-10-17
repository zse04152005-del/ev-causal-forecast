"""Result figures of the paper: Fig. 5 (context), Fig. 6 (accuracy by horizon), Fig. 7 (validation of the price response)
and Fig. 9 (scenarios). Fig. 8 (benchmark) is drawn by fig_ess_v3.py, Fig. S1 by fig_event_direction.py.

Inputs: results/comparisons/ (all_runs_flat, groups_mean, eid_summary, eps_summary, metrics_by_step, conformal_by_step,
        dm_wilcoxon; scripts/summarize_all.py, collect_by_step.py, significance_tests.py), results/scenarios/,
        results/EDA/, results/causal/5min/ and the UrbanEV shapefile for the map.
Outputs: Fig_R_context, Fig_R_horizon, Fig_R_validation, Fig_R_scenario as pdf / png (600 dpi) / svg, 190 mm wide

运行：python fig_results.py [comparisons 目录] [输出目录]
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import FS, INK, INK2, PAL, RULE, fix_svg_fonts, setup_rc  # noqa: E402


def find_root(start):
    p = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "results")):
            return p
        p = os.path.dirname(p)
    return None


ROOT = find_root(HERE)
CMP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT or HERE, "results", "comparisons")
OUT = sys.argv[2] if len(sys.argv) > 2 else (os.path.join(ROOT, "figures", "results") if ROOT else os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
ANCHOR, ANCHOR_SE = -0.213, 0.313         # 设计 D 全市合并估计；s.e. 为按小区整群自助法（第二轮，results/causal/round2）
C_OURS = PAL["causal"]["stroke"]          # 因果锚定（本文）
C_FREE = PAL["cov"]["stroke"]             # 只约束符号的自由学习
C_DATA = PAL["price"]["stroke"]           # 纯数据驱动 / PAG
C_PIAST = PAL["demand"]["stroke"]
C_GRAY = PAL["gray"]["stroke"]
setup_rc()


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(labelsize=7, colors=INK2)


def panel_label(ax, s, title, dx=0.0, y=1.06, gap=0.075):
    ax.text(dx, y, s, transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold", va="bottom", ha="left", color=INK)
    ax.text(dx + gap, y, title, transform=ax.transAxes, fontsize=FS["label"], fontweight="bold", va="bottom",
            ha="left", color=INK)


def neg_log_axis(ax, which, ticks, lo, hi):
    """坐标轴画 |β| 的对数刻度，刻度标签写成负数；越靠右 / 越靠上越接近 0。"""
    getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"set_{which}lim")(hi, lo)            # 反向：大的绝对值在左 / 下
    axis = getattr(ax, f"{which}axis")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_ticklabels([f"−{t:g}" for t in ticks])


def diag_angle(ax, lim):
    """对角线在屏幕上的倾角（坐标轴不是等比例时不等于 45°）。"""
    (x0, y0), (x1, y1) = ax.transData.transform([(lim[0], lim[0]), (lim[1], lim[1])])
    a = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    return a - 180 if a > 90 else a + 180 if a < -90 else a      # 坐标轴反向时保持文字正向


def save(fig, name):
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".svg")
    fix_svg_fonts(base + ".svg")
    plt.close(fig)
    print("→", base)


runs = pd.read_csv(os.path.join(CMP, "all_runs_flat.csv"))
runs["seed"] = runs.run.str.extract(r"_s(\d+)$").astype(float)

# ================================================================= data for Fig. 7 (loss profile and injected priors)
eid = pd.read_csv(os.path.join(CMP, "eid_summary.csv"))
prof = eid[(eid["round"] == "v2") & (eid.kind == "profile")].sort_values("beta_set")
seed_sd = runs[runs.group == "v2_main_A0"].val.std(ddof=1)          # seed-to-seed s.d. of the validation loss
eps = pd.read_csv(os.path.join(CMP, "eps_summary.csv"))
pp, pr = eps[eps.method == "pag_paper"], eps[eps.method == "pag_released"]
pc = eps[eps.method == "piast_clamp"]
S = [(pp, "arc_mean", C_DATA, "o", True, "PAG, published form"),
     (pr, "arc_mean", C_DATA, "o", False, "PAG, released code"),
     (pc, "piast_lambda_mean_tou", C_PIAST, "s", True, r"PIAST, learned coefficient $\lambda$"),
     (pc, "arc_mean", C_PIAST, "^", True, "PIAST, perturbation response")]
ours_b = [0.213, 0.45, 0.76, 1.48]
print(f"seed s.d. of the validation loss (x100): {seed_sd * 100:.4f} over {int((runs.group == 'v2_main_A0').sum())} seeds")

# ================================================================= 图 4：分步长精度与区间覆盖率（论文 4.4 节）
# 输入：metrics_by_step.csv、conformal_by_step.csv（scripts/collect_by_step.py）、dm_wilcoxon.csv（scripts/significance_tests.py）
f_ms = os.path.join(CMP, "metrics_by_step.csv")
if os.path.exists(f_ms):
    ms = pd.read_csv(f_ms)
    ms = ms[ms.step != "avg"].copy()
    ms["step"] = ms.step.astype(int)
    cf = pd.read_csv(os.path.join(CMP, "conformal_by_step.csv"))
    dm = pd.read_csv(os.path.join(CMP, "dm_wilcoxon.csv"))
    dm = dm[(dm.metric == "mae") & (dm.step != "all")].copy()
    dm["step"] = dm.step.astype(int)
    C_FCNN, C_LSTM, C_GRAPH = PAL["demand"]["stroke"], PAL["cov"]["stroke"], PAL["price"]["stroke"]
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 66 * MM),
                            gridspec_kw={"wspace": 0.36, "left": 0.062, "right": 0.988, "top": 0.86, "bottom": 0.175,
                                         "width_ratios": [1.1, 1.1, 0.9]})
    HT = [1, 3, 6, 9, 12, 24]

    def by_step(group, col="MAE"):
        d = ms[ms.group == group].groupby("step")[col].agg(["mean", "std"]).reindex(HT)
        return d["mean"].values * 100, d["std"].fillna(0).values * 100

    # (a) 各模型的 MAE 随步长
    ax = axs[0]
    clean(ax)
    SER = [("baseline_naive_last_utilization", "Last value", C_GRAY, (0, (1, 1.5)), 0.9, None),
           ("baseline_naive_seasonal_utilization", "Seasonal naive", C_GRAY, (0, (4, 2)), 0.9, None),
           ("r2_stgcn", "STGCN", C_GRAPH, "-", 1.0, "s"),
           ("r2_stid", "STID", C_LSTM, "-", 1.0, "^"),
           ("r2_fcnn", "FCNN", C_FCNN, "-", 1.0, "o"),
           ("v2_main_A0", "CPA-STGNN", C_OURS, "-", 1.5, "D")]
    for g, lab, col, ls, lw, mk in SER:
        m, s = by_step(g)
        if mk:
            ax.fill_between(HT, m - s, m + s, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(HT, m, color=col, ls=ls, lw=lw, marker=mk, ms=3.4 if mk else 0, mec="white", mew=0.5, zorder=4 if mk else 3,
                label=lab)
    ax.set_xticks(HT)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 11.5)
    ax.set_xlabel("Forecast horizon (h)", fontsize=FS["label"], color=INK)
    ax.set_ylabel(r"MAE ($\times 10^{-2}$)", fontsize=FS["label"], color=INK)
    ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", fontsize=FS["small"], frameon=False, handletextpad=0.4, borderaxespad=0.1,
              labelspacing=0.25, ncol=2, columnspacing=0.9, handlelength=1.8)
    panel_label(ax, "(a)", "Error by horizon", dx=-0.14, gap=0.115)

    # (b) 主模型相对于最接近的几个模型的 MAE 差（%），9 个保存的步长；实心 = DM 检验 p < 0.05
    ax = axs[1]
    clean(ax)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    CMPS = [("v2_main_A0", "r2_fcnn", "FCNN", C_FCNN, "o"),
            ("v2_main_A0", "r2_stid", "STID", C_LSTM, "^"),
            ("v2_main_A0", "r2_dlinear", "DLinear", C_GRAY, "v"),
            ("v2_main_A0", "v2_gwn_blind", "Graph WaveNet", C_GRAPH, "s")]
    for ref, vs, lab, col, mk in CMPS:
        d = dm[(dm.ref == ref) & (dm.vs == vs)].sort_values("step")
        ax.plot(d.step, d.rel_diff_pct, color=col, lw=1.0, zorder=3)
        sig = d.dm_p < 0.05
        ax.plot(d.step[sig], d.rel_diff_pct[sig], ls="none", marker=mk, ms=3.8, color=col, mec="white", mew=0.5, zorder=5)
        ax.plot(d.step[~sig], d.rel_diff_pct[~sig], ls="none", marker=mk, ms=3.4, mfc="white", mec=col, mew=0.8, zorder=4)
        ax.plot([], [], color=col, lw=1.0, marker=mk, ms=3.8, mec="white", mew=0.5, label=lab)
    ax.set_xticks([1, 3, 6, 9, 12, 24])
    ax.set_xlim(0, 25)
    ax.set_ylim(-11.5, 10.5)
    ax.text(24.6, -11.1, "CPA-STGNN more accurate", fontsize=FS["small"], color=INK2, ha="right", va="bottom")
    ax.text(24.6, 10.1, "CPA-STGNN less accurate", fontsize=FS["small"], color=INK2, ha="right", va="top")
    ax.set_xlabel("Forecast horizon (h)", fontsize=FS["label"], color=INK)
    ax.set_ylabel("MAE of CPA-STGNN relative to model (%)", fontsize=FS["label"], color=INK)
    ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
    ax.legend(loc="upper right", fontsize=FS["small"], frameon=False, handletextpad=0.4, borderaxespad=0.0,
              labelspacing=0.25, bbox_to_anchor=(1.0, 0.90))
    panel_label(ax, "(b)", "Difference to the closest models", dx=-0.14, gap=0.115)

    # (c) 90% 区间覆盖率：分位数头原始输出与在线校准之后
    ax = axs[2]
    clean(ax)
    ax.axhline(0.90, color=INK2, lw=0.6, ls=(0, (4, 2)), zorder=1)
    ax.text(24.6, 0.9035, "nominal", fontsize=FS["small"], color=INK2, ha="right", va="bottom")
    c0 = cf[cf.group == "v2_main_A0"].copy()
    c0["step"] = c0.step.astype(int)
    for meth, lab, col, mk, filled in (("quantile_raw", "Quantile head", C_GRAY, "o", False),
                                       ("aci", "After online calibration", C_OURS, "D", True)):
        d = c0[c0.method == meth].groupby("step").PICP.agg(["mean", "std"]).reindex(HT)
        ax.fill_between(HT, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(HT, d["mean"], color=col, lw=1.2, marker=mk, ms=3.6, mfc=col if filled else "white", mec="white" if filled else col,
                mew=0.6 if filled else 0.9, label=lab, zorder=4)
    ax.set_xticks(HT)
    ax.set_xlim(0, 25)
    ax.set_ylim(0.80, 0.93)
    ax.set_xlabel("Forecast horizon (h)", fontsize=FS["label"], color=INK)
    ax.set_ylabel("Coverage of the 90% interval", fontsize=FS["label"], color=INK)
    ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
    ax.legend(loc="lower right", fontsize=FS["small"], frameon=False, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25,
              bbox_to_anchor=(1.0, 0.48))
    panel_label(ax, "(c)", "Interval coverage", dx=-0.19, gap=0.14)
    save(fig, "Fig_R_horizon")

# ================================================================= 图 5：分时电价情景（论文 4.9 节）
# 输入：results/scenarios/scenario_summary.csv、scenario_profiles.csv（scripts/simulate_scenarios.py）
SCN = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(CMP)), "scenarios")
if os.path.exists(os.path.join(SCN, "scenario_profiles.csv")):
    sp = pd.read_csv(os.path.join(SCN, "scenario_profiles.csv"))
    ss = pd.read_csv(os.path.join(SCN, "scenario_summary.csv"))
    SC = [("flat", "No time-of-use tariff", C_GRAY, (0, (4, 2))),
          ("spread2", "Peak–valley spread doubled", PAL["demand"]["stroke"], "-"),
          ("evening", "Surcharge of 20%, 18:00–22:00", PAL["price"]["stroke"], "-")]
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 66 * MM),
                            gridspec_kw={"wspace": 0.36, "left": 0.062, "right": 0.988, "top": 0.86, "bottom": 0.175,
                                         "width_ratios": [1.0, 1.0, 1.05]})
    hrs = np.arange(25)

    def stepline(ax, v, **kw):
        ax.step(hrs, np.r_[v, v[-1]], where="post", **kw)

    ax = axs[0]
    clean(ax)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for k, lab, col, ls in SC:
        d = sp[sp.scenario == k].sort_values("hour")
        stepline(ax, d.dlog_change.values * 100, color=col, lw=1.2, ls=ls, label=lab, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(-9, 33)
    ax.set_yticks([-5, 0, 5, 10, 15, 20])
    ax.set_xlabel("Hour of day", fontsize=FS["label"], color=INK)
    ax.set_ylabel("Mean change in log price (×100)", fontsize=FS["label"], color=INK)
    ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
    ax.legend(loc="upper left", fontsize=FS["small"], frameon=False, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25,
              handlelength=1.8)
    panel_label(ax, "(a)", "Price change by scenario", dx=-0.14, gap=0.12)

    ax = axs[1]
    clean(ax)
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    for k, lab, col, ls in SC:
        d = sp[sp.scenario == k].sort_values("hour")
        lo, hi = (d.cf_lo / d.fact - 1).values * 100, (d.cf_hi / d.fact - 1).values * 100
        ax.fill_between(hrs, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=col, alpha=0.18, lw=0, zorder=2)
        stepline(ax, (d.cf / d.fact - 1).values * 100, color=col, lw=1.2, ls=ls, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ext = [((sp.cf_lo / sp.fact - 1) * 100).min(), ((sp.cf_hi / sp.fact - 1) * 100).max()]
    y_lo, y_hi = min(-10.5, np.floor(ext[0]) - 0.5), max(4.5, np.ceil(ext[1]) + 0.5)   # bands are never clipped
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Hour of day", fontsize=FS["label"], color=INK)
    ax.set_ylabel("Change in predicted charging (%)", fontsize=FS["label"], color=INK)
    ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
    ax.text(0.5, y_lo + 0.5, "bands: 90% interval,\nuncertainty of the elasticity", fontsize=FS["small"], color=INK2,
            ha="left", va="bottom", linespacing=1.0)
    panel_label(ax, "(b)", "Predicted response", dx=-0.14, gap=0.12)

    ax = axs[2]
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    EL = [("anchor", "−0.21, causal estimate", C_OURS, "D"), ("prior_0.45", "−0.45", C_GRAY, "o"),
          ("kuang_0.76", "−0.76", C_GRAY, "s"), ("naive_1.58", "−1.58, pooled regression", C_GRAY, "^")]
    for i, (k, lab, col, ls) in enumerate(SC):
        for j, (e, elab, ecol, mk) in enumerate(EL):
            r = ss[(ss.scenario == k) & (ss.elasticity == e)].iloc[0]
            yy = i + (j - 1.5) * 0.19
            if e == "anchor":
                ax.plot([r.up_pct_lo, r.up_pct_hi], [yy, yy], color=ecol, lw=1.4, zorder=3, solid_capstyle="butt")
            ax.plot([r.up_pct], [yy], ls="none", marker=mk, ms=4.6 if e == "anchor" else 4.0, color=ecol,
                    mfc=ecol if e == "anchor" else "white", mec="white" if e == "anchor" else INK2, mew=0.6 if e == "anchor" else 0.8,
                    zorder=4, label=elab if i == 0 else None)
    ax.set_ylim(len(SC) - 0.45, -0.55)
    ax.set_yticks(range(len(SC)))
    ax.set_yticklabels(["No time-of-\nuse tariff", "Spread\ndoubled", "Evening\nsurcharge"], fontsize=FS["small"], color=INK)
    ax.set_xlim(-27, 1.5)
    ax.set_xlabel("Change in charging where the price rises (%)", fontsize=FS["label"], color=INK)
    ax.grid(axis="x", color=RULE, lw=0.3, alpha=0.5)
    ax.legend(loc="center left", fontsize=FS["small"], frameon=False, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.3,
              title="Elasticity", title_fontsize=FS["small"], alignment="left", bbox_to_anchor=(0.0, 0.56))
    panel_label(ax, "(c)", "Dependence on the elasticity", dx=-0.22, gap=0.13)
    save(fig, "Fig_R_scenario")

# ================================================================= 图 6：数据背景与"从相关到因果"（论文 4.1、4.5 节）
# 输入：results/EDA/zone_tariff_type.csv、tou_profile_weekday.csv、naive_price_demand.csv；
#       results/causal/5min/spec_curve.csv、panel_supp.csv；小区边界 SZ_districts（UrbanEV 数据目录）
# 用法：python fig_results.py <comparisons> <输出目录> <scenarios> <EDA 目录> <因果估计/5min 目录> <UrbanEV data 目录>
RES_ROOT = os.path.dirname(os.path.abspath(CMP))
EDA = sys.argv[4] if len(sys.argv) > 4 else os.path.join(RES_ROOT, "EDA")
CAU = sys.argv[5] if len(sys.argv) > 5 else os.path.join(RES_ROOT, "causal", "5min")
GEO = sys.argv[6] if len(sys.argv) > 6 else os.path.join(ROOT or HERE, "data", "raw", "urbanev_github", "data")
if os.path.exists(os.path.join(EDA, "zone_tariff_type.csv")) and os.path.exists(os.path.join(GEO, "SZ_districts", "SZ_districts.shp")):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from shp import merc_to_ll, read_dbf, read_shp  # noqa: E402

    fig = plt.figure(figsize=(190 * MM, 64 * MM))

    def ftitle(x, lab, title):
        fig.text(x, 0.925, lab, fontsize=FS["panel"], fontweight="bold", va="bottom", ha="left", color=INK)
        fig.text(x + 0.027, 0.925, title, fontsize=FS["label"], fontweight="bold", va="bottom", ha="left", color=INK)
    # (a) 研究区：小区按电价类型着色
    ax = fig.add_axes([0.0, 0.17, 0.355, 0.73])
    zt = pd.read_csv(os.path.join(EDA, "zone_tariff_type.csv")).set_index("zone")
    recs = read_dbf(os.path.join(GEO, "SZ_districts", "SZ_districts.dbf"))
    shapes = read_shp(os.path.join(GEO, "SZ_districts", "SZ_districts.shp"))
    tz = [int(r["TAZID"]) for r in recs]
    CT = {"TOU": PAL["price"]["stroke"], "weak": PAL["price"]["strong"], "fixed": PAL["demand"]["strong"]}
    bg, polys, cols = [], [], []
    for k, parts in enumerate(shapes):
        rings = [np.c_[merc_to_ll(r[:, 0], r[:, 1])] for r in parts]
        if tz[k] in zt.index:
            polys += rings
            cols += [CT[zt.loc[tz[k], "tariff"]]] * len(rings)
        else:
            bg += rings
    ax.add_collection(PolyCollection(bg, facecolors="#F4F4F4", edgecolors="#DDDDDD", linewidths=0.25))
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.3))
    ax.set_xlim(113.75, 114.63)
    ax.set_ylim(22.44, 22.87)
    ax.set_aspect(1 / np.cos(np.radians(22.6)))
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cnt = zt.tariff.value_counts()
    ax.legend(handles=[Patch(color=CT["TOU"], label=f"Time-of-use ({cnt['TOU']})"),
                       Patch(color=CT["weak"], label=f"Weakly varying ({cnt['weak']})"),
                       Patch(color=CT["fixed"], label=f"Fixed price ({cnt['fixed']})"),
                       Patch(color="#F4F4F4", ec="#CCCCCC", lw=0.4, label="No charging data")],
              frameon=False, fontsize=FS["small"], loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2, handlelength=1.1,
              columnspacing=1.0, handletextpad=0.5, labelspacing=0.3)
    # 比例尺 10 km
    lon0, lat0 = 114.47, 22.50
    dlon = 10 / (111.32 * np.cos(np.radians(22.6)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=INK2, lw=1.0, solid_capstyle="butt")
    ax.text(lon0 + dlon / 2, lat0 + 0.008, "10 km", fontsize=FS["small"], color=INK2, ha="center", va="bottom")
    ftitle(0.008, "(a)", "Zones of Shenzhen by tariff type")

    # (b) 分时小区工作日的日内曲线：价格与利用率（上下两格，共用横轴）
    pf = pd.read_csv(os.path.join(EDA, "tou_profile_weekday.csv")).sort_values("hour")
    hrs_ = np.arange(25)
    a1 = fig.add_axes([0.435, 0.575, 0.185, 0.30])
    clean(a1)
    a1.axhline(0, color=RULE, lw=0.5)
    v = pf.dlog_price_mean.values * 100
    a1.step(hrs_, np.r_[v, v[-1]], where="post", color=PAL["price"]["stroke"], lw=1.3)
    a1.set_xlim(0, 24)
    a1.set_xticks([0, 6, 12, 18, 24])
    a1.set_xticklabels([])
    a1.set_ylim(-8, 8)
    a1.set_yticks([-5, 0, 5])
    a1.set_ylabel("Log price,\ndeviation (×100)", fontsize=FS["small"], color=INK)
    ftitle(0.375, "(b)", "Price and demand over the day")
    a2 = fig.add_axes([0.435, 0.175, 0.185, 0.30])
    clean(a2)
    a2.plot(pf.hour + 0.5, pf.util_mean, color=PAL["demand"]["stroke"], lw=1.3, marker="o", ms=2.2, mec="white", mew=0.3)
    a2.set_xlim(0, 24)
    a2.set_xticks([0, 6, 12, 18, 24])
    a2.set_ylim(0.12, 0.36)
    a2.set_yticks([0.15, 0.25, 0.35])
    a2.set_xlabel("Hour of day", fontsize=FS["label"], color=INK)
    a2.set_ylabel("Utilisation\nrate", fontsize=FS["small"], color=INK)
    fig.align_ylabels([a1, a2])

    # (c) 从相关到因果：斜率 / 弹性的阶梯
    ax = fig.add_axes([0.790, 0.175, 0.200, 0.70])
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    nv = pd.read_csv(os.path.join(EDA, "naive_price_demand.csv")).set_index("spec")
    scv = pd.read_csv(os.path.join(CAU, "spec_curve.csv"))
    psu = pd.read_csv(os.path.join(CAU, "panel_supp.csv"))

    def pk(design, exposures):
        r = scv[(scv.design == design) & (scv.W_min == 180) & (scv.est == "PPML") & (scv.donut_min == 0)
                & (scv.outcome == "duration") & (scv.exposures == exposures)]
        assert len(r) == 1
        return r.iloc[0]
    pp_ = psu[(psu["method"].str.startswith("PPML")) & (psu["sample"] == "TOU zones")].iloc[0]
    dm_ = psu[psu["method"].str.startswith("DML")].iloc[0]
    LAD = [("Pooled regression", nv.loc["pooled"], C_DATA, False, 0),
           ("+ zone and hour effects", nv.loc["zone+hour"], C_DATA, False, 1),
           ("+ zone×hour, date×hour", nv.loc[[i for i in nv.index if "date" in i][0]], C_DATA, False, 2),
           ("Switch events, design C", pk("C", False), C_GRAY, False, 3.35),
           ("Switch events, design D", pk("D", True), C_OURS, True, 4.35),
           ("Hourly panel, PPML", pp_, C_OURS, False, 5.35),
           ("Hourly panel, DML", dm_, C_OURS, False, 6.35)]
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    for lab, r, col, main, yy in LAD:
        b, se = float(r["beta"]), float(r["se"])
        ax.plot([b - 1.96 * se, b + 1.96 * se], [yy, yy], color=col, lw=1.3 if main else 1.0, zorder=3, solid_capstyle="butt")
        ax.plot([b], [yy], ls="none", marker="D" if main else "o", ms=5.0 if main else 4.0, color=col, mec="white", mew=0.6, zorder=4)
        ax.text(1.18, yy, f"−{abs(b):.2f}" if b < 0 else f"{b:.2f}", fontsize=FS["small"], color=INK, ha="right", va="center",
                fontweight="bold" if main else "normal")
    ax.set_ylim(6.9, -1.1)
    ax.set_yticks([r[4] for r in LAD])
    ax.set_yticklabels([r[0] for r in LAD], fontsize=FS["small"], color=INK)
    ax.set_xlim(-2.35, 1.22)
    ax.set_xticks([-2, -1, 0])
    ax.set_xticklabels(["−2", "−1", "0"])
    ax.set_xlabel("Slope of log demand\non log price (95% CI)", fontsize=FS["label"], color=INK)
    ax.axhline(2.42, color=RULE, lw=0.4, ls=(0, (2, 2)))
    ax.text(-2.33, -0.72, "naive regressions", fontsize=FS["small"], color=C_DATA, ha="left", va="center", style="italic")
    ax.text(-2.33, 2.80, "identification designs", fontsize=FS["small"], color=C_OURS, ha="left", va="center", style="italic")
    ftitle(0.655, "(c)", "From correlation to causation")
    save(fig, "Fig_R_context")

# ================================================================= 图 7：可识别性与先验敏感性合并图（论文 4.6 节）
fig, axs = plt.subplots(1, 2, figsize=(190 * MM, 70 * MM),
                        gridspec_kw={"wspace": 0.27, "left": 0.085, "right": 0.985, "top": 0.87, "bottom": 0.17,
                                     "width_ratios": [1.08, 1]})
ax = axs[0]
clean(ax)
x, y = -prof.beta_set.values, prof.val.values * 100
ymin = y.min()
ax.axhspan(ymin, ymin + 2 * seed_sd * 100, color=PAL["gray"]["fill"], lw=0, zorder=0)
ax.text(0.185, ymin + 1.45 * seed_sd * 100, "within 2 s.d. of\nseed-to-seed variation", fontsize=FS["small"],
        color=INK2, va="center", ha="left")
ax.axvline(-ANCHOR, color=C_OURS, lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.text(-ANCHOR * 0.94, 1.318, "causal estimate\n−0.21", fontsize=FS["small"], color=C_OURS, va="top", ha="left")
ax.plot(x, y, color=INK, lw=1.2, marker="o", ms=4.2, mfc="white", mec=INK, mew=1.0, zorder=4)
neg_log_axis(ax, "x", [3, 2, 1.2, 0.8, 0.4, 0.2, 0.1, 0.05], 0.04, 3.6)
ax.set_ylim(1.25, 1.32)
ax.set_xlabel(r"Elasticity fixed during training, $\beta$", fontsize=FS["label"], color=INK)
ax.set_ylabel(r"Validation pinball loss ($\times 10^{-2}$)", fontsize=FS["label"], color=INK)
ax.grid(axis="y", color=RULE, lw=0.3, alpha=0.5)
panel_label(ax, "(a)", "Forecast loss is flat in the elasticity")

ax = axs[1]
clean(ax)
lim = (0.07, 4.2)
ax.plot(lim, lim, color=RULE, lw=0.8, zorder=1)
ax.axhspan(lim[0], -(ANCHOR - ANCHOR_SE), color=PAL["causal"]["fill"], lw=0, zorder=0)
ax.axhline(-ANCHOR, color=C_OURS, lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.text(3.9, 0.118, "causal estimate −0.21 (±1 s.e.)", fontsize=FS["small"], color=C_OURS, va="center", ha="left")
for d, col_, c, mk, filled, lab in S:
    d = d.sort_values("prior_injected")
    ax.plot(-d.prior_injected, -d[col_], color=c, lw=1.1 if filled else 0, marker=mk, ms=4.6 if filled else 7.0,
            mfc=c if filled else "none", mec="white" if filled else c, mew=0.6 if filled else 0.9, label=lab, zorder=4)
ax.plot(ours_b, ours_b, ls="none", marker="D", ms=4.8, color=C_OURS, mec="white", mew=0.6, zorder=5,
        label="CPA-STGNN, anchor set to the prior")
neg_log_axis(ax, "x", [3, 1.48, 0.76, 0.45, 0.21, 0.1], *lim)
neg_log_axis(ax, "y", [3, 1.48, 0.76, 0.45, 0.21, 0.1], *lim)
ax.set_xlabel("Elasticity injected as prior", fontsize=FS["label"], color=INK)
ax.set_ylabel("Elasticity implied by the trained model", fontsize=FS["label"], color=INK)
ax.legend(loc="lower right", fontsize=FS["small"], frameon=False, handletextpad=0.4, borderaxespad=0.1,
          labelspacing=0.35, bbox_to_anchor=(1.0, 0.0))
ax.text(2.3, 2.3, "implied = injected", fontsize=FS["small"], color=INK2, rotation=diag_angle(ax, lim),
        rotation_mode="anchor", ha="center", va="bottom")
panel_label(ax, "(b)", "Injected priors do not determine the response", dx=-0.02)
save(fig, "Fig_R_validation")
