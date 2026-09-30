"""阶段 5.2 EDA 图（两张，英文，190 mm 宽）。

Fig_S_eda_temporal：(a) 日内曲线（工作日 / 休息日 / 法定假日）(b) 逐日全市利用率与数据切分 (c) 气温与雨天（去掉时刻 × 日类型平均后）
Fig_S_eda_price_space：(a) 分时小区日内价格结构（总价、电价、服务费）(b) 价格与需求的朴素估计 vs 因果估计
                       (c) 切换前后的原始需求曲线（分时 − 同时刻固定小区）(d) 局部空间自相关（LISA）

输入：05_实验结果/EDA/*.csv（scripts/eda.py 生成）、UrbanEV 的 SZ_districts shapefile
运行：python fig_eda.py [EDA 结果目录] [输出目录] [UrbanEV data 目录]（默认：项目根目录下的 05_实验结果/EDA、04_图表/数据与EDA图、02_数据/raw/urbanev_github/data）
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import FS, INK2, PAL, RULE, fix_svg_fonts, setup_rc  # noqa: E402

ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "03_代码")):
    ROOT = os.path.dirname(ROOT)                                       # 项目根目录（含 03_代码 的那一级）
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "05_实验结果", "EDA")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "04_图表", "数据与EDA图")
D1 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "02_数据", "raw", "urbanev_github", "data")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
C_DEM, C_PRICE, C_COV, C_CAUSAL, C_GRAY, C_OUT = (PAL[k]["stroke"] for k in ("demand", "price", "cov", "causal", "gray", "output"))
stats = json.load(open(os.path.join(RES, "eda_stats.json"), encoding="utf-8"))


def rd(name):
    return pd.read_csv(os.path.join(RES, name))


def panel_label(ax, s, title, x=0.0):
    ax.text(x, 1.10, s, transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold", va="bottom", ha="left")
    ax.text(x + 0.12, 1.10, title, transform=ax.transAxes, fontsize=FS["label"], fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def save(fig, name):
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".svg")
    fix_svg_fonts(base + ".svg")
    print("saved", base)


setup_rc()
# ================================================================ 图 1：时间维度
fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 66 * MM), gridspec_kw={"wspace": 0.42, "left": 0.07, "right": 0.99,
                                                                       "top": 0.86, "bottom": 0.33,
                                                                       "width_ratios": [1, 1.35, 1]})
ax = axs[0]
di = rd("diurnal.csv")
for dt, col, lab in (("weekday", C_DEM, "Workday"), ("weekend", C_GRAY, "Weekend"), ("holiday", C_COV, "Public holiday")):
    q = di[di.daytype == dt].sort_values("hour")
    if dt == "weekday":
        ax.fill_between(q.hour, q.q25, q.q75, color=PAL["demand"]["mid"], lw=0, alpha=0.8)
    ax.plot(q.hour, q["mean"], color=col, lw=1.2, label=lab)
ax.set_xlim(0, 23)
ax.set_xticks([0, 6, 12, 18, 23])
ax.set_xlabel("Hour of day", fontsize=7)
ax.set_ylabel("Mean utilization", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, handlelength=1.3,
          columnspacing=0.8)
clean(ax)
panel_label(ax, "(a)", "Diurnal profile")

ax = axs[1]
dd = rd("daily.csv")
dd["date"] = pd.to_datetime(dd["date"])
for _, r in dd[dd.holiday].iterrows():
    ax.axvspan(r.date, r.date + pd.Timedelta(days=1), color=PAL["cov"]["mid"], lw=0)
ax.plot(dd.date, dd.u, color=C_DEM, lw=0.9)
te, ve = pd.Timestamp(stats["train_end"]).normalize(), pd.Timestamp(stats["val_end"]).normalize()
for x in (te + pd.Timedelta(days=1), ve + pd.Timedelta(days=1)):
    ax.axvline(x, color=INK2, lw=0.6, ls=(0, (2, 2)))
yl = ax.get_ylim()
for x0, x1, lab in ((dd.date.min(), te, "Train"), (te, ve, "Val."), (ve, dd.date.max(), "Test")):
    ax.text(x0 + (x1 - x0) / 2, yl[1], lab, ha="center", va="bottom", fontsize=6.5, color=INK2)
ax.set_ylim(yl)
ax.set_xlabel("Date (2022-09 to 2023-02)", fontsize=7)
ax.set_ylabel("Daily mean utilization", fontsize=7)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
ax.legend([Line2D([], [], color=C_DEM, lw=0.9), Patch(color=PAL["cov"]["mid"])], ["City mean", "Public holiday"],
          frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handlelength=1.3)
clean(ax)
panel_label(ax, "(b)", "Daily series and data split", x=0.0)

ax = axs[2]
wt = rd("weather_temperature.csv")
wt = wt[wt["count"] >= 20]
ax.errorbar(wt.T_mid, wt["mean"], yerr=1.96 * wt.se, fmt="o", ms=2.6, color=C_COV, mfc="white", mec=C_COV, lw=0.7,
            capsize=1.2)
ax.plot(wt.T_mid, wt["mean"], color=C_COV, lw=0.8)
ax.axhline(0, color=RULE, lw=0.5)
w = stats["weather"]
ax.text(0.03, 0.04, f"Rain vs. dry: {w['rain_minus_dry_resid']:+.3f}", transform=ax.transAxes, fontsize=6.5, color=INK2)
ax.set_xlabel("Temperature (°C)", fontsize=7)
ax.set_ylabel("Utilization residual", fontsize=7)
ax.legend([Line2D([], [], color=C_COV, marker="o", ms=2.6, mfc="white", lw=0.8)],
          ["Mean residual (±95% CI), 2 °C bins"], frameon=False, fontsize=6.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.27), handlelength=1.6)
clean(ax)
panel_label(ax, "(c)", "Weather")
save(fig, "Fig_S_eda_temporal")
plt.close(fig)

# ================================================================ 图 2：价格与空间
fig = plt.figure(figsize=(190 * MM, 132 * MM))
gs = fig.add_gridspec(2, 2, left=0.075, right=0.99, top=0.93, bottom=0.12, wspace=0.62, hspace=0.62,
                      width_ratios=[1, 0.82])

ax = fig.add_subplot(gs[0, 0])
pdv = rd("price_dev_by_hour.csv")
ax.fill_between(pdv.hour, pdv.q25, pdv.q75, step="mid", color=PAL["price"]["mid"], lw=0)
ax.step(pdv.hour, pdv["median"], where="mid", color=C_PRICE, lw=1.3, label="Total price (median, IQR)")
ax.step(pdv.hour, pdv.electricity_mean, where="mid", color=C_PRICE, lw=0.8, ls=(0, (3, 1.5)), label="Electricity tariff (mean)")
ax.step(pdv.hour, pdv.service_mean, where="mid", color=INK2, lw=0.8, ls=(0, (1, 1.2)), label="Service fee (mean)")
ax.axhline(0, color=RULE, lw=0.5)
ax.set_xlim(-0.5, 23.5)
ax.set_xticks([0, 6, 12, 18, 23])
ax.set_xlabel("Hour of day", fontsize=7)
ax.set_ylabel("Log price minus zone-day mean", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.8,
          columnspacing=0.8)
clean(ax)
panel_label(ax, "(a)", "Intraday prices, TOU zones")

ax = fig.add_subplot(gs[0, 1])
nv = rd("naive_price_demand.csv")
labels = {"pooled": "Pooled OLS", "hour": "+ hour FE", "zone": "+ zone FE", "zone+hour": "+ zone & hour FE",
          "zone×hour + date×hour": "+ zone×hour & date×hour FE"}
rows = [(labels[r.spec], r.beta, r.se, C_GRAY, "o") for r in nv.itertuples()]
rows += [("Design D, 5-min (anchor)", -0.213, 0.202, C_CAUSAL, "D"),
         ("Panel PPML, hourly", -0.26, 0.25, C_CAUSAL, "D")]
for i, (lab, b, se, col, mk) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.errorbar(b, y, xerr=1.96 * se, fmt=mk, ms=3.2, color=col, lw=0.8, capsize=1.5,
                mfc=col if col == C_CAUSAL else "white", mec=col)
ax.axvline(0, color=RULE, lw=0.5)
ax.axhline(1.5, color=RULE, lw=0.4, ls=(0, (2, 2)))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.5)
ax.set_xlabel("Estimated log-log slope (95% CI)", fontsize=7)
ax.set_xlim(-2.3, 1.3)
ax.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5])
ax.text(1.28, 1.6, "Naive correlations", ha="right", va="bottom", fontsize=6.5, color=INK2)
ax.text(1.28, 1.4, "Causal estimates", ha="right", va="top", fontsize=6.5, color=C_CAUSAL)
clean(ax)
panel_label(ax, "(b)", "Correlation vs. causal estimates", x=-0.72)

ax = fig.add_subplot(gs[1, 0])
sc = rd("switch_curves.csv")
for dr, col, mk, lab in (("up", C_PRICE, "o", "Price rise"), ("down", C_DEM, "s", "Price cut")):
    q = sc[sc.dir == dr].sort_values("k")
    n = int(q.n_events.iloc[0])
    ax.fill_between(q.k, q["diff"] - 1.96 * q.diff_se, q["diff"] + 1.96 * q.diff_se, color=col, alpha=0.18, lw=0)
    ax.plot(q.k, q["diff"], color=col, lw=1.0, marker=mk, ms=2.4, label=f"{lab} (n = {n:,})")
ax.axhline(0, color=RULE, lw=0.5)
ax.axvline(-0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
ax.set_xlabel("Hours relative to switch", fontsize=7)
ax.set_ylabel("Δ log utilization, TOU − fixed", fontsize=7)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.6)
clean(ax)
panel_label(ax, "(c)", "Raw demand around price switches")

ax = fig.add_subplot(gs[1, 1])
lisa = rd("lisa_adjacency.csv").set_index("zone")
sys.path.insert(0, os.path.join(ROOT, "03_代码", "scripts", "audit"))
sys.path.insert(0, os.path.join(ROOT, "scripts_audit"))                     # 云端工作目录的布局
from shp import merc_to_ll, read_dbf, read_shp  # noqa: E402

recs = read_dbf(os.path.join(D1, "SZ_districts", "SZ_districts.dbf"))
shapes = read_shp(os.path.join(D1, "SZ_districts", "SZ_districts.shp"))
tz = [int(r["TAZID"]) for r in recs]
CL = {"HH": C_OUT, "LL": C_DEM, "HL": PAL["output"]["strong"], "LH": PAL["demand"]["strong"], "ns": "#E4E4E4"}
bg, polys, cols = [], [], []
for k, parts in enumerate(shapes):
    rings = [np.c_[merc_to_ll(r[:, 0], r[:, 1])] for r in parts]
    if tz[k] in lisa.index:
        for r in rings:
            polys.append(r)
            cols.append(CL[lisa.loc[tz[k], "cluster"]])
    else:
        bg += rings
ax.add_collection(PolyCollection(bg, facecolors="#F7F7F7", edgecolors="#E0E0E0", linewidths=0.25))
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.3))
ax.set_xlim(113.75, 114.63)
ax.set_ylim(22.44, 22.87)
ax.set_aspect(1 / np.cos(np.radians(22.6)))
ax.set_anchor("N")
ax.set_xticks([])
ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
cnt = lisa.cluster.value_counts()
mo = pd.read_csv(os.path.join(RES, "moran.csv"))
ia = mo[(mo.weights == "adjacency") & (mo.variable == "zone_mean_util_train")].iloc[0]
idist = mo[(mo.weights == "distance_5km") & (mo.variable == "zone_mean_util_train")].iloc[0]
ax.legend(handles=[Patch(color=CL[c], label=f"{c} ({int(cnt.get(c, 0))})") for c in ("HH", "LL", "HL", "LH")] +
          [Patch(color=CL["ns"], label=f"Not significant ({int(cnt.get('ns', 0))})")],
          frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, handlelength=1.2,
          columnspacing=0.8)
ax.text(0.0, 0.98, f"Global Moran's I = {ia.I:.3f} (p = {ia.p_perm:.3f}, adjacency)\n"
        f"= {idist.I:.3f} (p = {idist.p_perm:.3f}, 5-km band)", transform=ax.transAxes, fontsize=6.5, color=INK2,
        va="top")
panel_label(ax, "(d)", "Local spatial clusters (LISA)")
save(fig, "Fig_S_eda_price_space")
