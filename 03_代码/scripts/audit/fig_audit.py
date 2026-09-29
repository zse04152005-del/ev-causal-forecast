import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from shp import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm; from matplotlib.collections import PolyCollection; from matplotlib.patches import Patch
from _paths import cjk_font
plt.rcParams.update({"font.family": cjk_font(fm), "font.size": 9, "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
OUT = "."
def load(n):
    df = pd.read_csv(f"{D}/{n}.csv"); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time"); df.columns = df.columns.astype(int); return df
dur, pe, ps = load("duration"), load("e_price"), load("s_price")
st = pd.read_csv("zone_static.csv", index_col=0); fz = pd.read_csv("frozen_mask.csv", index_col=0, parse_dates=True)
C = {"TOU": "#C0504D", "fixed": "#4F81BD", "weak": "#F2B134", "grey": "#BFBFBF", "purple": "#7030A0"}
fig = plt.figure(figsize=(13, 9.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.36, wspace=0.34)
# (a) timeline
ax = fig.add_subplot(gs[0, 0]); day = dur.index.normalize()
dd = dur.sum(axis=1).groupby(day).sum() / 1000; fzd = fz.mean(axis=1).groupby(fz.index.normalize()).mean()
spl = [("训练集", "2022-09-01", "2023-01-05", "#F2F2F2"), ("验证集", "2023-01-06", "2023-01-23", "#E2EFDA"), ("测试集", "2023-01-24", "2023-02-28", "#FCE4D6")]
for name, a, b, col in spl:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color=col, zorder=0)
    xm = pd.Timestamp("2022-12-05") if name == "训练集" else pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
    ax.text(xm, 118, name, ha="center", fontsize=8.5, color="#404040")
hol = {"中秋": ("2022-09-10", "2022-09-12"), "国庆": ("2022-10-01", "2022-10-07"), "元旦": ("2022-12-31", "2023-01-02"), "春节": ("2023-01-21", "2023-01-27")}
for k, (a, b) in hol.items():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color="#FFD966", alpha=.55, zorder=1, lw=0)
    ax.text(pd.Timestamp(a), 104, k, fontsize=7.5, color="#7F6000")
ax.axvspan(pd.Timestamp("2022-10-26"), pd.Timestamp("2022-11-03"), facecolor="none", edgecolor="#7F7F7F", hatch="////", lw=0, zorder=1)
ax.annotate("数据冻结期 10-26~11-02\n（64 个小区数值冻结）", xy=(pd.Timestamp("2022-10-30"), 47), xytext=(pd.Timestamp("2022-11-12"), 50), fontsize=7.5, color="#595959", arrowprops=dict(arrowstyle="->", color="#595959", lw=.8))
ax.plot(dd.index, dd.values, color="#1F3864", lw=1.3, zorder=3, label="全市日充电时长（千桩·小时）")
ax.set_ylim(30, 125); ax.set_ylabel("千桩·小时 / 天")
ax2 = ax.twinx(); ax2.fill_between(fzd.index, 0, fzd.values * 100, color=C["purple"], alpha=.25, step="mid", zorder=2, label="冻结（插补）小区·小时占比")
ax2.set_ylim(0, 100); ax2.set_ylabel("冻结占比（%）", color=C["purple"]); ax2.spines["right"].set_visible(True); ax2.tick_params(axis="y", colors=C["purple"])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, fontsize=7.5, frameon=False)
ax.set_title("(a) 需求时间线、数据切分、节假日与数据质量", loc="left", fontsize=10, fontweight="bold")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y-%m"))
# (b) hourly price profile
ax = fig.add_subplot(gs[0, 1]); h = pe.index.hour
tou = st.index[st.pricing == "TOU"]; fix = st.index[st.pricing == "fixed"]
for ser, lab, col, ls in [((pe + ps)[tou], "总价（分时小区）", C["TOU"], "-"), (pe[tou], "电价（分时小区）", "#E46C0A", "--"), (ps[tou], "服务费（分时小区）", "#9BBB59", "--"), ((pe + ps)[fix], "总价（固定小区）", C["fixed"], ":")]:
    m = ser.groupby(h).mean().mean(axis=1); ax.step(np.arange(25), np.r_[m.values, m.values[-1]], where="post", color=col, ls=ls, lw=1.6, label=lab)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("时刻"); ax.set_ylabel("元 / kWh"); ax.set_ylim(0.4, 1.9)
ax.legend(fontsize=7.5, frameon=False, loc="center right")
ax.text(0.3, 0.47, "服务费与电价反向：谷时电价低、服务费高\n总价峰谷幅度约 ±6%，电价约 ±18%", fontsize=7.5, color="#404040")
ax.set_title("(b) 分时小区的日内价格结构（90 个严格分时小区均值）", loc="left", fontsize=10, fontweight="bold")
# (c) map
ax = fig.add_subplot(gs[1, 0])
recs = read_dbf(f"{D}/SZ_districts/SZ_districts.dbf"); shapes = read_shp(f"{D}/SZ_districts/SZ_districts.shp"); tz = [int(r["TAZID"]) for r in recs]
polys_bg, polys, cols = [], [], []
test_dead = set(fz.loc["2023-01-24":].mean()[lambda s: s > .5].index.astype(int))
for k, parts in enumerate(shapes):
    rings = [np.c_[merc_to_ll(r[:, 0], r[:, 1])] for r in parts]
    if tz[k] in st.index:
        for r in rings: polys.append(r); cols.append(C[st.loc[tz[k], "pricing"]])
    else:
        polys_bg += rings
ax.add_collection(PolyCollection(polys_bg, facecolors="#F2F2F2", edgecolors="#D9D9D9", linewidths=.3))
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=.35))
dead_polys = [np.c_[merc_to_ll(r[:, 0], r[:, 1])] for k, parts in enumerate(shapes) if tz[k] in test_dead for r in parts]
ax.add_collection(PolyCollection(dead_polys, facecolors="none", edgecolors="#262626", linewidths=.6, hatch="xxx"))
ax.set_xlim(113.75, 114.63); ax.set_ylim(22.44, 22.87); ax.set_aspect(1 / np.cos(np.radians(22.6))); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_visible(False)
n = st.pricing.value_counts()
ax.legend(handles=[Patch(color=C["TOU"], label=f"严格分时电价（{n['TOU']}）"), Patch(color=C["weak"], label=f"弱变动，按固定处理（{n['weak']}）"), Patch(color=C["fixed"], label=f"固定电价（{n['fixed']}）"),
                   Patch(facecolor="none", edgecolor="#262626", hatch="xxx", label=f"测试期 >50% 冻结（{len(test_dead)}）"), Patch(color="#F2F2F2", label="未纳入 UrbanEV 的小区")], fontsize=7.5, frameon=False, loc="lower right")
ax.set_title("(c) 275 个小区的定价类型与测试期数据冻结", loc="left", fontsize=10, fontweight="bold")
# (d) feasibility coefficients
ax = fig.add_subplot(gs[1, 1]); res = json.load(open("check7_feasibility.json"))
names = list(res.keys()); lab = ["A 分时 vs 固定\n（日期×时刻 FE）", "B A + 小区×时刻 FE\n（跨日变动）", "C 仅分时小区\n（切换时刻差异）", "D C + 小区×时刻 FE"]
for i, nm in enumerate(names):
    b, se = res[nm]["x"]; ax.errorbar(b, i, xerr=1.96 * se, fmt="o", color="#1F3864", ecolor="#1F3864", capsize=3, ms=5)
    ax.text(b, i + 0.25, f"{b:.2f}（se {se:.2f}）".replace("-", "-"), ha="center", fontsize=7.5)
for v, t, c, ls in [(-0.14, "PIAST 读出 -0.14（ST-EVCDP）", "#E46C0A", "--"), (-0.45, "PIAST 注入先验 -0.45", "#E46C0A", ":"), (-0.76, "Kuang 2024 读出 -0.76（ST-EVCDP）", "#9BBB59", "--"), (-1.48, "PAG 注入先验 -1.48", C["purple"], ":")]:
    ax.axvline(v, color=c, ls=ls, lw=1.1, label=t)
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
ax.axvline(0, color="#7F7F7F", lw=.8)
ax.set_yticks(range(4)); ax.set_yticklabels(lab, fontsize=7.5); ax.set_ylim(-0.6, 3.7); ax.invert_yaxis(); ax.set_xlim(-2.1, 0.5)
ax.set_xlabel("自身价格弹性（3 小时窗口，对数充电时长 / 对数总价）")
ax.set_title("(d) 可识别性可行性检验（仅训练期，小时数据，非最终结果）", loc="left", fontsize=10, fontweight="bold")
fig.savefig(f"{OUT}/数据核查图_v1.png", dpi=170, bbox_inches="tight"); print("saved")
