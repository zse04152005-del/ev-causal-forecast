"""论文图 4：CPA-STGNN 四个模块的结构图（与 03_代码 中的实现逐项对应，见 04_图表/README_图稿说明.md 的核对表）。

(a) 价格盲时空骨干（Graph WaveNet 编码器）+ 非交叉分位数头   src/models/backbones/graph_wavenet.py、heads/quantile.py
(b) 价格响应与截断反馈锚定                                     src/models/heads/price.py、causal/anchors.py、scripts/train.py
(c) 空间溢出环带（真实小区几何）                               src/data/prices.py（ring_members、ring_exposure）、causal/anchors.py
(d) 带质量掩码的自适应保形校准                                 src/conformal/aci.py

运行：python fig4_modules.py  → Fig4_CPA-STGNN_modules.{pdf,png,svg} 以及每个面板单独的文件 Fig4{a,b,c,d}_*.{pdf,png,svg}
      （在 04_图表/绘图脚本/ 下运行时写到 04_图表/模块架构图/；否则写到脚本旁边的 out/）
(c) 需要 assets/fig4c_zone525.json（由 make_fig4c_geometry.py 从 UrbanEV 数据生成）。
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import icons  # noqa: E402
from figlib import DOT_GRAD, FS, INK, INK2, LW, PAL, RULE, Canvas  # noqa: E402

# 输出目录：放在项目的 04_图表/绘图脚本/ 下运行时直接写到 04_图表/模块架构图/，否则写到脚本旁边的 out/
OUT = (os.path.abspath(os.path.join(HERE, "..", "模块架构图")) if os.path.isdir(os.path.join(HERE, "..", "模块架构图"))
       else os.path.join(HERE, "out"))
matplotlib.rcParams["hatch.linewidth"] = 0.3
B, P, G, C, O, N = (PAL[k] for k in ("demand", "price", "cov", "causal", "output", "gray"))  # noqa: E741
REAL = "--real" in sys.argv          # 用锚定文件中的真实层级与 δ 画 (b)(c)（阶段 6 定稿后使用）；默认画示意
PW_L, PW_R, PH = 98.0, 88.0, 70.0          # 左列（a、c）、右列（b、d）面板宽度与高度（毫米）


# ============================================================ 小工具
def tbox(c, x, y, w, h, s, fc="white", ec=INK, lw=LW["thin"], size=None, color=INK, r=0.7, weight="normal",
         ls="-", z=4):
    """带居中文字的小方框；返回中心。"""
    c.box(x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, ls=ls, z=z)
    c.text(x + w / 2, y + h / 2 + 0.05, s, size=size, color=color, weight=weight, z=z + 1, fit=(x, y, x + w, y + h))
    return x + w / 2, y + h / 2


def diamond(c, x, y, r, filled, color=None, z=5):
    col = color or P["stroke"]
    c.ax.add_patch(Polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], closed=True, fc=col if filled else "white",
                           ec=col, lw=LW["thin"], zorder=z, joinstyle="miter"))


def circ(c, x, y, r, s=None, fc="white", ec=INK, lw=LW["thin"], size=None, color=INK, z=4):
    c.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    if s:
        c.text(x, y + 0.05, s, size=size, color=color, z=z + 1)


# ============================================================ (a) 价格盲骨干 + 分位数头
def panel_a(c: Canvas, lead: float = 0.0):
    T = c.text
    # ---- 标题行（lead：总图里给 (a) 标签留出的位置）
    T(1.0 + lead, 2.7, "Price-blind ST backbone", ha="left", weight="bold", size=FS["label"], fit=(0.5, 48.0))

    # ---- 8 层叠放的时空层（仿 Graph WaveNet 原图：虚线框层叠）
    fx, fy, fw, fh = 2.5, 7.6, 42.0, 40.6
    dash = (0, (3.2, 1.2, 0.8, 1.2))
    for k in (2, 1):
        c.box(fx + 1.2 * k, fy - 1.2 * k, fw, fh, fc="white", ec=RULE, lw=LW["thin"], ls=dash, r=1.0, z=1 + 0.1 * (2 - k))
    c.box(fx, fy, fw, fh, fc="white", ec=INK2, lw=LW["thin"], ls=dash, r=1.0, z=1.5)
    T(fx + fw - 1.0, fy + 2.1, r"$\times 8$", ha="right", size=FS["label"], color=INK2)
    xc = 23.0                                                        # 主干竖线
    # 门控 TCN 子框
    gx0, gy0, gx1, gy1 = 8.9, 26.0, 37.1, 43.3
    c.box(gx0, gy0, gx1 - gx0, gy1 - gy0, fc="none", ec=INK2, lw=LW["hair"], ls=(0, (1.0, 1.0)), r=0.8, z=1.8)
    T(gx0 + 1.0, gy0 + 1.8, "Gated TCN", ha="left", size=FS["body"], color=INK2, fit=(gx0, xc - 0.8))
    ya, yt, yg = 40.1, 35.3, 30.9                                   # TCN 框、tanh/σ、⊗ 的纵坐标
    for x0, lab in ((16.0, "TCN-a"), (30.0, "TCN-b")):
        tbox(c, x0 - 4.6, ya - 1.7, 9.2, 3.4, lab, fc=B["mid"], ec=B["stroke"], lw=LW["thin"])
    for x0, lab in ((16.0, r"$\tanh$"), (30.0, r"$\sigma$")):      # 椭圆：7 pt 的 tanh 放得下
        c.ax.add_patch(Ellipse((x0, yt), 6.4, 3.9, fc=N["mid"], ec=N["stroke"], lw=LW["thin"], zorder=4))
        T(x0, yt + 0.05, lab, size=7.0 if "tanh" in lab else 7.5, z=5, fit=(x0 - 3.2, x0 + 3.2))
    c.op(xc, yg, "x", r=1.45)
    # 输入分叉：J1（残差分支）→ J0（两路 TCN）
    j1, j0 = 46.6, 44.8
    c.line([(xc, fy + fh + 0.02), (xc, j0)], arrow=False)
    c.line([(xc, j0), (16.0, j0), (16.0, ya + 1.7)], head=(1.0, 0.75))
    c.line([(xc, j0), (30.0, j0), (30.0, ya + 1.7)], head=(1.0, 0.75))
    for x0 in (16.0, 30.0):
        c.line([(x0, ya - 1.7), (x0, yt + 1.95)], head=(1.0, 0.75))
    c.line([(16.0, yt - 1.95), (16.0, yg), (xc - 1.45, yg)], head=(1.0, 0.75))
    c.line([(30.0, yt - 1.95), (30.0, yg), (xc + 1.45, yg)], head=(1.0, 0.75))
    for yy in (j1, j0):
        c.ax.add_patch(Circle((xc, yy), 0.42, fc=INK, ec="none", zorder=5))
    # 跳连接：门控输出 → 1×1 → 跳连接汇总
    ys = 24.5
    c.ax.add_patch(Circle((xc, ys), 0.42, fc=INK, ec="none", zorder=5))
    tbox(c, 36.0, ys - 1.5, 6.6, 3.0, r"1$\times$1", fc=B["mid"], ec=B["stroke"])
    c.line([(xc, ys), (36.0, ys)], head=(1.0, 0.75))
    # 多图扩散卷积 + 图结构小图标
    gcy = 20.4
    tbox(c, 10.3, gcy - 1.95, 25.4, 3.9, "Multi-graph GCN", fc=B["mid"], ec=B["stroke"])
    c.line([(xc, yg - 1.45), (xc, gcy + 1.95)], head=(1.0, 0.75))
    c.graph_icon(37.4, 13.4, s=5.0, color=N["stroke"])
    c.line([(38.0, 17.6), (35.8, gcy - 0.4)], color=N["stroke"], lw=LW["thin"], head=(0.9, 0.6))
    T(34.7, 14.3, r"$A_s$", ha="right", size=FS["body"], color=INK2)
    # 残差 ⊕ → BN → 下一层
    yr = 14.7
    c.op(xc, yr, "+", r=1.45)
    c.line([(xc, gcy - 1.95), (xc, yr + 1.45)], head=(1.0, 0.75))
    c.line([(xc, j1), (5.5, j1), (5.5, yr), (xc - 1.45, yr)], head=(1.0, 0.75))
    T(7.0, 36.0, "residual", rotation=90, size=FS["body"], color=INK2)
    tbox(c, xc - 3.6, 9.35, 7.2, 2.9, "BN", fc=N["mid"], ec=N["stroke"])
    c.line([(xc, yr - 1.45), (xc, 12.25)], head=(0.9, 0.7))
    c.line([(xc, 9.35), (xc, 4.9)], head=(1.0, 0.75))

    # ---- 跳连接汇总与输出层（列）
    ox, sy = 56.3, 27.4
    c.op(ox, sy, "+", r=1.45)
    for k, (x0, y0) in enumerate(((42.6, ys), (fx + fw + 1.2, ys - 1.2), (fx + fw + 2.4, ys - 2.4))):
        dx, dy = ox - x0, sy - y0
        L = math.hypot(dx, dy)
        c.line([(x0, y0), (ox - dx / L * 1.45, sy - dy / L * 1.45)], lw=LW["thin"] if k else LW["flow"],
               color=INK if k == 0 else INK2, head=(0.9, 0.65))
    T(ox, sy + 3.3, "skip sum,", size=FS["body"], color=INK2)
    T(ox, sy + 6.5, "last step", size=FS["body"], color=INK2)
    T(ox - 0.8, sy + 10.6, r"$d_\ell=1,2,4,8,$", size=FS["body"], color=INK2)
    T(ox - 0.8, sy + 13.9, r"$1,2,4,8$", size=FS["body"], color=INK2)
    prev, gap = sy - 1.45, 1.75
    for lab, pal in (("ReLU", N), (r"1$\times$1", B), ("ReLU", N), (r"1$\times$1", B)):
        top = prev - gap - 2.9
        tbox(c, ox - 4.2, top, 8.4, 2.9, lab, fc=pal["mid"], ec=pal["stroke"])
        c.line([(ox, prev), (ox, prev - gap)], head=(0.85, 0.62))
        prev = top
    zy = prev + 1.45                                                   # 最上面一个 1×1 的中心

    # ---- 分位数头（v1.3：向下放大到 y≈56，框、间距、分位数小图都加大）
    hx0, hy0, hx1, hy1 = 63.0, 0.8, 97.6, 56.3
    c.box(hx0, hy0, hx1 - hx0, hy1 - hy0, fc=B["fill"], ec=B["stroke"], lw=LW["box"], r=1.3, z=1)
    T(hx0 + 2.2, 3.7, "Quantile head", ha="left", weight="bold", size=FS["label"], fit=(hx0 + 1.5, hx1 - 1))
    bx0, bw, bh = 75.6, 10.4, 3.4
    ax_ = 92.0                                                         # ⊕ 与竖直链的横坐标
    rows = [(zy, None), (zy + 5.6, r"$\mathbf{c}_{t+h},\mathbf{w}_{t+h}$"), (zy + 11.2, r"$h$")]
    ay_ = rows[1][0]
    for k, (yy, lab) in enumerate(rows):
        pal = G if k == 1 else B
        tbox(c, bx0, yy - bh / 2, bw, bh, "Embed" if k == 2 else "Linear", fc="white", ec=pal["stroke"])
        if lab:
            T(bx0 - 1.0, yy + 0.1, lab, ha="right", size=FS["body"], color=G["stroke"] if k == 1 else INK,
              fit=(hx0 + 0.6, bx0))
        dx, dy = ax_ - (bx0 + bw), ay_ - yy
        L = math.hypot(dx, dy)
        c.line([(bx0 + bw, yy), (ax_ - dx / L * 1.6, ay_ - dy / L * 1.6)], head=(1.0, 0.72),
               color=G["stroke"] if k == 1 else INK)
    c.line([(ox + 4.2, zy), (bx0, zy)], color=B["stroke"], head=(1.0, 0.75))
    T(69.0, zy - 1.9, r"$\mathbf{z}_i$", size=FS["body"], color=B["stroke"])
    c.op(ax_, ay_, "+", r=1.6)
    prev = ay_ + 1.6
    gap, ch = 1.9, 3.4
    for k, (lab, pal) in enumerate((("ReLU", N), ("Linear", B), ("ReLU", N), ("Linear", B))):
        top = rows[2][0] + 3.3 if k == 0 else prev + gap            # 第一个框放在第三行输入之下，避免斜线穿过
        tbox(c, ax_ - 5.0, top, 10.0, ch, lab, fc="white" if pal is B else pal["mid"], ec=pal["stroke"])
        c.line([(ax_, prev), (ax_, top)], head=(0.95, 0.68))
        prev = top + ch
    sy0 = prev + gap
    tbox(c, 72.4, sy0, 24.6, 3.8, "softplus, cumsum", fc=B["mid"], ec=B["stroke"])
    c.line([(ax_, prev), (ax_, sy0)], head=(0.95, 0.68))
    # 非交叉分位数小图：7 段 softplus 增量累加
    qx = 64.6
    inc = [0.9, 0.8, 1.1, 1.4, 1.1, 0.9, 0.8]
    cum = np.cumsum(inc)
    base = hy1 - 1.4
    for m, v in enumerate(cum):
        hgt = v * 0.68
        c.rect(qx + m * 1.5, base - hgt, 1.1, hgt, fc=B["strong"] if m == 3 else B["mid"], ec=B["stroke"],
               lw=LW["hair"], z=3)
    T(76.0, base - 2.0, r"$b^{(0.05)}\leq\cdots\leq b^{(0.95)}$", ha="left", size=FS["body"], fit=(75.4, hx1 - 0.4))
    c.line([(85.0, sy0 + 3.8), (85.0, base - 4.4)], head=(0.9, 0.65))

    # ---- 输入嵌入（底部）
    ey = 55.6
    c.op(xc, ey, "+", r=1.45)
    c.line([(xc, ey - 1.45), (xc, fy + fh + 0.02)], head=(1.0, 0.75))
    T(xc + 1.2, 50.4, "left-pad to 31 steps", ha="left", size=FS["body"], color=INK2, fit=(xc, 50.0))
    c.stack(1.4, ey - 2.2, 7.2, 4.4, fc=B["fill"], ec=B["stroke"], n=3, off=0.7)
    c.dots(2.6, ey, n=4, dx=1.5, r=0.42, fc=B["stroke"], gap_at=3)
    tbox(c, 12.3, ey - 1.45, 7.0, 2.9, r"1$\times$1", fc=B["mid"], ec=B["stroke"])
    c.line([(9.3, ey), (12.3, ey)], color=B["stroke"], head=(0.9, 0.65))
    c.line([(19.3, ey), (xc - 1.45, ey)], color=B["stroke"], head=(0.9, 0.65))
    T(1.4, ey + 4.6, r"$y_{i,\tau},\ \tilde m_{i,\tau}$", ha="left", size=FS["body"], color=B["stroke"])
    T(1.4, ey + 8.0, r"$\tau=t-23,\dots,t$", ha="left", size=FS["body"], color=INK2)
    tbox(c, 27.3, ey - 1.45, 8.6, 2.9, "Linear", fc="white", ec=G["stroke"])
    c.line([(27.3, ey), (xc + 1.45, ey)], color=G["stroke"], head=(0.9, 0.65))
    c.line([(39.0, ey), (35.9, ey)], color=G["stroke"], head=(0.9, 0.65))
    T(39.6, ey + 0.1, r"$\mathbf{c}_\tau,\mathbf{w}_\tau$", ha="left", size=FS["body"], color=G["stroke"])
    tbox(c, xc - 4.3, 61.2, 8.6, 2.9, "Linear", fc="white", ec=G["stroke"])
    c.line([(xc, 61.2), (xc, ey + 1.45)], color=G["stroke"], head=(0.9, 0.65))
    T(xc, 67.9, r"static & POI $\mathbf{s}_i$", size=FS["body"], color=G["stroke"])
    c.line([(xc, 65.9), (xc, 64.1)], color=G["stroke"], head=(0.8, 0.6))

    # ---- 图结构支撑（右下）：四种图的小图标图例（v1.3，取代原来的四行文字）
    lx, ly = 49.6, 59.3
    T(lx, ly, r"Graph supports $A_s$ (2-hop)", ha="left", weight="bold", size=FS["body"], fit=(lx, PW_L - 0.3))
    ty, ts = 61.1, 4.8                                                 # 图标方块的上沿与边长
    pitch, cx0 = 12.2, 55.2

    def node(x, y, col, fc=None, r=0.45):
        c.ax.add_patch(Circle((x, y), r, fc=fc or col, ec=col, lw=LW["hair"], zorder=5))

    labels = ["adjacency", "distance", "POI sim.", "adaptive"]
    for k, lab in enumerate(labels):
        cx, cy = cx0 + k * pitch, ty + ts / 2
        adaptive = k == 3
        c.box(cx - ts / 2, ty, ts, ts, fc=B["fill"] if adaptive else "white", ec=B["stroke"] if adaptive else N["stroke"],
              lw=LW["thin"], r=0.6, z=3)
        if k == 0:                                                     # 对称邻接：实线图
            p = [(-1.5, -1.3), (0.4, -1.6), (1.6, -0.1), (-0.5, 0.6), (0.9, 1.6), (-1.5, 1.4)]
            for i, j in ((0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (3, 5)):
                c.line([(cx + p[i][0], cy + p[i][1]), (cx + p[j][0], cy + p[j][1])], color=N["stroke"],
                       lw=LW["hair"], arrow=False, z=4)
            for q in p:
                node(cx + q[0], cy + q[1], INK2)
        elif k == 1:                                                   # 高斯距离核：边越远越细，虚线圈表示带宽
            c.ax.add_patch(Circle((cx, cy), 1.75, fc="none", ec=N["stroke"], lw=LW["hair"], ls=(0, (1.0, 1.0)), zorder=4))
            for (dx, dy), w in (((1.5, -1.0), LW["box"]), ((-1.6, 0.9), LW["thin"]), ((0.6, 1.9), LW["hair"] * 0.6)):
                c.line([(cx, cy), (cx + dx, cy + dy)], color=N["stroke"], lw=w, arrow=False, z=4)
                node(cx + dx, cy + dy, INK2)
            node(cx, cy, INK2, r=0.55)
        elif k == 2:                                                   # POI 余弦相似：绿色节点 + 虚线边
            p = [(-1.5, -1.1), (1.3, -1.4), (1.5, 1.2), (-1.0, 1.4), (0.0, 0.0)]
            for i, j in ((0, 4), (1, 4), (2, 4), (3, 4), (0, 3)):
                c.line([(cx + p[i][0], cy + p[i][1]), (cx + p[j][0], cy + p[j][1])], color=G["stroke"], lw=LW["hair"],
                       ls=(0, (1.0, 0.8)), arrow=False, z=4)
            for q, fcol in zip(p, (G["strong"], G["stroke"], G["strong"], G["stroke"], G["stroke"])):
                node(cx + q[0], cy + q[1], G["stroke"], fc=fcol, r=0.5)
        else:                                                          # 自适应：由 E1E2ᵀ 学出的稠密权重矩阵
            n_, cs = 4, 1.05
            M = np.array([[.9, .3, .1, .2], [.3, .8, .4, .1], [.1, .4, .9, .3], [.2, .1, .3, .7]])
            for i in range(n_):
                for j in range(n_):
                    v = M[i, j]
                    col = tuple((1 - v) * np.array(matplotlib.colors.to_rgb(B["mid"])) + v * np.array(matplotlib.colors.to_rgb(B["stroke"])))
                    c.rect(cx - n_ * cs / 2 + j * cs, cy - n_ * cs / 2 + i * cs, cs, cs, fc=col, ec="white", lw=0.2, z=4)
        T(cx, ty + ts + 2.0, lab, size=FS["body"], color=B["stroke"] if adaptive else INK2, fit=(cx - pitch / 2 + 0.2, cx + pitch / 2 - 0.2))


# ============================================================ (b) 价格响应与截断反馈锚定
CTX = ["day", "night", "day", "night"]
# 锚定表的层级标记只是示意（正式锚定在阶段 6 生成后，可用 load_provenance() 读取真实层级再画）。
# 合并规则（anchors.merge_levels）：格子不合格 → 该情境的全市估计（各功能区共用）→ 全市总估计；
# 所以同一列里所有"非自身格子"必然处在同一层级
PROV_ILLUSTRATIVE = [["city", "context", "context", "context"],
                     ["city", "cell", "cell", "cell"],
                     ["city", "cell", "context", "cell"]]
GROUPS = ["Low-density", "Residential", "Commercial"]


def load_provenance():
    """锚定表每个格子取自哪个层级（cell / context / city）；读 configs/anchor/hourly_train.json，读不到时用示意。"""
    for p in (os.path.join(HERE, "..", "code", "configs", "anchor", "hourly_train.json"),
              os.path.join(HERE, "..", "..", "03_代码", "configs", "anchor", "hourly_train.json")):
        if os.path.exists(p):
            js = json.load(open(p, encoding="utf-8"))
            names = ["weekday_day", "weekday_night", "weekend_day", "weekend_night"]
            prov = [["city"] * 4 for _ in range(3)]
            for e in js["own"]:
                prov[int(e["group"])][names.index(e["context"])] = e.get("level", "cell")
            cross = [float(e["delta"]) for e in js.get("cross", [])][:3]
            return prov, cross, os.path.abspath(p)
    return ([["city", "context", "city", "context"], ["city", "cell", "cell", "cell"],
             ["city", "cell", "city", "cell"]], [0.37, 0.53, 0.18], None)


def glyph(c, x, y, kind, r=0.9, color=None, z=6):
    """层级标记：● 格子自身估计，◐ 并入情境层级，○ 并入全市层级。"""
    col = color or C["stroke"]
    if kind == "cell":
        c.ax.add_patch(Circle((x, y), r, fc=col, ec=col, lw=LW["thin"], zorder=z))
    elif kind == "context":
        c.ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=LW["thin"], zorder=z))
        c.ax.add_patch(Wedge((x, y), r, 90, 270, fc=col, ec=col, lw=0.0, zorder=z + 0.1))
    else:
        c.ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=LW["thin"], zorder=z))


def panel_b(c: Canvas, lead: float = 0.0):
    T = c.text
    prov = load_provenance()[0] if REAL else PROV_ILLUSTRATIVE
    W = PW_R
    T(1.0 + lead, 2.7, "Anchored price response", ha="left", weight="bold", size=FS["label"], fit=(0.5, W))

    # ---- 锚定表 β̂_{g,κ}：3 个功能区 × 4 个情境
    x0, cw, y0, rh = 19.2, 7.9, 11.4, 4.3
    T(x0 + cw, 6.6, "Workday", size=FS["body"], color=INK2)
    T(x0 + 3 * cw, 6.6, "Rest day", size=FS["body"], color=INK2)
    T(x0 - 1.0, 6.3, r"$\kappa(t{+}h)$", ha="right", size=FS["body"], color=INK2)
    T(x0 - 1.0, 9.9, r"$g(i)$", ha="right", size=FS["body"], color=INK2)
    for k in range(4):
        T(x0 + (k + 0.5) * cw, 9.6, CTX[k], size=FS["body"], color=INK2)
    for a in (0, 2):
        c.line([(x0 + a * cw + 0.4, 8.1), (x0 + (a + 2) * cw - 0.4, 8.1)], color=RULE, lw=LW["hair"], arrow=False)
    sel = (2, 1)                                                   # 示例格子：商业区 × 工作日夜间（格子自身估计）
    for g in range(3):
        yy = y0 + g * rh
        T(x0 - 1.0, yy + rh / 2 + 0.05, GROUPS[g], ha="right", size=FS["body"],
          weight="bold" if g == sel[0] else "normal", color=C["stroke"] if g == sel[0] else INK, fit=(0.3, x0))
        for k in range(4):
            c.rect(x0 + k * cw, yy, cw, rh, fc=C["fill"] if prov[g][k] != "cell" else C["mid"], ec=C["stroke"],
                   lw=LW["hair"], z=3)
            glyph(c, x0 + (k + 0.5) * cw, yy + rh / 2, prov[g][k], r=0.95)
    sx, sy = x0 + sel[1] * cw, y0 + sel[0] * rh
    c.rect(sx, sy, cw, rh, fc="none", ec=C["stroke"], lw=LW["causal"], z=7)

    # ---- 右侧：层级只用符号 + 一个词（合并阈值写在图注）；锁 = 锚定值固定、不可训练
    rx = x0 + 4 * cw + 4.2
    icons.lock(c, rx + 1.1, 7.0, 2.2)
    T(rx + 2.9, 6.9, "fixed", ha="left", size=FS["body"], color=C["stroke"], fit=(rx, W - 0.2))
    for k, kind in enumerate(("cell", "context", "city")):
        yy = y0 + (k + 0.5) * rh
        glyph(c, rx + 1.1, yy, kind, r=0.95)
        T(rx + 2.9, yy + 0.05, kind, ha="left", size=FS["body"], fit=(rx, W - 0.2))

    # ---- 查表结果进入 η（公式与 heads/price.py、cpastgnn.eta 一致；可选的 γU 写在图注）
    c.line([(sx + cw / 2, sy + rh), (sx + cw / 2, 29.0)], color=C["stroke"], lw=LW["causal"], ls=(0, (2.2, 1.2)),
           head=(1.1, 0.8))
    T(sx + cw / 2 - 5.0, 31.6, r"$\eta=\hat\beta_{g(i),\kappa(t+h)}\,\Delta\ell+\Sigma_k\,\hat\delta_k\,S^{(k)}$",
      ha="left", size=FS["body"], fit=(0.3, W - 0.2))
    if not REAL:
        T(x0 + cw - 0.6, y0 + 3 * rh + 1.9, "(illustrative)", ha="right", size=FS["body"], color=INK2, fit=(0.3, sx))

    # ---- 下半部分：梯度流对比（截断反馈 vs 联合训练），只用计算图表达，说明放图注
    fy0, fy1 = 34.7, 69.6
    for (a0, a1, title, cut) in ((0.4, 42.9, "Cut feedback (default)", True),
                                 (45.1, W - 0.4, "Joint training (ablations)", False)):
        c.box(a0, fy0, a1 - a0, fy1 - fy0, fc=C["fill"] if cut else "white", ec=C["stroke"] if cut else INK2,
              lw=LW["thin"], ls=(0, (2.2, 1.2)) if cut else "-", r=1.0, z=1)
        T(a0 + 1.4, fy0 + 2.5, title, ha="left", weight="bold", size=FS["body"], color=C["stroke"] if cut else INK2,
          fit=(a0, a1))
        mini_graph(c, a0, fy0, cut)


def mini_graph(c: Canvas, ox: float, oy: float, cut: bool):
    """一个小计算图：β、φ → ŷ → L_pin；灰色点线为梯度。"""
    T = c.text
    by, yy, py = oy + 13.4, oy + 19.6, oy + 25.8                    # β 行、ŷ、φ 行的纵坐标
    nx0, nw = ox + 1.6, 19.8
    xm = nx0 + nw + 4.6
    lx0, lw_ = xm + 5.2, 9.6
    if cut:
        tbox(c, nx0, by - 2.4, nw, 4.8, r"$\hat\beta_{g,\kappa},\ \hat\delta_k$", fc=C["mid"], ec=C["stroke"],
             lw=LW["box"])
        icons.lock(c, nx0 + nw - 0.4, by - 2.4, 2.0)
        c.line([(nx0 + nw, by), (xm, by), (xm, yy - 1.5)], color=C["stroke"], lw=LW["causal"], ls=(0, (2.2, 1.2)),
               head=(1.1, 0.8))
    else:
        tbox(c, nx0, by - 2.4, nw, 4.8, r"$\beta,\ \delta$", fc=P["fill"], ec=P["stroke"], lw=LW["box"])
        c.line([(nx0 + nw, by), (xm, by), (xm, yy - 1.5)], color=P["stroke"], lw=LW["flow"], head=(1.1, 0.8))
    c.box(nx0, py - 3.4, nw, 6.8, fc=B["mid"], ec=B["stroke"], lw=LW["box"], r=0.7, z=4)
    T(nx0 + nw / 2, py - 1.5, r"$\phi$", size=7.5, z=5)
    T(nx0 + nw / 2, py + 1.75, "backbone, head", size=FS["body"], z=5, fit=(nx0, nx0 + nw))
    c.line([(nx0 + nw, py), (xm, py), (xm, yy + 1.5)], color=B["stroke"], head=(1.1, 0.8))
    circ(c, xm, yy, 1.5, None, fc="white", ec=O["stroke"], lw=LW["box"])
    T(xm, yy + 0.05, r"$\hat y$", size=FS["body"], color=O["stroke"], z=7)
    tbox(c, lx0, yy - 1.9, lw_, 3.8, r"$\mathcal{L}_{\mathrm{pin}}$", fc=O["fill"], ec=O["stroke"], lw=LW["box"])
    c.line([(xm + 1.5, yy), (lx0, yy)], color=O["stroke"], head=(1.1, 0.8))
    # 梯度（灰色点线）：L_pin → φ 总是存在
    gx, gb = lx0 + lw_ / 2, py + 5.4
    c.line([(gx, yy + 1.9), (gx, gb), (nx0 + nw / 2, gb), (nx0 + nw / 2, py + 3.4)], color=INK2,
           lw=LW["flow"], ls=DOT_GRAD, head=(1.0, 0.7))
    T(gx - 1.0, gb - 2.3, r"$\nabla_\phi$", ha="right", size=FS["body"], color=INK2)
    gt = by - 6.4                                                  # 通向 β 的梯度路径所在高度
    if cut:
        c.line([(gx, yy - 1.9), (gx, gt), (xm + 1.3, gt)], color=INK2, lw=LW["flow"], ls=DOT_GRAD, arrow=False)
        c.scissors(xm - 0.6, gt, s=2.6, angle=180, color=C["stroke"])
        T(xm - 2.8, gt + 0.05, r"$\nabla_{\beta,\delta}=0$", ha="right", size=FS["body"], color=C["stroke"],
          fit=(ox, xm))
    else:
        c.line([(gx, yy - 1.9), (gx, gt), (nx0 + nw / 2, gt), (nx0 + nw / 2, by - 2.4)], color=INK2, lw=LW["flow"],
               ls=DOT_GRAD, head=(1.0, 0.7))
        T(gx - 1.0, gt + 2.1, r"$\nabla_{\beta,\delta}$", ha="right", size=FS["body"], color=INK2,
          fit=(nx0 + nw + 0.5, gx))


# ============================================================ (c) 空间溢出环带（真实几何）
RING_FILL = [P["strong"], "#F8DAC2", "#FCECDF"]


def load_geometry(zone: int = 525) -> dict:
    path = os.path.join(HERE, "assets", f"fig4c_zone{zone}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pav(v):
    """与 src/causal/anchors.project_monotone_nonneg 相同的 PAV 投影（δ1 ≥ δ2 ≥ … ≥ 0）。"""
    out = []
    for x in v:
        out.append([float(x), 1])
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            a, b = out.pop(), out.pop()
            out.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    res = []
    for val, w in out:
        res += [max(val, 0.0)] * w
    return res


def panel_c(c: Canvas, lead: float = 0.0):
    T = c.text
    geo = load_geometry()
    T(1.0 + lead, 2.7, "Distance-ring spillover exposure", ha="left", weight="bold", size=FS["label"], fit=(0.5, PW_L))

    # ---- 地图（以中心小区质心为原点，km；北向上）
    R = 7.2
    mx0, my0, msz = 0.6, 5.0, 54.0
    sc = msz / (2 * R)

    def P2(x, y):
        return mx0 + (x + R) * sc, my0 + (R - y) * sc
    clip = c.rect(mx0, my0, msz, msz, fc="none", ec=INK2, lw=LW["thin"], z=9)
    for e in geo["zones"]:
        if e.get("ring") == 0:
            fc, ec, lw, z = B["mid"], B["stroke"], LW["box"], 3
        elif e.get("ring") in (1, 2, 3):
            fc, ec, lw, z = RING_FILL[e["ring"] - 1], "#9A9A9A", LW["hair"], 2
        elif e["in_data"]:
            fc, ec, lw, z = "white", "#B5B5B5", LW["hair"], 1.5
        else:
            fc, ec, lw, z = "white", "#C8C8C8", LW["hair"], 1.4
        for part in e["parts"]:
            pts = [P2(x, y) for x, y in part]
            poly = Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round")
            c.ax.add_patch(poly)
            poly.set_clip_path(clip)
            if not e["in_data"]:                                    # 无充电数据：浅灰斜线（边线宽 0，只画阴影线）
                hp = Polygon(pts, closed=True, fc="none", ec="#C4C4C4", lw=0.0, hatch="//////", zorder=z + 0.05)
                c.ax.add_patch(hp)
                hp.set_clip_path(clip)
    cx, cy = P2(0, 0)
    dots = np.array([e["dot"] for e in geo["zones"] if e.get("ring") in (0, 1, 2, 3)])
    for k, r in enumerate((2, 4, 6)):
        circle = Circle((cx, cy), r * sc, fc="none", ec=INK, lw=LW["thin"], ls=(0, (3.0, 1.6)), zorder=6)
        c.ax.add_patch(circle)
        circle.set_clip_path(clip)
        # 环带标注放在离质心圆点最远的角度（偏好右上方），避免遮住圆点
        best, ang = -1e9, 60.0
        hw, hh = 0.80, 0.40                                           # 标注半宽、半高（km）
        for a in np.arange(16.0, 165.0, 1.0):
            p = np.array([r * math.cos(math.radians(a)), r * math.sin(math.radians(a))])
            dx = np.maximum(np.abs(dots[:, 0] - p[0]) - hw, 0.0)
            dy = np.maximum(np.abs(dots[:, 1] - p[1]) - hh, 0.0)
            d = min(np.min(np.hypot(dx, dy)), 0.6) - 0.001 * abs(a - 60.0)   # 间隙够 0.6 km 就优先靠近右上
            if d > best:
                best, ang = d, a
        lx, ly = cx + r * sc * math.cos(math.radians(ang)), cy - r * sc * math.sin(math.radians(ang))
        t = T(lx, ly, f"{r} km", size=FS["body"], color=INK, z=8)
        t.set_bbox(dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    # 质心圆点（放在 distance.csv 的精确距离上）
    for e in geo["zones"]:
        if e.get("ring") not in (1, 2, 3):
            continue
        x, y = P2(*e["dot"])
        kind = e["pricing"]
        fc, ec = {"TOU": (P["stroke"], P["stroke"]), "weak": (P["mid"], P["stroke"]), "fixed": ("white", INK2)}[kind]
        c.ax.add_patch(Circle((x, y), 0.72, fc=fc, ec=ec, lw=LW["thin"], zorder=7))
    c.ax.add_patch(Circle((cx, cy), 0.85, fc=B["stroke"], ec="white", lw=0.5, zorder=7))
    T(cx - 1.35, cy - 0.75, r"$i$", size=FS["label"], color=B["stroke"], z=8)
    # 指北针与比例尺
    nx, ny_ = mx0 + 3.6, my0 + 3.4
    c.ax.add_patch(Polygon([(nx, ny_), (nx - 1.3, ny_ + 4.0), (nx, ny_ + 3.1)], closed=True, fc=INK, ec=INK, lw=0.3,
                           zorder=9))
    c.ax.add_patch(Polygon([(nx, ny_), (nx + 1.3, ny_ + 4.0), (nx, ny_ + 3.1)], closed=True, fc="white", ec=INK,
                           lw=0.3, zorder=9))
    T(nx, ny_ + 5.6, "N", size=FS["body"], weight="bold", z=9)
    bx, byy = mx0 + 2.5, my0 + msz - 3.2
    c.rect(bx - 0.9, byy - 3.3, 2 * sc + 4.6, 5.0, fc="white", ec="none", z=8.5, alpha=0.85)
    c.rect(bx, byy, sc, 0.9, fc=INK, ec=INK, lw=0.3, z=9)
    c.rect(bx + sc, byy, sc, 0.9, fc="white", ec=INK, lw=0.3, z=9)
    for k, lab in enumerate(("0", "1", "2 km")):
        T(bx + k * sc + (1.3 if k == 2 else 0), byy - 1.5, lab, size=FS["body"], z=9.5)

    # ---- 图例（右上）
    lx = 58.2
    ly = 6.6
    cnt = geo["ring_counts"]
    ent = [(B["mid"], B["stroke"], r"zone $i$ (TOU, commercial)"),
           (RING_FILL[0], "#9A9A9A", f"ring 1: [0, 2) km, {sum(cnt['ring1'].values())} zones"),
           (RING_FILL[1], "#9A9A9A", f"ring 2: [2, 4) km, {sum(cnt['ring2'].values())} zones"),
           (RING_FILL[2], "#9A9A9A", f"ring 3: [4, 6) km, {sum(cnt['ring3'].values())} zones")]
    for k, (fc, ec, lab) in enumerate(ent):
        yy = ly + 3.4 * k
        c.rect(lx, yy - 1.1, 3.4, 2.2, fc=fc, ec=ec, lw=LW["hair"], z=4)
        T(lx + 4.5, yy + 0.05, lab, ha="left", size=FS["body"], fit=(lx, PW_L - 0.2))
    ent2 = [("white", "#B5B5B5", "zones beyond 6 km", None), ("white", "#C8C8C8", "no charging data", "//////")]
    for k, (fc, ec, lab, hatch) in enumerate(ent2):
        yy = ly + 3.4 * (4 + k)
        c.rect(lx, yy - 1.1, 3.4, 2.2, fc=fc, ec=ec, lw=LW["hair"], z=4)
        if hatch:
            c.rect(lx, yy - 1.1, 3.4, 2.2, fc="none", ec="#C4C4C4", lw=0.0, z=4.1, hatch=hatch)
        T(lx + 4.5, yy + 0.05, lab, ha="left", size=FS["body"], fit=(lx, PW_L - 0.2))
    mk = [((P["stroke"], P["stroke"]), "strict TOU"), ((P["mid"], P["stroke"]), "weak TOU"),
          (("white", INK2), r"fixed price ($\Delta\ell_j\equiv0$)")]
    for k, ((fc, ec), lab) in enumerate(mk):
        yy = ly + 3.4 * (6 + k)
        c.ax.add_patch(Circle((lx + 1.7, yy), 0.72, fc=fc, ec=ec, lw=LW["thin"], zorder=5))
        T(lx + 4.5, yy + 0.05, lab, ha="left", size=FS["body"], fit=(lx, PW_L - 0.2))

    # ---- δ_k 小图：因果模块的原始估计（空心）→ PAV 投影到 δ1 ≥ δ2 ≥ δ3 ≥ 0（实心，模型使用）
    raw = load_provenance()[1] if REAL else [0.30, 0.40, 0.14]     # 默认为示意值（非估计结果），说明 PAV 投影的作用
    proj = pav(raw)
    gx0, gy0, gw, gh = 64.0, 45.6, 31.0, 11.0
    c.line([(gx0, gy0), (gx0, gy0 + gh), (gx0 + gw, gy0 + gh)], color=INK, lw=LW["thin"], arrow=False)
    T(gx0 - 1.0, gy0 + gh / 2, r"$\delta_k$", ha="right", size=FS["label"])
    T(gx0 - 1.0, gy0 + gh, "0", ha="right", size=FS["body"], color=INK2)
    top = max(max(raw), max(proj)) * 1.12
    xs = [gx0 + gw * (k + 0.5) / 3 for k in range(3)]

    def yv(v):
        return gy0 + gh - gh * v / top
    for k in range(3):
        T(xs[k], gy0 + gh + 2.3, f"{2 * k}–{2 * k + 2} km", size=FS["body"], color=INK2)
    c.line([(xs[k], yv(proj[k])) for k in range(3)], color=P["stroke"], lw=LW["flow"], arrow=False, z=4)
    for k in range(3):
        if abs(raw[k] - proj[k]) > 1e-9:
            c.line([(xs[k], yv(raw[k])), (xs[k], yv(proj[k]))], color=INK2, lw=LW["hair"], ls=(0, (1.0, 0.8)),
                   arrow=False, z=4)
        diamond(c, xs[k], yv(raw[k]), 0.95, False)
        diamond(c, xs[k], yv(proj[k]), 0.95, True, z=6)
    for k, (filled, lab) in enumerate(((False, r"raw $\hat\delta_k$ from switch events"),
                                      (True, r"used: PAV to $\delta_1\geq\delta_2\geq\delta_3\geq0$"))):
        yy = gy0 - 7.4 + 3.7 * k
        diamond(c, lx + 1.7, yy, 0.95, filled)
        T(lx + 4.5, yy + 0.05, lab, ha="left", size=FS["body"], fit=(lx, PW_L - 0.2))
    if not REAL:
        T(gx0 + gw, gy0 - 0.4, "schematic", ha="right", size=FS["body"], color=INK2)

    # ---- 底部公式（与 src/data/prices.ring_members、ring_exposure 一致）
    T(0.6, 62.6, r"$S^{(k)}_{i,t}=\mathrm{mean}\{\Delta\ell_{j,t}:\ j\in\mathcal{R}_k(i)\}$,  "
      r"$\mathcal{R}_k(i)=\{j\neq i:\ r_{k-1}\leq d_{ij}<r_k\}$,  $d_{ij}$: centroid distance",
      ha="left", size=FS["body"], fit=(0.3, PW_L - 0.3))
    T(0.6, 67.2, r"$\eta^{\mathrm{spill}}_{i,t}=\Sigma_k\,\delta_k S^{(k)}_{i,t}$;  an empty ring gives $S^{(k)}_{i,t}=0$",
      ha="left", size=FS["body"], fit=(0.3, PW_L - 0.3))


# ============================================================ (d) 带质量掩码的自适应保形校准（ACI）
def panel_d(c: Canvas, lead: float = 0.0):
    T = c.text
    W = PW_R
    T(1.0 + lead, 2.7, "Masked adaptive conformal calibration", ha="left", weight="bold", size=FS["label"], fit=(0.5, W))

    # ---- 示意数据（固定随机种子；只表达机制，不是结果）
    rng = np.random.default_rng(7)
    n, now, hstep = 96, 82, 7                                     # 目标时刻 τ = 0..95；当前时刻 t；步长 h
    tau = np.arange(n)
    med = 0.46 + 0.24 * np.sin(2 * np.pi * tau / 24 - 1.2)
    y = np.clip(med + rng.normal(0, 0.075, n), 0.03, 0.97)
    frozen = (tau >= 28) & (tau <= 35)
    y[frozen] = y[27]
    raw_w = 0.085 + 0.01 * np.sin(tau / 9.0)
    qhat = 0.020 + 0.018 * (1 + np.sin(tau / 11.0 + 0.8)) / 2      # σ_i·q̂ 随 α 缓慢变化
    lo_r, hi_r = med - raw_w, med + raw_w
    lo_c, hi_c = lo_r - qhat, hi_r + qhat

    # ---- 上图：区间带与观测（横轴为目标时刻）
    px0, px1, py0, py1 = 8.0, W - 1.5, 6.4, 22.8
    xs = lambda v: px0 + (px1 - px0) * v / (n - 1)                 # noqa: E731
    ys = lambda v: py1 - (py1 - py0) * v                            # noqa: E731
    c.line([(px0, py0 - 0.5), (px0, py1), (px1, py1)], color=INK, lw=LW["thin"], arrow=False, z=3)
    T(px0 - 1.2, (py0 + py1) / 2, r"$y_{i,\tau}$", ha="right", size=FS["label"])
    m = tau <= now + hstep
    band = list(zip(xs(tau[m]), ys(hi_r[m]))) + list(zip(xs(tau[m][::-1]), ys(lo_r[m][::-1])))
    c.ax.add_patch(Polygon(band, closed=True, fc=B["mid"], ec="none", zorder=2))
    for arr in (lo_c, hi_c):
        c.line(list(zip(xs(tau[m]), ys(arr[m]))), color=O["stroke"], lw=LW["thin"], arrow=False, z=3)
    # 冻结段（灰底）
    fx0, fx1 = xs(27.5), xs(35.5)
    c.rect(fx0, py0 - 0.5, fx1 - fx0, py1 - py0 + 0.5, fc=N["fill"], ec="none", z=1)
    T(px0 + 1.2, py0 + 0.9, "schematic", ha="left", size=FS["body"], color=INK2, z=7)
    # 观测点：未冻结为黑点，冻结为灰色 ×，落在校准区间外的加红圈
    obs = tau <= now
    for k in tau[obs]:
        X, Y = xs(k), ys(y[k])
        if frozen[k]:
            d = 0.42
            c.line([(X - d, Y - d), (X + d, Y + d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
            c.line([(X - d, Y + d), (X + d, Y - d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
        else:
            c.ax.add_patch(Circle((X, Y), 0.36, fc=INK, ec="none", zorder=5))
            if y[k] < lo_c[k] or y[k] > hi_c[k]:
                c.ax.add_patch(Circle((X, Y), 0.85, fc="none", ec=O["stroke"], lw=LW["thin"], zorder=5))
    # 当前时刻 t 与预测目标 t+h
    xn, xf = xs(now), xs(now + hstep)
    c.line([(xn, py0 - 0.5), (xn, py1 + 0.9)], color=INK, lw=LW["thin"], ls=(0, (2.0, 1.2)), arrow=False, z=4)
    c.line([(xf, ys(lo_c[now + hstep])), (xf, ys(hi_c[now + hstep]))], color=O["stroke"], lw=LW["causal"],
           arrow=False, z=6)
    c.line([(xf, py1), (xf, py1 + 0.9)], color=INK, lw=LW["thin"], arrow=False)
    T(xn, py1 + 2.4, r"$t$", size=FS["body"])
    T(xf, py1 + 2.4, r"$t{+}h$", size=FS["body"])
    # 分数窗口（最近 168 个已揭晓的目标）与延迟 h
    wx0 = xs(now - 60)
    yb = py1 + 4.8
    c.line([(wx0, yb), (xn, yb)], color=INK2, lw=LW["thin"], head=(0.9, 0.55), both=True)
    T((wx0 + xn) / 2, yb + 2.1, "score window: last 168 targets", size=FS["body"], color=INK2,
      fit=(0.3, xn))
    c.line([(xn, yb), (xf, yb)], color=O["stroke"], lw=LW["thin"], head=(0.9, 0.55), both=True)
    T((xn + xf) / 2, yb + 2.1, r"$h$", size=FS["body"], color=O["stroke"])
    # 图例（一行）
    gy = yb + 5.6
    lx = 8.0
    c.rect(lx, gy - 1.0, 3.6, 2.0, fc=B["mid"], ec="none", z=3)
    T(lx + 4.6, gy + 0.05, r"raw $[\hat y^{(0.05)},\hat y^{(0.95)}]$", ha="left", size=FS["body"])
    lx2 = lx + 30.0
    c.line([(lx2, gy), (lx2 + 3.6, gy)], color=O["stroke"], lw=LW["thin"], arrow=False)
    T(lx2 + 4.6, gy + 0.05, "calibrated", ha="left", size=FS["body"])
    lx3 = lx2 + 17.0
    c.ax.add_patch(Circle((lx3 + 1.3, gy), 0.85, fc="none", ec=O["stroke"], lw=LW["thin"], zorder=5))
    c.ax.add_patch(Circle((lx3 + 1.3, gy), 0.36, fc=INK, ec="none", zorder=5))
    T(lx3 + 3.2, gy + 0.05, "miscovered", ha="left", size=FS["body"])
    lx4 = lx3 + 18.6
    d = 0.42
    c.line([(lx4 + 1.3 - d, gy - d), (lx4 + 1.3 + d, gy + d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
    c.line([(lx4 + 1.3 - d, gy + d), (lx4 + 1.3 + d, gy - d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
    T(lx4 + 3.2, gy + 0.05, "frozen", ha="left", size=FS["body"], fit=(lx4, W))

    # ---- 下图：α_{t,h} 在线更新（err 为各小区的未覆盖比例，所以轨迹是连续小步）
    ay0, ay1 = 38.0, 46.4
    c.line([(px0, ay0 - 0.5), (px0, ay1), (px1, ay1)], color=INK, lw=LW["thin"], arrow=False, z=3)
    T(px0 - 1.2, (ay0 + ay1) / 2, r"$\alpha_{t,h}$", ha="right", size=FS["label"])
    err = np.clip(0.10 + 0.07 * np.sin(tau / 7.5 + 0.5) + rng.normal(0, 0.05, n), 0, 1)
    a = np.empty(n)
    a[0] = 0.10
    for k in range(1, n):                                           # α 由所有小区共享：单个小区冻结时照常更新
        a[k] = a[k - 1] + 0.10 * (0.10 - err[k])
    aa = lambda v: ay1 - (ay1 - ay0) * (v - (a.min() - 0.01)) / (a.max() - a.min() + 0.02)   # noqa: E731
    step = []
    for k in range(now + 1):
        step += [(xs(max(k - 0.5, 0)), aa(a[k])), (xs(min(k + 0.5, now)), aa(a[k]))]
    c.line(step, color=O["stroke"], lw=LW["thin"], arrow=False, z=4)
    c.line([(px0, aa(0.10)), (px1, aa(0.10))], color=INK, lw=LW["thin"], ls=(0, (2.0, 1.2)), arrow=False, z=3)
    T(px1, aa(0.10) - 2.3, r"$\alpha=0.1$", ha="right", size=FS["body"], color=INK2)
    xnow = xs(now)                                                  # 当前的 α_{t,h}（红点）：下方的更新回到这里
    c.ax.add_patch(Circle((xnow, aa(a[now])), 0.6, fc=O["stroke"], ec="white", lw=0.3, zorder=6))
    vb = xs(44)
    c.line([(vb, ay0 - 0.5), (vb, ay1 + 1.0)], color=INK2, lw=LW["hair"], ls=(0, (1.0, 1.0)), arrow=False, z=3)
    T(vb - 1.0, ay1 + 2.1, "warm-up", ha="right", size=FS["body"], color=INK2, fit=(px0, vb))
    T(vb + 1.0, ay1 + 2.1, "online", ha="left", size=FS["body"], color=INK2, fit=(vb, W))

    # ---- 底部：一次校准的机制（公式写在图注；与 src/conformal/aci.py 对应）
    #      分数直方图与分位数 q̂ → 区间两侧各加宽 σ_i q̂ → 各小区是否覆盖 → err → 更新 α（回到上方轨迹）
    base_y, lab_y, mid_y = 62.4, 65.8, 57.8
    # ① 归一化分数 s 的直方图（冻结点不产生分数）；右尾 = 未覆盖部分，q̂ 为 1−α 分位数
    heights = [7.4, 6.9, 5.9, 4.8, 3.7, 2.7, 1.8, 1.1]
    bw, bs, bx0 = 1.55, 1.9, 3.6
    for k, hh in enumerate(heights):
        tail = k >= 6
        c.rect(bx0 + k * bs, base_y - hh, bw, hh, fc=O["mid"] if tail else N["strong"],
               ec=O["stroke"] if tail else N["stroke"], lw=LW["hair"], z=3)
    c.line([(bx0 - 0.8, base_y), (bx0 + 8 * bs + 0.4, base_y)], color=INK, lw=LW["thin"], arrow=False, z=4)
    qx = bx0 + 6 * bs - (bs - bw) / 2
    c.line([(qx, base_y), (qx, 53.0)], color=O["stroke"], lw=LW["thin"], ls=(0, (2.0, 1.0)), arrow=False, z=5)
    T(qx + 0.9, 53.6, r"$\hat q_{t,h}$", ha="left", size=FS["body"], color=O["stroke"])
    T(bx0 + 4 * bs, lab_y, r"$s_{\tau,i}$", size=FS["body"])
    T(qx + 0.5, 58.3, r"$\alpha_{t,h}$", ha="left", size=FS["body"], color=O["stroke"])   # 右尾面积 = α（与上方轨迹同一记号）
    # ② 区间加宽：原始分位数区间（蓝条）两端各向外推 σ_i q̂，得到校准区间（红色端帽）
    ix = 29.0
    c.line([(bx0 + 8 * bs + 1.2, mid_y), (ix - 2.4, mid_y)], color=INK, lw=LW["thin"], head=(0.9, 0.55))
    c.rect(ix - 0.75, 55.4, 1.5, 4.8, fc=B["strong"], ec=B["stroke"], lw=LW["hair"], z=3)
    c.line([(ix, 53.4), (ix, 62.2)], color=O["stroke"], lw=LW["thin"], arrow=False, z=4)
    for yy_ in (53.4, 62.2):
        c.line([(ix - 1.0, yy_), (ix + 1.0, yy_)], color=O["stroke"], lw=LW["box"], arrow=False, z=4)
    T(ix + 2.0, mid_y, r"$\pm\,\sigma_i\hat q_{t,h}$", ha="left", size=FS["body"], color=O["stroke"])
    T(ix, lab_y, r"$[\hat y^{\,\mathrm{lo}},\hat y^{\,\mathrm{hi}}]$", size=FS["body"])
    # ③ 各小区是否被覆盖（冻结小区不计入）→ err
    zx = [55.0, 57.6, 60.2, 62.8, 65.4, 68.0]
    c.line([(46.4, mid_y), (zx[0] - 1.6, mid_y)], color=INK, lw=LW["thin"], head=(0.9, 0.55))
    obs_y = [57.0, 58.4, 56.2, 52.6, 58.0, 57.4]
    for k, (zx_, oy_) in enumerate(zip(zx, obs_y)):
        c.line([(zx_, 55.0), (zx_, 60.6)], color=O["stroke"], lw=LW["thin"], arrow=False, z=4)
        for yy_ in (55.0, 60.6):
            c.line([(zx_ - 0.5, yy_), (zx_ + 0.5, yy_)], color=O["stroke"], lw=LW["thin"], arrow=False, z=4)
        if k == 4:                                                  # 冻结小区：灰色 ×，不计入 err
            d = 0.42
            c.line([(zx_ - d, oy_ - d), (zx_ + d, oy_ + d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
            c.line([(zx_ - d, oy_ + d), (zx_ + d, oy_ - d)], color=N["stroke"], lw=LW["thin"], arrow=False, z=5)
            continue
        c.ax.add_patch(Circle((zx_, oy_), 0.36, fc=INK, ec="none", zorder=5))
        if not 55.0 <= oy_ <= 60.6:
            c.ax.add_patch(Circle((zx_, oy_), 0.85, fc="none", ec=O["stroke"], lw=LW["thin"], zorder=5))
    c.line([(zx[0] - 1.2, 62.0), (zx[-1] + 1.4, 62.0)], color=INK2, lw=LW["hair"], head=(0.8, 0.45))   # 小区轴
    T(zx[-1] + 2.0, 62.0, r"$i$", ha="left", size=FS["body"], color=INK2)
    T((zx[0] + zx[-1]) / 2, lab_y, r"$\mathrm{err}_{t,h}$", size=FS["body"])
    # ④ 用 err 更新当前的 α_{t,h}（回到上方轨迹末端的红点）
    c.line([(zx[-1] + 1.4, mid_y), (xnow, mid_y), (xnow, aa(a[now]) + 0.9)], color=INK, lw=LW["thin"],
           head=(0.9, 0.55))
    T(xnow + 1.0, 52.6, r"$\gamma_{\mathrm{ACI}}$", ha="left", size=FS["body"])


# ============================================================ 组装与导出
PANELS = [
    ("a", panel_a, PW_L, "Fig4a_backbone_head", "Price-blind backbone and non-crossing quantile head"),
    ("b", panel_b, PW_R, "Fig4b_price_anchoring", "Anchored price response and cut feedback"),
    ("c", panel_c, PW_L, "Fig4c_spillover_rings", "Distance-ring spillover exposure on real zones"),
    ("d", panel_d, PW_R, "Fig4d_masked_aci", "Adaptive conformal calibration with the quality mask"),
]
GAP_X, GAP_Y, LEAD = 4.0, 4.0, 5.2


def main():
    os.makedirs(OUT, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else "abcd+"
    problems = 0
    # 每个面板单独成图（不带 (a)–(d) 字母）
    for tag, fn, w, name, _ in PANELS:
        if tag not in only:
            continue
        c = Canvas(w, PH)
        fn(c)
        issues = c.check()
        problems += len(issues)
        print(f"{name}: " + ("; ".join(issues) if issues else "检查通过"))
        c.save(os.path.join(OUT, name))
    if "+" not in only:
        return
    # 总图 190 mm 宽；(a)–(d) 字母放在各面板左上角
    root = Canvas(PW_L + GAP_X + PW_R, 2 * PH + GAP_Y)
    pos = {"a": (0.0, 0.0), "b": (PW_L + GAP_X, 0.0), "c": (0.0, PH + GAP_Y), "d": (PW_L + GAP_X, PH + GAP_Y)}
    subs = []
    for tag, fn, w, name, desc in PANELS:
        x, y = pos[tag]
        sub = root.sub(x, y, w, PH)
        fn(sub, lead=LEAD)
        subs.append((tag, sub))
        root.text(x + 0.3, y + 2.75, f"({tag})", size=FS["panel"], weight="bold", ha="left")
    for tag, sub in subs:
        issues = sub.check()
        problems += len(issues)
        if issues:
            print(f"总图 ({tag})：" + "; ".join(issues))
    issues = root.check()
    problems += len(issues)
    print(f"总图 {root.W:.0f} × {root.H:.1f} mm：" + ("; ".join(issues) if issues else "检查通过"))
    root.save(os.path.join(OUT, "Fig4_CPA-STGNN_modules"))
    print("问题总数", problems)


if __name__ == "__main__":
    main()
