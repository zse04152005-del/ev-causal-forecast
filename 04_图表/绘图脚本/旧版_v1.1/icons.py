"""方法类图用的矢量图标（仿顶刊框架图：图快照叠放、邻接矩阵、日历 / 天气 / 地点图标、特征立方体、损失曲线等）。

全部画在 figlib.Canvas 的毫米坐标上（原点左上角，y 向下），颜色取自 figlib.PAL，线宽取自 figlib.LW。
每个函数的 (x, y) 都是图标外接矩形的左上角，除非另外说明。
图 3 目前没有用到的备用图标：clock、jump（负荷转移 U）、loss_curve、coef_plot（事件研究）、snowflake。
snowflake 不要用来表示冻结的数据点：机器学习框架图里 ❄ 通常指"冻结的权重"，冻结点一律用 frozen_mask。
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Wedge

from figlib import INK, INK2, LW, PAL

B, P, G, C, O, N = (PAL[k] for k in ("demand", "price", "cov", "causal", "output", "gray"))  # noqa: E741


def _poly(c, pts, fc, ec, lw=LW["thin"], z=4, ls="-", alpha=1.0, join="round"):
    p = Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, alpha=alpha, joinstyle=join)
    c.ax.add_patch(p)
    return p


def _circ(c, x, y, r, fc, ec="none", lw=LW["thin"], z=4, alpha=1.0):
    p = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    c.ax.add_patch(p)
    return p


def _rbox(c, x, y, w, h, fc, ec, lw=LW["thin"], r=0.5, z=4, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    c.ax.add_patch(p)
    return p


# ------------------------------------------------------------------ 输入数据
GRAPH_NODES = [(0.14, 0.30), (0.40, 0.16), (0.70, 0.24), (0.88, 0.52), (0.58, 0.55), (0.30, 0.66), (0.72, 0.84),
               (0.14, 0.86)]
GRAPH_EDGES = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 4), (4, 5), (0, 5), (3, 6), (4, 6), (5, 7), (5, 6)]


def graph_frames(c, x, y, w, h, n=3, off=(1.9, -1.9), seed=4, z=4):
    """时空图快照叠放（仿 HSTGCN 的 X_{t-m+1} … X_t）：每层一张城市小区图，节点深浅 = 利用率。(x, y) 为最前一层左上角。"""
    rng = np.random.default_rng(seed)
    shades = [B["fill"], B["mid"], B["strong"], B["stroke"]]
    for k in range(n - 1, -1, -1):
        fx, fy = x + k * off[0], y + k * off[1]
        zz = z + (n - k) * 0.3
        _rbox(c, fx, fy, w, h, fc="white", ec=B["stroke"], lw=LW["thin"], r=0.7, z=zz)
        if k:                                                     # 后面几层只画边框，避免露出被裁切的节点
            continue
        P_ = [(fx + 0.9 + a * (w - 1.8), fy + 0.9 + b * (h - 1.8)) for a, b in GRAPH_NODES]
        for i, j in GRAPH_EDGES:
            c.line([P_[i], P_[j]], "#9FB8DC", LW["hair"], arrow=False, z=zz + 0.05)
        lev = rng.integers(0, 4, len(P_))
        for (px, py), lv in zip(P_, lev):
            _circ(c, px, py, min(w, h) * 0.075, fc=shades[lv], ec=B["stroke"], lw=0.35, z=zz + 0.1)


def matrix_stack(c, x, y, s, n=3, off=1.1, z=4, pal=None):
    """邻接 / 相似度矩阵叠放（仿 ASTGRN 的网格矩阵）：n 张 5×5 网格。(x, y) 为最前一张左上角，边长 s。"""
    pal = pal or N
    pattern = np.array([[3, 2, 0, 1, 0], [2, 3, 2, 0, 0], [0, 2, 3, 1, 2], [1, 0, 1, 3, 2], [0, 0, 2, 2, 3]])
    shades = ["white", pal["fill"], pal["mid"], pal["strong"]]
    m = pattern.shape[0]
    cs = s / m
    for k in range(n - 1, -1, -1):
        fx, fy = x + k * off, y - k * off
        zz = z + (n - k) * 0.3
        c.rect(fx, fy, s, s, fc="white", ec=pal["stroke"], lw=LW["thin"], z=zz)
        if k == 0:
            for i in range(m):
                for j in range(m):
                    c.rect(fx + j * cs, fy + i * cs, cs, cs, fc=shades[pattern[i, j]], ec=pal["stroke"], lw=0.3,
                           z=zz + 0.05)


def calendar(c, x, y, s, pal=None, z=4):
    """日历图标（仿 STAEformer）：表头 + 两个装订环 + 日期格。(x, y) 左上角，宽 s，高 0.92 s。"""
    pal = pal or G
    h = 0.92 * s
    _rbox(c, x, y + 0.08 * s, s, h - 0.08 * s, fc="white", ec=pal["stroke"], lw=LW["box"], r=0.12 * s, z=z)
    c.rect(x + 0.03 * s, y + 0.12 * s, s - 0.06 * s, 0.2 * s, fc=pal["stroke"], ec="none", z=z + 0.1)
    for fx in (0.3, 0.7):
        _rbox(c, x + fx * s - 0.05 * s, y, 0.1 * s, 0.2 * s, fc="white", ec=pal["stroke"], lw=LW["thin"],
              r=0.04 * s, z=z + 0.2)
    gx, gy, gw, gh = x + 0.14 * s, y + 0.42 * s, 0.72 * s, 0.4 * s
    for i in range(3):
        for j in range(4):
            hot = (i, j) == (1, 2)
            c.rect(gx + j * gw / 4 + 0.03 * s, gy + i * gh / 3 + 0.02 * s, gw / 4 - 0.06 * s, gh / 3 - 0.04 * s,
                   fc=pal["stroke"] if hot else pal["mid"], ec="none", z=z + 0.1)


def clock(c, cx, cy, r, pal=None, z=4):
    """时钟图标：小时特征。(cx, cy) 为圆心。"""
    pal = pal or G
    _circ(c, cx, cy, r, fc="white", ec=pal["stroke"], lw=LW["box"], z=z)
    for a in range(0, 360, 90):
        t = math.radians(a)
        c.line([(cx + 0.72 * r * math.cos(t), cy + 0.72 * r * math.sin(t)),
                (cx + 0.9 * r * math.cos(t), cy + 0.9 * r * math.sin(t))], pal["stroke"], LW["thin"], arrow=False,
               z=z + 0.1)
    c.line([(cx, cy), (cx, cy - 0.55 * r)], pal["stroke"], LW["box"], arrow=False, z=z + 0.2)
    c.line([(cx, cy), (cx + 0.42 * r, cy + 0.2 * r)], pal["stroke"], LW["box"], arrow=False, z=z + 0.2)
    _circ(c, cx, cy, 0.09 * r, fc=pal["stroke"], z=z + 0.3)


def weather(c, x, y, s, pal=None, z=4):
    """天气图标：太阳 + 云（云的外轮廓用"大一圈的描边色圆 + 填充色圆"叠出，内部不露圆弧）。(x, y) 左上角，边长 s。"""
    pal = pal or G
    sx, sy, sr = x + 0.36 * s, y + 0.34 * s, 0.17 * s
    for a in range(0, 360, 45):
        t = math.radians(a)
        c.line([(sx + 1.35 * sr * math.cos(t), sy + 1.35 * sr * math.sin(t)),
                (sx + 1.85 * sr * math.cos(t), sy + 1.85 * sr * math.sin(t))], pal["stroke"], LW["thin"], arrow=False,
               z=z)
    _circ(c, sx, sy, sr, fc=pal["mid"], ec=pal["stroke"], lw=LW["thin"], z=z + 0.1)
    parts = [(0.46, 0.68, 0.17), (0.64, 0.56, 0.21), (0.82, 0.70, 0.15)]
    base = (0.36, 0.70, 0.60, 0.16)                               # x, y, w, h（相对 s）
    lw_mm = 0.28
    for fc, grow, zz in ((pal["stroke"], lw_mm, z + 0.2), ("white", 0.0, z + 0.3)):
        for (px, py, pr) in parts:
            _circ(c, x + px * s, y + py * s, pr * s + grow, fc=fc, z=zz)
        _rbox(c, x + base[0] * s - grow, y + base[1] * s - grow, base[2] * s + 2 * grow, base[3] * s + 2 * grow,
              fc=fc, ec="none", r=0.08 * s, z=zz)


def pin(c, cx, cy, r, pal=None, z=4):
    """地点图钉（POI / 静态特征）：圆头 + 下尖，(cx, cy) 为圆头中心，尖端在 cy + 2.3 r。"""
    pal = pal or G
    tip = (cx, cy + 2.3 * r)
    d = tip[1] - cy
    phi = math.acos(r / d)                                      # 切点相对"指向尖端方向"的角度
    base_ang = math.pi / 2                                      # 尖端在圆心正下方（y 向下）
    angs = np.linspace(base_ang + phi, base_ang - phi + 2 * math.pi, 60)
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angs] + [tip]
    _poly(c, pts, fc=pal["mid"], ec=pal["stroke"], lw=LW["box"], z=z)
    _circ(c, cx, cy, 0.42 * r, fc="white", ec=pal["stroke"], lw=LW["thin"], z=z + 0.1)


def price_curve(c, x, y, w, h, pal=None, scenario=True, z=4):
    """分时电价曲线：粗实线 = 实际时刻表 p，灰色点线 = 训练期参考价 ℓ̄，同色细虚线 = 反事实情景 p′。(x, y) 左上角。"""
    pal = pal or P
    lv = [0.25, 0.25, 0.55, 0.9, 0.9, 0.55, 0.9, 0.55, 0.25]
    n = len(lv)
    pts = []
    for k, v in enumerate(lv):
        pts += [(x + w * k / n, y + h * (1 - v)), (x + w * (k + 1) / n, y + h * (1 - v))]
    c.line([(x, y + h), (x + w, y + h)], "#BDBDBD", LW["hair"], arrow=False, z=z)
    mean = float(np.mean(lv))
    c.line([(x, y + h * (1 - mean)), (x + w, y + h * (1 - mean))], INK2, LW["hair"], ls=(0, (1.2, 1.0)), arrow=False,
           z=z + 0.05)
    if scenario:
        lv2 = [0.2, 0.2, 0.55, 1.0, 1.0, 0.55, 1.0, 0.55, 0.2]
        pts2 = []
        for k, v in enumerate(lv2):
            pts2 += [(x + w * k / n, y + h * (1 - v)), (x + w * (k + 1) / n, y + h * (1 - v))]
        c.line(pts2, pal["stroke"], LW["thin"], ls=(0, (1.6, 1.0)), arrow=False, z=z + 0.1)
    c.line(pts, pal["stroke"], LW["flow"], arrow=False, z=z + 0.2)


def ring_map(c, cx, cy, r, z=4):
    """距离环带：三个同心环（由内到外颜色变浅），中心小区（蓝）+ 各环中的邻区（橙实心 = 分时，空心 = 固定电价）。"""
    fills = [P["fill"], P["mid"], P["strong"]]
    for k, fr in enumerate((1.0, 2 / 3, 1 / 3)):
        _circ(c, cx, cy, r * fr, fc=fills[k], ec=P["stroke"], lw=LW["hair"], z=z + 0.1 * k)
    pts = [(0.55, 30, 1), (0.52, 200, 0), (0.83, 100, 1), (0.8, 250, 1), (0.86, 330, 0), (0.24, 300, 1)]
    for rr, ang, tou in pts:
        t = math.radians(ang)
        px, py = cx + rr * r * math.cos(t), cy - rr * r * math.sin(t)
        _circ(c, px, py, 0.085 * r, fc=P["stroke"] if tou else "white", ec=P["stroke"], lw=0.35, z=z + 0.5)
    _circ(c, cx, cy, 0.11 * r, fc=B["stroke"], ec="white", lw=0.3, z=z + 0.6)


def jump(c, x, y, w, h, pal=None, z=4, ls="-"):
    """下一小时价格变化 U：两级阶梯 + 双向小箭头。(x, y) 左上角。"""
    pal = pal or P
    xm = x + w * 0.55
    c.line([(x, y + h * 0.8), (xm, y + h * 0.8), (xm, y + h * 0.2), (x + w, y + h * 0.2)], pal["stroke"], LW["flow"],
           ls=ls, arrow=False, z=z)
    c.line([(xm + 0.22 * w, y + h * 0.72), (xm + 0.22 * w, y + h * 0.28)], pal["stroke"], LW["thin"], z=z,
           head=(0.7, 0.45), both=True)


# ------------------------------------------------------------------ 网络部件
def trapezoid(c, x0, x1, top0, bot0, top1, bot1, fc, ec, lw=LW["box"], z=3):
    """编码器梯形（左宽右窄，仿 CV 框架图中的 Backbone）。左边在 x0 从 top0 到 bot0，右边在 x1 从 top1 到 bot1。"""
    return _poly(c, [(x0, top0), (x1, top1), (x1, bot1), (x0, bot0)], fc=fc, ec=ec, lw=lw, z=z)


def cuboid(c, x, y, w, h, d, pal=None, z=4, lw=LW["thin"]):
    """立方体（特征张量）：正面 (x, y, w, h)，向右上方有厚度 d。"""
    pal = pal or B
    top = [(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)]
    side = [(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)]
    _poly(c, top, fc=pal["fill"], ec=pal["stroke"], lw=lw, z=z)
    _poly(c, side, fc=pal["strong"], ec=pal["stroke"], lw=lw, z=z)
    c.rect(x, y, w, h, fc=pal["mid"], ec=pal["stroke"], lw=lw, z=z + 0.05)


def plates(c, x, y, w, h, n=3, gap=1.2, d=2.2, pal=None, z=4):
    """一叠扁平特征板（仿 DyFCLT 的特征图 F^l）：n 块扁立方体，自上而下叠放。(x, y) 为最上一块正面左上角。"""
    ph = (h - (n - 1) * gap) / n
    for k in range(n):
        cuboid(c, x, y + k * (ph + gap), w, ph, d, pal=pal, z=z + 0.1 * k)


# ------------------------------------------------------------------ 输出、损失、校准
def forecast_panel(c, x, y, w, h, z=3):
    """预测面板（仿 STAEformer 的输出帧）：历史曲线 | 虚线 "现在" | 未来中位数 + 校准后的区间带。(x, y) 左上角。"""
    t = np.linspace(0, 1, 120)
    sig = 0.5 + 0.22 * np.sin(2 * np.pi * (1.6 * t - 0.15)) + 0.04 * np.sin(2 * np.pi * 7 * t)
    now = 0.58
    X = x + t * w
    Y = y + h * (1 - sig)
    hist = t <= now
    fut = t >= now
    width = 0.07 + 0.16 * (t - now) / (1 - now)
    band = list(zip(X[fut], y + h * (1 - (sig[fut] + width[fut])))) + \
        list(zip(X[fut][::-1], y + h * (1 - (sig[fut] - width[fut])[::-1])))
    _poly(c, band, fc=O["mid"], ec="none", z=z)
    c.line(list(zip(X[hist], Y[hist])), INK2, LW["flow"], arrow=False, z=z + 0.1)
    c.line(list(zip(X[fut], Y[fut])), O["stroke"], LW["flow"], arrow=False, z=z + 0.2)
    c.line([(x + now * w, y), (x + now * w, y + h)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z + 0.1)


def loss_curve(c, x, y, w, h, pal=None, z=4):
    """损失曲线小图（仿 PIAST 的 Prediction Loss）：坐标轴 + 下降曲线。"""
    pal = pal or O
    c.line([(x, y), (x, y + h), (x + w, y + h)], INK, LW["thin"], arrow=False, z=z)
    t = np.linspace(0, 1, 50)
    v = 0.12 + 0.78 * np.exp(-4.0 * t)
    c.line(list(zip(x + 0.08 * w + t * 0.88 * w, y + h * (1 - v))), pal["stroke"], LW["flow"], arrow=False, z=z + 0.1)


def alpha_track(c, x, y, w, h, pal=None, z=4):
    """ACI 的 α 轨迹小图：阶梯线围绕虚线目标值波动。"""
    pal = pal or O
    rng = np.random.default_rng(3)
    a = np.cumsum(rng.normal(0, 0.1, 24))
    a = 0.5 + 0.34 * (a - a.mean()) / (np.abs(a - a.mean()).max() + 1e-9)
    pts = []
    for k, v in enumerate(a):
        pts += [(x + w * k / len(a), y + h * (1 - v)), (x + w * (k + 1) / len(a), y + h * (1 - v))]
    c.line([(x, y + h * 0.5), (x + w, y + h * 0.5)], INK2, LW["hair"], ls=(0, (1.2, 1.0)), arrow=False, z=z)
    c.line(pts, pal["stroke"], LW["thin"], arrow=False, z=z + 0.1)
    c.line([(x, y), (x, y + h), (x + w, y + h)], INK, LW["thin"], arrow=False, z=z)


def snowflake(c, cx, cy, r, color=None, crossed=True, z=6):
    """雪花（备用）。不要用来表示冻结的数据点：框架图中 ❄ 一般读作"冻结的权重"，冻结点用 frozen_mask。"""
    col = color or "#6E8FB5"
    for a in (90, 30, 150):
        t = math.radians(a)
        dx, dy = r * math.cos(t), r * math.sin(t)
        c.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], col, LW["thin"], arrow=False, z=z)
        for sgn in (1, -1):
            ex, ey = cx + sgn * 0.62 * dx, cy + sgn * 0.62 * dy
            for bb in (35, -35):
                tb = t + math.radians(bb) + (0 if sgn > 0 else math.pi)
                c.line([(ex, ey), (ex + 0.3 * r * math.cos(tb), ey + 0.3 * r * math.sin(tb))], col, LW["hair"],
                       arrow=False, z=z)
    if crossed:
        c.line([(cx - 1.05 * r, cy + 1.05 * r), (cx + 1.05 * r, cy - 1.05 * r)], O["stroke"], LW["box"], arrow=False,
               z=z + 0.1)


# ------------------------------------------------------------------ 因果识别
def switch_rd(c, x, y, w, h, z=4):
    """切换事件（断点）小图：上方橙色价格阶梯在切换时刻上跳，下方蓝色需求点在切换后下移，紫色为两侧拟合线。"""
    xm = x + 0.5 * w
    c.line([(x, y + h), (x + w, y + h)], INK, LW["thin"], arrow=False, z=z)
    c.line([(xm, y), (xm, y + h)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z)
    c.line([(x + 0.05 * w, y + 0.26 * h), (xm, y + 0.26 * h), (xm, y + 0.08 * h), (x + 0.95 * w, y + 0.08 * h)],
           P["stroke"], LW["flow"], arrow=False, z=z + 0.1)
    rng = np.random.default_rng(11)
    for side, lvl in ((-1, 0.56), (1, 0.76)):
        xs = np.linspace(0.08, 0.42, 6) if side < 0 else np.linspace(0.58, 0.92, 6)
        for u in xs:
            _circ(c, x + u * w, y + h * (lvl + rng.normal(0, 0.045)), 0.045 * h, fc=B["stroke"], z=z + 0.1)
        x0, x1 = (x + 0.06 * w, xm - 0.03 * w) if side < 0 else (xm + 0.03 * w, x + 0.94 * w)
        c.line([(x0, y + lvl * h), (x1, y + lvl * h)], C["stroke"], LW["box"], arrow=False, z=z + 0.2)


def coef_plot(c, x, y, w, h, z=4):
    """事件研究系数图：切换前系数约为 0，切换后为负，竖线为置信区间（紫色）。"""
    y0 = y + 0.36 * h
    c.line([(x, y0), (x + w, y0)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z)
    c.line([(x + 0.43 * w, y), (x + 0.43 * w, y + h)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z)
    vals = [0.02, -0.04, 0.03, 0.42, 0.55, 0.5, 0.58]
    ci = [0.12, 0.12, 0.1, 0.16, 0.2, 0.22, 0.26]
    for k, (v, e) in enumerate(zip(vals, ci)):
        px = x + (k + 0.5) * w / len(vals)
        py = y0 + v * h * 0.95
        c.line([(px, py - e * h), (px, py + e * h)], C["stroke"], LW["thin"], arrow=False, z=z + 0.1)
        _circ(c, px, py, 0.055 * h, fc=C["stroke"], z=z + 0.2)
    c.line([(x, y + h), (x + w, y + h)], INK, LW["thin"], arrow=False, z=z)


def pool_tree(c, x, y, w, h, z=4):
    """合并层级：底层格子（功能区 × 情境）→ 情境层 → 全市层。"""
    top = (x + 0.5 * w, y + 0.12 * h)
    mids = [(x + 0.27 * w, y + 0.5 * h), (x + 0.73 * w, y + 0.5 * h)]
    leaves = [(x + (0.12 + 0.253 * k) * w, y + 0.88 * h) for k in range(4)]
    for m in mids:
        c.line([top, m], C["stroke"], LW["thin"], arrow=False, z=z)
    for k, lf in enumerate(leaves):
        c.line([mids[k // 2], lf], C["stroke"], LW["thin"], arrow=False, z=z)
    r = 0.085 * h
    _circ(c, *top, r * 1.25, fc="white", ec=C["stroke"], lw=LW["box"], z=z + 0.1)          # ○ 全市
    for m in mids:                                                                          # ◐ 情境
        _circ(c, *m, r * 1.15, fc="white", ec=C["stroke"], lw=LW["box"], z=z + 0.1)
        c.ax.add_patch(Wedge(m, r * 1.15, 90, 270, fc=C["stroke"], ec="none", zorder=z + 0.2))
    for lf in leaves:                                                                       # ● 格子
        _circ(c, *lf, r * 1.05, fc=C["stroke"], ec=C["stroke"], z=z + 0.1)


def lock(c, cx, cy, s, pal=None, z=7):
    """挂锁：锚定值固定（缓冲区，不可训练）。(cx, cy) 为锁体中心，s 为锁体宽。"""
    pal = pal or C
    c.ax.add_patch(Arc((cx, cy - 0.42 * s), 0.62 * s, 0.8 * s, theta1=180, theta2=360, color=pal["stroke"],
                       lw=LW["box"] + 0.4, zorder=z))                   # y 轴向下：180–360° 是屏幕上方的半圆
    for sgn in (-1, 1):
        c.line([(cx + sgn * 0.31 * s, cy - 0.42 * s), (cx + sgn * 0.31 * s, cy - 0.2 * s)], pal["stroke"],
               LW["box"] + 0.4, arrow=False, z=z)
    _rbox(c, cx - 0.5 * s, cy - 0.28 * s, s, 0.72 * s, fc=pal["stroke"], ec=pal["stroke"], r=0.1 * s, z=z + 0.1)
    _circ(c, cx, cy + 0.02 * s, 0.1 * s, fc="white", z=z + 0.2)
    c.rect(cx - 0.035 * s, cy + 0.02 * s, 0.07 * s, 0.2 * s, fc="white", ec="none", z=z + 0.2)


def anchor_table(c, x, y, w, h, rows=3, cols=4, z=4):
    """锚定表（功能区 × 情境）：部分格子为自身估计（深），其余并入上一层级（浅）。"""
    own = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (0, 1)}
    cw, rh = w / cols, h / rows
    for i in range(rows):
        for j in range(cols):
            c.rect(x + j * cw, y + i * rh, cw, rh, fc=C["strong"] if (i, j) in own else C["fill"], ec=C["stroke"],
                   lw=0.35, z=z)
    c.rect(x, y, w, h, fc="none", ec=C["stroke"], lw=LW["thin"], z=z + 0.1)


def slope_plot(c, x, y, w, h, z=4):
    """固定效应回归小图：去除固定效应后的散点（横轴价格跳变，纵轴需求变化）与负斜率拟合线，斜率即 β̂。"""
    c.line([(x, y + 0.5 * h), (x + w, y + 0.5 * h)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z)
    c.line([(x + 0.5 * w, y), (x + 0.5 * w, y + h)], INK2, LW["hair"], ls=(0, (1.5, 1.0)), arrow=False, z=z)
    rng = np.random.default_rng(5)
    u = rng.uniform(-0.42, 0.42, 16)
    v = -0.62 * u + rng.normal(0, 0.09, 16)
    for a, b in zip(u, v):
        _circ(c, x + (0.5 + a) * w, y + (0.5 - b) * h, 0.045 * h, fc=C["mid"], ec=C["stroke"], lw=0.3, z=z + 0.1)
    c.line([(x + 0.06 * w, y + (0.5 - 0.62 * 0.44) * h), (x + 0.94 * w, y + (0.5 + 0.62 * 0.44) * h)], C["stroke"],
           LW["flow"], arrow=False, z=z + 0.2)


def frozen_mask(c, cx, cy, w, z=6):
    """冻结（插补）数据点被掩码：一段数值不变的灰色 × 点 + 红色斜杠（与图 4(d) 的冻结点记号一致）。"""
    n = 4
    d = 0.11 * w
    for k in range(n):
        px = cx - 0.36 * w + k * 0.24 * w
        c.line([(px - d, cy - d), (px + d, cy + d)], N["stroke"], LW["thin"], arrow=False, z=z)
        c.line([(px - d, cy + d), (px + d, cy - d)], N["stroke"], LW["thin"], arrow=False, z=z)
    c.line([(cx - 0.5 * w, cy + 0.36 * w), (cx + 0.5 * w, cy - 0.36 * w)], O["stroke"], LW["box"], arrow=False,
           z=z + 0.1)


def pinball_icon(c, x, y, w, h, pal=None, z=4):
    """pinball（分位数）损失：以 0 为顶点、两侧斜率不同的折线 ρ_q(u)。"""
    pal = pal or O
    x0, y0 = x + 0.55 * w, y + 0.88 * h
    c.line([(x, y0), (x + w, y0)], INK, LW["thin"], arrow=False, z=z)
    c.line([(x0, y), (x0, y + h)], INK, LW["hair"], arrow=False, z=z)
    c.line([(x + 0.04 * w, y + 0.08 * h), (x0, y0), (x + 0.98 * w, y + 0.5 * h)], pal["stroke"], LW["flow"],
           arrow=False, z=z + 0.1)
