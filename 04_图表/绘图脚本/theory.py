"""模型框架图的"理论图形"图元（仿 PAG、PIAST、iTransformer、Graph WaveNet 等范例的画法）。

- 节点令牌卡片：叠放的浅色卡片上一排节点圆点 x_1 … x_N，右下角斜箭头表示时间维（PAG / PIAST 的输入层画法）
- 网络层：浅灰蓝圆角矩形 + 深灰细边（PAG、PIAST 的层）
- 网格图结构图标（PAG 的 Graph）
- 行标签与竖向双箭头（PAG 左侧的 Raw data / Features / Networks）
- 层间的尖角 ^（PAG 在数据行与网络模块之间的连接）
坐标与 figlib.Canvas 相同：毫米，原点左上角，y 向下。
"""
from __future__ import annotations

from matplotlib.patches import Circle, Ellipse

from figlib import FS, INK, INK2, LW

LAYER_FC = "#DDE3EC"          # 网络层填充（浅灰蓝）
LAYER_EC = "#4A4A4A"          # 网络层边框
DASH_ROW = (0, (3.0, 2.0))    # 数据行虚线框


def layer(c, x, y, w, h, label, fc=LAYER_FC, ec=LAYER_EC, size=None, weight="normal", r=0.9, lw=LW["thin"],
          color=INK, z=4):
    """一个网络层（圆角矩形 + 居中文字）；返回中心。"""
    c.box(x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, z=z)
    c.text(x + w / 2, y + h / 2 + 0.05, label, size=size or FS["body"], weight=weight, color=color, z=z + 1,
           fit=(x, y, x + w, y + h))
    return x + w / 2, y + h / 2


def ellipse_op(c, cx, cy, w, h, label, ec=LAYER_EC, z=5):
    """激活函数椭圆（tanh、σ）。"""
    c.ax.add_patch(Ellipse((cx, cy), w, h, fc="white", ec=ec, lw=LW["thin"], zorder=z))
    c.text(cx, cy + 0.05, label, size=FS["body"], z=z + 1, fit=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def token_card(c, x, y, w, h, colors, labels=None, card_fc="#E6EEF7", card_ec="#9DB3CF", n_back=2, off=1.0,
               r_node=1.15, axis=None, axis_color=INK, node_y=None, z=2):
    """叠放的节点卡片。colors：每个位置一个颜色（或两个颜色的元组 = 拼接后的节点），None = 省略号。
    labels：每个位置下方的记号（mathtext），None 表示不写。axis：右下角斜箭头旁的记号（如 't'）。返回各节点中心。"""
    for k in range(n_back, 0, -1):
        c.box(x + k * off, y - k * off, w, h, fc=card_fc, ec=card_ec, lw=LW["hair"], r=1.0, z=z + 0.01 * (n_back - k))
    c.box(x, y, w, h, fc=card_fc, ec=card_ec, lw=LW["hair"], r=1.0, z=z + 0.1)
    n = len(colors)
    slot = w / n
    ny = node_y if node_y is not None else (y + h * 0.40 if labels else y + h / 2)
    centers = []
    for i, col in enumerate(colors):
        cx = x + slot * (i + 0.5)
        if col is None:
            for j in (-1, 0, 1):
                c.ax.add_patch(Circle((cx + j * 0.75, ny), 0.2, fc=INK2, ec="none", zorder=z + 0.3))
            centers.append(None)
            continue
        if isinstance(col, tuple):                                   # 拼接后的节点：两个半重叠的圆
            d = r_node * 0.62
            c.ax.add_patch(Circle((cx - d, ny), r_node * 0.9, fc=col[0], ec="white", lw=0.3, zorder=z + 0.3))
            c.ax.add_patch(Circle((cx + d, ny), r_node * 0.9, fc=col[1], ec="white", lw=0.3, zorder=z + 0.35))
        else:
            c.ax.add_patch(Circle((cx, ny), r_node, fc=col, ec="white", lw=0.3, zorder=z + 0.3))
        centers.append((cx, ny))
        if labels and labels[i]:
            c.text(cx, y + h * 0.80, labels[i], size=FS["body"], z=z + 0.4)
    if axis:
        ax0, ay0 = x + w + 0.2, y + h - 0.6
        c.line([(ax0, ay0), (ax0 + 2.6, ay0 - 2.6)], axis_color, LW["thin"], head=(0.9, 0.5), z=z + 0.4)
        c.text(ax0 + 3.3, ay0 - 0.6, axis, size=FS["body"], color=axis_color, ha="left", z=z + 0.4)
    return centers


def grid_graph(c, x, y, s, color=INK, z=4):
    """网格状图结构（仿 PAG 的 Graph 图标）：3×3 网格、节点在交点上，另有两条斜边与一条外伸边。(x, y) 左上角。"""
    g = [0.12, 0.47, 0.82]
    P = {(i, j): (x + g[i] * s, y + g[j] * s) for i in range(3) for j in range(3)}
    for j in range(3):
        c.line([P[(0, j)], P[(2, j)]], color, LW["thin"], arrow=False, z=z)
        c.line([P[(j, 0)], P[(j, 2)]], color, LW["thin"], arrow=False, z=z)
    c.line([P[(1, 1)], P[(2, 2)]], color, LW["thin"], arrow=False, z=z)
    c.line([P[(0, 1)], P[(1, 0)]], color, LW["thin"], arrow=False, z=z)
    c.line([P[(2, 0)], (x + 0.98 * s, y - 0.02 * s)], color, LW["thin"], arrow=False, z=z)
    c.line([(x + 0.0 * s, y + 0.62 * s), P[(0, 2)]], color, LW["thin"], arrow=False, z=z)
    for p in P.values():
        c.ax.add_patch(Circle(p, 0.055 * s, fc="white", ec=color, lw=LW["thin"], zorder=z + 0.1))


def caret(c, x, y, w=5.0, h=2.0, color=INK, lw=1.6, z=5):
    """层间的尖角 ^（顶点在 (x, y)，开口向下），表示数据向上进入上一层。"""
    c.line([(x - w / 2, y + h), (x, y), (x + w / 2, y + h)], color, lw, arrow=False, z=z)


def row_label(c, x, y0, y1, label, xa=None):
    """左侧的行标签 + 竖向双箭头（仿 PAG）。"""
    xa = xa if xa is not None else x + 7.3
    c.line([(xa, y0), (xa, y1)], INK, LW["thin"], head=(1.1, 0.6), both=True)
    c.text(x, (y0 + y1) / 2, label, size=FS["body"], ha="center", linespacing=1.25)
