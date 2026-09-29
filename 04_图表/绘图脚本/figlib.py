"""论文方法类图的绘图工具（设计语言 v1.1，见 04_图表/README_图稿说明.md）。

- 画布用毫米坐标，原点在左上角，y 向下（与草图、SVG 的习惯一致）；按期刊版心宽度（双栏 190 mm）直接设计，
  字号就是最终印刷字号（正文 7 pt，模块标题 7.5–8 pt）
- 文字 Liberation Sans（与 Arial 等宽度），公式 STIX（与 Times 一致）；PDF 以 TrueType（Type 42）嵌入
- 导出：SVG（文字保留为可编辑文本，字体名改写为 Arial / STIXGeneral 以便在 Mac、Windows 上打开）、PDF（矢量）、PNG（600 dpi）
"""
from __future__ import annotations

import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

MM = 1 / 25.4

# ---------------------------------------------------------------- 设计语言 v1.1（色板经色盲模拟验证，见图稿说明）
INK = "#222222"          # 文字
INK2 = "#555555"         # 次要文字
RULE = "#9A9A9A"         # 分隔线、次要边框
PAL = {
    "demand": dict(stroke="#4072B7", fill="#EEF5FF", mid="#D6E8FF", strong="#B6D3FC"),
    "price":  dict(stroke="#CA710A", fill="#FDF2EA", mid="#FBE1CC", strong="#F2C8A7"),
    "cov":    dict(stroke="#3B9D5C", fill="#EDF8EF", mid="#D4EED9", strong="#B2DEBC"),
    "causal": dict(stroke="#6D2D98", fill="#F7F2FD", mid="#EEE0FB", strong="#DDC6F2"),
    "output": dict(stroke="#B73138", fill="#FFF1EF", mid="#FFDCDA", strong="#F9C1BD"),
    "gray":   dict(stroke="#8C8C8C", fill="#F3F3F3", mid="#E6E6E6", strong="#D2D2D2"),
}
LW = dict(box=0.8, flow=1.0, causal=1.5, thin=0.6, hair=0.4)   # pt
DASH_CAUSAL = (0, (4.0, 2.0))
DOT_GRAD = (0, (1.0, 1.6))
FS = dict(body=7.0, small=7.0, label=7.5, title=8.0, panel=8.5)


def pick_font(cands=("Liberation Sans", "Arial", "Helvetica", "DejaVu Sans")) -> str:
    """Liberation Sans 与 Arial 字宽相同：云端用前者，Mac / Windows 自动退到 Arial，版面不变。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in cands if n in names), "DejaVu Sans")


def setup_rc():
    plt.rcParams.update({
        "font.family": pick_font(),
        "font.size": FS["body"],
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })


class Canvas:
    """毫米坐标画布：c = Canvas(190, 120)；所有 y 从上往下量。"""

    def __init__(self, width_mm: float, height_mm: float, _parent=None, _at=(0.0, 0.0)):
        setup_rc()
        self.W, self.H = width_mm, height_mm
        if _parent is None:
            self.fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
            rect = [0, 0, 1, 1]
        else:                                   # 子画布：同一张图里的一块区域，自己的毫米坐标（面板复用）
            self.fig = _parent.fig
            PW, PH = _parent.W, _parent.H
            x0, y0 = _at
            rect = [x0 / PW, 1 - (y0 + height_mm) / PH, width_mm / PW, height_mm / PH]
        self.ax = self.fig.add_axes(rect)
        self.ax.set_xlim(0, width_mm)
        self.ax.set_ylim(height_mm, 0)          # y 向下
        self.ax.set_axis_off()
        self.ax.set_aspect("equal", adjustable="box")
        self._fits = []

    def sub(self, x, y, w, h) -> "Canvas":
        """在 (x, y) 处开一块 w × h 毫米的子画布（坐标从子画布左上角算起）；面板函数既可画进总图，也可单独成图。"""
        return Canvas(w, h, _parent=self, _at=(x, y))

    # ------------------------------------------------------------ 基本图元
    def box(self, x, y, w, h, fc="white", ec=INK, lw=LW["box"], ls="-", r=1.2, z=2, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                           ls=ls, zorder=z, alpha=alpha, joinstyle="round")
        self.ax.add_patch(p)
        return p

    def rect(self, x, y, w, h, fc="white", ec=INK, lw=LW["box"], ls="-", z=2, hatch=None, alpha=1.0):
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, hatch=hatch, alpha=alpha)
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, size=None, color=INK, ha="center", va="center", weight="normal", style="normal",
             z=6, rotation=0, family=None, linespacing=1.15, fit=None):
        """fit=(x0, x1) 或 (x0, y0, x1, y1)：登记这段文字必须落在的范围，check() 时核对。"""
        kw = dict(fontsize=size or FS["body"], color=color, ha=ha, va=va, fontweight=weight, fontstyle=style,
                  zorder=z, rotation=rotation, linespacing=linespacing)
        if family:
            kw["family"] = family
        t = self.ax.text(x, y, s, **kw)
        if fit is not None:
            self._fits.append((t, fit))
        return t

    def check(self, tol=0.15, ignore_pairs=()):
        """自动检查：文字是否超出登记的范围、文字之间是否重叠（毫米）。返回问题列表。"""
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()

        def bb(t):
            e = t.get_window_extent(r)
            (x0, y0), (x1, y1) = inv.transform((e.x0, e.y0)), inv.transform((e.x1, e.y1))
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        issues = []
        for t in self.ax.texts:                      # 所有文字都必须落在画布之内（单独导出时不被裁掉）
            if t.get_text().strip():
                x0, y0, x1, y1 = bb(t)
                if x0 < -tol or y0 < -tol or x1 > self.W + tol or y1 > self.H + tol:
                    issues.append(f"超出画布 {t.get_text()!r}: x [{x0:.1f}, {x1:.1f}], y [{y0:.1f}, {y1:.1f}]")
        for t, fit in self._fits:
            x0, y0, x1, y1 = bb(t)
            if len(fit) == 2:
                fx0, fx1 = fit
                if x0 < fx0 - tol or x1 > fx1 + tol:
                    issues.append(f"超出范围 {t.get_text()!r}: [{x0:.1f}, {x1:.1f}] ⊄ [{fx0:.1f}, {fx1:.1f}]")
            else:
                fx0, fy0, fx1, fy1 = fit
                if x0 < fx0 - tol or x1 > fx1 + tol or y0 < fy0 - tol or y1 > fy1 + tol:
                    issues.append(f"超出范围 {t.get_text()!r}")
        texts = [t for t in self.ax.texts if t.get_text().strip()]
        boxes = [bb(t) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a, b = boxes[i], boxes[j]
                ov_x = min(a[2], b[2]) - max(a[0], b[0])
                ov_y = min(a[3], b[3]) - max(a[1], b[1])
                if ov_x > tol and ov_y > tol:
                    pair = (texts[i].get_text(), texts[j].get_text())
                    if pair not in ignore_pairs and pair[::-1] not in ignore_pairs:
                        issues.append(f"文字重叠 {pair[0]!r} × {pair[1]!r}")
        return issues

    def line(self, pts, color=INK, lw=LW["flow"], ls="-", z=3, arrow=True, head=(1.25, 0.95), both=False):
        """折线（毫米坐标）；arrow=True 时终点带实心三角箭头。head = (长, 半宽) 毫米。"""
        pts = [tuple(p) for p in pts]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
        if not arrow:
            self.ax.add_patch(matplotlib.patches.PathPatch(Path(pts, codes), fc="none", ec=color, lw=lw, ls=ls,
                                                           zorder=z, capstyle="butt", joinstyle="miter"))
            return
        # 线段缩短到箭头根部，再单独画实心箭头（虚线箭头的头部也保持实心）
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        L = math.hypot(x1 - x0, y1 - y0)
        hl, hw = head
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        base = (x1 - ux * hl * 0.85, y1 - uy * hl * 0.85)
        body = pts[:-1] + [base]
        if both:
            (a0, b0), (a1, b1) = pts[1], pts[0]
            L2 = math.hypot(a1 - a0, b1 - b0)
            vx, vy = (a1 - a0) / L2, (b1 - b0) / L2
            body[0] = (a1 - vx * hl * 0.85, b1 - vy * hl * 0.85)
            self._head(a1, b1, vx, vy, hl, hw, color, z)
        self.ax.add_patch(matplotlib.patches.PathPatch(Path(body, [Path.MOVETO] + [Path.LINETO] * (len(body) - 1)),
                                                       fc="none", ec=color, lw=lw, ls=ls, zorder=z,
                                                       capstyle="butt", joinstyle="miter"))
        self._head(x1, y1, ux, uy, hl, hw, color, z)

    def _head(self, x1, y1, ux, uy, hl, hw, color, z):
        px, py = -uy, ux
        tri = [(x1, y1), (x1 - ux * hl + px * hw, y1 - uy * hl + py * hw), (x1 - ux * hl - px * hw, y1 - uy * hl - py * hw)]
        self.ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, lw=0.3, zorder=z + 0.1, joinstyle="miter"))

    def curve(self, p0, p1, color=INK, lw=LW["flow"], ls="-", rad=0.3, z=3, head=True):
        """弯曲箭头（用于从说明文字指向图元）。"""
        a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>,head_length=3.2,head_width=1.8"
                            if head else "-", color=color, lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0,
                            mutation_scale=1)
        self.ax.add_patch(a)
        return a

    def op(self, x, y, kind="+", r=1.55, color=INK, lw=LW["box"], fc="white", z=5):
        """运算符圆圈：'+' 逐元素相加，'x' 逐元素相乘，'c' 拼接。"""
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=lw, zorder=z))
        k = r * 0.58
        if kind == "+":
            self.line([(x - k, y), (x + k, y)], color, lw, arrow=False, z=z + 0.2)
            self.line([(x, y - k), (x, y + k)], color, lw, arrow=False, z=z + 0.2)
        elif kind == "x":
            k2 = k * 0.78
            self.line([(x - k2, y - k2), (x + k2, y + k2)], color, lw, arrow=False, z=z + 0.2)
            self.line([(x - k2, y + k2), (x + k2, y - k2)], color, lw, arrow=False, z=z + 0.2)
        elif kind == "c":
            self.text(x, y + 0.05, "C", size=6.2, color=color, weight="bold", z=z + 0.2)

    def dots(self, x, y, n=4, dx=2.3, r=0.75, fc="#4072B7", ec=None, gap_at=None, z=4):
        """一排节点圆点（仿 PAG 输入层），gap_at 处画省略号。"""
        xs = []
        k = 0
        cx = x
        while k < n:
            if gap_at is not None and k == gap_at:
                for j in range(3):
                    self.ax.add_patch(Circle((cx - 0.55 + j * 0.55, y), 0.16, fc=INK2, ec="none", zorder=z))
                cx += dx
            self.ax.add_patch(Circle((cx, y), r, fc=fc, ec=ec or fc, lw=0.3, zorder=z))
            xs.append(cx)
            cx += dx
            k += 1
        return xs

    def stack(self, x, y, w, h, fc, ec, n=3, off=0.9, lw=LW["thin"], r=0.8, z=2):
        """叠层片（时间窗口 / 特征张量），返回最前一层的 (x, y, w, h)。"""
        for k in range(n - 1, -1, -1):
            self.box(x + k * off, y - k * off, w, h, fc=fc, ec=ec, lw=lw, r=r, z=z + (n - k) * 0.01)
        return (x, y, w, h)

    def graph_icon(self, x, y, s=6.0, color="#6E6E6E", fc="#FFFFFF", z=4):
        """小图结构图标（节点 + 边），(x, y) 为左上角，边长 s 毫米。"""
        pts = [(0.15, 0.25), (0.55, 0.12), (0.88, 0.35), (0.35, 0.62), (0.75, 0.82), (0.12, 0.88)]
        edges = [(0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4), (3, 5)]
        P = [(x + a * s, y + b * s) for a, b in pts]
        for i, j in edges:
            self.line([P[i], P[j]], color, LW["thin"], arrow=False, z=z)
        for px, py in P:
            self.ax.add_patch(Circle((px, py), s * 0.075, fc=fc, ec=color, lw=LW["thin"], zorder=z + 0.1))

    def price_steps(self, x, y, w, h, color, lw=LW["flow"], z=4, levels=(0.55, 0.55, 0.25, 0.25, 0.8, 0.8, 0.45, 0.45)):
        """分时电价阶梯小图标，(x, y) 左上角。"""
        n = len(levels)
        pts = []
        for k, v in enumerate(levels):
            xa, xb = x + w * k / n, x + w * (k + 1) / n
            yy = y + h * (1 - v)
            pts += [(xa, yy), (xb, yy)]
        self.line(pts, color, lw, arrow=False, z=z)

    def jump_icon(self, x, y, w, h, color, z=4):
        """下一小时价格变化 U：两级阶梯 + 跳变处的双向小箭头，(x, y) 左上角。"""
        xm = x + w * 0.55
        self.line([(x, y + h * 0.78), (xm, y + h * 0.78), (xm, y + h * 0.18), (x + w, y + h * 0.18)], color,
                  LW["flow"], arrow=False, z=z)
        self.line([(xm + 1.3, y + h * 0.70), (xm + 1.3, y + h * 0.27)], color, LW["thin"], z=z, head=(0.8, 0.5),
                  both=True)

    def fan(self, x, y, w, h, color, fill, z=4, seed=3, n_q=3, center=None):
        """分位数扇形小图标：中位数线 + 逐层变浅的区间带，(x, y) 左上角。"""
        import numpy as np
        t = np.linspace(0, 1, 40)
        c = 0.5 + 0.18 * np.sin(2 * np.pi * (t * 1.2 + 0.1)) if center is None else center(t)
        widths = np.linspace(0.08, 0.30, n_q)[::-1]
        for k, wd in enumerate(widths):
            spread = wd * (0.55 + 0.45 * t)
            lo, hi = c - spread, c + spread
            xs = x + t * w
            poly = list(zip(xs, y + h * (1 - hi))) + list(zip(xs[::-1], y + h * (1 - lo[::-1])))
            self.ax.add_patch(Polygon(poly, closed=True, fc=fill, ec="none", alpha=0.45 + 0.25 * k, zorder=z))
        self.line(list(zip(x + t * w, y + h * (1 - c))), color, LW["flow"], arrow=False, z=z + 0.1)

    def scissors(self, x, y, s=3.2, angle=0.0, color=INK, lw=LW["box"], z=7):
        """剪刀图标（截断反馈 ✂），中心 (x, y)，尺寸 s 毫米，angle 为顺时针角度。"""
        import numpy as np
        th = math.radians(angle)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])

        def T(p):
            q = R @ np.array(p) * s
            return (x + q[0], y + q[1])
        # 两片刀刃（细长三角），在中心交叉
        for sgn in (1, -1):
            blade = [T((-0.05, 0.0)), T((0.62, -0.20 * sgn)), T((0.55, -0.10 * sgn))]
            self.ax.add_patch(Polygon(blade, closed=True, fc=color, ec=color, lw=0.3, zorder=z))
            self.line([T((0.0, 0.0)), T((-0.30, 0.20 * sgn))], color, lw, arrow=False, z=z)
            cx, cy = T((-0.42, 0.28 * sgn))
            self.ax.add_patch(Circle((cx, cy), 0.15 * s, fc="white", ec=color, lw=lw, zorder=z))
        self.ax.add_patch(Circle(T((0.0, 0.0)), 0.045 * s, fc="white", ec=color, lw=0.4, zorder=z + 0.1))

    def title_bar(self, x, y, w, h, label, fc="#D9D9D9", color=INK, size=None, weight="bold", r=1.0, z=3):
        self.box(x, y, w, h, fc=fc, ec="none", lw=0, r=r, z=z)
        self.text(x + w / 2, y + h / 2 + 0.05, label, size=size or FS["title"], color=color, weight=weight, z=z + 1)

    def badge(self, x, y, label, color, r=1.45, size=6.4, z=7):
        """模块编号圆标（①②③…）。"""
        self.ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=z))
        self.text(x, y + 0.05, label, size=size, color="white", weight="bold", z=z + 0.1)

    # ------------------------------------------------------------ 导出
    def save(self, base: str, dpi: int = 600):
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        self.fig.savefig(base + ".pdf")
        self.fig.savefig(base + ".png", dpi=dpi)
        self.fig.savefig(base + ".svg")
        fix_svg_fonts(base + ".svg")
        plt.close(self.fig)


def fix_svg_fonts(path: str):
    """SVG 中的字体名改为跨平台可用的字体栈（Liberation Sans 与 Arial 等宽，STIX 在 Mac 上自带）。"""
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"font-family:\s*'?Liberation Sans'?", "font-family: Arial, 'Liberation Sans', Helvetica, sans-serif", s)
    s = re.sub(r"font-family=\"Liberation Sans\"", "font-family=\"Arial, 'Liberation Sans', Helvetica, sans-serif\"", s)
    s = re.sub(r"font-family:\s*'?STIXGeneral'?", "font-family: STIXGeneral, 'STIX Two Text', 'Times New Roman', serif", s)
    s = re.sub(r"font-family:\s*'?STIXNonUnicode'?", "font-family: STIXGeneral, 'STIX Two Math', 'Times New Roman', serif", s)
    open(path, "w", encoding="utf-8").write(s)
