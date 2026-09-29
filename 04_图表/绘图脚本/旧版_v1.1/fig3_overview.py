"""论文图 3：CPA-STGNN 总框架（双栏 190 mm × 108 mm；v4：以图标代替文字，仿顶刊框架图的画法，不用模块编号）。

与代码的对应（逐条核对见 04_图表/README_图稿说明.md）：
  上层（蓝）  价格盲时空骨干 Graph WaveNet → 节点表示 Z → 非交叉分位数头 → 基线 b^(q)
              src/models/backbones/graph_wavenet.py、src/models/heads/quantile.py
  中层（橙）  价格项：Δℓ·β̂_{g,κ}、S^(k)·δ̂_k → ⊕ → η → exp → ⊗ b^(q) → min{1,·}（可选的 γU 默认关闭，图中不画）
              src/models/heads/price.py、src/models/cpastgnn.py
  下层（紫）  因果识别：切换事件 → 双向固定效应（设计 D）→ 合并层级 → 锚定值（固定缓冲区，梯度截断）
              src/causal/switch_did.py、src/causal/anchors.py、scripts/estimate_anchors.py
  右列（红）  掩码 pinball 损失（只训练骨干与分位数头）；带掩码的自适应保形校准 → 90% 区间
              src/models/losses.py、src/conformal/aci.py
用法：python fig3_overview.py [输出目录]（默认：04_图表/模型总框架图/，找不到时为脚本旁边的 out/）
"""
import os
import sys

from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icons as I  # noqa: E402
from figlib import DOT_GRAD, FS, INK, INK2, LW, PAL, Canvas  # noqa: E402

B, P, G, C, O, K = (PAL[k] for k in ("demand", "price", "cov", "causal", "output", "gray"))  # noqa: E741
W, H = 190.0, 108.0
BUS = (0, (2.6, 1.3))                     # 固定锚定值：紫色粗虚线
BUS_LW = LW["causal"] + 0.3
TAIL = 35.5                               # 所有输入箭头的起点


def build(out_dir: str, check: bool = True) -> str:
    c = Canvas(W, H)
    T = c.text
    L = c.line

    def dot(x, y, col):
        c.ax.add_patch(Circle((x, y), 0.5, fc=col, ec="none", zorder=5))

    # ================================================================ 上层：价格盲需求通路（蓝）
    I.graph_frames(c, 4.0, 8.2, 15.0, 11.0)                      # 小区图快照：利用率 y 与陈旧标记 m̃
    T(26.5, 13.7, r"$y,\ \tilde m$", ha="left", size=FS["label"], color=B["stroke"])
    I.matrix_stack(c, 6.0, 23.0, 7.6)                            # 图结构 A_s
    T(26.5, 26.8, r"$A_s$", ha="left", size=FS["label"], color=K["stroke"])
    I.calendar(c, 3.6, 32.4, 7.0)                                # 日历、天气、POI 与静态特征
    I.weather(c, 11.8, 31.4, 8.4)
    I.pin(c, 23.0, 34.0, 1.8)
    T(26.5, 36.2, r"$\mathbf{c},\mathbf{w},\mathbf{s}$", ha="left", size=FS["label"], color=G["stroke"])

    tx0, tx1 = 44.0, 84.0                                        # 编码器梯形
    I.trapezoid(c, tx0, tx1, 7.0, 37.0, 13.0, 31.0, fc=B["mid"], ec=B["stroke"])
    T(63.5, 16.6, "Price-blind", weight="bold", size=FS["label"], fit=(tx0, tx1))
    T(63.5, 20.5, "ST backbone", weight="bold", size=FS["label"], fit=(tx0, tx1))
    T(63.5, 25.4, r"Graph WaveNet $\times 8$", size=FS["body"], color=INK2, fit=(tx0, tx1))
    L([(TAIL, 13.7), (tx0, 13.7)], B["stroke"])
    L([(TAIL, 26.8), (tx0, 26.8)], K["stroke"])
    L([(TAIL, 36.2), (tx0, 36.2)], G["stroke"])

    lab_y = 11.2                                                 # Z 与 b^(q) 标注同一基线
    I.plates(c, 88.0, 14.6, 8.0, 15.0, n=3, gap=1.2, d=2.0)       # 节点表示 Z
    T(93.0, lab_y, r"$Z$", size=FS["label"], color=B["stroke"])
    L([(tx1, 22.0), (87.8, 22.0)], B["stroke"])

    hx0, hx1, hy0, hy1 = 103.0, 109.8, 11.5, 32.5                # 分位数头（竖排文字的细长块）
    hxc = (hx0 + hx1) / 2
    c.box(hx0, hy0, hx1 - hx0, hy1 - hy0, fc=B["fill"], ec=B["stroke"], lw=LW["box"], r=1.0, z=3)
    T(hxc, (hy0 + hy1) / 2, "Quantile head", rotation=90, size=FS["label"], weight="bold", fit=(hx0, hy0, hx1, hy1))
    L([(98.2, 22.0), (hx0, 22.0)], B["stroke"])
    dot(40.2, 36.2, G["stroke"])                                 # 未来日历与（观测）天气 → 分位数头
    L([(40.2, 36.2), (40.2, 41.0), (hxc, 41.0), (hxc, hy1)], G["stroke"])
    T(95.0, 38.9, r"$\mathbf{c}_{t+h},\mathbf{w}_{t+h}$", size=FS["label"], color=G["stroke"])

    c.fan(113.2, 16.8, 12.6, 10.4, B["stroke"], B["strong"])     # 基线分位数 b^(q)
    T(119.5, lab_y, r"$b^{(q)}$", size=FS["label"], color=B["stroke"])
    L([(hx1, 22.0), (113.0, 22.0)], B["stroke"])

    # ================================================================ 中层：结构化价格响应（橙）
    # 默认模型只有本地价格项与空间溢出项；可选的负荷转移项 γU（默认关闭，γ 未做识别）不画，在图注中说明
    rows = [55.5, 67.5]
    ymid = (rows[0] + rows[1]) / 2
    I.price_curve(c, 4.0, rows[0] - 4.3, 18.0, 8.6)              # 实线 p，虚线 p′（情景）
    T(12.0, rows[0] - 6.2, r"$p'$", size=FS["body"], color=P["stroke"])
    I.ring_map(c, 12.8, rows[1], 4.6)
    for yy, lab in zip(rows, (r"$\Delta\ell$", r"$S^{(k)}$")):
        T(26.5, yy, lab, ha="left", size=FS["label"], color=P["stroke"])

    gx0, gx1, gy0, gy1 = 44.0, 106.0, 46.0, 75.0
    c.box(gx0, gy0, gx1 - gx0, gy1 - gy0, fc=P["fill"], ec=P["stroke"], lw=LW["box"], r=1.6, z=1)
    T(gx0 + 2.4, gy0 + 3.3, "Price response", ha="left", weight="bold", size=FS["label"], fit=(gx0, gx1))
    chx, chw, chh = 89.5, 12.0, 5.2                              # 锚定参数所在的列
    ox = 101.8                                                   # ⊕
    for yy, par in zip(rows, (r"$\hat\beta_{g,\kappa}$", r"$\hat\delta_k$")):
        L([(TAIL, yy), (chx - chw / 2, yy)], P["stroke"])
        c.box(chx - chw / 2, yy - chh / 2, chw, chh, fc=C["mid"], ec=C["stroke"], lw=LW["box"], r=0.9, z=4)
        T(chx, yy - 0.05, par, size=FS["label"], z=6)
    L([(chx + chw / 2, rows[0]), (ox, rows[0]), (ox, ymid - 1.55)], P["stroke"])
    L([(chx + chw / 2, rows[1]), (ox, rows[1]), (ox, ymid + 1.55)], P["stroke"])
    c.op(ox, ymid, "+", color=P["stroke"])

    ex0, ex1 = 111.0, 122.0
    L([(ox + 1.55, ymid), (ex0, ymid)], P["stroke"])
    T(108.7, ymid - 2.2, r"$\eta$", size=FS["label"], color=P["stroke"])
    c.box(ex0, ymid - 3.0, ex1 - ex0, 6.0, fc="white", ec=P["stroke"], lw=LW["thin"], r=1.0, z=2)
    T((ex0 + ex1) / 2, ymid, r"$\exp(\cdot)$", size=FS["label"])
    mx = 128.5                                                   # ⊗：基线 × exp(η)
    L([(ex1, ymid), (mx - 1.7, ymid)], P["stroke"])
    L([(125.8, 22.0), (mx, 22.0), (mx, ymid - 1.7)], B["stroke"])
    c.op(mx, ymid, "x", r=1.7)
    nx0, nx1 = 134.5, 146.5
    L([(mx + 1.7, ymid), (nx0, ymid)], INK)
    c.box(nx0, ymid - 3.0, nx1 - nx0, 6.0, fc="white", ec=INK, lw=LW["thin"], r=1.0, z=2)
    T((nx0 + nx1) / 2, ymid, r"$\min\{1,\cdot\}$", size=FS["label"])

    # ================================================================ 右列：损失、校准、输出（红）
    jx = 150.6
    L([(nx1, ymid), (jx, ymid)], O["stroke"], arrow=False)
    dot(jx, ymid, O["stroke"])
    T(151.2, ymid + 4.3, r"$\hat y^{(q)}$", size=FS["label"], color=O["stroke"])
    ax0, ax1, ay0, ay1 = 155.0, 188.0, 48.5, 75.0                # 自适应保形校准
    L([(jx, ymid), (ax0, ymid)], O["stroke"])
    c.box(ax0, ay0, ax1 - ax0, ay1 - ay0, fc=O["fill"], ec=O["stroke"], lw=LW["box"], r=1.6, z=1)
    T((ax0 + ax1) / 2, ay0 + 3.6, "Adaptive conformal", weight="bold", size=FS["label"], fit=(ax0, ax1))
    I.alpha_track(c, 159.5, ay0 + 8.0, 17.0, 11.0)
    T(168.0, ay0 + 22.8, r"$\alpha_{t,h}$", size=FS["label"])
    I.frozen_mask(c, 182.6, ay0 + 13.2, 6.0)
    ox0, ox1, oy0, oy1 = 155.0, 188.0, 3.0, 40.0                 # 校准后的预测
    L([((ax0 + ax1) / 2, ay0), ((ax0 + ax1) / 2, oy1)], O["stroke"])
    c.box(ox0, oy0, ox1 - ox0, oy1 - oy0, fc="white", ec=O["stroke"], lw=LW["box"], r=1.6, z=1)
    T((ox0 + ox1) / 2, oy0 + 3.6, "Calibrated forecast", weight="bold", size=FS["label"], fit=(ox0, ox1))
    I.forecast_panel(c, 158.5, oy0 + 9.0, 26.0, 18.5)
    T((ox0 + ox1) / 2, oy1 - 3.4, r"$[\hat y^{\,\mathrm{lo}},\,\hat y^{\,\mathrm{hi}}]$", size=FS["label"],
      fit=(ox0, ox1))

    lx0, lx1, ly0, ly1 = 132.0, 146.8, 27.0, 44.0                # 训练损失（掩码 pinball）
    L([(jx, ymid), (jx, (ly0 + ly1) / 2), (lx1, (ly0 + ly1) / 2)], O["stroke"], LW["thin"])
    c.box(lx0, ly0, lx1 - lx0, ly1 - ly0, fc="white", ec=O["stroke"], lw=LW["box"], r=1.2, z=1)
    I.pinball_icon(c, lx0 + 1.6, ly0 + 2.0, 7.2, 6.4)
    I.frozen_mask(c, lx0 + 11.8, ly0 + 5.3, 4.2)
    T((lx0 + lx1) / 2, ly1 - 4.0, r"$\mathcal{L}_{\mathrm{pin}}$", size=FS["title"])
    gy = 4.0                                                     # 梯度（灰色点线）：只到分位数头与骨干
    gxs = (lx0 + lx1) / 2
    L([(gxs, ly0), (gxs, gy), (hxc, gy), (hxc, hy0)], INK2, LW["thin"], ls=DOT_GRAD, head=(1.0, 0.75))
    L([(hxc, gy), (60.0, gy), (60.0, 9.3)], INK2, LW["thin"], ls=DOT_GRAD, head=(1.0, 0.75))
    T(83.0, 2.0, r"$\nabla_{\phi}\,\mathcal{L}_{\mathrm{pin}}$", size=FS["label"], color=INK2)

    # ================================================================ 下层：因果识别（紫）
    qx0, qx1, qy0, qy1 = 3.0, 106.0, 81.0, 106.6
    c.box(qx0, qy0, qx1 - qx0, qy1 - qy0, fc=C["fill"], ec=C["stroke"], lw=LW["box"], r=1.8, z=1)
    T(qx0 + 2.4, qy0 + 3.4, "Causal identification", ha="left", weight="bold", size=FS["label"], fit=(qx0, qx1))
    iy0, iy1, ly = 87.8, 98.6, 102.9
    stages = [(7.0, 17.0, "Switch events"), (33.5, 14.0, "Two-way FE"), (58.5, 13.0, "Pooling"),
              (80.5, 18.0, r"$\hat\beta,\ \hat\delta\ \pm$ se")]
    I.switch_rd(c, stages[0][0], iy0, stages[0][1], iy1 - iy0)
    I.slope_plot(c, stages[1][0], iy0, stages[1][1], iy1 - iy0)
    I.pool_tree(c, stages[2][0], iy0, stages[2][1], iy1 - iy0)
    I.anchor_table(c, stages[3][0], iy0 + 2.2, stages[3][1], iy1 - iy0 - 3.6)
    for x0, w0, lab in stages:
        T(x0 + w0 / 2, ly, lab, size=FS["body"], fit=(qx0, qx1))
    for k in range(3):
        a = stages[k][0] + stages[k][1] + 1.4
        b = stages[k + 1][0] - 1.4
        L([(a, (iy0 + iy1) / 2), (b, (iy0 + iy1) / 2)], C["stroke"], LW["thin"], head=(1.0, 0.7))
    # 锚定值 → 价格参数（紫色粗虚线，箭头落在各参数框底边）；锁 = 不可训练；梯度在此截断（灰色点线 + 剪刀）
    L([(chx, iy0 + 2.2), (chx, rows[1] + chh / 2)], C["stroke"], BUS_LW, ls=BUS, head=(1.5, 1.05))
    L([(chx, rows[1] - chh / 2), (chx, rows[0] + chh / 2)], C["stroke"], BUS_LW, ls=BUS, head=(1.2, 0.9))
    I.lock(c, chx - 3.4, ymid + 0.3, 2.4)
    cx_ = 99.0
    L([(cx_, gy1), (cx_, qy0)], INK2, LW["thin"], ls=DOT_GRAD, arrow=False)
    c.scissors(cx_, 78.0, s=2.6, angle=90, color=C["stroke"])

    # ================================================================ 图例（右下，只解释符号）
    gx_, gy_ = 112.0, 81.0
    c.box(gx_, gy_, 188.0 - gx_, qy1 - gy_, fc="white", ec="#BFBFBF", lw=LW["hair"], r=1.2, z=1)
    col1, col2 = gx_ + 4.0, gx_ + 40.0
    ys = [gy_ + 6.4, gy_ + 12.8, gy_ + 19.2]
    L([(col1, ys[0]), (col1 + 6.0, ys[0])], C["stroke"], BUS_LW, ls=BUS, arrow=False)
    T(col1 + 7.6, ys[0], "fixed causal anchor", ha="left", size=FS["body"], fit=(gx_, col2 - 0.5))
    L([(col1, ys[1]), (col1 + 6.0, ys[1])], INK2, LW["thin"], ls=DOT_GRAD, arrow=False)
    T(col1 + 7.6, ys[1], "gradient", ha="left", size=FS["body"], fit=(gx_, col2 - 0.5))
    c.scissors(col1 + 3.0, ys[2], s=2.4, angle=0, color=C["stroke"])
    T(col1 + 7.6, ys[2], "gradient cut", ha="left", size=FS["body"], fit=(gx_, col2 - 0.5))
    I.lock(c, col2 + 3.0, ys[0] + 0.4, 2.4)
    T(col2 + 7.6, ys[0], "not trainable", ha="left", size=FS["body"], fit=(col2, 188.0))
    I.frozen_mask(c, col2 + 3.0, ys[1], 5.2)
    T(col2 + 7.6, ys[1], "frozen points masked", ha="left", size=FS["body"], fit=(col2, 188.0))
    c.op(col2 + 1.4, ys[2], "+", r=1.2)
    c.op(col2 + 4.6, ys[2], "x", r=1.2)
    T(col2 + 7.6, ys[2], "add, multiply", ha="left", size=FS["body"], fit=(col2, 188.0))

    if check:
        for msg in c.check():
            print("  [检查]", msg)
    base = os.path.join(out_dir, "Fig3_CPA-STGNN_overall")
    c.save(base)
    return base


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.join(here, "..", "模型总框架图")          # 在 04_图表/绘图脚本/ 下运行时直接写到 04_图表/模型总框架图/
    default = os.path.abspath(proj_dir) if os.path.isdir(proj_dir) else os.path.join(here, "out")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    print(build(out))
