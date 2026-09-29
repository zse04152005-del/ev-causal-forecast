"""论文图 3：CPA-STGNN 总框架（双栏 190 mm × 125 mm；v2：按 PAG、PIAST、iTransformer 范例的"理论图形"重绘）。

版式（自下而上，仿 PAG）：
  Raw data ：利用率 y（节点卡片）、日历 / 天气 / POI（图标卡片）、电价 p（节点卡片，虚线卡片 = 反事实情景 p′）
  Features ：拼接后的节点令牌 x_1 … x_N 与图结构 A_s；目标时刻的价格偏离 Δℓ 与距离环带暴露 S^(k)
  Model    ：左 = 价格盲时空骨干（Graph WaveNet 的一层：门控 TCN → 跳连接 / 多图 GCN → 残差 → BN，× 8）
                   各层跳连接之和 → 分位数头（另输入未来日历与观测天气）→ 基线 b^(q)
             中 = 价格响应：β̂_{g,κ}Δℓ + Σδ̂_k S^(k) → ⊕ → η → exp → ⊗ b^(q) → min{1,·} → ŷ^(q)
  右侧点线框：因果识别（切换事件 → 双向固定效应 → 合并层级 → 锚定值），紫色粗虚线送入价格响应（固定缓冲区，梯度截断）
  顶部     ：掩码 pinball 损失（梯度只回到分位数头与骨干）、自适应保形校准 → 校准后的 90% 区间
与代码的对应见 04_图表/README_图稿说明.md 第六节。可选的负荷转移项 γU 默认关闭，图中不画（图注说明）。
用法：python fig3_overview.py [输出目录]（默认：04_图表/模型总框架图/，找不到时为脚本旁边的 out/）
"""
import os
import sys

from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icons as I  # noqa: E402
import theory as Th  # noqa: E402
from figlib import DOT_GRAD, FS, INK, INK2, LW, PAL, Canvas  # noqa: E402

B, P, G, C, O, K = (PAL[k] for k in ("demand", "price", "cov", "causal", "output", "gray"))  # noqa: E741
W, H = 190.0, 125.0
BUS = (0, (2.6, 1.3))                     # 固定锚定值：紫色粗虚线
BUS_LW = LW["causal"] + 0.3
EDGE = "#6E6E6E"                          # 数据行虚线框


def build(out_dir: str, check: bool = True) -> str:
    c = Canvas(W, H)
    T = c.text
    L = c.line

    def dot(x, y, col, r=0.55):
        c.ax.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=7))

    def blayer(x, y, w, h, label, **kw):                           # 骨干中的层：与图 4(a) 同样的蓝色
        kw.setdefault("fc", B["mid"])
        kw.setdefault("ec", B["stroke"])
        return Th.layer(c, x, y, w, h, label, **kw)

    # ================================================================ 行标签（仿 PAG）
    NET, FEA, RAW = (15.0, 71.0), (74.0, 92.0), (95.0, 123.0)
    Th.row_label(c, 7.6, *NET, "Model", xa=15.4)
    Th.row_label(c, 7.6, *FEA, "Features", xa=15.4)
    Th.row_label(c, 7.6, *RAW, "Raw data", xa=15.4)

    # ================================================================ 原始数据行
    c.rect(18.0, RAW[0], 118.0, RAW[1] - RAW[0], fc="none", ec=EDGE, lw=LW["thin"], ls=Th.DASH_ROW, z=1)
    cy_, ch_ = 100.5, 10.0
    blue, green, orange = B["stroke"], G["stroke"], P["stroke"]
    top_raw = cy_ - 2 * 1.0                                        # 最上一层卡片的上沿
    # 利用率（各小区，过去 24 小时）
    Th.token_card(c, 22.0, cy_, 26.0, ch_, [blue, blue, blue, None, blue],
                  [r"$y_1$", r"$y_2$", r"$y_3$", None, r"$y_N$"], card_fc=B["mid"], card_ec=B["strong"], axis=r"$t$")
    T(35.0, 116.3, "Utilization", size=FS["body"])
    # 日历、天气（全市共用）与 POI / 静态特征：用图标卡片而不是逐小区的节点
    for k in range(2, 0, -1):
        c.box(58.0 + k, cy_ - k, 26.0, ch_, fc=G["mid"], ec=G["strong"], lw=LW["hair"], r=1.0, z=2 + 0.01 * (2 - k))
    c.box(58.0, cy_, 26.0, ch_, fc=G["mid"], ec=G["strong"], lw=LW["hair"], r=1.0, z=2.1)
    I.calendar(c, 60.6, cy_ + 1.8, 6.6, z=3)
    I.weather(c, 67.6, cy_ + 0.8, 8.4, z=3)
    I.pin(c, 79.6, cy_ + 3.3, 1.75, z=3)
    c.line([(84.2, cy_ + ch_ - 0.6), (86.8, cy_ + ch_ - 3.2)], INK, LW["thin"], head=(0.9, 0.5), z=3)
    T(87.5, cy_ + ch_ - 1.2, r"$t$", size=FS["body"], ha="left")
    T(71.0, 116.3, "Calendar, weather, POI", size=FS["body"], fit=(52.0, 96.0))
    # 电价（目标时刻）；后面的虚线卡片 = 反事实情景 p′（只替换价格路径）
    c.box(98.8, cy_ + 2.0, 26.0, ch_, fc="none", ec=P["stroke"], lw=LW["thin"], ls=(0, (1.6, 1.0)), r=1.0, z=1.9)
    T(97.6, cy_ + ch_ + 1.2, r"$p'$", size=FS["label"], color=P["stroke"], ha="right")
    Th.token_card(c, 101.0, cy_, 26.0, ch_, [orange, orange, orange, None, orange],
                  [r"$p_1$", r"$p_2$", r"$p_3$", None, r"$p_N$"], card_fc=P["mid"], card_ec=P["strong"],
                  axis=r"$t{+}h$")
    T(114.0, 116.3, "Price", size=FS["body"])

    # 拼接（仿 PAG 的 Concatenate 圆圈）：利用率 + 协变量 → 节点令牌
    xc = 40.0                                                       # 骨干一列的中线
    jx, jy = xc, 93.5
    L([(35.0, top_raw), (35.0, jy), (jx - 1.0, jy)], INK, LW["thin"], arrow=False)
    L([(71.0, top_raw), (71.0, jy), (jx + 1.0, jy)], INK, LW["thin"], arrow=False)
    c.ax.add_patch(Circle((jx, jy), 1.0, fc="white", ec=INK, lw=LW["thin"], zorder=6))
    # 电价 → Δℓ 与 S^(k)
    L([(114.0, top_raw), (114.0, jy)], INK, LW["thin"], arrow=False)
    dot(114.0, jy, INK, 0.45)

    # ================================================================ 特征行
    c.rect(18.0, FEA[0], 70.0, FEA[1] - FEA[0], fc="none", ec=EDGE, lw=LW["thin"], ls=Th.DASH_ROW, z=1)
    c.rect(92.0, FEA[0], 44.0, FEA[1] - FEA[0], fc="none", ec=EDGE, lw=LW["thin"], ls=Th.DASH_ROW, z=1)
    pair = (blue, green)
    Th.token_card(c, 23.0, 78.2, 34.0, 10.4, [pair, pair, pair, None, pair],
                  [r"$\mathbf{x}_1$", r"$\mathbf{x}_2$", r"$\mathbf{x}_3$", None, r"$\mathbf{x}_N$"],
                  card_fc="#E7EEF6", card_ec="#A9BCD3", axis=r"$t$")
    L([(jx, jy - 1.0), (jx, 88.8)], INK, LW["thin"], head=(1.0, 0.6))
    Th.grid_graph(c, 66.5, 77.0, 11.0)
    T(79.6, 86.6, r"$A_s$", size=FS["label"], ha="left")

    Th.token_card(c, 96.5, 77.6, 15.0, 7.4, [orange, orange, None, orange], card_fc=P["mid"], card_ec=P["strong"],
                  n_back=2, off=0.9, r_node=1.05)
    T(105.2, 89.6, r"$\Delta\ell$", size=FS["label"], color=P["stroke"], ha="left")
    I.ring_map(c, 125.0, 81.6, 4.4)
    T(126.2, 89.6, r"$S^{(k)}$", size=FS["label"], color=P["stroke"], ha="left")
    L([(114.0, jy), (104.0, jy), (104.0, 85.3)], INK, LW["thin"], head=(1.0, 0.6))
    L([(114.0, jy), (125.0, jy), (125.0, 86.3)], INK, LW["thin"], head=(1.0, 0.6))

    # ================================================================ 模型：价格盲时空骨干（仿 Graph WaveNet 原图）
    c.rect(18.0, NET[0], 70.0, NET[1] - NET[0], fc="white", ec=INK, lw=LW["box"], z=1)
    T(20.0, 18.4, "Price-blind ST backbone", ha="left", weight="bold", size=FS["label"], fit=(18.0, 88.0))
    x0s, ws = 25.0, 30.0
    Th.caret(c, xc, NET[1] + 0.8, w=5.5, h=2.0)                    # 节点令牌与图结构 → 骨干
    blayer(x0s, 64.6, ws, 4.2, "Linear embedding")
    gy0, gy1 = 24.4, 63.9                                           # 一个时空层（重复 8 次）
    c.rect(21.0, gy0, 38.0, gy1 - gy0, fc="none", ec=INK2, lw=LW["hair"], ls=(0, (2.0, 1.4)), z=2)
    T(21.0, gy0 - 2.0, r"$\times 8$", size=FS["label"], ha="left", weight="bold")
    sy = 62.6                                                       # 残差与门控的分叉点
    L([(xc, 64.6), (xc, sy)], INK, LW["thin"], arrow=False)
    dot(xc, sy, INK, 0.45)
    for xx, lab, act, ew in ((31.5, "TCN-a", "tanh", 8.0), (48.5, "TCN-b", r"$\sigma$", 4.8)):
        L([(xc, sy), (xx, sy), (xx, 59.8)], INK, LW["thin"], head=(0.9, 0.55))
        blayer(xx - 5.5, 55.8, 11.0, 4.0, lab)
        Th.ellipse_op(c, xx, 51.8, ew, 3.4, act)
        L([(xx, 55.8), (xx, 53.5)], INK, LW["thin"], head=(0.9, 0.55))
    ug = 47.2                                                       # 门控：tanh ⊙ σ
    L([(31.5, 50.1), (31.5, ug), (xc - 1.35, ug)], INK, LW["thin"], head=(0.9, 0.55))
    L([(48.5, 50.1), (48.5, ug), (xc + 1.35, ug)], INK, LW["thin"], head=(0.9, 0.55))
    c.op(xc, ug, "x", r=1.35, lw=LW["thin"])
    blayer(x0s, 36.8, ws, 4.0, "Multi-graph GCN")
    L([(xc, ug - 1.35), (xc, 40.8)], INK, LW["thin"], head=(0.9, 0.55))
    js = 43.4                                                       # 跳连接取自门控输出（进入图卷积之前）
    dot(xc, js, INK, 0.45)
    yp = 33.0
    c.op(xc, yp, "+", r=1.3, lw=LW["thin"])
    L([(xc, 36.8), (xc, yp + 1.3)], INK, LW["thin"], head=(0.9, 0.55))
    L([(xc, sy), (22.6, sy), (22.6, yp), (xc - 1.3, yp)], INK, LW["thin"], head=(0.9, 0.55))    # 残差
    blayer(34.0, 26.4, 12.0, 3.0, "BN")
    L([(xc, yp - 1.3), (xc, 29.4)], INK, LW["thin"], head=(0.9, 0.55))
    L([(xc, 26.4), (xc, gy0)], INK, LW["thin"], head=(0.9, 0.55))                                # → 下一层
    # 分位数头：输入为各层跳连接之和与未来日历、观测天气；输出基线分位数 b^(q)
    hx0, hx1, hy0, hy1 = 63.0, 70.5, gy0, 50.0
    hxc = (hx0 + hx1) / 2
    c.box(hx0, hy0, hx1 - hx0, hy1 - hy0, fc=B["strong"], ec=B["stroke"], lw=LW["box"], r=0.9, z=4)
    T(hxc, (hy0 + hy1) / 2, "Quantile head", rotation=90, weight="bold", size=FS["label"], z=6,
      fit=(hx0, hy0, hx1, hy1))
    L([(xc, js), (hx0, js)], INK, LW["thin"], head=(1.0, 0.6))
    T(52.2, js + 1.55, "skip", size=FS["body"], color=INK2)
    L([(hxc, 57.4), (hxc, hy1)], G["stroke"], LW["flow"], head=(1.0, 0.7))
    T(hxc + 1.5, 55.9, r"$\mathbf{c}_{t+h},\mathbf{w}_{t+h}$", size=FS["label"], color=G["stroke"], ha="left",
      fit=(hx0, 88.0))
    yq = 33.5
    c.fan(73.4, yq - 4.7, 13.2, 9.4, B["stroke"], B["strong"])
    T(80.0, yq + 7.2, r"$b^{(q)}$", size=FS["label"], color=B["stroke"])
    L([(hx1, yq), (73.2, yq)], B["stroke"], LW["flow"], head=(1.0, 0.7))

    # ================================================================ 模型：价格响应
    bx0, bx1 = 92.0, 136.0
    xb = 114.5
    c.rect(bx0, NET[0], bx1 - bx0, NET[1] - NET[0], fc="white", ec=INK, lw=LW["box"], z=1)
    T(bx0 + 2.0, 18.4, "Price response", ha="left", weight="bold", size=FS["label"], fit=(bx0, bx1))
    # 固定锚定值组（紫色虚线框 + 锁）
    c.box(94.4, 57.6, 40.0, 10.8, fc=C["fill"], ec=C["stroke"], lw=LW["thin"], ls=(0, (2.2, 1.2)), r=1.0, z=2)
    yc = 63.0
    for xx, lab in ((104.0, r"$\hat\beta_{g,\kappa}$"), (125.0, r"$\hat\delta_k$")):
        Th.layer(c, xx - 7.2, yc - 2.9, 14.4, 5.8, lab, fc=C["mid"], ec=C["stroke"], lw=LW["box"], size=FS["label"])
    I.lock(c, xb, 58.0, 2.5)
    L([(104.0, 75.8), (104.0, yc + 2.9)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    L([(125.0, 77.2), (125.0, yc + 2.9)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    ya = 52.0                                                       # ⊕
    L([(104.0, yc - 2.9), (104.0, ya), (xb - 1.45, ya)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    L([(125.0, yc - 2.9), (125.0, ya), (xb + 1.45, ya)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    c.op(xb, ya, "+", r=1.45, color=P["stroke"])
    ey0 = 41.6
    Th.layer(c, xb - 8.0, ey0, 16.0, 4.4, r"$\exp(\cdot)$", fc=P["fill"], ec=P["stroke"], size=FS["label"])
    L([(xb, ya - 1.45), (xb, ey0 + 4.4)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    T(xb + 1.6, 48.3, r"$\eta$", size=FS["label"], color=P["stroke"], ha="left")
    c.op(xb, yq, "x", r=1.5)
    L([(xb, ey0), (xb, yq + 1.5)], P["stroke"], LW["flow"], head=(1.0, 0.7))
    L([(86.6, yq), (xb - 1.5, yq)], B["stroke"], LW["flow"], head=(1.0, 0.7))          # b^(q) → ⊗
    my0 = 23.6
    Th.layer(c, xb - 8.0, my0, 16.0, 4.4, r"$\min\{1,\cdot\}$", fc=O["fill"], ec=O["stroke"], size=FS["label"])
    L([(xb, yq - 1.5), (xb, my0 + 4.4)], INK, LW["flow"], head=(1.0, 0.7))
    jyq = 13.0                                                      # ŷ^(q)：分到损失与保形校准
    T(xb + 1.3, 19.6, r"$\hat y^{(q)}$", size=FS["label"], color=O["stroke"], ha="left")

    # ================================================================ 因果识别（右侧点线框，仿 PAG 的预训练模块）
    qx0, qx1, qy0, qy1 = 142.0, 189.0, 49.6, RAW[1]
    c.rect(qx0, qy0, qx1 - qx0, qy1 - qy0, fc="white", ec=INK, lw=LW["box"], ls=(0, (1.0, 1.2)), z=1)
    T(qx0 + 2.0, qy0 + 3.3, "Causal identification", ha="left", weight="bold", size=FS["label"], fit=(qx0, qx1))
    bxl, bxr, lab_x = 146.0, 186.0, 177.6
    for y0_, y1_ in ((56.0, 70.0), (75.2, 86.4), (91.6, 102.8), (108.0, 120.4)):
        c.box(bxl, y0_, bxr - bxl, y1_ - y0_, fc="white", ec=C["stroke"], lw=LW["thin"], r=1.0, z=2)
    I.anchor_table(c, 149.0, 58.6, 18.0, 8.8)
    T(lab_x, 63.0, r"$\hat\beta,\ \hat\delta\ \pm$ se", size=FS["body"], fit=(168.0, bxr))
    I.pool_tree(c, 151.5, 76.0, 13.0, 9.6)
    T(lab_x, 80.8, "Pooling", size=FS["body"], fit=(168.0, bxr))
    I.slope_plot(c, 150.0, 92.3, 16.0, 9.8)
    T(lab_x, 97.2, "Two-way FE", size=FS["body"], fit=(168.0, bxr))
    I.switch_rd(c, 149.0, 109.0, 18.0, 10.2)
    T(lab_x, 114.2, "Switch\nevents", size=FS["body"], fit=(168.0, bxr))
    for yy in (73.0, 88.8, 105.2):
        Th.caret(c, 157.5, yy, w=5.0, h=1.9, color=C["stroke"])
    L([(136.0, 114.2), (bxl, 114.2)], INK, LW["thin"], head=(1.0, 0.6))              # 训练期数据 → 切换事件
    # 锚定值 → 价格响应（紫色粗虚线 = 固定值）；反向梯度被截断（灰色点线 + 剪刀）
    L([(bxl, yc), (134.4, yc)], C["stroke"], BUS_LW, ls=BUS, head=(1.5, 1.05))
    L([(134.4, 66.8), (141.6, 66.8)], INK2, LW["thin"], ls=DOT_GRAD, arrow=False)
    c.scissors(138.6, 66.8, s=2.3, angle=180, color=C["stroke"])

    # ================================================================ 顶部：训练损失、保形校准、输出
    ly0, ly1 = 2.2, 10.8
    lx0, lx1 = hxc - 13.8, hxc + 13.8
    c.box(lx0, ly0, lx1 - lx0, ly1 - ly0, fc="white", ec=O["stroke"], lw=LW["box"], r=1.0, z=2)
    I.pinball_icon(c, lx0 + 1.8, ly0 + 1.4, 7.6, 5.8)
    T(lx0 + 15.0, (ly0 + ly1) / 2, r"$\mathcal{L}_{\mathrm{pin}}$", size=FS["title"])
    I.frozen_mask(c, lx0 + 23.3, (ly0 + ly1) / 2, 4.6)
    L([(xb, my0), (xb, jyq)], O["stroke"], LW["flow"], arrow=False)
    dot(xb, jyq, O["stroke"])
    L([(xb, jyq), (90.0, jyq), (90.0, 6.5), (lx1, 6.5)], O["stroke"], LW["thin"], head=(1.0, 0.6))
    # 梯度只回到分位数头与骨干（φ），不到锚定值
    L([(hxc, ly1), (hxc, hy0)], INK2, LW["thin"], ls=DOT_GRAD, head=(1.0, 0.65))
    T(hxc + 1.4, 19.4, r"$\nabla_{\phi}\,\mathcal{L}_{\mathrm{pin}}$", size=FS["label"], color=INK2, ha="left")

    ax0, ax1 = 99.0, 130.0
    c.box(ax0, ly0, ax1 - ax0, ly1 - ly0, fc=O["fill"], ec=O["stroke"], lw=LW["box"], r=1.0, z=2)
    T((ax0 + ax1) / 2, ly0 + 2.6, "Adaptive conformal", weight="bold", size=FS["label"], fit=(ax0, ax1))
    I.alpha_track(c, ax0 + 3.0, ly0 + 4.4, 9.0, 3.4, n=10)
    T(ax0 + 16.0, ly0 + 6.2, r"$\alpha_{t,h}$", size=FS["body"], ha="left")
    I.frozen_mask(c, ax0 + 26.4, ly0 + 6.2, 3.6)
    L([(xb, jyq), (xb, ly1)], O["stroke"], LW["flow"], head=(1.0, 0.7))
    ox0, ox1, oy0, oy1 = 142.0, 189.0, ly0, 35.0
    c.box(ox0, oy0, ox1 - ox0, oy1 - oy0, fc="white", ec=O["stroke"], lw=LW["box"], r=1.2, z=1)
    T((ox0 + ox1) / 2, oy0 + 3.5, "Calibrated forecast", weight="bold", size=FS["label"], fit=(ox0, ox1))
    I.forecast_panel(c, ox0 + 5.0, oy0 + 8.0, ox1 - ox0 - 10.0, 17.5)
    T((ox0 + ox1) / 2, oy1 - 3.2, r"$[\hat y^{\,\mathrm{lo}},\,\hat y^{\,\mathrm{hi}}]$", size=FS["label"],
      fit=(ox0, ox1))
    L([(ax1, 6.5), (ox0, 6.5)], O["stroke"], LW["flow"], head=(1.0, 0.7))

    # ================================================================ 图例（右列，只解释符号）
    lgx, lg2, lgy = 143.4, 168.6, [38.4, 42.2, 46.0]
    L([(lgx, lgy[0]), (lgx + 5.0, lgy[0])], C["stroke"], BUS_LW, ls=BUS, arrow=False)
    T(lgx + 6.3, lgy[0], "fixed anchor", ha="left", size=FS["body"], fit=(lgx, lg2 - 2.2))
    L([(lgx, lgy[1]), (lgx + 5.0, lgy[1])], INK2, LW["thin"], ls=DOT_GRAD, arrow=False)
    T(lgx + 6.3, lgy[1], "gradient", ha="left", size=FS["body"], fit=(lgx, lg2 - 2.2))
    c.scissors(lgx + 2.5, lgy[2], s=2.2, angle=0, color=C["stroke"])
    T(lgx + 6.3, lgy[2], "gradient cut", ha="left", size=FS["body"], fit=(lgx, lg2 - 2.2))
    I.lock(c, lg2, lgy[0] + 0.35, 2.3)
    T(lg2 + 3.0, lgy[0], "not trainable", ha="left", size=FS["body"], fit=(lg2, W))
    I.frozen_mask(c, lg2, lgy[1], 4.2)
    T(lg2 + 3.0, lgy[1], "frozen, masked", ha="left", size=FS["body"], fit=(lg2, W))
    c.ax.add_patch(Circle((lg2, lgy[2]), 1.0, fc="white", ec=INK, lw=LW["thin"], zorder=6))
    T(lg2 + 3.0, lgy[2], "concatenate", ha="left", size=FS["body"], fit=(lg2, W))

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
