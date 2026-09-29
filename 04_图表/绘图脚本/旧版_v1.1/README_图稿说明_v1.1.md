# 图稿说明（阶段 4：模型框架图与模块图）

> 版本：v1.1（2026-09-28：按你的反馈重绘图 3——能用图标表达的不写文字、图标仿顶刊框架图、版面简化、去掉 ①⑤a 之类的模块编号；图 4 同样去掉编号圆标；第三轮独立审查通过）。v1（2026-09-28）。对应代码：`03_代码`（模型设计文档 v1.1、决策 D1/D7/D8/D9 确认后的版本）。
>
> 本文件说明：设计语言、每张图表达的计算逻辑、创新点与图中元素的对应、建议图注、每个图元与代码的核对表、独立审查记录，以及审查中发现的待决问题。

---

## 一、文件清单

| 位置 | 文件 | 说明 |
|---|---|---|
| `04_图表/模型总框架图/` | `Fig3_CPA-STGNN_overall.{pdf,svg,png}` | 论文图 3，190 × 108 mm |
| `04_图表/模块架构图/` | `Fig4_CPA-STGNN_modules.{pdf,svg,png}` | 论文图 4，190 × 144 mm，四个面板 (a)–(d) |
| `04_图表/模块架构图/` | `Fig4a_backbone_head`、`Fig4b_price_anchoring`、`Fig4c_spillover_rings`、`Fig4d_masked_aci`（各 `.pdf/.svg/.png`） | 四个面板单独成图（98 或 88 × 70 mm，不带 (a)–(d) 字母），用于幻灯片或期刊要求分图提交时 |
| `04_图表/绘图脚本/` | `figlib.py` | 共用绘图工具：毫米坐标画布、设计语言常量、文字越界与重叠自动检查、导出 |
| `04_图表/绘图脚本/` | `fig3_overview.py`、`fig4_modules.py` | 两张图的绘图脚本 |
| `04_图表/绘图脚本/` | `icons.py` | 图 3 用的矢量图标库（图快照、矩阵、日历、天气、定位、电价曲线、环带、特征片、分位数扇形、pinball、冻结点、锁、因果识别四个小图等），画法见 2.4 |
| `04_图表/绘图脚本/` | `make_fig4c_geometry.py`、`assets/fig4c_zone525.json` | 图 4(c) 的真实小区几何（由 UrbanEV 边界与距离矩阵生成的 120 KB 小文件） |

- **PDF**：投稿用的矢量文件，字体以 TrueType（Type 42）嵌入（已用 `pdffonts` 核对，全部 `emb yes`）
- **SVG**：可编辑。文字保留为文字，字体名写成 `Arial, 'Liberation Sans', Helvetica` 与 `STIXGeneral, 'Times New Roman'` 的字体栈，Mac、Windows 上用 Illustrator / Inkscape 打开不会乱码
- **PNG**：600 dpi，用于 Word 与预览

**重新生成**（Mac / Windows 均可，需要 `numpy`、`matplotlib`；Liberation Sans 与 Arial 字宽相同，没有前者时自动用 Arial，版面不变）：

```bash
cd 04_图表/绘图脚本
python fig3_overview.py          # → 04_图表/模型总框架图/
python fig4_modules.py           # → 04_图表/模块架构图/（总图 + 四个单独面板）
python fig4_modules.py --real    # 阶段 6 正式锚定生成后：(b) 的层级标记与 (c) 的 δ 小图改用锚定文件中的真实值
python make_fig4c_geometry.py    # 只在需要更换 (c) 的示例小区时运行（--zone 编号），需要 02_数据 中的 UrbanEV 原始数据
```

每次运行都会自动检查：文字是否超出所在方框、文字之间是否重叠、文字是否超出画布；问题会打印出来，当前版本全部为 0。

---

## 二、设计语言 v1.1

### 2.1 颜色语义（经色盲模拟验证）

| 角色 | 线条 / 描边 | 浅填充 | 中间色 | 深填充 | 用于 |
|---|---|---|---|---|---|
| 需求与潜在特征 | `#4072B7` | `#EEF5FF` | `#D6E8FF` | `#B6D3FC` | 需求输入、骨干网络、分位数头、基线 $b^{(q)}$ |
| 价格信号 | `#CA710A` | `#FDF2EA` | `#FBE1CC` | `#F2C8A7` | 电价曲线、Δℓ、环带暴露 S、价格响应模块、环带着色 |
| 协变量 | `#3B9D5C` | `#EDF8EF` | `#D4EED9` | `#B2DEBC` | 日历、天气、静态与 POI 特征 |
| 因果锚定 | `#6D2D98` | `#F7F2FD` | `#EEE0FB` | `#DDC6F2` | 因果识别模块、锚定值、截断反馈 |
| 输出 | `#B73138` | `#FFF1EF` | `#FFDCDA` | `#F9C1BD` | ŷ、训练损失、保形校准与区间 |
| 中性（图结构、辅助） | `#8C8C8C` | `#F3F3F3` | `#E6E6E6` | `#D2D2D2` | 图结构、tanh / σ / ReLU / BN、冻结点 |

- 五个有色角色用数据可视化规范的调色板校验脚本检查（OKLab 色差 × 100，全部两两组合）：亮度带、饱和度下限、与底色对比度全部通过；**色盲模拟下最小色差 8.1**（绿—橙，红绿色盲），蓝黄色盲 7.8；**正常色觉下最小色差 15.3**（红—橙）。中性灰不属于分类色，不参与这项检查
- v1 色板的问题（已弃用）：紫与蓝在红绿色盲下色差只有 1.8，绿与橙 4.6
- 颜色从不单独承担含义：每种数据流都有数学符号标注（$y$、$A_s$、$\Delta\ell$、$S^{(k)}$ 等）或图例，因果锚定另用粗虚线，梯度另用点线

### 2.2 线型与符号

| 图元 | 含义 |
|---|---|
| 实线箭头（按上表着色） | 前向数据流 |
| **紫色粗虚线**（1.5–1.8 pt） | 固定的因果锚定值进入模型（缓冲区，不是可训练参数） |
| 灰色点线箭头 | 训练梯度 ∇ |
| ✂（剪刀） | 截断反馈：预测损失的梯度不回传到锚定值 |
| 🔒（紫色挂锁） | 固定缓冲区，不可训练（图 3） |
| 一串灰色 × + 红色斜线 | 冻结（插补）的数据点被掩码：不进损失、不产生保形分数（图 3；与图 4(d) 的 × 记号一致） |
| ⊕ / ⊗ | 逐元素相加 / 相乘 |
| 梯形 + "Graph WaveNet ×8"（图 3）；点画线叠放框 + "×8"（图 4(a)） | 重复 8 层的时空编码器（后者仿 Graph WaveNet 原图） |
| 虚线方框 | 示意部分（图 4） |

### 2.3 版式

| 项目 | 规格 |
|---|---|
| 宽度 | 双栏 190 mm；按最终尺寸设计，**不缩放** |
| 字体 | 文字 Liberation Sans（与 Arial 等宽）；公式 STIX（与 Times 一致），变量斜体 |
| 字号 | 正文与说明 **7 pt**（全图没有更小的正文字）；模块标题与数学符号 7.5 pt，$\mathcal{L}_{\mathrm{pin}}$ 8 pt；面板字母 8.5 pt 粗体；一级上下标 4.9–5.6 pt（所在字号的 70%）；**不使用二级上下标**（例如用 $b^{(0.05)}$ 而不是 $b^{(q_1)}$） |
| 文字量 | 总框架图只写模块名称和数学符号，能用图标表达的不写文字，说明性的话（过去 24 小时、观测天气、训练期均值、合并阈值、可选项等）一律放图注 |
| 编号 | **不用编号圆标**（①、⑤a 之类）：模块靠位置、颜色和图标区分，图注里用模块名称指代 |
| 线宽 | 方框 0.8 pt，数据流 1.0 pt，细线 0.6 pt，因果锚定 1.5–1.8 pt |
| 面板字母 | 左上角粗体 "(a)"–"(d)"，面板标题写在图内，详细说明放图注 |
| 示意与结果 | 方法图中不出现估计结果的数值；示意曲线与示意格局都在图中标 "schematic" / "illustrative" |

### 2.4 图标（图 3，`icons.py`）

参照 `01_模型总框架图/` 与 `总框架图范例_顶刊/` 中范例的画法（只借鉴画法，不复制范例图形）：线条图标、描边为主、浅色填充，线宽与数据流一致（0.6–1.0 pt），颜色跟随所属数据流；不用写实插画、emoji 或与模型无关的装饰图标。

| 图标 | 表示 | 画法参照 |
|---|---|---|
| 三帧叠放的小区图快照（只在最前一帧画节点，节点深浅 = 利用率） | 时空输入 $y,\tilde m$ | HSTGCN 的 $X_{t-m+1}\dots X_t$ |
| 叠放的方阵 | 图结构 $A_s$ | ASTGRN 的网格矩阵 |
| 日历、天气（太阳 + 云）、定位（统一为描边风格） | 协变量 $\mathbf c,\mathbf w,\mathbf s$ | STAEformer 的日历图标 |
| 电价阶梯曲线（实线 p、点线参考价、虚线情景 $p'$） | 价格输入 $\Delta\ell$ 与反事实情景 | — |
| 同心环带（中心小区蓝点，邻区实心 = 分时、空心 = 固定电价） | 环带暴露 $S^{(k)}$ | 与图 4(c) 一致 |
| 梯形 | 编码器（价格盲骨干） | 视觉类框架图中的 Backbone 梯形 |
| 三块叠放的扁立方片 | 节点表示 $Z$ | DyFCLT 的特征图 |
| 竖排文字的细长块 | 分位数头 | 范例中的单层 / 头 |
| 分位数扇形（多层浅蓝带） | 基线分位数 $b^{(q)}$ | — |
| 非对称 V 形折线 | pinball 损失 $\rho_q(u)$ | 损失框的画法仿 PIAST；用检查函数的形状，比下降的损失曲线更能看出是分位数损失 |
| 围绕点线波动的阶梯线 | ACI 的 $\alpha_{t,h}$ 在线更新 | — |
| 历史曲线 + 虚线"现在" + 预测与区间带 | 校准后的预测 | STAEformer 的输出帧 |
| 价格阶跃 + 两侧需求散点 | 分时电价切换事件 | 断点 / 事件研究图 |
| 残差散点 + 负斜率拟合线 | 双向固定效应回归（每个格子一个斜率） | — |
| 三层树：叶 ●、中 ◐、根 ○ | 合并层级：格子 → 情境 → 全市（符号与图 4(b) 锚定表一致） | — |
| 深浅格子的小表 | 锚定表 $\hat\beta,\hat\delta\pm$se | — |

---

## 三、每张图的计算逻辑

### 图 3：CPA-STGNN 总框架（v1.1 重绘版）

**版式**：左列是输入图标；向右三条横向通道——上层蓝色需求通路、中层橙色价格通路、下层紫色因果识别——在右列红色的损失、校准与输出处汇合。图中只写模块名称和数学符号，其余说明放图注（第五节）。

1. **输入（左列，图标 + 符号，所有输入箭头从同一条竖线出发）**
   - 三帧叠放的小区图快照 → $y,\tilde m$：过去 L = 24 小时的利用率 $y_{i,t}$ 与陈旧标记 $\tilde m_{i,t}$（截至当前已连续不变 ≥ 6 小时）
   - 叠放的方阵 → $A_s$：三张固定图（邻接、距离、POI 相似度）与自适应图
   - 日历、天气、定位图标 → $\mathbf{c},\mathbf{w},\mathbf{s}$：日历、天气、静态与 POI 特征
   - 电价阶梯曲线 → $\Delta\ell$：粗实线为实际电价 p，点线为训练期参考价，虚线为反事实情景 $p'$；$\Delta\ell_{i,t+h}=\ell_{i,t+h}-\bar\ell_i$（$\bar\ell_i$ 为训练期均值）
   - 同心环带 → $S^{(k)}$：中心小区（蓝点）与三个距离环带内的邻区（实心 = 分时，空心 = 固定电价）
2. **价格盲时空骨干（蓝色梯形）**：标题 "Price-blind ST backbone"，Graph WaveNet 编码器，门控空洞 TCN + 多图 GCN × 8 层，**看不到任何价格** → 节点表示 $Z$（三块叠放的特征片）
3. **分位数头（竖排文字的细长块）**：输入 $Z$、未来日历与观测天气 $\mathbf{c}_{t+h},\mathbf{w}_{t+h}$（绿线从协变量分出，从下方进入）与步长嵌入（图中不画）→ 累加 softplus → 基线分位数 $b^{(q)}$（扇形，$b^{(0.05)}\le\cdots\le b^{(0.95)}$）
4. **价格响应（橙色模块 "Price response"）**：$\Delta\ell$ 进入 $\hat\beta_{g,\kappa}$、$S^{(k)}$ 进入 $\hat\delta_k$（两个紫色参数块）→ ⊕ → η → exp(·) → 与 $b^{(q)}$ 相乘（⊗，蓝线从上方落下）→ $\min\{1,\cdot\}$ → $\hat y^{(q)}$。**只有 η 依赖价格**，所以反事实预测 = 事实预测 × exp(Δη)（推论 1）。可选的负荷转移项 γU 默认关闭、γ 目前不做识别，**图中不画**，在图注里说明
5. **因果识别（底部紫色带 "Causal identification"）**：四个小图，从左到右
   - Switch events：价格在切换时刻上跳（橙），两侧需求散点与均值线 = 训练期的分时电价切换事件（切换前后各 3 小时价格不变、未冻结、排除法定假日前后 1 天）
   - Two-way FE：去掉固定效应后的残差散点与负斜率拟合线 = 设计 D（日期×时刻 + 小区×时刻固定效应、环带暴露、聚类标准误），每个格子一个斜率
   - Pooling：三层树，叶 ● = 格子、中间 ◐ = 情境（各功能区共用）、根 ○ = 全市，与图 4(b) 锚定表的符号一致
   - $\hat\beta,\hat\delta\pm$se：锚定表（深色格子 = 自身估计，浅色 = 并入上一层级）

   锚定值经紫色粗虚线送入两个参数块（箭头落在参数块底边）；挂锁表示固定缓冲区、不可训练；价格模块与因果识别之间的灰色点线加剪刀表示梯度在此截断
6. **训练（红框 $\mathcal{L}_{\mathrm{pin}}$）**：$\hat y^{(q)}$ 从右侧进入；非对称 V 形是 pinball 检查函数，旁边的冻结点记号表示只用未冻结点；梯度 $\nabla_\phi\mathcal{L}_{\mathrm{pin}}$（灰色点线）只回到分位数头与骨干
7. **自适应保形校准（红框 "Adaptive conformal"）**：$\alpha_{t,h}$ 围绕目标值在线更新的阶梯线；冻结点记号表示冻结点不产生分数、不计入 err → **校准后的预测（"Calibrated forecast"）**：历史曲线（灰）、"现在"虚线、预测中位数与 90% 区间 $[\hat y^{\mathrm{lo}},\hat y^{\mathrm{hi}}]$（红）
8. **图例（右下）**：只解释符号——fixed causal anchor、gradient、gradient cut、not trainable、frozen points masked、add / multiply；颜色的含义由各通道的符号标注和图注说明

**从旧版图中移到图注的内容**：过去 24 小时、陈旧标记的定义、"(observed)" 未来天气、"(training period)"、"only η depends on price"、设计 D 的细节、合并阈值（10 个分时小区、se 0.5）、可选项 γU、90% 区间。

### 图 4(a)：价格盲骨干与非交叉分位数头

- 输入嵌入：$[y,\tilde m]$ 经 1×1 卷积，加上历史日历与天气的线性投影（广播到各小区）和静态特征的线性投影（广播到各时刻），左侧补零到感受野 31 步
- 每一层：TCN-a → tanh 与 TCN-b → σ 逐元素相乘（门控，核宽 2，空洞率 $d_\ell$ = 1, 2, 4, 8, 1, 2, 4, 8）→ 跳连接 1×1 → 多图扩散卷积（4 张图、各 2 步扩散、拼接后 1×1，dropout 0.3）→ 加残差 → BN
- 输出：各层跳连接相加取最后时刻 → ReLU → 1×1（256）→ ReLU → 1×1（32）→ $\mathbf{z}_i$
- 图结构：对称化邻接、距离高斯核、POI 余弦相似度（前 10 个邻居），三者加自环后行归一化；再加自适应图 $\tilde A=\mathrm{softmax}(\mathrm{ReLU}(E_1E_2^\top))$
- 分位数头：Linear($\mathbf{z}_i$) + Linear($\mathbf{c}_{t+h},\mathbf{w}_{t+h}$) + 步长嵌入 $\mathbf{e}_h$ → ReLU → Linear(64) → ReLU → Linear(7) → softplus → 累加

### 图 4(b)：锚定的价格响应与截断反馈

- 锚定表：3 个 POI 功能区（低密度、居住混合、高密度商业）× 4 个情境（工作日 / 休息日 × 白天 [7, 21) 时 / 夜间）；小区 i 按功能区 g(i) 取行，预测时刻 t+h 按情境 κ(t+h) 取列
- 合并规则：格子内分时小区少于 10 个或 se > 0.5 → 用该情境的全市估计（各功能区共用，◐）→ 仍不合格则用全市总估计（○）。所以同一列中所有"非自身格子"处于同一层级。图中格局是示意（`--real` 可改画真实格局）
- 左下（默认，截断反馈）：$\hat\beta_{g,\kappa}$、$\hat\delta_k$ 为缓冲区，pinball 损失只训练 φ（骨干与头）
- 右下（消融，联合训练）：A1 `free`：$\beta=-\mathrm{softplus}(\theta_{g,\kappa}+\xi_{i,\kappa})$，δ 由 softplus 增量保证单调非负，另加 $\lambda_h\,\overline{\xi^2}$；A2 `none`：$\beta=\theta_{g,\kappa}+\xi_{i,\kappa}$、δ 无约束；A13 `soft`：A1 再加锚定损失 $\lambda_a\mathcal{L}_{\mathrm{anc}}$（各格子分时小区的平均弹性 × 口径因子，与 $\hat\beta$ 按 1/se² 比较）。命题 1：固定时刻表下 pinball 损失不能识别 β

### 图 4(c)：距离环带溢出（真实几何）

- 示例小区 i = 525（深圳，严格分时，高密度商业区）；邻区按**质心距离**分入 [0, 2)、[2, 4)、[4, 6) km 三个环带（5 / 5 / 13 个小区，与 `ring_members` 完全一致；圆点画在 distance.csv 的精确距离上，与质心相差不超过 12 m）
- $S^{(k)}_{i,t}$ = 环带内全部邻区 $\Delta\ell_{j,t}$ 的平均；固定电价邻区 $\Delta\ell_j\equiv0$（整个样本期价格不变）；空环带时 $S=0$
- $\eta^{\mathrm{spill}}=\Sigma_k\delta_kS^{(k)}$；因果模块的原始 $\hat\delta_k$ 先经 PAV 投影到 $\delta_1\ge\delta_2\ge\delta_3\ge0$ 再使用（小图数值为示意）

### 图 4(d)：带质量掩码的自适应保形校准（示意，不按比例）

- 按目标时刻 τ 记号：$s_{\tau,i}=\max(\hat y^{(0.05)}_{\tau,i}-y_{\tau,i},\,y_{\tau,i}-\hat y^{(0.95)}_{\tau,i})/\sigma_i$，$\hat y_{\tau,i}$ 是 τ−h 时刻发出的 h 步预测，$\sigma_i$ 为小区训练期的中位绝对偏差
- $\hat q_{t,h}$：最近 168 个已揭晓目标（$t-167\le\tau\le t$）、全部小区、未冻结分数的 $\lceil(n+1)(1-\alpha_{t,h})\rceil/n$ 分位数
- 区间：$[\hat y^{\mathrm{lo}},\hat y^{\mathrm{hi}}]=[\hat y^{(0.05)}-\sigma_i\hat q_{t,h},\,\hat y^{(0.95)}+\sigma_i\hat q_{t,h}]$
- 更新：$\alpha_{t,h}=\alpha_{t-1,h}+\gamma_{\mathrm{ACI}}(\alpha-\mathrm{err}_{t,h})$，err 为未冻结小区中未被覆盖的比例；α = 0.1，$\gamma_{\mathrm{ACI}}$ = 0.005；每个步长一个 α、各小区共用；验证期预热后在测试期在线更新
- 冻结点既不产生分数，也不计入 err；单个小区冻结时 α 照常更新（只有某一时刻全部小区都冻结时才暂停）

---

## 四、创新点与图中元素的对应

| 创新点 | 图 3 中的元素 | 图 4 中的元素 |
|---|---|---|
| 价格盲基线 + 乘性价格项（反事实只改 η，推论 1） | 骨干标题 "Price-blind"，价格只进入 "Price response" 模块；电价曲线上的情景虚线 $p'$ | (a) 整个面板没有价格输入 |
| 从分时电价切换事件识别弹性（设计 D） | 底部 "Causal identification" 的四个小图：Switch events → Two-way FE → Pooling → $\hat\beta,\hat\delta\pm$se | (b) 锚定表 "TOU switch events, design D" |
| 截断反馈（模块化 / cut Bayes） | 紫色粗虚线 + 挂锁（固定缓冲区）；灰色点线 + 剪刀（梯度截断）；$\nabla_\phi\mathcal{L}_{\mathrm{pin}}$ 只回到骨干与头 | (b) 截断反馈与联合训练两个小计算图的对比 |
| 功能区 × 情境的异质弹性与合并规则 | 参数下标 $g,\kappa$；Pooling 三层树（● 格子 → ◐ 情境 → ○ 全市） | (b) 锚定表的 ● ◐ ○ 与查表箭头 |
| 真实几何上的距离环带溢出、单调约束 | 环带图标、$S^{(k)}$ 与 $\hat\delta_k$ | (c) 地图、环带成员、PAV 小图 |
| 数据质量掩码贯穿全流程 | 损失框与保形校准框中的冻结点记号（灰色 × + 红色斜线）；切换事件"未冻结"写在图注 | (d) 冻结点不计分、不计入 err |
| 口径对齐（切换前后 3 小时窗口的对数均值） | —（图注写 "prices constant for 3 h on each side"） | —（第 6 节正文与图 5 说明） |

---

## 五、建议图注（英文）

**Fig. 3.** Overall architecture of CPA-STGNN. Top (blue): a price-blind spatio-temporal backbone, a Graph WaveNet encoder with eight gated TCN and multi-graph convolution layers, encodes the last 24 h of zone utilization $y$ and a stale-data flag $\tilde m$ over three fixed graphs (adjacency, distance, POI similarity) and an adaptive graph ($A_s$), together with calendar, weather, and static and POI features ($\mathbf c,\mathbf w,\mathbf s$). A non-crossing quantile head maps the node embeddings $Z$, the future calendar and observed weather ($\mathbf c_{t+h},\mathbf w_{t+h}$) and a horizon embedding to baseline quantiles $b^{(q)}$. Middle (orange): prices enter only through the price response $\eta=\hat\beta_{g(i),\kappa(t+h)}\Delta\ell+\sum_k\hat\delta_kS^{(k)}$, where $\Delta\ell$ is the log-price deviation from the zone's training-period mean and $S^{(k)}$ averages the deviations of the neighbours in distance ring $k$ (fixed-price neighbours contribute zero). The forecast is $\hat y^{(q)}=\min\{1,\,b^{(q)}\exp(\eta)\}$, so a counterfactual forecast replaces the price path $p$ by a scenario $p'$ (dashed) and leaves $b^{(q)}$ unchanged. An optional load-shift term $\gamma U$ (off by default) is not shown. Bottom (purple): the coefficients are identified once from TOU price-switch events in the training period (prices constant for 3 h on each side, unfrozen, public holidays ±1 d excluded) by a two-way fixed-effects regression (design D: date × hour and zone × hour fixed effects, ring exposures, clustered standard errors). Cells (POI cluster $g$ × context $\kappa$) with fewer than 10 TOU zones or se > 0.5 are pooled to the context level (◐) and, if still failing, to the city level (○). The anchors enter as fixed buffers (dashed purple, lock), so the gradient of the masked pinball loss updates only the backbone and the head (dotted; scissors: cut feedback). Right (red): adaptive conformal inference updates $\alpha_{t,h}$ online and calibrates the 90% interval $[\hat y^{\mathrm{lo}},\hat y^{\mathrm{hi}}]$. Frozen (imputed) observations (crossed markers) are excluded from both the loss and the conformal scores.

**Fig. 4.** Modules of CPA-STGNN. (a) Price-blind backbone: eight gated TCN layers (kernel 2, dilations 1, 2, 4, 8, 1, 2, 4, 8), each followed by a multi-graph two-hop diffusion convolution (dropout 0.3), a residual connection and batch normalization. The skip outputs are summed at the last time step. The quantile head adds linear projections of $\mathbf z_i$ and of the future calendar and observed weather to a horizon embedding, then applies a two-layer MLP and a cumulative softplus. (b) Anchored price response: $\hat\beta$ is looked up by the POI cluster $g(i)$ (low-density, residential-mixed, high-density commercial) and the context $\kappa(t+h)$ (workday or rest day × day 07:00–21:00 or night). Cells with fewer than 10 TOU zones or se > 0.5 are pooled to the context level (shared by all clusters) and then to the city level; the cell pattern shown is illustrative. Under cut feedback (default) $\hat\beta$ and $\hat\delta$ are fixed buffers. In the joint-training ablations, $\beta=-\mathrm{softplus}(\theta_{g,\kappa}+\xi_{i,\kappa})$ with a shrinkage penalty on $\xi$ (A1), $\beta=\theta_{g,\kappa}+\xi_{i,\kappa}$ (A2), or A1 plus an anchor loss toward $\hat\beta\pm$se (A13); by Proposition 1 the pinball loss alone cannot identify $\beta$. (c) Spillover exposure for an example TOU zone in Shenzhen: neighbours are assigned to rings [0, 2), [2, 4) and [4, 6) km by centroid distance, and $S^{(k)}$ averages their price deviations, with fixed-price neighbours contributing zero. The ring coefficients are projected onto $\delta_1\ge\delta_2\ge\delta_3\ge0$ by pool-adjacent-violators (inset values are schematic). (d) Adaptive conformal calibration with the data-quality mask (schematic, not to scale). For each horizon $h$, the normalized CQR scores of the last 168 revealed targets of all zones give $\hat q_{t,h}$. $\alpha_{t,h}$ is updated from the miscoverage share of unfrozen zones with step $\gamma_{\mathrm{ACI}}=0.005$. Frozen observations produce no score and do not count in the error. The window and $\alpha$ are warmed up on the validation period.

（正文中 ACI 的步长请同样写成 $\gamma_{\mathrm{ACI}}$，与负荷转移系数 γ 区分。）

---

## 六、图元与代码的核对表

路径相对于 `03_代码/`。行号为 2026-09-28 版本。

| 图 | 图元 | 代码 |
|---|---|---|
| 3、4(a) | 输入通道 $[y,\tilde m]$；历史日历 + 天气 12 维；未来协变量 | `src/data/windows.py` `batch()` 第 41–56 行；`future_weather: oracle`（`configs/default.yaml` 第 18 行） |
| 3、4(a) | 陈旧标记（≥ 6 小时不变，只用过去信息） | `src/data/quality.py` `stale_flag()` |
| 4(a) | 1×1 起始卷积 + cov_proj + static_proj，左侧补零到 31 | `src/models/backbones/graph_wavenet.py` 第 86–90 行 |
| 4(a) | tanh ⊙ σ 门控 → 跳连接 1×1 → GCN → 残差 → BN | 同上第 94–103 行 |
| 4(a) | 跳连接最后时刻 → ReLU → 1×1 → ReLU → 1×1 | 同上第 104–106 行 |
| 4(a) | 多图扩散卷积（拼接 $[x, A_sx, A_s^2x]$ → 1×1，dropout） | 同上 `GraphConv` 第 25–33 行 |
| 4(a) | 三张固定图与自适应图 | `src/data/graphs.py` `build_supports()`；`graph_wavenet.py` `supports()` |
| 3、4(a) | 分位数头与非交叉输出 | `src/models/heads/quantile.py` 第 38–43 行；`quantiles`（default.yaml 第 52 行） |
| 3、4(b) | $\eta=\beta\Delta\ell+\Sigma\delta S\,[+\gamma U]$，$\min\{1,\cdot\}$ | `src/models/cpastgnn.py` `eta()`、`forward()` 第 66–99 行；`use_shift: false`（default.yaml 第 56 行） |
| 3、4(b) | 截断反馈：β、δ、γ 为缓冲区 | `src/models/heads/price.py` 第 34–35、78–79、105–106 行 |
| 4(b) | A1 / A2 / A13 的参数化 | `price.py` `beta_table()`、`Spillover.delta()`；`scripts/train.py` 第 118–121 行 |
| 4(b) | 锚定损失、ξ 收缩 | `src/models/losses.py` `anchor_loss()`、`hier_loss()` |
| 3、4(b) | 功能区与情境 | `src/data/clusters.py`；`src/data/calendar.py` `context_index()` |
| 3、4(b) | 合并规则 | `src/causal/anchors.py` `merge_levels()` 第 125–143 行（default.yaml 第 112–113 行：`min_zones: 10`、`max_se: 0.5`） |
| 3 | 切换事件、设计 D、环带暴露、聚类标准误 | `src/causal/switch_did.py` `build_panel()`、`estimate_design()`；`scripts/estimate_anchors.py` |
| 4(c) | 环带成员、暴露、U | `src/data/prices.py` `ring_members()`、`ring_exposure()`、`shift_feature()` |
| 4(c) | PAV 投影 | `src/causal/anchors.py` `project_monotone_nonneg()`；`price.py` 第 77 行 |
| 3、4(d) | ACI（分数、窗口、延迟、α 更新、冻结点） | `src/conformal/aci.py` 第 35–67 行；`conformal:`（default.yaml） |
| 3 | 掩码 pinball 损失 | `src/models/losses.py` `pinball_loss()` |

---

## 七、独立审查记录（2026-09-28）

由没有参与绘图的审查代理，对照代码与设计文档逐项核对图 3、图 4，共三轮（第三轮针对重绘后的图 3）：

- **第一轮发现并已修正**（这里的 ③④ 等是旧版图 3 的模块编号）：
  - 图 3、图 4(b) 把 $\hat\gamma$ 画成因果识别的输出，但代码没有识别 γ（`estimate_anchors.py` 写死 γ = 0），④ 默认关闭 → 删去，④ 的参数框改成灰色虚线表示可选
  - 图 4(d) 在单个小区冻结期间画了"α 不更新"，与代码不符（α 由所有小区共用，err 只在未冻结小区上算）→ α 连续更新，改为"frozen: excluded"
  - 图 3 与图 4(d) 的 $\hat y^{\mathrm{lo}}$、$\hat y^{\mathrm{hi}}$ 含义不一致，$\alpha_t$ 与 $\alpha_{t,h}$ 不一致 → 统一：原始区间写 $[\hat y^{(0.05)},\hat y^{(0.95)}]$，校准区间写 $[\hat y^{\mathrm{lo}},\hat y^{\mathrm{hi}}]$
  - 未来天气是观测值，没有标注 → 图 3 标 "(observed)"，图注说明
  - 图 4(b) 的锚定表直接读了小时数据的可行性锚定（其中一个格子 β̂ 为正）→ 改为示意格局，并标 "illustrative"
  - 图 4(b) 联合训练框只写了 β → 改为 "learned β, δ"，并注明命题 1
  - 图 4(c) δ 小图用了小时数据的估计值、标记形状与地图图例重复 → 改为示意值与菱形标记，单独图例
  - 图 4(d) 分数下标 j 实际是预测起点，目标在 j+h → 改为按目标时刻 τ 记号
  - 图 3 有 48 处文字小于 7 pt；二级上标（$b^{(q_1)}$、$\Sigma_{j\in\mathcal{R}_k(i)}$）实际只有 3.4 pt → 全部改为 7 pt，去掉二级上下标
  - 若干细节：校准区间图标改为单一区间带；$\hat y^{(q)}$ 标注压线；③ 的下标越出框；"no holidays" 改为 "holidays ±1 d excluded"；两图 GCN 名称统一为 "Multi-graph GCN"
- **第二轮发现并已修正**：示意锚定表的格局违反合并规则（同一列的非自身格子层级不同）→ 改为符合规则的格局，并在脚本中加自检；另外三处细节
- **第二轮确认**：所有高、中级问题已解决；Fig. 3、Fig. 4 没有小于 7 pt 的正文字，没有文字与线条冲突，PDF 字体全部嵌入
- **第三轮（图 3 按你的反馈重绘之后）**：审查代理对照代码再核一遍，**内容与代码没有出入**（输入通道、分位数头的未来协变量、$b^{(q)}\exp(\eta)$ 再截到 1、`use_shift: false`、截断模式下的缓冲区、损失与 ACI 都用 `valid` 掩码、设计 D 的固定效应、合并顺序）；你的四点要求：文字已明显减少、图标基本符合范例画法、版面基本安静、没有编号。按它的建议又改了：
  - 合并树的深浅与图 4(b) 相反 → 改为叶 ● 格子、中 ◐ 情境、根 ○ 全市
  - "Two-way FE" 原来画成事件研究系数图，而锚定值是每个格子一个窗口 DiD 斜率 → 改为残差散点 + 负斜率拟合线
  - 用雪花表示冻结点会被读成"冻结的权重" → 改用与图 4(d) 一致的灰色 × + 红色斜线
  - 损失图标由下降曲线改为 pinball 检查函数；定位图标由实心改为与日历、天气一致的描边
  - 因果识别外框原为 1.5 pt 紫色虚线，与"固定锚定值"图例几乎一样且最抢眼 → 改为 0.8 pt 实线
  - 挂锁压住了锚定表的角 → 移到两个参数块之间；锚定虚线的箭头落在参数块底边，避免被读成 β̂ 与 δ̂ 之间的连线
  - $p'$ 的虚线颜色太浅、标注离曲线远 → 虚线改用价格色，标注贴近峰段
  - 损失框移到 ⊗ 右上方的空白处并居中于红线，梯度点线不再与其他线交叉；输入箭头起点统一；⊗ → min 的箭头加长到 4 mm 以上；图快照只在最前一帧画节点；$Z$ 与 $b^{(q)}$ 同一基线
  - $\mathbf{c}_{t+h},\mathbf{w}_{t+h}$ 与 $\nabla_\phi\mathcal{L}_{\mathrm{pin}}$ 改为 7.5 pt（下标不小于 4.9 pt）；"Quantile head" 与其他模块标题同为 7.5 pt；矩阵网格线加粗到 0.3 pt
  - 图例去掉颜色条目，只留符号；"own price"、"spatial spillover"、"(training period)" 删去，写进图注
- 图 4 同时去掉了编号圆标，其余不变；两张图重新生成后自动检查 0 个问题

---

## 八、审查中发现的待决问题（与图无关，需要你决定）

1. **溢出暴露的口径不完全一致**：因果估计（`switch_did.build_panel` 第 73 行）的暴露量只用**严格分时**邻区、且只计 |Δlog p| > 1% 的跳变；模型的 $S^{(k)}$（`prices.ring_exposure`）用**全部**邻区的 Δℓ（含 27 个弱变动小区）。设计文档 16.1 写的是"完全同一口径"。差别很小（弱变动小区的价格变动中位数只有 1.6%），但两处应统一。建议阶段 6 让因果估计直接用 `ring_exposure` 算出的 S 在切换前后窗口的差
2. **`final_main.yaml` 会用到非零的 δ**：它加载 `hourly_train.json`，其中 δ 投影后为 0.45 / 0.45 / 0.18；而设计文档 16.3 写"溢出系数暂不用于模型，默认 δ = 0"。二选一：主实验改用 δ = 0 的锚定文件（或关闭空间溢出项），或者修改文档说明使用设计 D 的 δ
3. **合并规则没有符号检查**：功能区 2 × 休息日夜间 β̂ = +0.023（se 0.25）仍按格子层级保留，加载时只给警告。阶段 6 定稿前决定正号格子是否并入上一层级
4. **全市层级是最后的兜底**：即使其 se（0.51）也超过 0.5 阈值仍然使用。这是设计如此，论文中说明即可
5. **γ 没有识别设计**：负荷转移项 γU 目前只是可选结构（图 3 不画）；若要启用，需要补充 γ 的识别方法（例如切换前的预期性调整）
