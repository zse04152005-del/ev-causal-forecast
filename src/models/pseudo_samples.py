"""PAG（Qu et al. 2024, IEEE T-ITS）的"物理先验伪样本"，按论文公式 (8)、(9) 实现（纯 numpy，云端可测）。

公式 (8)：本区 i 的价格相对变化 Δp_i/p_i 服从正态分布，需求响应 Δy_i = ε · (Δp_i/p_i) · y_i（ε 为注入的先验弹性）
公式 (9)：邻区 j ∈ N(i) 的响应 Δy_j = −Δy_i / |N(i)|（本区需求减少的部分平均转移到邻区）
伪样本 = 历史价格整段乘以 (1 + Δp/p)，目标需求加上 Δy（截断到 [0, clip_max]）。

与发布代码（github.com/IntelligentSystemsLab/ST-EVCDP，functions.PseudoDataset）的差别，论文中需要说明：
- 发布代码 self.law = −law 后又用 label_chg = self.law · prc_chg，注入 −1.48 时本区直接项为 +1.48（涨价、本区需求上升），
  计入两跳回流后本区净系数在 ST-EVCDP 图上为 +1.07～+1.48；这里按论文公式 (8) 取 ε 本身
- 发布代码对整个伪标签取 tan(·)（包括未受价格冲击的小区：0.8 → 1.03），这里不做
- 发布代码的邻区响应沿图传播两跳，按接收小区含自环的度归一（需求总量不守恒）；这里按论文公式 (9) 只传一跳、按 |N(i)| 均分
- 发布代码另一种 simplified 模式（CreateFastDataset）用 label_chg / self.law，隐含弹性约 −1/1.48 ≈ −0.68（量级颠倒），
  且只保留涨幅 > 60% 的冲击（约 11.5% 的小区）
- 发布代码不截断价格冲击（约 2.3% 的受冲击小区价格变为负），这里截断在 −0.9
- 发布代码每个数据集只抽一次价格冲击；这里每个批次重新抽取（论文说"一组正态分布的冲击"）
价格冲击的比例与幅度沿用发布代码：40% 的小区，Δp/p ~ N(0, 0.5²)，并截断在 (−0.9, +∞) 以保证价格为正。
"""
from __future__ import annotations

import numpy as np


def draw_price_shocks(N: int, rng: np.random.Generator, prop: float = 0.4, sd: float = 0.5,
                      eligible: np.ndarray | None = None) -> np.ndarray:
    """返回各小区价格相对变化 Δp/p [N]；不受冲击的小区为 0。eligible 限定可被冲击的小区（默认全部）。"""
    hit = rng.random(N) < prop
    if eligible is not None:
        hit &= np.asarray(eligible, dtype=bool)
    chg = np.where(hit, rng.normal(0.0, sd, N), 0.0)
    return np.maximum(chg, -0.9)


def pseudo_targets(y: np.ndarray, shocks: np.ndarray, law: float, adj: np.ndarray,
                   clip_max: float | None = 1.0) -> np.ndarray:
    """y [..., N]（未来需求）→ 伪目标 [..., N]：公式 (8) 本区响应 + 公式 (9) 一跳邻区转移。"""
    a = (np.asarray(adj) > 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    dy_own = law * shocks * y                                                  # 公式 (8)
    share = np.divide(a, deg[:, None], out=np.zeros_like(a), where=deg[:, None] > 0)   # share[i, j] = 1/|N(i)|
    dy_nb = -np.einsum("...i,ij->...j", dy_own, share)                         # 公式 (9)
    out = np.maximum(y + dy_own + dy_nb, 0.0)
    if clip_max is not None:
        out = np.minimum(out, clip_max)
    return out


def mix_rows(n: int, gamma: float) -> np.ndarray:
    """批内哪些行换成伪样本：后 round(γ·n) 行（与发布代码 data_mix 一样按行混合；γ 为伪样本比例）。"""
    k = int(round(float(np.clip(gamma, 0.0, 1.0)) * n))
    m = np.zeros(n, dtype=bool)
    if k:
        m[n - k:] = True
    return m
