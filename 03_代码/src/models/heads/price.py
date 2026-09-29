"""价格项：模块② 本地价格响应、模块③ 空间溢出、模块④ 负荷转移（设计文档 5.3–5.5）。

四种锚定方式（设计文档 3.4，决策 D7）：
- cut  ：截断反馈（默认）。β、δ、γ 固定为因果估计值，是缓冲区而非参数，预测损失不改变它们
- soft ：软锚定。β = −softplus(θ_g,κ + ξ_i,κ)，另加锚定损失（对照 A13）
- free ：只保留符号约束 β ≤ 0，不加锚定损失（E-ID 实验、消融 A1）
- none ：连符号约束都没有（消融 A2）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...causal.anchors import AnchorSet, project_monotone_nonneg
from .quantile import inv_softplus

MODES = ("cut", "soft", "free", "none")


class PriceResponse(nn.Module):
    def __init__(self, groups: np.ndarray, anchors: AnchorSet, mode: str = "cut", n_ctx: int = 4,
                 init_beta: float | None = None, lag_weights=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"未知锚定方式：{mode}")
        self.mode = mode
        self.register_buffer("groups", torch.tensor(np.array(groups), dtype=torch.long))
        N = len(groups)
        b0 = np.asarray(anchors.beta, dtype=np.float64)
        if init_beta is not None:
            b0 = np.full_like(b0, float(init_beta))
        if mode == "cut":
            self.register_buffer("beta_gc", torch.tensor(b0, dtype=torch.float32))
        elif mode in ("soft", "free"):
            th = np.vectorize(lambda b: inv_softplus(max(-b, 1e-3)))(b0)
            self.theta = nn.Parameter(torch.tensor(th, dtype=torch.float32))
            self.xi = nn.Parameter(torch.zeros(N, n_ctx))
        else:
            self.theta = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
            self.xi = nn.Parameter(torch.zeros(N, n_ctx))
        if lag_weights is not None:
            w = torch.as_tensor(np.asarray(lag_weights, dtype=np.float64) / np.sum(lag_weights), dtype=torch.float32)
            self.register_buffer("lag_w", w)
        else:
            self.lag_w = None

    @property
    def n_lags(self) -> int:
        return 0 if self.lag_w is None else len(self.lag_w) - 1

    def beta_table(self) -> torch.Tensor:
        """[N, C] 每个小区、每个情境的弹性。"""
        if self.mode == "cut":
            return self.beta_gc[self.groups]
        if self.mode in ("soft", "free"):
            return -F.softplus(self.theta[self.groups] + self.xi)
        return self.theta[self.groups] + self.xi

    def forward(self, dl: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """dl [B, H, N]（有滞后核时为 [B, H+M, N]）；ctx [B, H] → η_own [B, H, N]。"""
        H = ctx.shape[1]
        beta = self.beta_table().T[ctx]                                   # [B, H, N]
        if self.lag_w is not None:
            M = self.n_lags
            dl = sum(self.lag_w[m] * dl[:, M - m: M - m + H] for m in range(M + 1))
        return beta * dl


class Spillover(nn.Module):
    """δ1 ≥ δ2 ≥ … ≥ δK ≥ 0：邻区涨价使本区需求上升，且越近影响越大。"""

    def __init__(self, anchors: AnchorSet, mode: str = "cut"):
        super().__init__()
        self.mode = mode
        d0 = project_monotone_nonneg(np.asarray(anchors.delta, dtype=np.float64))
        if mode == "cut":
            self.register_buffer("delta_fixed", torch.tensor(d0, dtype=torch.float32))
        elif mode in ("soft", "free"):
            inc = np.append(d0[:-1] - d0[1:], d0[-1])                   # 从外向内的增量
            self.omega = nn.Parameter(torch.tensor([inv_softplus(max(v, 1e-4)) for v in inc], dtype=torch.float32))
        else:
            self.delta_free = nn.Parameter(torch.tensor(np.asarray(anchors.delta, dtype=np.float64), dtype=torch.float32))

    def delta(self) -> torch.Tensor:
        if self.mode == "cut":
            return self.delta_fixed
        if self.mode in ("soft", "free"):
            return torch.flip(torch.cumsum(torch.flip(F.softplus(self.omega), [0]), 0), [0])
        return self.delta_free

    def forward(self, spill: torch.Tensor) -> torch.Tensor:
        """spill [B, H, N, K] → [B, H, N]。"""
        return (spill * self.delta()).sum(-1)


class Shift(nn.Module):
    """预期性负荷转移：η = γ · (Δℓ_{t+1} − Δℓ_t)，γ ≥ 0。"""

    def __init__(self, anchors: AnchorSet, mode: str = "cut"):
        super().__init__()
        self.mode = mode
        g0 = max(float(anchors.gamma), 0.0)
        if mode == "cut":
            self.register_buffer("gamma_fixed", torch.tensor(g0))
        elif mode in ("soft", "free"):
            self.omega = nn.Parameter(torch.tensor(inv_softplus(max(g0, 1e-4))))
        else:
            self.gamma_free = nn.Parameter(torch.tensor(float(anchors.gamma)))

    def gamma(self) -> torch.Tensor:
        if self.mode == "cut":
            return self.gamma_fixed
        if self.mode in ("soft", "free"):
            return F.softplus(self.omega)
        return self.gamma_free

    def forward(self, shift: torch.Tensor) -> torch.Tensor:
        return self.gamma() * shift
