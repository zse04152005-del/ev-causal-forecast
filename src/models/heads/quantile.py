"""模块⑤a：非交叉分位数基线头（设计文档 5.2）。

u = MLP([z_i ; e(c_{t+h}) ; e_h]) ∈ R^Q，b^(q1) = softplus(u1)，b^(qm) = b^(q_{m-1}) + softplus(um)。
输出恒为正且随分位数水平单调递增。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    return y + math.log(-math.expm1(-y))


class QuantileHead(nn.Module):
    def __init__(self, D: int, cov_fut_dim: int, H: int, Q: int, hidden: int = 64, init_quantiles=None,
                 extra_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.z = nn.Linear(D, hidden)
        self.cov = nn.Linear(cov_fut_dim, hidden)
        self.hemb = nn.Embedding(H, hidden)
        self.extra = nn.Linear(extra_dim, hidden) if extra_dim else None
        self.mlp = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, Q))
        last = self.mlp[-1]
        nn.init.normal_(last.weight, std=0.01)
        if init_quantiles is not None:
            q = [max(float(v), 1e-3) for v in init_quantiles]
            inc = [q[0]] + [max(q[m] - q[m - 1], 1e-3) for m in range(1, len(q))]
            with torch.no_grad():
                last.bias.copy_(torch.tensor([inv_softplus(v) for v in inc]))

    def forward(self, z: torch.Tensor, cov_fut: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """z [B, N, D]；cov_fut [B, H, C]；extra [B, H, N, E] → [B, H, N, Q]。"""
        h = self.z(z)[:, None] + self.cov(cov_fut)[:, :, None] + self.hemb.weight[None, :, None]
        if self.extra is not None and extra is not None:
            h = h + self.extra(extra)
        return torch.cumsum(F.softplus(self.mlp(h)), dim=-1)
