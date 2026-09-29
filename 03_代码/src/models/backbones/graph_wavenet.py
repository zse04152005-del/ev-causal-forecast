"""价格盲时空骨干：Graph WaveNet 编码器（Wu et al., IJCAI 2019；设计文档 5.1）。

与原版的区别：只作编码器，输出节点表示 Z [B, N, D]，预测交给分位数头。
张量布局沿用原版 [B, C, N, T]；时间卷积核 (1, k) 作用在最后一维。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """多图扩散卷积：对每个支撑矩阵做 1..order 步扩散，与自身拼接后 1×1 卷积。

    out_i = Σ_j A[i, j] x_j（A 行归一化，行 = 接收方）。
    """

    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mlp = nn.Conv2d(c_in * (n_supports * order + 1), c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        out = [x]
        for A in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("bcjt,ij->bcit", h, A)
                out.append(h)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, self.training)


class GraphWaveNetEncoder(nn.Module):
    def __init__(self, n_nodes: int, c_in: int, cov_dim: int, static: torch.Tensor, supports: list, D: int = 32,
                 dilations=(1, 2, 4, 8, 1, 2, 4, 8), kernel_size: int = 2, order: int = 2, dropout: float = 0.3,
                 adaptive: bool = True, adaptive_dim: int = 10, skip_dim: int | None = None,
                 end_dim: int | None = None):
        super().__init__()
        skip_dim = skip_dim or 4 * D
        end_dim = end_dim or 8 * D
        self.N = n_nodes
        self.register_buffer("static", static.float())                      # [N, S]
        self.register_buffer("supports_fixed", torch.stack([s.float() for s in supports])
                             if supports else torch.zeros(0, n_nodes, n_nodes))
        self.adaptive = adaptive
        if adaptive:
            self.E1 = nn.Parameter(torch.randn(n_nodes, adaptive_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(n_nodes, adaptive_dim) * 0.1)
        n_sup = len(supports) + int(adaptive)

        self.start = nn.Conv2d(c_in, D, kernel_size=(1, 1))
        self.cov_proj = nn.Linear(cov_dim, D)
        self.static_proj = nn.Linear(static.shape[1], D)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.bns = nn.ModuleList()
        for d in dilations:
            self.filters.append(nn.Conv2d(D, D, kernel_size=(1, kernel_size), dilation=(1, d)))
            self.gates.append(nn.Conv2d(D, D, kernel_size=(1, kernel_size), dilation=(1, d)))
            self.skips.append(nn.Conv2d(D, skip_dim, kernel_size=(1, 1)))
            if n_sup > 0:
                self.gconvs.append(GraphConv(D, D, n_sup, order, dropout))
            else:
                self.residuals.append(nn.Conv2d(D, D, kernel_size=(1, 1)))
            self.bns.append(nn.BatchNorm2d(D))
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        self.end1 = nn.Conv2d(skip_dim, end_dim, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(end_dim, D, kernel_size=(1, 1))
        self.out_dim = D

    def supports(self) -> list:
        sup = list(self.supports_fixed.unbind(0))
        if self.adaptive:
            sup.append(F.softmax(F.relu(self.E1 @ self.E2.T), dim=1))
        return sup

    def forward(self, x: torch.Tensor, cov_hist: torch.Tensor, **_) -> torch.Tensor:
        """x [B, L, N, C_in]；cov_hist [B, L, cov_dim] → Z [B, N, D]。"""
        B, L, N, _ = x.shape
        h = self.start(x.permute(0, 3, 2, 1))                                  # [B, D, N, L]
        h = h + self.cov_proj(cov_hist).permute(0, 2, 1).unsqueeze(2)          # 广播到各节点
        h = h + self.static_proj(self.static).T.unsqueeze(0).unsqueeze(-1)     # 广播到各时刻
        if L < self.receptive_field:
            h = F.pad(h, (self.receptive_field - L, 0))
        sup = self.supports()
        skip = None
        for i in range(len(self.filters)):
            res = h
            h = torch.tanh(self.filters[i](h)) * torch.sigmoid(self.gates[i](h))
            s = self.skips[i](h)
            skip = s if skip is None else s + skip[..., -s.size(3):]
            if self.gconvs:
                h = self.gconvs[i](h, sup)
            else:
                h = self.residuals[i](h)
            h = h + res[..., -h.size(3):]
            h = self.bns[i](h)
        z = F.relu(skip[..., -1:])
        z = self.end2(F.relu(self.end1(z)))                                    # [B, D, N, 1]
        return z.squeeze(-1).permute(0, 2, 1)                                  # [B, N, D]
