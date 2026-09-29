"""ASTGCN（Guo et al. 2019, AAAI）编码器，只用"最近时段"分量（UrbanEV 官方通过 torch_geometric_temporal 调用的也是单分量）。

每个块 = 时间注意力 → 空间注意力 → 带空间注意力的切比雪夫图卷积 → 时间卷积 → 残差 + LayerNorm。
按原文实现，不依赖 torch_geometric_temporal；原文用 3 个时间分量并融合，这里只有最近分量。
默认 2 个块、K = 3、64 个滤波器（原文设置；UrbanEV 官方用 1 个块、K = 1、32 个滤波器，可在配置里改）。
最后用覆盖全部时间步的卷积得到 D 维表示，后接统一分位数头。
参数初始化按官方代码（build_baseline 中的 official_init：二维以上 xavier_uniform，一维 U(0, 1)）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.graphs import cheb_polynomials, scaled_laplacian


class TemporalAttention(nn.Module):
    def __init__(self, N: int, C: int, T: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.randn(N) * 0.1)
        self.U2 = nn.Parameter(torch.randn(C, N) * 0.1)
        self.U3 = nn.Parameter(torch.randn(C) * 0.1)
        self.be = nn.Parameter(torch.zeros(1, T, T))
        self.Ve = nn.Parameter(torch.randn(T, T) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, C, T] → E [B, T, T]（按第 1 维归一化，与原文一致）。"""
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)       # [B, T, N]
        rhs = torch.matmul(self.U3, x)                                                  # [B, N, T]
        e = torch.matmul(self.Ve, torch.sigmoid(torch.matmul(lhs, rhs) + self.be))
        return F.softmax(e, dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, N: int, C: int, T: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(T) * 0.1)
        self.W2 = nn.Parameter(torch.randn(C, T) * 0.1)
        self.W3 = nn.Parameter(torch.randn(C) * 0.1)
        self.bs = nn.Parameter(torch.zeros(1, N, N))
        self.Vs = nn.Parameter(torch.randn(N, N) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, C, T] → S [B, N, N]"""
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)                           # [B, N, T]
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)                                # [B, T, N]
        s = torch.matmul(self.Vs, torch.sigmoid(torch.matmul(lhs, rhs) + self.bs))
        return F.softmax(s, dim=1)


class ChebConvWithSAt(nn.Module):
    def __init__(self, cheb: torch.Tensor, c_in: int, c_out: int):
        super().__init__()
        self.register_buffer("cheb", cheb)
        K = cheb.shape[0]
        self.theta = nn.Parameter(torch.empty(K, c_in, c_out))
        nn.init.xavier_uniform_(self.theta.view(K * c_in, c_out))

    def forward(self, x: torch.Tensor, s_att: torch.Tensor) -> torch.Tensor:
        """x [B, N, C, T]，s_att [B, N, N] → [B, N, C_out, T]；每阶：(T_k ⊙ S)ᵀ x Θ_k。"""
        out = 0
        for k in range(self.cheb.shape[0]):
            tk = (self.cheb[k][None] * s_att).transpose(1, 2)                           # [B, N, N]
            out = out + torch.einsum("bnm,bmct,co->bnot", tk, x, self.theta[k])
        return F.relu(out)


class ASTGCNBlock(nn.Module):
    def __init__(self, N: int, c_in: int, T: int, cheb: torch.Tensor, chev_filter: int, time_filter: int):
        super().__init__()
        self.tat = TemporalAttention(N, c_in, T)
        self.sat = SpatialAttention(N, c_in, T)
        self.gcn = ChebConvWithSAt(cheb, c_in, chev_filter)
        self.time_conv = nn.Conv2d(chev_filter, time_filter, (1, 3), padding=(0, 1))
        self.res_conv = nn.Conv2d(c_in, time_filter, (1, 1))
        self.norm = nn.LayerNorm(time_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, C, T] → [B, N, F, T]"""
        B, N, C, T = x.shape
        e = self.tat(x)
        x_tat = torch.matmul(x.reshape(B, N * C, T), e).reshape(B, N, C, T)
        g = self.gcn(x, self.sat(x_tat))                                                # [B, N, F', T]
        t = self.time_conv(g.permute(0, 2, 1, 3))                                       # [B, F, N, T]
        r = self.res_conv(x.permute(0, 2, 1, 3))
        y = F.relu(r + t).permute(0, 3, 2, 1)                                           # [B, T, N, F]
        return self.norm(y).permute(0, 2, 3, 1)                                         # [B, N, F, T]


class ASTGCNEncoder(nn.Module):
    def __init__(self, L: int, C: int, D: int, N: int, adj: np.ndarray, K: int = 3, chev_filter: int = 64,
                 time_filter: int = 64, blocks: int = 2):
        super().__init__()
        cheb = torch.tensor(cheb_polynomials(scaled_laplacian(adj), K))
        mods, c = [], C
        for _ in range(blocks):
            mods.append(ASTGCNBlock(N, c, L, cheb, chev_filter, time_filter))
            c = time_filter
        self.blocks = nn.ModuleList(mods)
        self.final = nn.Conv2d(L, D, (1, time_filter))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.permute(0, 2, 3, 1)                                                       # [B, N, C, L]
        for blk in self.blocks:
            x = blk(x)
        return self.final(x.permute(0, 3, 1, 2))[..., 0].transpose(1, 2)                # [B, N, D]
