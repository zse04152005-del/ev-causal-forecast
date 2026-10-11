"""STGCN（Yu, Yin & Zhu 2018, IJCAI）编码器：切比雪夫图卷积版本，按官方 TF 代码（github.com/VeritasYin/STGCN_IJCAI-18）实现。

ST-Conv 块 = 门控时间卷积（GLU）→ 带残差的切比雪夫图卷积 + ReLU → 带残差的时间卷积 + ReLU → LayerNorm → Dropout；
输出层 = 覆盖剩余时间长度的 GLU 时间卷积 → LayerNorm → 1×1 时间卷积 + sigmoid → 全连接。
（官方代码第二个时间卷积用 ReLU、图卷积带残差、输出层多一个 sigmoid 卷积；论文图 2 画的是两个 GLU。这里以官方代码为准。）
与官方的差别：全连接层输出 D 维表示而不是预测值，后接统一分位数头。
通道 (c_t, c_s, c_o)：时间卷积 → c_t，图卷积 → c_s，时间卷积 → c_o；默认 [64, 16, 64]（论文设置；官方代码为
[[1, 32, 64], [64, 32, 128]]，即 c_s = c_t）。Kt = 3、Ks = 3。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ...data.graphs import cheb_polynomials, scaled_laplacian


class Align(nn.Module):
    """通道对齐：c_in > c_out 用 1×1 卷积，c_in < c_out 补零。"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_out = c_out
        self.conv = nn.Conv2d(c_in, c_out, 1) if c_in > c_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            return self.conv(x)
        if x.shape[1] < self.c_out:
            pad = torch.zeros(x.shape[0], self.c_out - x.shape[1], *x.shape[2:], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        return x


class TemporalConv(nn.Module):
    """x [B, C, T, N] → [B, C_out, T − Kt + 1, N]；act ∈ {glu, relu, sigmoid}（与官方 temporal_conv_layer 相同）。"""

    def __init__(self, c_in: int, c_out: int, Kt: int, act: str = "relu"):
        super().__init__()
        if act not in ("glu", "relu", "sigmoid"):
            raise ValueError(act)
        self.Kt, self.c_out, self.act = Kt, c_out, act
        self.conv = nn.Conv2d(c_in, 2 * c_out if act == "glu" else c_out, (Kt, 1))
        self.align = Align(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = self.conv(x)
        if self.act == "sigmoid":
            return torch.sigmoid(xc)
        res = self.align(x)[:, :, self.Kt - 1:, :]
        if self.act == "glu":
            p, q = xc.chunk(2, dim=1)
            return (p + res) * torch.sigmoid(q)
        return torch.relu(xc + res)


class SpatialConv(nn.Module):
    """x [B, C, T, N] → [B, C_out, T, N]：ReLU(Σ_k θ_k T_k(L̃) x + 对齐后的 x)。"""

    def __init__(self, c_in: int, c_out: int, cheb: torch.Tensor):
        super().__init__()
        self.register_buffer("cheb", cheb)                               # [K, N, N]
        K = cheb.shape[0]
        self.theta = nn.Parameter(torch.empty(K, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.theta.view(K * c_in, c_out))
        self.align = Align(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = torch.einsum("knm,bctm->bkctn", self.cheb, x)
        g = torch.einsum("bkctn,kco->botn", xk, self.theta) + self.bias[None, :, None, None]
        return torch.relu(g + self.align(x))


class STBlock(nn.Module):
    def __init__(self, c_in: int, channels, Kt: int, cheb: torch.Tensor, N: int, dropout: float):
        super().__init__()
        c_t, c_s, c_o = channels
        self.t1 = TemporalConv(c_in, c_t, Kt, "glu")
        self.sc = SpatialConv(c_t, c_s, cheb)
        self.t2 = TemporalConv(c_s, c_o, Kt, "relu")
        self.norm = nn.LayerNorm([N, c_o])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t2(self.sc(self.t1(x)))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.drop(x)


class STGCNEncoder(nn.Module):
    def __init__(self, L: int, C: int, D: int, N: int, adj: np.ndarray, Kt: int = 3, Ks: int = 3,
                 blocks=((64, 16, 64), (64, 16, 64)), dropout: float = 0.1):
        super().__init__()
        cheb = torch.tensor(cheb_polynomials(scaled_laplacian(adj), Ks))
        mods, c = [], C
        for ch in blocks:
            mods.append(STBlock(c, ch, Kt, cheb, N, dropout))
            c = ch[-1]
        self.blocks = nn.ModuleList(mods)
        t_left = L - 2 * (Kt - 1) * len(blocks)
        if t_left < 1:
            raise ValueError(f"STGCN：输入长度 L={L} 太短（{len(blocks)} 个块、Kt={Kt} 共消耗 {L - t_left} 步）")
        self.out_t = TemporalConv(c, c, t_left, "glu")
        self.out_norm = nn.LayerNorm([N, c])
        self.out_sig = TemporalConv(c, c, 1, "sigmoid")
        self.out_fc = nn.Conv2d(c, D, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.permute(0, 3, 1, 2)                                          # [B, C, L, N]
        for blk in self.blocks:
            x = blk(x)
        x = self.out_t(x)                                                  # [B, c, 1, N]
        x = self.out_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.out_sig(x)
        return self.out_fc(x)[:, :, 0].transpose(1, 2)                     # [B, N, D]
