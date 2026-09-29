"""AGCRN（Bai et al. 2020, NeurIPS）编码器。

按官方实现（github.com/LeiBAI/AGCRN）：节点嵌入 E 生成自适应图 softmax(ReLU(EEᵀ))，节点自适应参数（NAPL）的图卷积嵌入 GRU 门控；
默认 2 层、隐藏 64、嵌入维 10、cheb_k = 2（官方 PEMS 设置）。官方输出层把最后隐状态用卷积映射到 H 步，这里改为映射到 D 维表示，
后接统一分位数头。不使用预定义图（原文设计如此）。
参数初始化按官方 Run.py（build_baseline 中的 official_init：二维以上 xavier_uniform，一维 U(0, 1)，含节点嵌入 E）。设计文档第 10 节的"即插即用验证"也用这个编码器。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, c_in, c_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, c_out))
        nn.init.xavier_normal_(self.weights_pool.view(embed_dim * cheb_k * c_in, c_out))
        nn.init.xavier_normal_(self.bias_pool)

    def forward(self, x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """x [B, N, C]，E [N, d] → [B, N, C_out]"""
        N = E.shape[0]
        s = F.softmax(F.relu(E @ E.T), dim=1)
        sup = [torch.eye(N, device=x.device, dtype=x.dtype), s]
        for _ in range(2, self.cheb_k):
            sup.append(2 * s @ sup[-1] - sup[-2])
        sup = torch.stack(sup[: self.cheb_k])                                          # [K, N, N]
        w = torch.einsum("nd,dkio->nkio", E, self.weights_pool)
        b = E @ self.bias_pool
        xg = torch.einsum("knm,bmc->bnkc", sup, x)
        return torch.einsum("bnki,nkio->bno", xg, w) + b


class AGCRNCell(nn.Module):
    def __init__(self, c_in: int, hidden: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(c_in + hidden, 2 * hidden, cheb_k, embed_dim)
        self.update = AVWGCN(c_in + hidden, hidden, cheb_k, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.gate(torch.cat([x, state], dim=-1), E))
        z, r = torch.split(zr, self.hidden, dim=-1)
        hc = torch.tanh(self.update(torch.cat([x, z * state], dim=-1), E))
        return r * state + (1 - r) * hc


class AGCRNEncoder(nn.Module):
    def __init__(self, L: int, C: int, D: int, N: int, hidden: int = 64, layers: int = 2, embed_dim: int = 10,
                 cheb_k: int = 2):
        super().__init__()
        self.E = nn.Parameter(torch.randn(N, embed_dim))
        self.cells = nn.ModuleList([AGCRNCell(C if i == 0 else hidden, hidden, cheb_k, embed_dim)
                                    for i in range(layers)])
        self.out = nn.Linear(hidden, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = X.shape
        seq = X
        for cell in self.cells:
            h = torch.zeros(B, N, cell.hidden, device=X.device, dtype=X.dtype)
            outs = []
            for t in range(L):
                h = cell(seq[:, t], h, self.E)
                outs.append(h)
            seq = torch.stack(outs, dim=1)
        return self.out(seq[:, -1])
