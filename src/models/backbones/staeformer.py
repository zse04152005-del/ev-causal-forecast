"""更换骨干消融（A10）：STAEformer 编码器（Liu et al., CIKM 2023）的精简实现。

输入嵌入 + 时刻嵌入 + 星期嵌入 + 自适应时空嵌入 → 时间注意力 × layers → 空间注意力 × layers
→ 把每个节点的 L 个时间步展平后线性映射为节点表示 Z [B, N, D]。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _SelfAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [batch, seq, d]
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))


class STAEformerEncoder(nn.Module):
    def __init__(self, n_nodes: int, L: int, c_in: int, cov_dim: int, static: torch.Tensor, D: int = 32,
                 d_model: int = 24, tod_dim: int = 24, dow_dim: int = 24, adaptive_dim: int = 80, heads: int = 4,
                 layers: int = 3, ff_dim: int = 256, dropout: float = 0.1, **_):
        super().__init__()
        self.register_buffer("static", static.float())
        self.inp = nn.Linear(c_in, d_model)
        self.cov = nn.Linear(cov_dim, d_model)
        self.tod = nn.Embedding(24, tod_dim)
        self.dow = nn.Embedding(7, dow_dim)
        self.static_proj = nn.Linear(static.shape[1], d_model)
        self.adaptive = nn.Parameter(torch.empty(L, n_nodes, adaptive_dim))
        nn.init.xavier_uniform_(self.adaptive)
        dm = d_model + tod_dim + dow_dim + adaptive_dim
        heads = heads if dm % heads == 0 else 1
        self.t_layers = nn.ModuleList([_SelfAttnBlock(dm, heads, ff_dim, dropout) for _ in range(layers)])
        self.s_layers = nn.ModuleList([_SelfAttnBlock(dm, heads, ff_dim, dropout) for _ in range(layers)])
        self.out = nn.Linear(L * dm, D)
        self.L, self.dm, self.out_dim = L, dm, D

    def forward(self, x: torch.Tensor, cov_hist: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor, **_) -> torch.Tensor:
        B, L, N, _ = x.shape
        h = self.inp(x) + self.cov(cov_hist).unsqueeze(2) + self.static_proj(self.static)[None, None]
        te = self.tod(tod).unsqueeze(2).expand(B, L, N, -1)
        de = self.dow(dow).unsqueeze(2).expand(B, L, N, -1)
        ae = self.adaptive[None].expand(B, -1, -1, -1)
        h = torch.cat([h, te, de, ae], dim=-1)                                  # [B, L, N, dm]
        t = h.permute(0, 2, 1, 3).reshape(B * N, L, self.dm)
        for blk in self.t_layers:
            t = blk(t)
        s = t.reshape(B, N, L, self.dm).permute(0, 2, 1, 3).reshape(B * L, N, self.dm)
        for blk in self.s_layers:
            s = blk(s)
        s = s.reshape(B, L, N, self.dm).permute(0, 2, 1, 3).reshape(B, N, L * self.dm)
        return self.out(s)                                                     # [B, N, D]
