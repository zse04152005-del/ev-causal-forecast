"""UrbanEV 数据集论文（Li et al. 2025, Scientific Data）官方基线的编码器移植。

来源：github.com/IntelligentSystemsLab/UrbanEV，code/baselines.py 与 utils.load_net 中的超参数。
与官方代码的差别（全部为适配统一协议，编码器结构不变）：
1. 官方输出层是 Linear(…→1)，只预测第 pred_len 步；这里改为 Linear(…→D)，后接统一的 H 步分位数头
2. 官方 GCN-LSTM 在多特征输入时 view 维度不一致（n_fea > 1 会报错）；这里先按官方 Conv2d 把特征压成 1 维，再送入 LSTM 与 GCN
3. 邻接矩阵用对称化后加自环的 D^{-1/2}(A+I)D^{-1/2}（官方直接用 adj.csv，含自环但不对称）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ...data.graphs import sym_norm_adj


def _per_node(X: torch.Tensor) -> torch.Tensor:
    """[B, L, N, C] → [B, N, L, C]"""
    return X.permute(0, 2, 1, 3)


class FCNNEncoder(nn.Module):
    """官方 Fcnn：把 L×C 个输入拉平后做一次线性变换（各小区共享权重）。"""

    def __init__(self, L: int, C: int, D: int):
        super().__init__()
        self.linear = nn.Linear(L * C, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, C = X.shape
        return self.linear(_per_node(X).reshape(B, N, L * C))


class LSTMEncoder(nn.Module):
    """官方 Lstm：各小区共享的 2 层 LSTM（隐藏 16），全部时刻输出拉平后线性变换。"""

    def __init__(self, L: int, C: int, D: int, hidden: int = 16, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, num_layers=layers, batch_first=True)
        self.linear = nn.Linear(L * hidden, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, C = X.shape
        out, _ = self.lstm(_per_node(X).reshape(B * N, L, C))
        return self.linear(out.reshape(B, N, -1))


class GCNEncoder(nn.Module):
    """官方 Gcn：Conv2d(N, N, (1, C)) 压缩特征 → [Linear(L→hidden) → Â· → ReLU] × layers → 与需求历史拼接 → 线性。"""

    def __init__(self, L: int, C: int, D: int, N: int, adj: np.ndarray, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.register_buffer("A", torch.tensor(sym_norm_adj(adj)))
        self.encoder = nn.Conv2d(N, N, (1, C))
        self.layers = nn.ModuleList([nn.Linear(L if i == 0 else hidden, hidden) for i in range(layers)])
        self.act = nn.ReLU()
        self.decoder = nn.Linear(hidden + L, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, C = X.shape
        x = _per_node(X)                                   # [B, N, L, C]：官方把节点当作 Conv2d 的通道
        g = self.encoder(x).reshape(B, N, L)
        for lin in self.layers:
            g = self.act(self.A @ lin(g))
        return self.decoder(torch.cat([x[..., 0], g], dim=-1))


class GCNLSTMEncoder(nn.Module):
    """官方 Gcnlstm（load_net：gcn_out 32、gcn_layers 1、lstm_hidden 32、lstm_layers 1）：
    Conv2d 压缩特征 → LSTM 最后隐状态 + GCN 输出 + 需求历史 → 线性。"""

    def __init__(self, L: int, C: int, D: int, N: int, adj: np.ndarray, gcn_out: int = 32, gcn_layers: int = 1,
                 lstm_hidden: int = 32, lstm_layers: int = 1):
        super().__init__()
        self.register_buffer("A", torch.tensor(sym_norm_adj(adj)))
        self.encoder = nn.Conv2d(N, N, (1, C))
        self.gcn = nn.ModuleList([nn.Linear(L if i == 0 else gcn_out, gcn_out) for i in range(gcn_layers)])
        self.lstm = nn.LSTM(1, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(L + gcn_out + lstm_hidden, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, C = X.shape
        x = _per_node(X)
        h = self.encoder(x).reshape(B, N, L)
        out, _ = self.lstm(h.reshape(B * N, L, 1))
        last = out[:, -1].reshape(B, N, -1)
        g = h
        for lin in self.gcn:
            g = self.act(self.A @ lin(g))
        return self.decoder(torch.cat([x[..., 0], last, g], dim=-1))
