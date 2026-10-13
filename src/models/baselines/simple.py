"""Simple strong baselines added in the second review round: STID and DLinear (encoders for the common protocol).

STID (Shao et al., CIKM 2022, "Spatial-temporal identity"): a multilayer perceptron on the flattened history plus
learnable identity embeddings. The original uses embeddings of the node, the time of day and the day of week of the
last input step; here the node embedding is learned and the time of day / day of week enter through an embedding of
the calendar covariates of the last step (hour and weekday as sin/cos, holiday and make-up workday flags), which carry
the same information. Default sizes follow the original (embedding 32, 3 MLP layers with residual connections).

DLinear (Zeng et al., AAAI 2023): the demand history of each zone is split into a moving-average trend and a remainder,
and each part is mapped linearly from L inputs to the representation; weights are shared across zones.
In the original the linear maps produce the H forecasts directly; here they produce the D-dimensional representation
that the common quantile head turns into forecasts, as for every other encoder.

Interface: encoder(X) -> z with X [B, L, N, C] (channels: standardised demand, "unchanged" flag, optional price,
12 covariates broadcast to all zones) and z [B, N, D].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STIDEncoder(nn.Module):
    def __init__(self, L: int, C: int, D: int, N: int, embed: int = 32, layers: int = 3, dropout: float = 0.15,
                 n_cov: int = 12):
        super().__init__()
        self.n_cov = n_cov
        c_series = C - n_cov                                     # demand, flag (and price if given)
        self.series = nn.Linear(L * c_series, embed)
        self.node = nn.Parameter(torch.empty(N, embed))
        nn.init.xavier_uniform_(self.node)
        self.time = nn.Linear(6, embed) if n_cov >= 6 else None  # calendar part of the covariates of the last step
        h = 3 * embed
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h))
                                  for _ in range(layers)])
        self.out = nn.Linear(h, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, N, C = X.shape
        series = X[..., : C - self.n_cov].permute(0, 2, 1, 3).reshape(B, N, -1)
        node = self.node.unsqueeze(0).expand(B, N, -1)
        if self.time is not None:
            tim = self.time(X[:, -1, :, C - self.n_cov: C - self.n_cov + 6])         # [B, N, embed]
        else:
            tim = torch.zeros_like(node)
        h = torch.cat([self.series(series), node, tim], dim=-1)
        for blk in self.mlp:
            h = h + blk(h)
        return self.out(h)


class DLinearEncoder(nn.Module):
    def __init__(self, L: int, C: int, D: int, kernel: int = 7):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(L, D)
        self.season = nn.Linear(L, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0].permute(0, 2, 1)                                             # [B, N, L] standardised demand
        pad = (self.kernel - 1) // 2
        xp = torch.cat([x[..., :1].expand(-1, -1, pad), x, x[..., -1:].expand(-1, -1, self.kernel - 1 - pad)], dim=-1)
        trend = F.avg_pool1d(xp.reshape(-1, 1, xp.shape[-1]), self.kernel, stride=1).reshape(x.shape)
        return self.trend(trend) + self.season(x - trend)
