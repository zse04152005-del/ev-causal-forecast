"""损失函数（设计文档 7.1）：掩码分位数损失、层级收缩、软锚定损失（口径对齐）。"""
from __future__ import annotations

import numpy as np
import torch

from ..causal.anchors import AnchorSet


def pinball_loss(y_q: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, quantiles) -> torch.Tensor:
    """y_q [B, H, N, Q]；y、valid [B, H, N]。只对 valid 的点求平均。"""
    q = torch.as_tensor(quantiles, dtype=y_q.dtype, device=y_q.device)
    u = torch.nan_to_num(y).unsqueeze(-1) - y_q
    loss = torch.maximum(q * u, (q - 1) * u).mean(-1)
    loss = torch.where(valid, loss, torch.zeros_like(loss))          # 掩码点即使含非有限值也不影响
    return loss.sum() / valid.to(y_q.dtype).sum().clamp(min=1.0)


def masked_mse(y: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """点预测的掩码均方误差（PIAST、PAG 与 loss=mse 的基线）。"""
    d = (y - torch.nan_to_num(t)) ** 2
    d = torch.where(v, d, torch.zeros_like(d))
    return d.sum() / v.to(y.dtype).sum().clamp(min=1.0)


def hier_loss(model) -> torch.Tensor:
    pm = getattr(model, "price", None)
    if pm is None or not hasattr(pm, "xi"):
        return torch.zeros((), device=next(model.parameters()).device)
    return (pm.xi ** 2).mean()


def anchor_loss(model, anchors: AnchorSet, tou_mask: np.ndarray, groups: np.ndarray, window_factor: float = 1.0) -> torch.Tensor:
    """软锚定（仅 soft 模式）：各格子内分时小区的平均弹性 × 口径因子，与因果估计按 1/se² 比较。"""
    dev = next(model.parameters()).device
    total = torch.zeros((), device=dev)
    n = 0
    if model.price is not None:
        bt = model.price.beta_table()                                        # [N, C]
        tou = torch.as_tensor(np.array(tou_mask), device=dev)                # 复制一份：原数组只读时 torch 会警告
        g = torch.as_tensor(np.array(groups), device=dev)
        for gi in range(anchors.beta.shape[0]):
            m = tou & (g == gi)
            if int(m.sum()) == 0:
                continue
            bbar = bt[m].mean(0) * window_factor                             # [C]
            tgt = torch.as_tensor(anchors.beta[gi], dtype=bbar.dtype, device=dev)
            se = torch.as_tensor(anchors.se[gi], dtype=bbar.dtype, device=dev)
            total = total + (((bbar - tgt) / se) ** 2).sum()
            n += len(tgt)
    if model.spill is not None:
        d = model.spill.delta()
        tgt = torch.as_tensor(anchors.delta, dtype=d.dtype, device=dev)
        se = torch.as_tensor(anchors.delta_se, dtype=d.dtype, device=dev)
        total = total + (((d - tgt) / se) ** 2).sum()
        n += len(tgt)
    if model.shift is not None:
        gm = model.shift.gamma()
        total = total + ((gm - anchors.gamma) / anchors.gamma_se) ** 2
        n += 1
    return total / max(n, 1)
