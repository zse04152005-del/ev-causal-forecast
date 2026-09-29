"""CPA-STGNN 组装（设计文档 4.1）：

    ŷ^(q) = min{ clip, b^(q) · exp( β_{g(i),κ} Δℓ + Σ_k δ_k S^(k) + γ U ) }

- 骨干网络与分位数头**看不到价格**（原则 P1）；价格只经 η 进入
- forward(batch, price_override=...) 只替换价格特征，基线 b 不变 → 反事实预测
- 消融开关：use_price（A5）、price_input（A6）、use_spill（A7）、use_shift（A8）、backbone（A10）
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from ..causal.anchors import AnchorSet, window_factor
from .backbones.graph_wavenet import GraphWaveNetEncoder
from .backbones.staeformer import STAEformerEncoder
from .heads.price import PriceResponse, Shift, Spillover
from .heads.quantile import QuantileHead

PRICE_KEYS = ("dl_fut", "spill_fut", "shift_fut", "dl_fut_ext")


class CPASTGNN(nn.Module):
    def __init__(self, mcfg, *, N: int, L: int, H: int, K: int, cov_hist_dim: int, cov_fut_dim: int,
                 static: np.ndarray, supports: list, groups: np.ndarray, anchors: AnchorSet, anchor_mode: str = "cut",
                 graph_cfg=None, init_quantiles=None, init_beta: float | None = None, anchor_window: int = 3):
        super().__init__()
        self.price_input = bool(mcfg.price_input)
        self.use_price = bool(mcfg.use_price) and not self.price_input
        self.clip_max = mcfg.get("clip_max")
        self.quantiles = list(mcfg.quantiles)
        self.price_lags = int(mcfg.get("price_lags", 0) or 0)
        c_in = 2 + int(self.price_input)
        st = torch.tensor(np.array(static, dtype=np.float32))          # 复制：避免只读数组警告
        g = graph_cfg or {}
        if mcfg.backbone == "graph_wavenet":
            self.backbone = GraphWaveNetEncoder(
                N, c_in, cov_hist_dim, st, [torch.tensor(np.array(s, dtype=np.float32)) for s in supports], D=mcfg.D,
                dilations=tuple(mcfg.dilations), kernel_size=mcfg.kernel_size, order=mcfg.diffusion_steps,
                dropout=mcfg.dropout, adaptive=g.get("adaptive", True), adaptive_dim=g.get("adaptive_dim", 10))
        elif mcfg.backbone == "staeformer":
            self.backbone = STAEformerEncoder(N, L, c_in, cov_hist_dim, st, D=mcfg.D, **dict(mcfg.staeformer))
        else:
            raise ValueError(f"未知骨干网络：{mcfg.backbone}")
        extra_dim = (2 + K) if self.price_input else 0
        self.head = QuantileHead(mcfg.D, cov_fut_dim, H, len(self.quantiles), mcfg.head_hidden, init_quantiles,
                                 extra_dim=extra_dim)
        lag_w = None
        if self.price_lags > 0:
            lag_w = anchors.lag_weights if anchors.lag_weights is not None else np.ones(self.price_lags + 1)
            if len(lag_w) != self.price_lags + 1:
                raise ValueError(f"滞后权重长度 {len(lag_w)} 与 model.price_lags + 1 = {self.price_lags + 1} 不一致")
        # 口径对齐（设计文档 6.5）：锚定值是切换后 anchor_window 小时的平均响应；有滞后核时长期弹性 = 锚定值 / 窗口因子
        self.wf = window_factor(lag_w, anchor_window)
        wf_emp = (anchors.meta or {}).get("window_factor")
        if lag_w is not None and wf_emp:                  # 锚定文件给出经验窗口因子（5 分钟半合成验证）时以它为准
            self.wf = float(wf_emp)
        price_anchors = anchors
        if lag_w is not None:
            price_anchors = dataclasses.replace(anchors, beta=np.asarray(anchors.beta, dtype=np.float64) / self.wf)
        self.price = PriceResponse(groups, price_anchors, anchor_mode, init_beta=init_beta, lag_weights=lag_w) \
            if self.use_price else None
        self.spill = Spillover(anchors, anchor_mode) if (self.use_price and mcfg.use_spill) else None
        self.shift = Shift(anchors, anchor_mode) if (self.use_price and mcfg.use_shift) else None

    def eta(self, pf: dict) -> torch.Tensor:
        dl = pf["dl_fut"]
        eta = torch.zeros_like(dl)
        if self.price is not None:
            eta = eta + self.price(pf["dl_fut_ext"] if self.price_lags > 0 else dl, pf["ctx_fut"])
        if self.spill is not None:
            eta = eta + self.spill(pf["spill_fut"])
        if self.shift is not None:
            eta = eta + self.shift(pf["shift_fut"])
        return eta

    def forward(self, batch: dict, price_override: dict | None = None) -> dict:
        pf = dict(batch)
        if price_override:
            unknown = set(price_override) - set(PRICE_KEYS)
            if unknown:
                raise KeyError(f"price_override 只能替换价格特征，收到：{unknown}")
            pf.update(price_override)
            if self.price_lags > 0 and "dl_fut" in price_override and "dl_fut_ext" not in price_override:
                # 只替换未来价格时，滞后核里的过去价格保持真实值
                pf["dl_fut_ext"] = torch.cat([batch["dl_fut_ext"][:, : self.price_lags], price_override["dl_fut"]], dim=1)
        x = batch["x"]
        if self.price_input:
            x = torch.cat([x, batch["dl_hist"].unsqueeze(-1)], dim=-1)
        z = self.backbone(x, batch["cov_hist"], tod=batch.get("tod_hist"), dow=batch.get("dow_hist"))
        extra = None
        if self.price_input:
            extra = torch.cat([pf["dl_fut"].unsqueeze(-1), pf["spill_fut"], pf["shift_fut"].unsqueeze(-1)], dim=-1)
        b_q = self.head(z, batch["cov_fut"], extra)                         # [B, H, N, Q]
        eta = self.eta(pf) if self.use_price else torch.zeros_like(b_q[..., 0])
        y_q = b_q * torch.exp(eta).unsqueeze(-1)
        if self.clip_max is not None:
            y_q = torch.clamp(y_q, max=float(self.clip_max))
        return {"b_q": b_q, "eta": eta, "y_q": y_q}

    def price_parameters(self) -> dict:
        """报告用：当前弹性表 [N, C]、δ、γ（numpy）。"""
        out = {}
        if self.price is not None:
            out["beta"] = self.price.beta_table().detach().cpu().numpy()
        if self.spill is not None:
            out["delta"] = self.spill.delta().detach().cpu().numpy()
        if self.shift is not None:
            out["gamma"] = float(self.shift.gamma().detach().cpu())
        return out
