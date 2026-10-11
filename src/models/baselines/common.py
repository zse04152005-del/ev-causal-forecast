"""深度基线的公共部件：输入组装、统一输出头（图矩阵在 src/data/graphs.py）（设计文档第 10 节、阶段 3.5）。

统一比较协议：所有深度基线与 CPA-STGNN 使用同一切分、同一窗口（L、H）、同一质量掩码、同一套未来日历与天气输入、
同一个非交叉分位数输出头和 pinball 损失，**只替换编码器**。这样事实精度的差异只来自编码器与价格进入方式。
PIAST 例外：保留原文的点预测、MSE 与三阶段物理约束训练，区间由验证期残差分位数给出（见 piast.py）。

编码器接口：encoder(X) → z，X 为 [B, L, N, C]，z 为 [B, N, D]。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..heads.quantile import QuantileHead

PRICE_INPUTS = ("none", "hist", "hist_fut")
PRICE_KEYS = {"dl_hist", "dl_fut", "spill_fut", "shift_fut", "dl_fut_ext"}      # price_override 允许替换的键


def readout_keys(price_input: str) -> tuple:
    """价格响应读出时要一起上调的价格输入（模型看得到的全部价格）。"""
    return {"none": (), "hist": ("dl_hist",), "hist_fut": ("dl_hist", "dl_fut")}[price_input]


# ---------------------------------------------------------------- 输入组装
def n_input_channels(price_input: str, use_cov: bool) -> int:
    return 2 + int(price_input != "none") + (12 if use_cov else 0)


def assemble_inputs(batch: dict, price_input: str, use_cov: bool) -> torch.Tensor:
    """[B, L, N, C]：标准化需求 + 因果"已不变"标记（+ 历史价格 Δℓ）（+ 广播到各节点的历史日历与天气）。"""
    x = batch["x"]
    parts = [x]
    if price_input != "none":
        parts.append(batch["dl_hist"].unsqueeze(-1))
    if use_cov:
        B, L, N, _ = x.shape
        parts.append(batch["cov_hist"][:, :, None, :].expand(B, L, N, batch["cov_hist"].shape[-1]))
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------- 统一包装
class BaselineModel(nn.Module):
    """编码器 + 统一输出头。

    loss="pinball"：输出 y_q [B, H, N, Q]（非交叉分位数）；loss="mse"：输出点预测 y [B, H, N]，区间由训练脚本用验证期残差补上。
    price_input="hist_fut" 时，未来价格 Δℓ_{t+h} 作为输出头的额外输入（对照 P3：反事实时直接替换价格输入）。
    price_override 可替换未来价格，也可替换历史价格 dl_hist（价格响应读出时"模型看得到的价格全部上调"）。
    """

    def __init__(self, encoder: nn.Module, D: int, H: int, cov_fut_dim: int, quantiles, head_hidden: int = 64,
                 init_quantiles=None, price_input: str = "none", use_cov: bool = True, loss: str = "pinball",
                 clip_max: float | None = 1.0, dropout: float = 0.0):
        super().__init__()
        if price_input not in PRICE_INPUTS:
            raise ValueError(f"price_input 只能是 {PRICE_INPUTS}，收到 {price_input}")
        if loss not in ("pinball", "mse"):
            raise ValueError(f"loss 只能是 pinball 或 mse，收到 {loss}")
        self.encoder = encoder
        self.price_input, self.use_cov, self.loss = price_input, use_cov, loss
        self.readout = readout_keys(price_input)
        self.point = loss == "mse"
        self.quantiles = list(quantiles)
        self.clip_max = clip_max
        Q = 1 if self.point else len(self.quantiles)
        init = None
        if init_quantiles is not None:
            iq = list(init_quantiles)
            init = [iq[len(iq) // 2]] if self.point else iq
        self.head = QuantileHead(D, cov_fut_dim, H, Q, head_hidden, init, extra_dim=int(price_input == "hist_fut"),
                                 dropout=dropout)

    def forward(self, batch: dict, price_override: dict | None = None) -> dict:
        pf = dict(batch)
        if price_override:
            unknown = set(price_override) - PRICE_KEYS
            if unknown:
                raise KeyError(f"price_override 只能替换价格特征，收到：{unknown}")
            pf.update(price_override)
        z = self.encoder(assemble_inputs(pf, self.price_input, self.use_cov))          # 只有显式替换 dl_hist 时历史价格才变
        extra = pf["dl_fut"].unsqueeze(-1) if self.price_input == "hist_fut" else None
        out = self.head(z, batch["cov_fut"], extra)                                     # [B, H, N, Q]，恒为正
        if self.clip_max is not None:
            out = torch.clamp(out, max=float(self.clip_max))
        if self.point:
            return {"y": out[..., 0]}
        return {"y_q": out}
