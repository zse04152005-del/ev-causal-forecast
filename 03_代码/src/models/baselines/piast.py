"""PIAST（Kuang et al. 2024, Applied Energy 363:123059）移植：先验敏感性实验 E-PS 与反事实对照 P2。

来源：github.com/kuanghx3/PIAST（proposed_model.py、proposed_main.py、args.py；MIT 许可）。
网络结构与原代码一致：
  需求历史与"预测时刻价格（沿时间重复）"拼成 2 通道 → Conv2d(1,1,(kcnn,2)) 融合 →
  两层共享参数的多头 GAT 注意力（按邻接掩码、第 0 维 softmax）+ 共享线性层 → 与融合特征堆叠成 3 通道 →
  2 层 LSTM(3→3) + 时间注意力 → 两个 MLP 解码器 → 点预测。
物理约束（三阶段训练，与原代码一致）：
  con1 = ∂y/∂p − (y/p)·λ（λ 为每个小区一个可学习弹性），L3 = mean((λ − 先验)²)；
  阶段 1：MSE + con1² + 1.0·L3；阶段 2：MSE + con1² + 0.1·L3；阶段 3：只用 MSE（λ 不再得到梯度，相当于冻结）。
  每步之后把 λ 截断到 [−1, 0]（原代码 clamp_(-1, 0)；配置 clamp=null 可取消，用于 E-PS 对照）。

发布代码中的三处实现问题（released_code_quirks=true 时按原样复现，默认修正）：
  1. GAT 各头的权重放在普通 dict 里，没有注册为模型参数 → 从不训练（修正：注册为参数）
  2. 注意力分数用 h[edges] 索引 b·n 行的特征，只用到批内第 1 个样本 → 全批共用一张注意力图（修正：逐样本计算）
  3. ∂y/∂p 用 autograd.grad(..., create_graph=False) → 约束项不对网络的价格响应产生梯度，只训练 λ 与 y 的水平（修正：create_graph=True）
另有一处设计上的性质（两种模式都保留，论文中需说明）：原代码用 autograd.grad(y, p, ones) 求导，得到的是雅可比矩阵的
列和 Σ_j ∂y_j/∂p_i（本区价格对所有小区需求的影响之和，含溢出），而不是本区自身的 ∂y_i/∂p_i；写了对角线版本 net_con1 但没有调用。

多步预测的适配：原文只预测一个步长（predict_time）。这里把 H 个步长当作批维度，第 h 步输入 p_{t+h}，
并在融合特征上加一个可学习的步长嵌入（零初始化；H = 1 时不加，与原文完全一致）。
价格用水平值 p = exp(ℓ) / price_scale（原代码为总价 / 1.5），弹性 ∂y/∂p·p/y 与尺度无关。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATAttention(nn.Module):
    def __init__(self, adj_mask: np.ndarray, s: int, heads: int = 4, alpha: float = 0.2, train_heads: bool = True,
                 per_sample: bool = True):
        super().__init__()
        a = np.asarray(adj_mask) > 0
        e0, e1 = np.nonzero(a)
        n = a.shape[0]
        self.n, self.per_sample = n, per_sample
        self.register_buffer("flat_idx", torch.tensor(e0 * n + e1, dtype=torch.long))
        self.register_buffer("e0", torch.tensor(e0, dtype=torch.long))
        self.register_buffer("e1", torch.tensor(e1, dtype=torch.long))
        mask = np.where(a, 0.0, -1e9).astype(np.float32)
        self.register_buffer("mask", torch.tensor(mask))
        W = torch.empty(heads, s, s)
        av = torch.empty(heads, 1, 2 * s)
        for h in range(heads):
            nn.init.xavier_normal_(W[h], gain=1.414)
            nn.init.xavier_normal_(av[h], gain=1.414)
        av = av[:, 0].clone()                                                # [heads, 2s]
        if train_heads:
            self.W, self.a = nn.Parameter(W), nn.Parameter(av)
        else:                                                                # 发布代码：不参与训练
            self.register_buffer("W", W)
            self.register_buffer("a", av)
        self.linear = nn.Linear(heads, 1)
        self.leaky = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [b, n, s] → 注意力矩阵 [b, n, n]"""
        b = x.shape[0]
        xs = x if self.per_sample else x[:1]
        h = torch.einsum("bns,kst->bknt", xs, self.W)                                  # [b', heads, n, s]
        s = h.shape[-1]
        # a·[h_i ‖ h_j] = a_前半·h_i + a_后半·h_j：先按节点算两半，再按边取值（与拼接写法等价，显存省约 2s 倍）
        src = torch.einsum("bknt,kt->bkn", h, self.a[:, :s])
        dst = torch.einsum("bknt,kt->bkn", h, self.a[:, s:])
        e = self.leaky(src[:, :, self.e0] + dst[:, :, self.e1])                        # [b', heads, E]
        e = self.linear(e.transpose(1, 2)).squeeze(-1)                                 # [b', E]
        dense = xs.new_zeros(xs.shape[0], self.n * self.n).scatter(1, self.flat_idx.expand(xs.shape[0], -1), e)
        att = F.softmax(dense.view(-1, self.n, self.n) + self.mask, dim=1)             # 原代码 softmax(dim=0)
        return att if self.per_sample else att.expand(b, self.n, self.n)


class PIASTNet(nn.Module):
    def __init__(self, N: int, L: int, H: int, adj: np.ndarray, kcnn: int = 2, mlp_hidden: int = 64, heads: int = 4,
                 dropout: float = 0.2, train_heads: bool = True, per_sample: bool = True):
        super().__init__()
        self.L, self.H, self.seq = L, H, L - kcnn + 1
        s = self.seq
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))
        mask = (np.asarray(adj) > 0) | np.eye(N, dtype=bool)                          # 加自环：孤立小区只关注自身
        self.gat = GATAttention(mask, s, heads, train_heads=train_heads, per_sample=per_sample)
        self.gcn = nn.Linear(s, s)
        self.lstm = nn.LSTM(3, 3, num_layers=2, batch_first=True)
        self.Q = nn.Linear(s, s, bias=False)
        self.K = nn.Linear(s, s, bias=False)
        self.V = nn.Linear(s, s, bias=False)
        self.dec1 = nn.Sequential(nn.Linear(s, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(),
                                  nn.Linear(mlp_hidden // 2, 1))
        self.dec2 = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.hemb = nn.Parameter(torch.zeros(H, s)) if H > 1 else None

    def forward(self, occ: torch.Tensor, prc: torch.Tensor) -> torch.Tensor:
        """occ [B, L, N]（需求历史）；prc [B, H, N]（各预测时刻的价格水平）→ [B, H, N]"""
        B, L, N = occ.shape
        H = prc.shape[1]
        o = occ.permute(0, 2, 1)[:, None].expand(B, H, N, L)
        p = prc[..., None].expand(B, H, N, L)
        fea = torch.stack([o, p], dim=-1).reshape(B * H * N, 1, L, 2)
        fea = self.conv2d(fea).reshape(B, H, N, self.seq)
        if self.hemb is not None:
            fea = fea + self.hemb[None, :, None, :]
        fea = fea.reshape(B * H, N, self.seq)
        c1 = self.dropout(self.act(self.gcn(self.gat(fea) @ fea)))
        c2 = self.dropout(self.act(self.gcn(self.gat(c1) @ c1)))
        z = torch.stack([fea, c1, c2], dim=3).reshape(B * H * N, self.seq, 3)
        lo, _ = self.lstm(z)
        y_long = lo[:, -1, 0]
        xl = lo.permute(0, 2, 1)                                                        # [., 3, seq]
        q, k, v = self.Q(xl), self.K(xl).permute(0, 2, 1), self.V(xl)
        att = F.softmax(q @ k / k.shape[2], dim=2)
        a_out = self.dec1(att @ v).squeeze(2)                                           # [., 3]
        y = self.dec2(torch.cat([y_long[:, None], a_out], dim=1)).squeeze(1)
        return y.reshape(B, H, N)


class PIAST(nn.Module):
    """包装：价格水平换算、每区弹性 λ、物理约束残差。输出点预测 {"y": [B, H, N]}。"""

    point = True
    loss = "mse"
    readout = ("dl_fut",)                                                           # 模型只看预测时刻价格

    def __init__(self, N: int, L: int, H: int, adj: np.ndarray, ref_lp: np.ndarray, kcnn: int = 2,
                 mlp_hidden: int = 64, heads: int = 4, dropout: float = 0.2, price_scale: float = 1.5,
                 released_code_quirks: bool = False, seed_lambda: int | None = None):
        super().__init__()
        self.quirks = bool(released_code_quirks)
        self.net = PIASTNet(N, L, H, adj, kcnn, mlp_hidden, heads, dropout, train_heads=not self.quirks,
                            per_sample=not self.quirks)
        g = torch.Generator().manual_seed(seed_lambda) if seed_lambda is not None else None
        self.lambda_1 = nn.Parameter(-torch.rand(N, generator=g))                     # 原代码：torch.rand(nodes) * (−1)
        self.register_buffer("ref_lp", torch.tensor(np.asarray(ref_lp, dtype=np.float32)))
        self.price_scale = float(price_scale)

    def price_level(self, dl_fut: torch.Tensor) -> torch.Tensor:
        return torch.exp(dl_fut + self.ref_lp) / self.price_scale

    def forward(self, batch: dict, price_override: dict | None = None) -> dict:
        dl = batch["dl_fut"]
        if price_override:
            unknown = set(price_override) - {"dl_hist", "dl_fut", "spill_fut", "shift_fut", "dl_fut_ext"}
            if unknown:
                raise KeyError(f"price_override 只能替换未来价格特征，收到：{unknown}")
            dl = price_override.get("dl_fut", dl)
        return {"y": self.net(batch["x"][..., 0], self.price_level(dl))}

    def physics(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (y, con1)：con1 = ∂(Σy)/∂p − (y/p)·λ，与原代码 net_con2 相同的列和导数。"""
        p = self.price_level(batch["dl_fut"]).detach().requires_grad_(True)
        # 约束项要对 LSTM 二阶求导：cuDNN 的 LSTM 不支持二阶导，torch 2.0.0 的 CPU mkldnn LSTM 也不支持，这里临时改用原生实现
        with torch.backends.cudnn.flags(enabled=False), torch.backends.mkldnn.flags(enabled=False):
            y = self.net(batch["x"][..., 0], p)
            y_p = torch.autograd.grad(y, p, grad_outputs=torch.ones_like(y), retain_graph=True,
                                      create_graph=not self.quirks)[0]
        con1 = y_p - y / p * self.lambda_1[None, None, :]
        return y, con1

    def clamp_lambda(self, lo: float | None, hi: float | None) -> None:
        if lo is None and hi is None:
            return
        with torch.no_grad():
            self.lambda_1.clamp_(min=lo, max=hi)
