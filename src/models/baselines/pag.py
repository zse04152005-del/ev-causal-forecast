"""PAG（Qu, Kuang, Wang, Li & You 2024, IEEE T-ITS）移植：对照 P1 与先验敏感性实验 E-PS。

来源：github.com/IntelligentSystemsLab/ST-EVCDP（models.PAG、learner.py、functions.py；MIT 许可；论文脚注给出的仓库）。
网络结构与发布代码一致：
  需求历史与价格历史拼成 2 通道 → Conv2d(1,1,(kcnn,2)) → 两层共享参数的 GAT 注意力 + 共享线性层 →
  动量残差 x1 ← (1−β)x1 + β·x0，x2 ← (1−β)x2 + β·x1（β = 0.5，论文公式 4）→ 两个跳数特征作为 2 通道输入 LSTM(2→2, 2 层) →
  TPA 时间模式注意力（fc1: seq−1→k，fc2: k→m，sigmoid 打分）→ fc3 输出。GAT 注意力与 PIAST 同源（piast.GATAttention）。
价格只以**历史窗口**进入模型（不含预测时刻价格），所以价格响应读出时上调的是历史价格。
多步预测：发布代码只预测一个步长；这里 fc3 直接输出 H 个步长（其余结构不变）。

物理先验元学习预训练（论文算法 1，FOMAML）：
  每个先验弹性 ε_s 一个伪样本缓冲区（论文 −1.48 与 −0.228；发布代码为 −1.48 与 −0.74）；训练期前一半为支持集、后一半为查询集
  （发布代码的做法；论文写"第 1–12 天 / 第 13–24 天"，后者包含验证期）；每个外层轮次对每个缓冲区：复制全局模型 →
  在支持集上用 Adam 训练一遍 → 在查询集上求梯度；全局参数 φ ← φ − λ·Σ_s g_s / S（λ = 0.005，论文表 I）。
  伪样本比例 γ_e = 1 − e/1000（论文表 I），伪样本按论文公式 (8)(9) 生成（src/models/pseudo_samples.py）；
  查询集同样来自混合缓冲区（论文算法 1 第 7 行 g2 = ∇l_s(θ_s, R2_s)）。
  与发布代码的其他差别：发布代码查询集用真实数据；支持集的混合比例在每遍内随批次变化（伪样本比例从约 1 降到约 0.5）；
  外层步长为 0.02（因下述问题实际不生效）。
  预训练后在真实训练集上用 MSE 微调（Adam，学习率 1e-3，权重衰减 1e-5，早停）。

发布代码的问题（论文中需要说明；pretrain="none" 即发布代码实际的行为）：
  - learner.physics_informed_meta_learning 的全局更新写成 `param = param - 0.02 * grad`，只是给局部变量重新赋值，
    全局模型参数不变 → 默认的 completed 模式下微调从初始权重开始，预训练对参数不起作用（只消耗随机数），
    我们据此推断发布代码的 PAG 相当于论文中不做预训练的 PAG-；只有 simplified 模式真正在伪样本上训练
  - 伪样本弹性符号相反、伪标签取 tan（见 pseudo_samples.py 文件头）
  - GAT 注意力的两处问题与 PIAST 相同（released_code_quirks=true 时复现）
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from ..losses import masked_mse
from ..pseudo_samples import draw_price_shocks, mix_rows, pseudo_targets
from ..torch_utils import to_tensors
from .piast import GATAttention


class PAGNet(nn.Module):
    def __init__(self, N: int, L: int, H: int, adj: np.ndarray, kcnn: int = 2, k: int = 6, m: int = 2, heads: int = 4,
                 beta: float = 0.5, dropout: float = 0.5, train_heads: bool = True, per_sample: bool = True):
        super().__init__()
        if m != 2:
            raise ValueError("PAG 的 LSTM 输入是两个跳数的特征，m 必须为 2（与发布代码一致）")
        self.seq, self.H, self.beta = L - kcnn + 1, H, beta
        s = self.seq
        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))
        mask = (np.asarray(adj) > 0) | np.eye(N, dtype=bool)
        self.gat = GATAttention(mask, s, heads, alpha=0.2, train_heads=train_heads, per_sample=per_sample)
        self.gcn = nn.Linear(s, s)
        self.lstm = nn.LSTM(m, m, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(s - 1, k)
        self.fc2 = nn.Linear(k, m)
        self.fc3 = nn.Linear(k + m, H)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, occ: torch.Tensor, prc: torch.Tensor) -> torch.Tensor:
        """occ、prc [B, L, N]（需求历史、价格历史）→ [B, H, N]"""
        B, L, N = occ.shape
        data = torch.stack([occ.permute(0, 2, 1), prc.permute(0, 2, 1)], dim=3).reshape(B * N, 1, L, 2)
        data = self.conv2d(data).reshape(B, N, self.seq)
        c1 = self.dropout(self.act(self.gcn(self.gat(data) @ data)))
        c2 = self.dropout(self.act(self.gcn(self.gat(c1) @ c1)))
        c1 = (1 - self.beta) * c1 + self.beta * data                                  # 论文公式 (4)
        c2 = (1 - self.beta) * c2 + self.beta * c1
        x = torch.stack([c1.reshape(B * N, self.seq), c2.reshape(B * N, self.seq)], dim=2)   # [BN, seq, 2]
        lo, _ = self.lstm(x)
        ht = lo[:, -1, :]                                                             # [BN, m]
        hc = self.fc1(lo[:, :-1, :].transpose(1, 2))                                  # [BN, m, k]
        a = torch.sigmoid(torch.bmm(self.fc2(hc), ht.unsqueeze(2))).transpose(1, 2)   # [BN, 1, m]
        vt = a @ hc                                                                   # [BN, 1, k]
        y = self.fc3(torch.cat([vt, ht.unsqueeze(1)], dim=2))                         # [BN, 1, H]
        return y.reshape(B, N, self.H).permute(0, 2, 1)


class PAG(nn.Module):
    point = True
    loss = "mse"
    readout = ("dl_hist",)                                                            # 模型只看历史价格

    def __init__(self, N: int, L: int, H: int, adj: np.ndarray, ref_lp: np.ndarray, kcnn: int = 2, k: int = 6,
                 heads: int = 4, beta: float = 0.5, dropout: float = 0.5, price_scale: float = 1.0,
                 released_code_quirks: bool = False):
        super().__init__()
        self.net = PAGNet(N, L, H, adj, kcnn, k, 2, heads, beta, dropout, train_heads=not released_code_quirks,
                          per_sample=not released_code_quirks)
        self.register_buffer("ref_lp", torch.tensor(np.asarray(ref_lp, dtype=np.float32)))
        self.price_scale = float(price_scale)

    def price_level(self, dl: torch.Tensor) -> torch.Tensor:
        return torch.exp(dl + self.ref_lp) / self.price_scale

    def forward(self, batch: dict, price_override: dict | None = None) -> dict:
        dl = batch["dl_hist"]
        if price_override:
            unknown = set(price_override) - {"dl_hist", "dl_fut", "spill_fut", "shift_fut", "dl_fut_ext"}
            if unknown:
                raise KeyError(f"price_override 只能替换价格特征，收到：{unknown}")
            dl = price_override.get("dl_hist", dl)
        return {"y": self.net(batch["x"][..., 0], self.price_level(dl))}


def pseudo_batch(tb: dict, law: float, gamma: float, rng: np.random.Generator, adj: np.ndarray,
                 clip_max: float | None, prop: float = 0.4, sd: float = 0.5) -> dict:
    """把批内比例为 γ 的行换成伪样本：历史价格整段乘以 (1 + Δp/p)，目标按公式 (8)(9) 改写。"""
    B, _, N = tb["dl_hist"].shape
    rows = mix_rows(B, gamma)
    k = int(rows.sum())
    if k == 0:
        return tb
    shocks = np.stack([draw_price_shocks(N, rng, prop, sd) for _ in range(k)])        # [k, N]
    dev = tb["dl_hist"].device
    r = torch.as_tensor(rows, device=dev)
    y = torch.nan_to_num(tb["y_fut"][r]).cpu().numpy().astype(np.float64)             # [k, H, N]
    y_ps = pseudo_targets(y, shocks[:, None, :], law, adj, clip_max)
    out = dict(tb)
    dl = tb["dl_hist"].clone()
    dl[r] = dl[r] + torch.as_tensor(np.log1p(shocks)[:, None, :], dtype=dl.dtype, device=dev)
    yf = tb["y_fut"].clone()
    yf[r] = torch.as_tensor(y_ps, dtype=yf.dtype, device=dev)
    out["dl_hist"], out["y_fut"] = dl, yf
    return out


def fomaml_pretrain(model: PAG, W_sup, W_qry, laws, epochs: int, device, rng: np.random.Generator, adj: np.ndarray,
                    clip_max: float | None, bs: int, outer_lr: float = 0.005, inner_lr: float = 1e-3,
                    weight_decay: float = 1e-5, gamma_den: float = 1000.0, max_batches: int | None = None,
                    log=None) -> list:
    """论文算法 1（一阶 MAML）。返回每轮的查询集平均损失。"""
    hist = []
    params = dict(model.named_parameters())
    for e in range(int(epochs)):
        gamma = max(0.0, 1.0 - e / float(gamma_den))
        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        q_losses = []
        for law in laws:
            temp = copy.deepcopy(model)
            temp.train()
            opt = torch.optim.Adam(temp.parameters(), lr=inner_lr, weight_decay=weight_decay)
            for b in W_sup.iterate(bs, shuffle=True, rng=rng, max_batches=max_batches):
                tb = pseudo_batch(to_tensors(b, device), float(law), gamma, rng, adj, clip_max)
                loss = masked_mse(temp(tb)["y"], tb["y_fut"], tb["valid_fut"])
                opt.zero_grad()
                loss.backward()
                opt.step()
            opt.zero_grad()
            nq, ql = 0, 0.0
            for b in W_qry.iterate(bs, max_batches=max_batches):
                tb = pseudo_batch(to_tensors(b, device), float(law), gamma, rng, adj, clip_max)
                loss = masked_mse(temp(tb)["y"], tb["y_fut"], tb["valid_fut"])
                loss.backward()                                                       # 梯度在各批之间累加
                nq += 1
                ql += loss.item()
            for n, p in temp.named_parameters():
                if p.grad is not None:
                    acc[n] += p.grad / max(nq, 1)
            q_losses.append(ql / max(nq, 1))
        with torch.no_grad():
            for n, p in params.items():
                p -= outer_lr * acc[n] / len(laws)
        hist.append(float(np.mean(q_losses)))
        if log is not None and (e == 0 or (e + 1) % 10 == 0 or e + 1 == epochs):
            log.info(f"PAG 元学习预训练 第 {e + 1:3d} 轮  伪样本比例 {gamma:.3f}  查询集损失 {hist[-1]:.5f}")
        if not math.isfinite(hist[-1]):
            if log is not None:
                log.warning("PAG 预训练损失不是有限值，提前结束预训练")
            break
    return hist
