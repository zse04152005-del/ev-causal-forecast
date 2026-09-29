"""深度基线注册表（阶段 3.5）。用法见 scripts/train_baseline.py。

| 名称 | 来源 | 图 |
|---|---|---|
| fcnn、lstm、gcn、gcnlstm | UrbanEV 数据集论文官方基线（结构移植） | gcn、gcnlstm 用邻接图 |
| stgcn | Yu et al. 2018 | 邻接图（切比雪夫） |
| astgcn | Guo et al. 2019 | 邻接图（切比雪夫 + 空间注意力） |
| agcrn | Bai et al. 2020 | 自适应图 |
| pag | Qu et al. 2024（ST-EVCDP 发布代码移植 + 论文版元学习预训练） | 邻接掩码 + GAT |
| piast | Kuang et al. 2024（官方代码移植） | 邻接掩码 + GAT |

Graph WaveNet 与 STAEformer 已作为 CPA-STGNN 的骨干实现：`scripts/train.py model.use_price=false` 即为它们的价格盲版本。
"""
from __future__ import annotations

from .agcrn import AGCRNEncoder
from .astgcn import ASTGCNEncoder
from .common import BaselineModel, n_input_channels
from .pag import PAG, fomaml_pretrain
from .piast import PIAST
from .stgcn import STGCNEncoder
from .urbanev import FCNNEncoder, GCNEncoder, GCNLSTMEncoder, LSTMEncoder

NAMES = ("fcnn", "lstm", "gcn", "gcnlstm", "stgcn", "astgcn", "agcrn", "pag", "piast")


def build_baseline(cfg, P, n_cov_fut: int, init_quantiles=None, seed: int | None = None):
    bc = cfg.baseline
    name = bc.name
    if name not in NAMES:
        raise ValueError(f"未知基线 {name}；可选 {NAMES}")
    L, H, N = int(cfg.data.L), int(cfg.data.H), P.N
    if P.adj is None:
        raise ValueError("Prepared 缺少邻接矩阵 adj")
    adj = P.adj
    if name == "pag":
        gc = bc.pag
        return PAG(N, L, H, adj, P.ref_lp, kcnn=int(gc.kcnn), k=int(gc.k), heads=int(gc.heads), beta=float(gc.beta),
                   dropout=float(gc.dropout), price_scale=float(gc.price_scale),
                   released_code_quirks=bool(gc.released_code_quirks))
    if name == "piast":
        pc = bc.piast
        return PIAST(N, L, H, adj, P.ref_lp, kcnn=int(pc.kcnn), mlp_hidden=int(pc.mlp_hidden), heads=int(pc.heads),
                     dropout=float(pc.dropout), price_scale=float(pc.price_scale),
                     released_code_quirks=bool(pc.released_code_quirks), seed_lambda=seed)
    C = n_input_channels(bc.price_input, bool(bc.use_cov))
    D = int(bc.D)
    sub = bc.get(name) or {}
    if name == "fcnn":
        enc = FCNNEncoder(L, C, D)
    elif name == "lstm":
        enc = LSTMEncoder(L, C, D, hidden=int(sub.get("hidden", 16)), layers=int(sub.get("layers", 2)))
    elif name == "gcn":
        enc = GCNEncoder(L, C, D, N, adj, hidden=int(sub.get("hidden", 32)), layers=int(sub.get("layers", 1)))
    elif name == "gcnlstm":
        enc = GCNLSTMEncoder(L, C, D, N, adj, gcn_out=int(sub.get("gcn_out", 32)),
                             gcn_layers=int(sub.get("gcn_layers", 1)), lstm_hidden=int(sub.get("lstm_hidden", 32)),
                             lstm_layers=int(sub.get("lstm_layers", 1)))
    elif name == "stgcn":
        enc = STGCNEncoder(L, C, D, N, adj, Kt=int(sub.get("Kt", 3)), Ks=int(sub.get("Ks", 3)),
                           blocks=[tuple(int(c) for c in b) for b in sub.get("blocks", [[64, 16, 64], [64, 16, 64]])],
                           dropout=float(bc.dropout))
    elif name == "astgcn":
        enc = ASTGCNEncoder(L, C, D, N, adj, K=int(sub.get("K", 3)), chev_filter=int(sub.get("chev_filter", 64)),
                            time_filter=int(sub.get("time_filter", 64)), blocks=int(sub.get("blocks", 2)))
    else:  # agcrn
        enc = AGCRNEncoder(L, C, D, N, hidden=int(sub.get("hidden", 64)), layers=int(sub.get("layers", 2)),
                           embed_dim=int(sub.get("embed_dim", 10)), cheb_k=int(sub.get("cheb_k", 2)))
    if name in ("astgcn", "agcrn"):
        official_init(enc)
    clip = cfg.model.get("clip_max")
    if cfg.data.target == "volume":
        clip = None
    return BaselineModel(enc, D, H, n_cov_fut, cfg.model.quantiles, int(cfg.model.head_hidden), init_quantiles,
                         price_input=bc.price_input, use_cov=bool(bc.use_cov), loss=bc.loss, clip_max=clip,
                         dropout=float(bc.dropout))


def official_init(module) -> None:
    """ASTGCN（ASTGCN-r-pytorch）与 AGCRN（Run.py）官方的初始化：二维以上参数 xavier_uniform，一维参数 U(0, 1)。"""
    import torch.nn as nn
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)


def readout_keys_of(model) -> tuple:
    """价格响应读出时要上调的价格输入（模型看得到的全部价格）；空元组 = 价格盲，不做读出。"""
    return tuple(getattr(model, "readout", ()))


__all__ = ["NAMES", "build_baseline", "official_init", "readout_keys_of", "fomaml_pretrain", "BaselineModel", "PAG", "PIAST"]
