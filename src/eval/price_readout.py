"""从任意"价格感知"预测模型中读出它隐含的价格弹性（E-PS、对照 P1–P3 用；设计文档 11.2）。

做法：测试期把选定小区的未来价格统一乘以 (1 + pct)（对数价格加 log(1 + pct)），比较前后的预测：
    ε̂_i = log( Σ_t ŷ'_{i,t} / Σ_t ŷ_{i,t} ) / log(1 + pct)
这是弧弹性，包含模型隐含的全部价格通道（本区价格、以及同时涨价的邻区通过溢出产生的影响），
与原文 PIAST 的"价格调整实验"口径一致。价格盲模型读出恒为 0。
"""
from __future__ import annotations

import numpy as np


def arc_elasticity(y_base: np.ndarray, y_up: np.ndarray, pct: float, zones_mask: np.ndarray | None = None) -> dict:
    """y_base、y_up [n, H, N]（中位数或点预测，目标单位）→ 各小区弧弹性与汇总。"""
    b = np.clip(np.nan_to_num(y_base, nan=0.0), 1e-9, None).sum(axis=(0, 1))
    u = np.clip(np.nan_to_num(y_up, nan=0.0), 1e-9, None).sum(axis=(0, 1))
    eps = np.log(u / b) / np.log1p(pct)
    m = np.ones_like(eps, dtype=bool) if zones_mask is None else np.asarray(zones_mask, dtype=bool)
    sel = eps[m]
    return {"pct": pct, "n_zones": int(m.sum()), "mean": float(sel.mean()) if sel.size else float("nan"),
            "median": float(np.median(sel)) if sel.size else float("nan"),
            "share_negative": float((sel < 0).mean()) if sel.size else float("nan"),
            "by_zone": eps.astype(float).tolist()}
