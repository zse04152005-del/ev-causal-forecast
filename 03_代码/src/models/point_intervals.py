"""点预测模型（PIAST、loss=mse 的基线）的分位数：按步长汇总校准集（验证期）有效点残差的经验分位数，加到点预测上。

之后与所有模型一样再经过滚动分割保形与 ACI（src/eval/runner.py），所以区间覆盖率在各模型之间可比。
"""
from __future__ import annotations

import numpy as np


def residual_quantiles(point: np.ndarray, y: np.ndarray, valid: np.ndarray, quantiles) -> np.ndarray:
    """point、y、valid [n, H, N] → [H, Q]；某步没有有效点时残差分位数取 0。"""
    H = point.shape[1]
    q = np.asarray(quantiles, dtype=np.float64)
    out = np.zeros((H, len(q)))
    for h in range(H):
        r = (y[:, h] - point[:, h])[valid[:, h] & np.isfinite(y[:, h])]
        if r.size:
            out[h] = np.quantile(r, q)
    return out


def point_to_quantiles(point: np.ndarray, rq: np.ndarray, clip_max: float | None = 1.0) -> np.ndarray:
    """point [n, H, N]，rq [H, Q] → [n, H, N, Q]：截断到 [0, clip_max] 后排序，保证不交叉。"""
    yq = point[..., None] + rq[None, :, None, :]
    yq = np.maximum(yq, 0.0)
    if clip_max is not None:
        yq = np.minimum(yq, clip_max)
    return np.sort(yq, axis=-1).astype(np.float32)
