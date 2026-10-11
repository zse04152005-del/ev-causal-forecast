"""模块⑤b：带数据质量掩码的自适应保形校准（ACI，Gibbs & Candès 2021；设计文档 5.6）。

- 分数（CQR 形式）：s = max(lo − y, y − hi) / σ_i，σ_i 为小区训练期的中位绝对偏差 → 各小区共用一个分数分布
- 每个步长 h 单独维护 α_h；预测起点 s 只能使用目标已经观测到的分数（起点 ≤ s − ceil(h/stride)）
- 冻结点既不进入分数窗口，也不参与 α 的更新
- 先用验证期的起点"预热"分数窗口与 α，再对测试期在线校准
"""
from __future__ import annotations

import math

import numpy as np


def zone_scale(y_train: np.ndarray, valid_train: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    out = np.zeros(y_train.shape[1])
    for j in range(y_train.shape[1]):
        v = y_train[valid_train[:, j], j]
        out[j] = np.median(np.abs(v - np.median(v))) if len(v) else floor
    return np.maximum(out, floor)


def _cq(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    if alpha <= 0:
        return float(np.max(scores))
    if alpha >= 1:
        return float(np.min(scores))
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def aci_calibrate(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, valid: np.ndarray, sigma: np.ndarray,
                  alpha: float = 0.1, gamma: float = 0.005, window: int = 168, h: int = 1, stride: int = 1,
                  adaptive: bool = True) -> dict:
    """单个步长的在线校准。lo、hi、y、valid：[n_origins, N]（按时间排序）。

    返回校准后的 lo_c、hi_c 与 α 轨迹；adaptive=False 时 α 固定（滚动窗口的分割保形）。
    """
    n, N = lo.shape
    delay = max(1, math.ceil(h / stride))
    raw = np.maximum(lo - y, y - hi) / sigma[None, :]
    raw = np.where(valid, raw, np.nan)
    lo_c = np.full_like(lo, np.nan)
    hi_c = np.full_like(hi, np.nan)
    alphas = np.full(n, np.nan)
    a = alpha
    for s in range(n):
        if adaptive and s - delay >= 0:
            j = s - delay                                     # 刚刚揭晓的起点
            if np.isfinite(lo_c[j]).any():
                vj = valid[j]
                if vj.any():
                    covered = (y[j] >= lo_c[j]) & (y[j] <= hi_c[j])
                    err = 1.0 - covered[vj].mean()
                    a = a + gamma * (alpha - err)
        a_eff = min(max(a, 1e-3), 1 - 1e-3)
        first = max(0, s - delay - window + 1)
        pool = raw[first: s - delay + 1] if s - delay >= 0 else raw[:0]
        sc = pool[np.isfinite(pool)]
        qv = _cq(sc, a_eff) if sc.size else 0.0
        lo_c[s] = lo[s] - sigma * qv
        hi_c[s] = hi[s] + sigma * qv
        alphas[s] = a
    return {"lo": lo_c, "hi": hi_c, "alpha": alphas}
