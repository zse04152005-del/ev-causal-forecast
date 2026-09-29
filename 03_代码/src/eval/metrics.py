"""掩码评价指标（设计文档第 9、10 节；只对 valid=True 的点计算）。

点预测取中位数分位数。区间：90% = [q05, q95]，80% = [q10, q90]。
CRPS 用分位数损失近似：CRPS ≈ 2 × 各分位数水平平均的 pinball 损失。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mm(x: np.ndarray, m: np.ndarray) -> float:
    n = m.sum()
    return float((x * m).sum() / n) if n > 0 else float("nan")


def pinball(yq: np.ndarray, y: np.ndarray, quantiles) -> np.ndarray:
    q = np.asarray(quantiles)
    u = y[..., None] - yq
    return np.maximum(q * u, (q - 1) * u).mean(-1)


def metrics_block(yq: np.ndarray, y: np.ndarray, valid: np.ndarray, quantiles, mape_min: float = 0.05) -> dict:
    """yq [..., Q]、y [...]、valid [...] → 指标字典。"""
    q = list(np.round(quantiles, 4))
    med = yq[..., q.index(0.5)]
    e = med - y
    m = valid.astype(np.float64)
    out = {
        "MAE": _mm(np.abs(e), m),
        "RMSE": float(np.sqrt(_mm(e ** 2, m))),
        "WAPE": float((np.abs(e) * m).sum() / max((np.abs(y) * m).sum(), 1e-12)),
        "MAPE": _mm(np.abs(e) / np.maximum(np.abs(y), mape_min), m * (np.abs(y) >= mape_min)),
        "pinball": _mm(pinball(yq, y, quantiles), m),
    }
    out["CRPS_q"] = 2 * out["pinball"]
    for level, (lo, hi) in {"90": (0.05, 0.95), "80": (0.1, 0.9)}.items():
        if lo in q and hi in q:
            L, U = yq[..., q.index(lo)], yq[..., q.index(hi)]
            a = 1 - (hi - lo)
            cover = (y >= L) & (y <= U)
            out[f"PICP{level}"] = _mm(cover, m)
            out[f"MPIW{level}"] = _mm(U - L, m)
            winkler = (U - L) + (2 / a) * np.maximum(L - y, 0) + (2 / a) * np.maximum(y - U, 0)
            out[f"IS{level}"] = _mm(winkler, m)
    return out


def metrics_table(yq: np.ndarray, y: np.ndarray, valid: np.ndarray, quantiles, steps) -> pd.DataFrame:
    """yq [n, H, N, Q] → 每个报告步长一行，外加全部步长平均（step = 'avg'）。"""
    rows = []
    for s in steps:
        rows.append({"step": s, **metrics_block(yq[:, s - 1], y[:, s - 1], valid[:, s - 1], quantiles)})
    rows.append({"step": "avg", **metrics_block(yq, y, valid, quantiles)})
    return pd.DataFrame(rows)


def interval_metrics(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, valid: np.ndarray, alpha: float) -> dict:
    m = valid.astype(np.float64)
    cover = (y >= lo) & (y <= hi)
    w = hi - lo
    winkler = w + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return {"PICP": _mm(cover, m), "MPIW": _mm(w, m), "IS": _mm(winkler, m)}
