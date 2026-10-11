"""不依赖 PyTorch 的基线（UrbanEV 基准中的 Last Observation 等）：

- naive_last     ：ŷ_{t0+h} = y_{t0}
- naive_seasonal ：ŷ_{t0+h} = y_{t0+h-24}（前一天同一时刻）
- profile        ：训练期"小区 × 一周中的小时"平均
分位数：在训练窗口上按步长汇总残差的经验分位数（只用有效点），加到点预测上，再截断到 [0, clip]。
它们也用来在云端（没有 PyTorch）端到端检查评价流程。
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import Prepared
from ..data.windows import WindowDataset


def _point(P: Prepared, name: str, t0: np.ndarray, H: int, prof: np.ndarray | None) -> np.ndarray:
    f = t0[:, None] + np.arange(1, H + 1)[None, :]
    if name == "naive_last":
        return np.repeat(P.y[t0][:, None, :], H, axis=1)
    if name == "naive_seasonal":
        return P.y[f - 24]
    if name == "profile":
        how = (P.dow * 24 + P.tod)[f]
        return prof[how]
    raise ValueError(name)


class NaiveForecaster:
    def __init__(self, name: str, quantiles, clip_max: float | None = 1.0):
        self.name, self.quantiles, self.clip_max = name, np.asarray(quantiles), clip_max
        self.resid_q = None
        self.profile = None

    def fit(self, P: Prepared, W_train: WindowDataset) -> "NaiveForecaster":
        tr = P.split.train_end
        if self.name == "profile":
            how = (P.dow * 24 + P.tod)[: tr + 1]
            y, v = P.y[: tr + 1], P.valid[: tr + 1]
            prof = np.zeros((168, P.N))
            for k in range(168):
                m = how == k
                num = (y[m] * v[m]).sum(0)
                den = v[m].sum(0)
                prof[k] = np.divide(num, den, out=np.zeros(P.N), where=den > 0)
            self.profile = prof
        t0 = W_train.origins
        pt = _point(P, self.name, t0, W_train.H, self.profile)
        f = t0[:, None] + np.arange(1, W_train.H + 1)[None, :]
        r = P.y[f] - pt
        v = P.valid[f]
        self.resid_q = np.stack([np.quantile(r[:, h][v[:, h]], self.quantiles) for h in range(W_train.H)])  # [H, Q]
        return self

    def predict(self, P: Prepared, W: WindowDataset) -> np.ndarray:
        pt = _point(P, self.name, W.origins, W.H, self.profile)                # [n, H, N]
        yq = pt[..., None] + self.resid_q[None, :, None, :]
        yq = np.maximum(yq, 0.0)
        if self.clip_max is not None:
            yq = np.minimum(yq, self.clip_max)
        return np.sort(yq, axis=-1).astype(np.float32)
