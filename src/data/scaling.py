"""只在训练期、只用有效点拟合的标准化器。"""
from __future__ import annotations

import numpy as np


class Standardizer:
    def __init__(self, per_zone: bool = False):
        self.per_zone = per_zone
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray, valid: np.ndarray | None = None) -> "Standardizer":
        """x [T, N]；valid [T, N] 布尔。"""
        v = np.ones_like(x, dtype=bool) if valid is None else valid
        if self.per_zone:
            w = v.astype(np.float64)
            n = w.sum(axis=0).clip(min=1)
            m = (x * w).sum(axis=0) / n
            s = np.sqrt((((x - m) ** 2) * w).sum(axis=0) / n)
            self.mean, self.std = m, np.where(s > 1e-9, s, 1.0)
        else:
            m, s = x[v].mean(), x[v].std()
            self.mean, self.std = np.float64(m), np.float64(s if s > 1e-9 else 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean
