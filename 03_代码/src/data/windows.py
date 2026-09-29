"""滑动窗口数据集（设计文档 4.3 的张量字典）。

不依赖 PyTorch：batch() 返回 numpy 数组字典，训练脚本再转成张量。
预测起点 t0 = 最后一个已观测小时；目标 t0+1..t0+H 必须整体落在该部分（train/val/test）内，
历史 t0-L+1..t0 可以早于该部分（标准做法，不构成泄露）。
"""
from __future__ import annotations

import numpy as np

from .dataset import Prepared


class WindowDataset:
    def __init__(self, P: Prepared, part: str, L: int = 24, H: int = 24, stride: int = 1,
                 future_weather: str = "oracle", price_lags: int = 0):
        self.P, self.part, self.L, self.H = P, part, L, H
        self.future_weather = future_weather
        self.price_lags = price_lags
        lo, hi = P.split.part_range(part)
        first = max(L - 1, lo - 1, price_lags)
        last = hi - H
        if last < first:
            raise ValueError(f"{part} 期太短，放不下一个 L={L}、H={H} 的窗口")
        self.origins = np.arange(first, last + 1, stride, dtype=np.int64)
        self._h = np.arange(-L + 1, 1)
        self._f = np.arange(1, H + 1)

    def __len__(self) -> int:
        return len(self.origins)

    @property
    def n_cov_fut(self) -> int:
        return 12 if self.future_weather == "oracle" else 6

    def batch(self, idx) -> dict:
        P = self.P
        t0 = self.origins[np.asarray(idx)]
        hist = t0[:, None] + self._h[None, :]
        fut = t0[:, None] + self._f[None, :]
        cov_fut = P.cal[fut]
        if self.future_weather == "oracle":
            cov_fut = np.concatenate([cov_fut, P.weather[fut]], axis=-1)
        out = {
            "t0": t0,
            "x": np.stack([P.y_in[hist], P.stale[hist]], axis=-1),            # [B, L, N, 2]
            "cov_hist": np.concatenate([P.cal[hist], P.weather[hist]], -1),    # [B, L, 12]
            "tod_hist": P.tod[hist], "dow_hist": P.dow[hist],                  # [B, L]
            "dl_hist": P.dl[hist],                                             # [B, L, N]（仅消融 A6 使用）
            "cov_fut": cov_fut,                                                # [B, H, 6|12]
            "dl_fut": P.dl[fut],                                               # [B, H, N]
            "spill_fut": P.spill[fut],                                         # [B, H, N, K]
            "shift_fut": P.shift[fut],                                         # [B, H, N]
            "ctx_fut": P.ctx[fut],                                             # [B, H]
            "y_fut": P.y[fut],                                                 # [B, H, N]
            "valid_fut": P.valid[fut],                                         # [B, H, N]
        }
        if self.price_lags > 0:
            ext = t0[:, None] + np.arange(1 - self.price_lags, self.H + 1)[None, :]
            out["dl_fut_ext"] = P.dl[ext]                                      # [B, H+M, N]
        return out

    def iterate(self, batch_size: int, shuffle: bool = False, rng: np.random.Generator | None = None,
                max_batches: int | None = None):
        order = np.arange(len(self))
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        n = 0
        for s in range(0, len(order), batch_size):
            if max_batches is not None and n >= max_batches:
                break
            yield self.batch(order[s: s + batch_size])
            n += 1
