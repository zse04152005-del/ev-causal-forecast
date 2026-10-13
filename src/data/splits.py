"""按天的时间顺序切分（设计文档 9.6）：训练 2022-09-01~2023-01-05，验证 01-06~01-23，测试 01-24~02-28。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_end: int          # 训练期最后一个小时的下标（含）
    val_end: int            # 验证期最后一个小时的下标（含）
    T: int

    def part_range(self, part: str) -> tuple[int, int]:
        """返回该部分目标时刻的下标范围 [lo, hi]（含两端）。"""
        if part == "train":
            return 0, self.train_end
        if part == "val":
            return self.train_end + 1, self.val_end
        if part == "test":
            return self.val_end + 1, self.T - 1
        raise ValueError(part)

    def describe(self, time) -> dict:
        t = pd.DatetimeIndex(time)
        return {p: (str(t[a].date()), str(t[b].date())) for p in ("train", "val", "test")
                for a, b in [self.part_range(p)]}


def make_split(time, mode: str = "ratio_days", ratios=(0.7, 0.1, 0.2), train_end: str | None = None,
               val_end: str | None = None) -> Split:
    t = pd.DatetimeIndex(time)
    days = t.normalize()
    ud = days.unique()
    if mode == "ratio_days":
        nd = len(ud)
        ntr = int(round(nd * ratios[0]))
        nva = int(round(nd * ratios[1]))
        last_train_day, last_val_day = ud[ntr - 1], ud[ntr + nva - 1]
    elif mode == "dates":
        last_train_day, last_val_day = pd.Timestamp(train_end), pd.Timestamp(val_end)
    else:
        raise ValueError(mode)
    tr = int(np.flatnonzero(days <= last_train_day)[-1])
    va = int(np.flatnonzero(days <= last_val_day)[-1])
    return Split(train_end=tr, val_end=va, T=len(t))
