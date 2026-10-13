"""数据质量：冻结（插补）值检测（设计文档 9.7）。

UrbanEV 的预处理用前后向填充补缺失、用相邻值替换异常值，会产生长时间不变的冻结段。
- frozen_mask：双向定义（需要知道整段有多长），只用于**掩码**训练损失、锚定估计和评价指标
- stale_flag：只用过去信息的"已连续不变"标记，可以作为**模型输入**，不泄露未来
"""
from __future__ import annotations

import numpy as np


def _same_as_prev(occ: np.ndarray, dur: np.ndarray) -> np.ndarray:
    same = np.zeros(occ.shape, dtype=bool)
    same[1:] = (occ[1:] == occ[:-1]) & (dur[1:] == dur[:-1])
    return same


def frozen_mask(occ: np.ndarray, dur: np.ndarray, min_run_frac: int = 6, min_run_any: int = 24) -> np.ndarray:
    """[T, N] 布尔：属于冻结段的小区·小时。

    冻结段 = 占用与时长同时连续不变，且
    （时长为非整数、大于 0、持续 ≥ min_run_frac 小时）或（持续 ≥ min_run_any 小时）。
    与数据核查脚本 check7.py 的定义完全一致。
    """
    same = _same_as_prev(occ, dur)
    run_id = np.cumsum(~same, axis=0)                    # 每段从 same=False 的位置开始
    out = np.zeros(occ.shape, dtype=bool)
    for j in range(occ.shape[1]):
        rid = run_id[:, j]
        length = np.bincount(rid)
        starts = np.flatnonzero(~same[:, j])             # 每段起点，与 rid 的取值一一对应（rid 从 1 开始）
        v0 = dur[starts, j]
        frac = np.abs(v0 - np.round(v0)) > 1e-6
        flag_run = ((length[1:] >= min_run_frac) & frac & (v0 > 0)) | (length[1:] >= min_run_any)
        flag = np.concatenate([[False], flag_run])
        out[:, j] = flag[rid]
    return out


def stale_flag(occ: np.ndarray, dur: np.ndarray, hours: int = 6) -> np.ndarray:
    """[T, N] float32：截至 t（含 t）占用与时长已连续 hours 个小时相同 → 1。只用过去信息。"""
    same = _same_as_prev(occ, dur)
    run = np.zeros(occ.shape, dtype=np.int64)
    for t in range(occ.shape[0]):
        run[t] = np.where(same[t], run[t - 1] + 1, 1) if t > 0 else 1
    return (run >= hours).astype(np.float32)


def outage_hours(dur: np.ndarray, hour: np.ndarray, ratio: float = 0.6, window_days: int = 15) -> np.ndarray:
    """[T] 布尔：全市时长低于同一时刻前后 window_days 天中位数的 ratio 倍（全市层面的采集异常）。"""
    tot = dur.sum(axis=1)
    out = np.zeros(len(tot), dtype=bool)
    half = window_days // 2
    for h in range(24):
        idx = np.flatnonzero(hour == h)
        x = tot[idx]
        med = np.array([np.median(x[max(0, k - half): k + half + 1]) for k in range(len(x))])
        out[idx] = x < ratio * med
    return out
