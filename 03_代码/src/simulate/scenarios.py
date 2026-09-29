"""定价情景仿真（设计文档 8.3–8.4）。

每个情景把真实总价矩阵 p [T, N] 变成情景价格 p'，只改分时小区（mask），其余不变：
- S1 widen：峰谷价差扩大 factor 倍（相对当天该小区的平均价）
- S2 flat_period：指定时段改为当天平均价（"增加一个平段"）
- S3 surcharge：对指定小区全天加价 pct
- S4 shift：把每天的价格时刻表整体平移 hours 小时（谷时开始提前 / 推后）
- S5 flatten_service：服务费不再与电价反向 —— 服务费取当天平均，总价只随电价变化
利用率 → 负荷：kW = ŷ × 桩数 × 平均功率。
参数不确定性：β⁽ˢ⁾ ~ N(β̂, se²) 截断在 ≤ 0，由推论 1 解析传播（不需要重新训练）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily(p: np.ndarray, time) -> tuple[np.ndarray, np.ndarray]:
    day = np.asarray(pd.DatetimeIndex(time).normalize().factorize()[0])
    mean = np.zeros_like(p)
    for d in np.unique(day):
        m = day == d
        mean[m] = p[m].mean(axis=0, keepdims=True)
    return day, mean


def widen(p, time, mask, factor: float = 1.2):
    _, mean = _daily(p, time)
    out = p.copy()
    out[:, mask] = mean[:, mask] + factor * (p[:, mask] - mean[:, mask])
    return out


def flat_period(p, time, mask, hours=(21, 22)):
    _, mean = _daily(p, time)
    out = p.copy()
    h = np.isin(pd.DatetimeIndex(time).hour, hours)
    out[np.ix_(h, mask)] = mean[np.ix_(h, mask)]
    return out


def surcharge(p, mask, pct: float = 0.10):
    out = p.copy()
    out[:, mask] = p[:, mask] * (1 + pct)
    return out


def shift_schedule(p, time, mask, hours: int = 1):
    day, _ = _daily(p, time)
    out = p.copy()
    for d in np.unique(day):
        m = np.flatnonzero(day == d)
        if len(m) == 24:
            out[np.ix_(m, mask)] = np.roll(p[np.ix_(m, mask)], hours, axis=0)
    return out


def flatten_service(e_price, s_price, time, mask):
    _, s_mean = _daily(s_price, time)
    s = s_price.copy()
    s[:, mask] = s_mean[:, mask]
    return e_price + s


def to_load_kw(y_util: np.ndarray, capacity: np.ndarray, power_kw: np.ndarray) -> np.ndarray:
    return y_util * capacity * power_kw


def draw_betas(beta: np.ndarray, se: np.ndarray, S: int = 200, seed: int = 0) -> np.ndarray:
    """截断正态（β ≤ 0）的参数抽样，返回 [S, *beta.shape]。"""
    rng = np.random.default_rng(seed)
    out = rng.normal(beta, se, size=(S,) + np.shape(beta))
    bad = out > 0
    while bad.any():
        out[bad] = rng.normal(np.broadcast_to(beta, out.shape)[bad], np.broadcast_to(se, out.shape)[bad])
        bad = out > 0
    return out


def peak_to_average(load_city: np.ndarray, time) -> float:
    """全市负荷的日峰均比（各天平均）。load_city [T]。"""
    day = np.asarray(pd.DatetimeIndex(time).normalize().factorize()[0])
    r = [load_city[day == d].max() / max(load_city[day == d].mean(), 1e-9) for d in np.unique(day)]
    return float(np.mean(r))
