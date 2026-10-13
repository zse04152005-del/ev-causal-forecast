"""日历：法定假日与调休（全项目唯一一份日期表）、日历特征、情境编号。

情境 κ（设计文档 2.1）：工作日 / 休息日 × 白天 [7, 21) / 夜间。
法定假日算休息日，调休上班日算工作日；夜间 0–6 点按该时刻所在日期判断。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = {                       # 法定假日（放假日期，含首尾）
    "中秋节": ("2022-09-10", "2022-09-12"),
    "国庆节": ("2022-10-01", "2022-10-07"),
    "元旦": ("2022-12-31", "2023-01-02"),
    "春节": ("2023-01-21", "2023-01-27"),
}
MAKEUP_WORKDAYS = ["2022-10-08", "2022-10-09", "2023-01-28", "2023-01-29"]   # 调休上班的周末
CONTEXT_NAMES = ["weekday_day", "weekday_night", "weekend_day", "weekend_night"]


def _dates(time) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(time).normalize()


def holiday_days() -> pd.DatetimeIndex:
    out = []
    for a, b in HOLIDAYS.values():
        out.extend(pd.date_range(a, b, freq="D"))
    return pd.DatetimeIndex(out)


def is_holiday(time) -> np.ndarray:
    return np.asarray(_dates(time).isin(holiday_days()))


def is_makeup_workday(time) -> np.ndarray:
    return np.asarray(_dates(time).isin(pd.DatetimeIndex(MAKEUP_WORKDAYS)))


def is_restday(time) -> np.ndarray:
    t = pd.DatetimeIndex(time)
    weekend = np.asarray(t.dayofweek >= 5)
    return (weekend & ~is_makeup_workday(t)) | is_holiday(t)


def calendar_features(time) -> np.ndarray:
    """[T, 6]：小时 sin/cos、星期 sin/cos、法定假日、调休上班日。"""
    t = pd.DatetimeIndex(time)
    h = np.asarray(t.hour, dtype=np.float64)
    d = np.asarray(t.dayofweek, dtype=np.float64)
    return np.stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7),
        is_holiday(t).astype(np.float64), is_makeup_workday(t).astype(np.float64),
    ], axis=1)


def context_index(time, day_start: int = 7, night_start: int = 21) -> np.ndarray:
    """0 工作日白天，1 工作日夜间，2 休息日白天，3 休息日夜间。"""
    t = pd.DatetimeIndex(time)
    h = np.asarray(t.hour)
    night = (h < day_start) | (h >= night_start)
    return (2 * is_restday(t).astype(np.int64) + night.astype(np.int64)).astype(np.int64)


def holiday_window(time, pm_days: int = 1) -> np.ndarray:
    """法定假日及前后各 pm_days 天（因果估计中排除，节假日定价与需求变化同时发生）。"""
    days = holiday_days()
    ext = days.union(days - pd.Timedelta(days=pm_days)).union(days + pd.Timedelta(days=pm_days))
    return np.asarray(_dates(time).isin(ext))
