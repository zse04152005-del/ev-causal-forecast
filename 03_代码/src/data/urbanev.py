"""读取 UrbanEV GitHub 版逐小时小区级数据（02_数据/raw/urbanev_github/data/）。

注意（数据核查结论，见设计文档 9.1）：
- occupancy.csv 是"不可用或忙碌的桩数"，不是百分比；最大值等于该小区桩数
- duration.csv 是正在充电的桩·小时；volume.csv 由额定功率推算（kWh）
- 站点坐标为 GCJ-02，POI 为 WGS84
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

TS_FILES = {
    "occupancy": "occupancy.csv",
    "duration": "duration.csv",
    "volume": "volume.csv",
    "volume11": "volume-11kW.csv",
    "e_price": "e_price.csv",
    "s_price": "s_price.csv",
}
WEATHER_FILES = {"central": "weather_central.csv", "airport": "weather_airport.csv"}
WEATHER_COLS = ["T", "U", "nRAIN"]          # 设计文档 9.8：去掉气压，保留气温、湿度、降雨等级


@dataclass
class UrbanEVData:
    time: pd.DatetimeIndex
    zones: np.ndarray                        # [N] int
    series: dict                             # name -> [T, N] float64
    weather: dict                            # 'central' / 'airport' -> DataFrame [T, 3]
    adj: np.ndarray                          # [N, N] 0/1（原始，不对称）
    dist_km: np.ndarray                      # [N, N]
    stations: pd.DataFrame                   # inf.csv（筛选后，17,532 桩）
    stations_raw: pd.DataFrame               # inf_raw.csv
    capacity: np.ndarray                     # [N] 桩数（inf.csv 按 TAZID 求和）
    data_dir: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.time)

    @property
    def N(self) -> int:
        return len(self.zones)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.series[name]

    def subset(self, idx) -> "UrbanEVData":
        """按小区下标取子集（冒烟实验用）；所有与小区有关的数组一起切。"""
        idx = np.asarray(idx)
        keep = set(self.zones[idx].tolist())
        return replace(
            self,
            zones=self.zones[idx],
            series={k: v[:, idx] for k, v in self.series.items()},
            adj=self.adj[np.ix_(idx, idx)],
            dist_km=self.dist_km[np.ix_(idx, idx)],
            stations=self.stations[self.stations.TAZID.isin(keep)].copy(),
            stations_raw=self.stations_raw[self.stations_raw.TAZID.isin(keep)].copy(),
            capacity=self.capacity[idx],
        )

    def poi(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.data_dir, "poi.csv"))


def _read_ts(path: str) -> tuple[pd.DatetimeIndex, list[int], np.ndarray]:
    df = pd.read_csv(path)
    if df.columns[0] != "time":
        raise ValueError(f"{path} 第一列应为 time")
    t = pd.DatetimeIndex(pd.to_datetime(df["time"]))
    zones = [int(c) for c in df.columns[1:]]
    return t, zones, df.iloc[:, 1:].to_numpy(dtype=np.float64)


def load_urbanev(data_dir: str) -> UrbanEVData:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到 UrbanEV 数据目录：{data_dir}")
    time = zones = None
    series = {}
    for name, fn in TS_FILES.items():
        t, z, arr = _read_ts(os.path.join(data_dir, fn))
        if time is None:
            time, zones = t, z
        else:
            if not t.equals(time):
                raise ValueError(f"{fn} 的时间索引与 occupancy.csv 不一致")
            if z != zones:
                raise ValueError(f"{fn} 的小区顺序与 occupancy.csv 不一致")
        if np.isnan(arr).any():
            raise ValueError(f"{fn} 含缺失值")
        series[name] = arr
    step = np.diff(time.values).astype("timedelta64[m]").astype(int)
    if not (step == 60).all():
        raise ValueError("时间索引不是连续的逐小时序列")

    weather = {}
    for key, fn in WEATHER_FILES.items():
        w = pd.read_csv(os.path.join(data_dir, fn))
        w["time"] = pd.to_datetime(w["time"], format="%Y/%m/%d %H:%M")
        w = w.set_index("time").reindex(time)
        if w[WEATHER_COLS].isna().any().any():
            raise ValueError(f"{fn} 与需求数据的时间没有完全对齐")
        weather[key] = w[WEATHER_COLS].astype(float)

    zone_ids = np.array(zones, dtype=np.int64)
    adj = pd.read_csv(os.path.join(data_dir, "adj.csv"))
    dist = pd.read_csv(os.path.join(data_dir, "distance.csv"))
    for name, m in (("adj.csv", adj), ("distance.csv", dist)):
        if [int(c) for c in m.columns] != zones:
            raise ValueError(f"{name} 的小区顺序与时间序列不一致")
    stations = pd.read_csv(os.path.join(data_dir, "inf.csv"))
    stations_raw = pd.read_csv(os.path.join(data_dir, "inf_raw.csv"))
    cap = stations.groupby("TAZID").charge_count.sum().reindex(zone_ids)
    if cap.isna().any():
        raise ValueError("有小区在 inf.csv 中没有站点")
    return UrbanEVData(
        time=time, zones=zone_ids, series=series, weather=weather,
        adj=adj.to_numpy(dtype=np.int64), dist_km=dist.to_numpy(dtype=np.float64) / 1000.0,
        stations=stations, stations_raw=stations_raw, capacity=cap.to_numpy(dtype=np.float64),
        data_dir=data_dir,
    )
