"""分区面、坐标换算与小区静态特征（设计文档 9.4、9.5）。

- 分区面 SZ_districts 为 WGS84（Web 墨卡托），TAZID 字段对应 UrbanEV 的小区编号
- 数据集作者用 GCJ-02 原始站点坐标直接落区；为保持同一口径，POI（WGS84）先转 GCJ-02 再落区
- 小区 348 的属性表缺面积和质心：面积用 inf.csv，质心按多边形几何计算
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._shp import (assign_points, gcj_to_wgs, ll_to_merc, merc_to_ll, read_dbf, read_shp, ring_area,
                   wgs_to_gcj)

__all__ = ["load_districts", "build_zone_static", "gcj_to_wgs", "wgs_to_gcj", "merc_to_ll", "ll_to_merc"]


def load_districts(data_dir: str):
    base = os.path.join(data_dir, "SZ_districts", "SZ_districts")
    recs = read_dbf(base + ".dbf")
    shapes = read_shp(base + ".shp")
    tazid = np.array([int(r["TAZID"]) for r in recs])
    return tazid, shapes, pd.DataFrame(recs)


def _centroid_ll(parts) -> tuple[float, float]:
    r = max(parts, key=lambda q: abs(ring_area(q)))
    a = ring_area(r)
    x, y = r[:, 0], r[:, 1]
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    gx = ((x + np.roll(x, -1)) * c).sum() / (6 * a)
    gy = ((y + np.roll(y, -1)) * c).sum() / (6 * a)
    lon, lat = merc_to_ll(gx, gy)
    return float(lon), float(lat)


def build_zone_static(data_dir: str, zones: np.ndarray, stations: pd.DataFrame) -> pd.DataFrame:
    """重新生成小区静态特征（POI 落区约 20 秒）。结果另存到 assets/zone_static.csv 以保证跨机器一致。"""
    tazid, shapes, att = load_districts(data_dir)
    pos = {t: k for k, t in enumerate(tazid)}
    poi = pd.read_csv(os.path.join(data_dir, "poi.csv"))
    lo, la = wgs_to_gcj(poi.longitude.to_numpy(), poi.latitude.to_numpy())
    mx, my = ll_to_merc(lo, la)
    k = assign_points(shapes, mx, my)
    poi["TAZID"] = np.where(k >= 0, tazid[np.maximum(k, 0)], -1)
    names = {"business and residential": "poi_business_res", "food and beverage services": "poi_food",
             "lifestyle services": "poi_life"}
    cnt = (poi[poi.TAZID.isin(zones)].groupby(["TAZID", "primary_types"]).size().unstack(fill_value=0)
           .rename(columns=names).reindex(index=zones, columns=list(names.values()), fill_value=0))
    st = pd.DataFrame(index=pd.Index(zones, name="zone"))
    st["area_km2"] = (stations.groupby("TAZID").area.first() / 1e6).reindex(zones).to_numpy()
    st["road_km"] = att.set_index(att.TAZID.astype(int)).LENG_ROAD.reindex(zones).to_numpy()
    cen = np.array([_centroid_ll(shapes[pos[z]]) for z in zones])
    st["lon"], st["lat"] = cen[:, 0], cen[:, 1]
    st = st.join(cnt)
    st["piles"] = stations.groupby("TAZID").charge_count.sum().reindex(zones).to_numpy()
    st["stations"] = stations.groupby("TAZID").size().reindex(zones).to_numpy()
    return st


def static_matrix(st: pd.DataFrame, power_kw: np.ndarray) -> tuple[np.ndarray, list]:
    """8 维静态特征（取对数后按小区标准化）：桩数、站数、面积、路网密度、三类 POI 密度、平均功率。"""
    area = st.area_km2.to_numpy()
    cols = {
        "log_piles": np.log(st.piles.to_numpy()),
        "log_stations": np.log(st.stations.to_numpy()),
        "log_area": np.log(area),
        "road_density": st.road_km.to_numpy() / area,
        "log_dens_business_res": np.log1p(st.poi_business_res.to_numpy() / area),
        "log_dens_food": np.log1p(st.poi_food.to_numpy() / area),
        "log_dens_life": np.log1p(st.poi_life.to_numpy() / area),
        "log_power_kw": np.log(power_kw),
    }
    x = np.stack(list(cols.values()), axis=1)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    return x.astype(np.float32), list(cols)


def poi_log_density(st: pd.DataFrame) -> np.ndarray:
    area = st.area_km2.to_numpy()
    return np.log1p(st[["poi_business_res", "poi_food", "poi_life"]].to_numpy() / area[:, None])
