"""把原始数据整理成建模用的数组（设计文档第 2、4 节）。

prepare(cfg) 返回 Prepared：所有与时间、小区有关的特征都在这里一次算好，
训练、评价、因果估计、仿真共用同一份，保证口径一致、没有信息泄露：
- 输入标准化、参考价格、平均功率只用训练期
- 质量掩码（冻结值）只用于损失和指标；模型输入只用"截至当前已不变"的因果标记
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.paths import project_root, resolve
from . import calendar as cal
from .geo import build_zone_static, poi_log_density, static_matrix
from .graphs import build_supports, symmetric_adjacency
from .prices import (PRICING_CODES, classify_pricing, price_matrix, reference_log_price, ring_exposure,
                     ring_members, shift_feature)
from .quality import frozen_mask, stale_flag
from .scaling import Standardizer
from .splits import Split, make_split
from .urbanev import UrbanEVData, load_urbanev

ASSET_NAME = "zone_static.csv"


@dataclass
class Prepared:
    time: pd.DatetimeIndex
    zones: np.ndarray
    split: Split
    y: np.ndarray               # [T, N] 目标（原始尺度：利用率 / 占用率 / kWh）
    y_in: np.ndarray            # [T, N] 标准化后的输入
    valid: np.ndarray           # [T, N] 参与损失与指标的点（未冻结）
    frozen: np.ndarray          # [T, N] 冻结值（无论是否启用掩码都计算）
    stale: np.ndarray           # [T, N] 输入用的因果"已不变"标记
    cal: np.ndarray             # [T, 6]
    weather: np.ndarray         # [T, 6]（训练期标准化）
    lp: np.ndarray              # [T, N] 对数价格
    ref_lp: np.ndarray          # [N] 参考对数价格（训练期均值）
    dl: np.ndarray              # [T, N] Δℓ
    spill: np.ndarray           # [T, N, K]
    shift: np.ndarray           # [T, N]
    ctx: np.ndarray             # [T] 情境 0..3
    tod: np.ndarray             # [T] 小时
    dow: np.ndarray             # [T] 星期
    groups: np.ndarray          # [N] 功能区 0..G-1
    pricing: np.ndarray         # [N] 0 固定 / 1 弱变动 / 2 严格分时
    tou_share: np.ndarray       # [N]
    static: np.ndarray          # [N, 8]
    static_names: list
    supports: list              # 每个 [N, N] float32，行归一化
    ring_members: np.ndarray    # [K, N, N]
    capacity: np.ndarray        # [N] 桩数
    power_kw: np.ndarray        # [N] 训练期平均功率（电量 ÷ 时长）
    dist_km: np.ndarray         # [N, N]
    raw: dict                   # 原始序列（duration / occupancy / volume / volume11 / e_price / s_price）
    target: str = "utilization"
    y_scaler: Standardizer = None
    meta: dict = field(default_factory=dict)
    adj: np.ndarray | None = None  # [N, N] 对称 0/1 邻接（无自环），GCN 类基线与 PIAST 使用

    @property
    def T(self) -> int:
        return len(self.time)

    @property
    def N(self) -> int:
        return len(self.zones)

    @property
    def K(self) -> int:
        return self.spill.shape[2]

    @property
    def G(self) -> int:
        return int(self.groups.max()) + 1

    def tou_mask(self) -> np.ndarray:
        return self.pricing == PRICING_CODES["TOU"]

    def price_features(self, lp_alt: np.ndarray) -> dict:
        """给定另一条对数价格路径 [T, N]，按同一口径算出 Δℓ、S^(k)、U（反事实与仿真用）。"""
        dl = lp_alt - self.ref_lp[None, :]
        return {"dl": dl.astype(np.float32), "spill": ring_exposure(dl, self.ring_members).astype(np.float32),
                "shift": shift_feature(dl).astype(np.float32)}


def _select_zones(data: UrbanEVData, zones) -> UrbanEVData:
    if zones is None:
        return data
    if isinstance(zones, int):
        return data.subset(np.arange(min(zones, data.N)))
    idx = [int(np.flatnonzero(data.zones == int(z))[0]) for z in zones]
    return data.subset(np.array(idx))


def load_static(root: str, cfg, data: UrbanEVData, full_data: UrbanEVData) -> pd.DataFrame:
    """读取 assets/zone_static.csv（缺失时现算并保存）；按当前小区顺序返回。"""
    path = os.path.join(resolve(cfg.paths.assets_dir, root), ASSET_NAME)
    if os.path.exists(path):
        st = pd.read_csv(path, index_col=0)
    else:
        from .clusters import functional_groups
        st = build_zone_static(full_data.data_dir, full_data.zones, full_data.stations)
        st["group"] = functional_groups(st)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        st.to_csv(path)
    missing = set(data.zones.tolist()) - set(st.index.astype(int).tolist())
    if missing:
        raise ValueError(f"{ASSET_NAME} 缺少小区：{sorted(missing)[:10]}")
    st.index = st.index.astype(int)
    return st.loc[data.zones]


def build_target(data: UrbanEVData, target: str) -> np.ndarray:
    if target == "utilization":
        return data["duration"] / data.capacity[None, :]
    if target == "occupancy":
        return data["occupancy"] / data.capacity[None, :]
    if target == "volume":
        return data["volume"].copy()
    raise ValueError(f"未知目标变量：{target}")


def prepare(cfg, data: UrbanEVData | None = None) -> Prepared:
    root = project_root(cfg.paths.get("root"))
    full = data if data is not None else load_urbanev(resolve(cfg.paths.data_dir, root))
    d = _select_zones(full, cfg.data.get("zones"))
    dc = cfg.data
    split = make_split(d.time, dc.split.mode, dc.split.ratios, dc.split.get("train_end"), dc.split.get("val_end"))
    tr = split.train_end

    # 目标与质量
    y = build_target(d, dc.target)
    frozen = frozen_mask(d["occupancy"], d["duration"])
    valid = ~frozen if dc.quality_mask else np.ones_like(frozen)
    valid &= np.isfinite(y)
    stale = stale_flag(d["occupancy"], d["duration"], int(dc.stale_input_hours))
    scaler = Standardizer(per_zone=(dc.scaler == "per_zone")).fit(y[: tr + 1], valid[: tr + 1])
    y_in = scaler.transform(y)

    # 日历与天气
    c = cal.calendar_features(d.time)
    w = np.concatenate([d.weather["central"].to_numpy(), d.weather["airport"].to_numpy()], axis=1)
    wm, ws = w[: tr + 1].mean(axis=0), w[: tr + 1].std(axis=0)
    w = (w - wm) / np.where(ws > 1e-9, ws, 1.0)
    ctx = cal.context_index(d.time, dc.context.day_start, dc.context.night_start)

    # 价格
    p = price_matrix(d["e_price"], d["s_price"], dc.price)
    lp = np.log(p)
    share, label = classify_pricing(p, d.time, dc.tou_strict_share)
    pricing = np.array([PRICING_CODES[s] for s in label], dtype=np.int64)
    ref = reference_log_price(lp, tr)
    dl = lp - ref[None, :]
    members = ring_members(d.dist_km, dc.rings_km)
    spill = ring_exposure(dl, members)
    shift = shift_feature(dl)

    # 静态特征、功能区、图
    st = load_static(root, cfg, d, full)
    dur_tr, vol_tr = d["duration"][: tr + 1].sum(axis=0), d["volume"][: tr + 1].sum(axis=0)
    power = np.divide(vol_tr, dur_tr, out=np.full(d.N, np.nan), where=dur_tr > 0)
    power = np.where(np.isfinite(power), power, np.nanmedian(power))
    static, static_names = static_matrix(st, power)
    g = cfg.graph
    supports = build_supports(d.adj, d.dist_km, poi_log_density(st), g.use_adj, g.use_dist, g.use_poi,
                              g.poi_topk, g.dist_eps)

    t = pd.DatetimeIndex(d.time)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    return Prepared(
        time=t, zones=d.zones, split=split, y=f32(y), y_in=f32(y_in), valid=valid, frozen=frozen, stale=stale,
        cal=f32(c), weather=f32(w), lp=lp, ref_lp=ref, dl=f32(dl), spill=f32(spill), shift=f32(shift),
        ctx=ctx, tod=np.asarray(t.hour, dtype=np.int64), dow=np.asarray(t.dayofweek, dtype=np.int64),
        groups=st["group"].to_numpy(dtype=np.int64), pricing=pricing, tou_share=share, static=static,
        static_names=static_names, supports=supports, ring_members=members, capacity=d.capacity,
        power_kw=power, dist_km=d.dist_km, raw=dict(d.series), target=dc.target, y_scaler=scaler,
        adj=symmetric_adjacency(d.adj).astype(np.float32),
        meta={"split": split.describe(t), "n_tou": int((pricing == 2).sum()), "n_weak": int((pricing == 1).sum()),
              "n_fixed": int((pricing == 0).sum()), "frozen_share": float(frozen.mean())},
    )
