"""E-SS 半合成反事实基准（设计文档 11.3）。

以冻结比例低的**固定电价小区**的真实序列为"无价格效应"的基线（它们的价格从未变过），
给其中一半小区合成分时电价时刻表，再按已知弹性合成需求：
    y* = y · exp(β_i Δℓ* + Σ_k δ_k S*^(k))
- ρ（内生性强度）：高价时段与该小区自身需求高峰的重合程度（0 = 随机，1 = 恰好在高峰）
- day_shift_prob / shift_hours：每天以该概率把时刻表整体平移 ±shift_hours 小时
- flat_day_prob：每天以该概率不执行分时电价（全天取平均价）
  后两项破坏假设 A，使"同一小区、同一时刻、不同日子"的变动可用于识别（设计 B/D）
真实反事实已知：给定任意价格路径，y_cf = y · exp(β Δℓ' + δ·S')。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..data.dataset import Prepared
from ..data.graphs import row_normalize
from ..data.prices import PRICING_CODES, ring_exposure, ring_members, shift_feature


@dataclass
class SemiSynth:
    idx: np.ndarray            # 在原 Prepared 中的小区下标
    treated: np.ndarray        # [n] 布尔
    beta: np.ndarray           # [n] 真实弹性
    delta: np.ndarray          # [K] 真实交叉弹性
    lp: np.ndarray             # [T, n] 合成对数价格
    ref_lp: np.ndarray         # [n]
    y_base: np.ndarray         # [T, n] 原始序列（无价格效应的基线）
    y: np.ndarray              # [T, n] 合成需求
    members: np.ndarray        # [K, n, n]
    rho: float
    day_shift_prob: float
    flat_day_prob: float = 0.0

    def true_outcome(self, lp_alt: np.ndarray, clip: float | None = 1.0) -> np.ndarray:
        """任意价格路径下的真实需求（已知的反事实）。"""
        dl = lp_alt - self.ref_lp[None, :]
        eta = self.beta[None, :] * dl + (ring_exposure(dl, self.members) * self.delta).sum(-1)
        y = self.y_base * np.exp(eta)
        return np.minimum(y, clip) if clip is not None else y


def make_semisynthetic(P: Prepared, rho: float = 0.5, betas_by_group=(-0.2, -0.8, -1.5), delta=(0.0, 0.0, 0.0),
                       amp: float = 0.15, high_hours: int = 10, frac_treated: float = 0.5,
                       day_shift_prob: float = 0.0, shift_hours: int = 3, flat_day_prob: float = 0.0,
                       max_frozen: float = 0.2, seed: int = 0, clip: float | None = 1.0) -> SemiSynth:
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero((P.pricing == PRICING_CODES["fixed"]) & (P.frozen.mean(axis=0) < max_frozen))
    n, T = len(pool), P.T
    treated = np.zeros(n, dtype=bool)
    treated[rng.choice(n, int(round(n * frac_treated)), replace=False)] = True
    tr = P.split.train_end
    y0 = P.y[:, pool].astype(np.float64)
    v0 = P.valid[:, pool]
    hour = P.tod
    day = (np.asarray(P.time.normalize() - P.time.normalize()[0]).astype("timedelta64[D]").astype(int))
    n_days = day.max() + 1
    lp = np.repeat(P.lp[:1, pool], T, axis=0).astype(np.float64)      # 固定电价：用小区真实的固定价格
    for j in np.flatnonzero(treated):
        prof = np.array([y0[: tr + 1][(hour[: tr + 1] == h) & v0[: tr + 1, j], j].mean() for h in range(24)])
        prof = np.nan_to_num(prof, nan=np.nanmean(prof))
        rank = np.argsort(np.argsort(prof)) / 23.0                     # 0 = 最低，1 = 最高
        score = rho * rank + (1 - rho) * rng.random(24)
        high = np.zeros(24)
        high[np.argsort(-score)[:high_hours]] = 1.0
        shifts = np.where(rng.random(n_days) < day_shift_prob, rng.choice([-shift_hours, shift_hours], n_days), 0)
        sched = np.stack([np.roll(high, s) for s in shifts])            # [days, 24]
        sched[rng.random(n_days) < flat_day_prob] = high_hours / 24.0    # 不执行分时电价的日子
        lp[:, j] += amp * (sched[day, hour] - high_hours / 24.0)
    ref = lp[: tr + 1].mean(axis=0)
    dl = lp - ref[None, :]
    groups = P.groups[pool]
    beta = np.array([betas_by_group[g] for g in groups], dtype=np.float64)
    members = ring_members(P.dist_km[np.ix_(pool, pool)], [[0, 2], [2, 4], [4, 6]][: len(delta)])
    eta = beta[None, :] * dl + (ring_exposure(dl, members) * np.asarray(delta)).sum(-1)
    y = y0 * np.exp(eta)
    if clip is not None:
        y = np.minimum(y, clip)
    return SemiSynth(idx=pool, treated=treated, beta=beta, delta=np.asarray(delta, float), lp=lp, ref_lp=ref,
                     y_base=y0, y=y, members=members, rho=rho, day_shift_prob=day_shift_prob,
                    flat_day_prob=flat_day_prob)


def to_prepared(P: Prepared, S: SemiSynth) -> Prepared:
    """把半合成数据装进一个只含所选小区的 Prepared，可直接交给训练、因果估计与评价流程。"""
    i = S.idx
    tr = P.split.train_end
    dl = S.lp - S.ref_lp[None, :]
    scaler = type(P.y_scaler)(per_zone=P.y_scaler.per_zone).fit(S.y[: tr + 1], P.valid[: tr + 1][:, i])
    cap = P.capacity[i]
    raw = {k: v[:, i].copy() for k, v in P.raw.items()}
    raw["duration"] = S.y * cap[None, :]                               # 因果估计用"时长"口径
    sup = [row_normalize(s[np.ix_(i, i)].astype(np.float64), self_loop=False).astype(np.float32) for s in P.supports]
    pricing = np.where(S.treated, PRICING_CODES["TOU"], PRICING_CODES["fixed"]).astype(np.int64)
    return replace(
        P, zones=P.zones[i], y=S.y.astype(np.float32), y_in=scaler.transform(S.y).astype(np.float32),
        valid=P.valid[:, i], frozen=P.frozen[:, i], stale=P.stale[:, i], lp=S.lp, ref_lp=S.ref_lp,
        dl=dl.astype(np.float32), spill=ring_exposure(dl, S.members).astype(np.float32),
        shift=shift_feature(dl).astype(np.float32), groups=P.groups[i], pricing=pricing,
        tou_share=S.treated.astype(float), static=P.static[i], supports=sup, ring_members=S.members,
        capacity=cap, power_kw=P.power_kw[i], dist_km=P.dist_km[np.ix_(i, i)], raw=raw, y_scaler=scaler,
        adj=None if P.adj is None else P.adj[np.ix_(i, i)],
        meta={**P.meta, "semisynthetic": {"rho": S.rho, "day_shift_prob": S.day_shift_prob,
                                           "flat_day_prob": S.flat_day_prob, "n": len(i),
                                           "n_treated": int(S.treated.sum())}},
    )
