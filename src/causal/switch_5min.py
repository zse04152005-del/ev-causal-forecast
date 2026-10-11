"""站点级 5 分钟数据上的切换点估计（阶段 6，设计文档 6.2–6.4、16.2；研究大纲 3.3.2）。

与小时数据版 `switch_did.py` 同一套估计量与固定效应（设计 A–D），但观测单位改为"站点 × 5 分钟"：

    J_{s,t} = log(ū_{[t+d, t+d+W-1]} + ε) − log(ū_{[t-d-W, t-d-1]} + ε)          窗口均值的对数差（d = 甜甜圈步数）
    x_{s,t} = 站点 s 在 t 开始的一次"价格过渡"的对数价格净变动（没有过渡时为 0）
    ΔS^(k)_{z,t} = 窗口后减窗口前的环带暴露 S^(k)（S 用邻区 Δℓ 水平值，与模型的 S 完全同一口径）

价格过渡：站点价格在 gap 步（默认 3 步 = 15 分钟）内连续变动，合并为一次过渡（例如 19:00 → 19:05 → 19:10 分批变价）；
过渡起点 s、终点 e；x = ln p_e − ln p_{s-1}；来回抖动（净变动 ≤ 阈值）不算事件。
干净窗口：前窗内价格不变；过渡结束后到后窗末尾价格不变；两个窗口内没有无效值（冻结段、负值、缺失）。
观测集：分时站点在自己有过切换的"时刻槽"（一天 288 个 5 分钟槽之一）上的全部干净观测（有切换的日子与没有切换的日子），
        另可加入一批固定电价站点作对照（设计 A/B）。
固定效应：设计 C/主对照 = 日期×时刻槽；设计 D（主设计）= 日期×时刻槽 + 站点×时刻槽。
标准误：默认按小区聚类（CR1）；另提供按日期聚类与小区、日期双向聚类。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import calendar as cal
from .switch_did import _demean

STEPS_PER_DAY = 288


@dataclass
class Spec5:
    window: int = 36                 # 前后各 W 个 5 分钟点（36 = 3 小时，与小时数据的锚定口径一致）
    donut: int = 0                   # 甜甜圈：窗口与过渡起点之间空出的步数
    gap: int = 3                     # 合并分批变价的间隔（步）
    span_max: int = 6                # 过渡起点到终点的最大跨度（步）
    jump_threshold: float = 0.01     # |Δ ln p| 不超过此值不算价格变动
    eps: float = 0.01                # 平滑常数（利用率单位；小区级小时数据 0.5 桩·小时 ÷ 平均桩数约 0.008）
    exclude_holiday_pm1: bool = True
    min_slot_events: int = 1         # 某站点的时刻槽至少有几次切换才进入观测集
    outcome: str = "duration"        # duration | occupancy | volume
    n_fixed_controls: int = 0        # 加入多少个固定电价站点作对照（设计 A/B）
    seed: int = 0
    placebo_shift: int = 0           # 安慰剂：把每次过渡的时刻平移这么多步（价格跳变量 x 不变），真实效应应为 0


# ============================================================ 数据读取
def _load(dirpath: str, name: str):
    z = np.load(os.path.join(dirpath, name + ".npz"), allow_pickle=True)
    return z["time"], z["cols"].astype(int), z["X"]


@dataclass
class FiveMin:
    time: pd.DatetimeIndex
    st_ids: np.ndarray               # [Ns]
    st_zone: np.ndarray              # [Ns] 小区编号（TAZID）
    st_piles: np.ndarray             # [Ns]
    P: np.ndarray                    # [T, Ns] 总价（电价 + 服务费）
    dur: np.ndarray
    occ: np.ndarray
    vol: np.ndarray
    zone_ids: np.ndarray             # [Nz] 小区 5 分钟表的列（与 prepare 的小区顺序对齐）
    zone_P: np.ndarray               # [T, Nz]


def load_fivemin(dirpath: str, main_only: bool = True) -> FiveMin:
    time, cols, E = _load(dirpath, "station_e_price")
    _, _, S = _load(dirpath, "station_s_price")
    _, _, D = _load(dirpath, "station_duration")
    _, _, OCC = _load(dirpath, "station_occupancy")
    _, _, V = _load(dirpath, "station_volume")
    inf = pd.read_csv(os.path.join(dirpath, "station_inf.csv")).set_index("station_id")
    keep = np.ones(len(cols), dtype=bool)
    if main_only:
        main = set(pd.read_csv(os.path.join(dirpath, "station_information_zone.csv")).station_id)
        keep = np.array([c in main for c in cols])
    _, zc, ZE = _load(dirpath, "zone_e_price")
    _, _, ZS = _load(dirpath, "zone_s_price")
    return FiveMin(time=pd.DatetimeIndex(time), st_ids=cols[keep], st_zone=inf.loc[cols[keep], "TAZID"].to_numpy(),
                   st_piles=inf.loc[cols[keep], "charge_count"].to_numpy(), P=(E + S)[:, keep], dur=D[:, keep],
                   occ=OCC[:, keep], vol=V[:, keep], zone_ids=zc, zone_P=ZE + ZS)


# ============================================================ 价格过渡
def find_transitions(p: np.ndarray, spec: Spec5, t_hi: int) -> pd.DataFrame:
    """p [T, Ns] 总价。返回过渡表 [j, s, e, x]（只含起点 ≤ t_hi 的，净变动 |x| > 阈值的）。"""
    lp = np.log(p)
    T, Ns = lp.shape
    rows = []
    for j in range(Ns):
        ch = np.flatnonzero(np.abs(lp[1:, j] - lp[:-1, j]) > 1e-6) + 1        # ch[k]：k-1 → k 价格变了
        if len(ch) == 0:
            continue
        starts = [ch[0]]
        ends = []
        for a, b in zip(ch[:-1], ch[1:]):
            if b - a > spec.gap:
                ends.append(a)
                starts.append(b)
        ends.append(ch[-1])
        for s, e in zip(starts, ends):
            if s > t_hi:
                continue
            x = lp[e, j] - lp[s - 1, j]
            if abs(x) > spec.jump_threshold and e - s <= spec.span_max:
                rows.append((j, int(s), int(e), float(x)))
    return pd.DataFrame(rows, columns=["j", "s", "e", "x"])


def station_types(p: np.ndarray, time: pd.DatetimeIndex, t_hi: int, strict_share: float = 0.5) -> np.ndarray:
    """0 固定 / 1 弱变动 / 2 严格分时：训练期内有日内价格变动的天数占比（与小区级 classify_pricing 同一口径）。"""
    day = np.asarray(time[: t_hi + 1].normalize())
    df = pd.DataFrame(np.round(p[: t_hi + 1], 4))
    share = (df.groupby(day).nunique() > 1).mean(axis=0).to_numpy()
    return np.where(share >= strict_share, 2, np.where(share > 0, 1, 0))


# ============================================================ 无效值
def _station_stale(occ: np.ndarray, dur: np.ndarray, run_active: int, run_any: int) -> np.ndarray:
    """站点级"数值长时间完全不变"：占用与时长同时不变，且（占用 > 0 并持续 ≥ run_active 步）或（持续 ≥ run_any 步）。

    与小区级冻结定义不同：站点只有十几根桩，同一批车连续停几个小时时占用与时长本来就不变，
    所以门槛设得高（默认 24 小时 / 48 小时）；小区级的插补段由小时冻结掩码负责。
    """
    same = np.zeros(occ.shape, dtype=bool)
    same[1:] = (occ[1:] == occ[:-1]) & (dur[1:] == dur[:-1])
    out = np.zeros(occ.shape, dtype=bool)
    for j in range(occ.shape[1]):
        rid = np.cumsum(~same[:, j])
        length = np.bincount(rid)
        starts = np.flatnonzero(~same[:, j])
        active = occ[starts, j] > 0
        fl = np.concatenate([[False], ((length[1:] >= run_active) & active) | (length[1:] >= run_any)])
        out[:, j] = fl[rid]
    return out


def invalid_mask(fm: FiveMin, zone_frozen_hourly: np.ndarray | None, zone_index: np.ndarray | None,
                 run_active: int = 288, run_any: int = 576) -> np.ndarray:
    """[T, Ns]：站点级长时间不变、负值/超容量/缺失、小区级小时冻结（若给出，放大 12 倍）。"""
    dur = fm.dur
    bad = ~np.isfinite(dur) | (dur < -1e-6) | (dur > fm.st_piles[None, :] / 12.0 + 1e-6)
    bad |= _station_stale(fm.occ, dur, run_active, run_any)
    if zone_frozen_hourly is not None:
        zf = np.repeat(zone_frozen_hourly, 12, axis=0)                     # [T, Nz]
        bad |= zf[:, zone_index]
    return bad


# ============================================================ 环带暴露（5 分钟）
def zone_ring_exposure(zone_P: np.ndarray, members: np.ndarray, ref_lp: np.ndarray) -> np.ndarray:
    """[T, Nz, K] float32：S = 环带内邻区 Δℓ（价格对参考价格的对数偏离）的均值，与 src/data/prices.ring_exposure 同一口径。"""
    dl = (np.log(zone_P) - ref_lp[None, :]).astype(np.float32)
    K = members.shape[0]
    out = np.zeros(dl.shape + (K,), dtype=np.float32)
    for k in range(K):
        n = members[k].sum(axis=1)
        s = dl @ members[k].T.astype(np.float32)
        out[:, :, k] = np.divide(s, n[None, :], out=np.zeros_like(s), where=n[None, :] > 0)
    return out


def _csum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1,) + a.shape[1:], dtype=np.float64), np.cumsum(a, axis=0, dtype=np.float64)], axis=0)


# ============================================================ 面板
def build_panel_5min(fm: FiveMin, groups_by_zone: dict, zone_pos: dict, S_zone: np.ndarray, invalid: np.ndarray,
                     t_hi: int, spec: Spec5 | None = None, day_start: int = 7, night_start: int = 21) -> tuple[pd.DataFrame, dict]:
    """构造窗口观测面板；列与 switch_did.build_panel 一致（zone、group、ctx、dh、zh、x、J、tou、S1..SK、novel）。

    groups_by_zone：小区编号 → 功能区；zone_pos：小区编号 → S_zone 的列下标；t_hi：训练期最后一个 5 分钟点。
    返回 (面板, 信息字典)。
    """
    spec = spec or Spec5()
    W, d = spec.window, spec.donut
    T = t_hi + 1
    P = fm.P[:T]
    types = station_types(fm.P, fm.time, t_hi)
    tr = find_transitions(P, spec, t_hi)
    if spec.placebo_shift:
        tr = tr.assign(s=tr.s + spec.placebo_shift, e=tr.e + spec.placebo_shift)
        tr = tr[(tr.s >= 1) & (tr.e < t_hi)].reset_index(drop=True)
    tou_j = np.flatnonzero(types == 2)
    rng = np.random.default_rng(spec.seed)
    fixed_pool = np.flatnonzero(types == 0)
    ctrl = np.sort(rng.choice(fixed_pool, size=min(spec.n_fixed_controls, len(fixed_pool)), replace=False)) \
        if spec.n_fixed_controls > 0 else np.array([], dtype=int)
    use = np.concatenate([tou_j, ctrl]).astype(int)

    lp = np.log(P)
    dP = np.zeros((T, len(fm.st_ids)), dtype=bool)
    dP[1:] = np.abs(lp[1:] - lp[:-1]) > 1e-6
    cs = _csum(dP)                                                        # cs[k] = 变动数（下标 < k）
    Y = {"duration": fm.dur, "occupancy": fm.occ, "volume": fm.vol}[spec.outcome][:T].astype(np.float64)
    U = np.clip(Y / (fm.st_piles[None, :] / 12.0), 0, None) if spec.outcome != "volume" else Y
    if spec.outcome == "volume":
        U = Y / np.maximum(fm.st_piles[None, :], 1)
    U = np.where(invalid[:T], 0.0, U)
    cu = _csum(U)
    Yraw = np.where(invalid[:T], 0.0, np.clip(Y, 0, None))               # 窗口计数用（原始单位：时长 = 桩·小时）
    cy = _csum(Yraw)
    cbad = _csum(invalid[:T].astype(np.float64))
    Sz = S_zone[:T]
    K = Sz.shape[2]
    cS = _csum(Sz.astype(np.float64).reshape(T, -1))                      # [T+1, Nz*K]

    t_idx = pd.DatetimeIndex(fm.time[:T])
    slot = (t_idx.hour * 60 + t_idx.minute).to_numpy() // 5
    day = np.asarray((t_idx.normalize() - t_idx.normalize()[0]).days)
    hol = cal.holiday_window(t_idx, 1) if spec.exclude_holiday_pm1 else np.zeros(T, dtype=bool)
    ctx_all = cal.context_index(t_idx, day_start, night_start)
    tt_all = np.arange(T)

    frames = []
    ev_by_j = {j: g for j, g in tr.groupby("j")}
    for j in use:
        g_ev = ev_by_j.get(j)
        # 该站点的"事件时刻槽"
        if g_ev is not None:
            sl = slot[g_ev.s.to_numpy()]
            vals, cnts = np.unique(sl, return_counts=True)
            ev_slots = vals[cnts >= spec.min_slot_events]
        else:
            ev_slots = np.array([], dtype=int)
        if types[j] == 0:                                                # 固定站点：取全体分时站点常见的时刻槽
            ev_slots = COMMON_SLOTS(tr, slot, fm, types, spec)
        if len(ev_slots) == 0:
            continue
        is_slot = np.isin(slot, ev_slots)
        t_lo, t_hi_ = W + d, T - (W + d)
        cand = tt_all[(tt_all >= t_lo) & (tt_all <= t_hi_) & is_slot & ~hol]
        # 事件行与非事件行
        x_at = np.zeros(T)
        e_at = np.full(T, -1)
        if g_ev is not None:
            x_at[g_ev.s.to_numpy()] = g_ev.x.to_numpy()
            e_at[g_ev.s.to_numpy()] = g_ev.e.to_numpy()
        # 关键：t 时刻若是某次过渡"中间"的变动点（dP[t] 真但 t 不是起点）→ 排除；由干净窗口条件自动排除
        xs = x_at[cand]
        es = e_at[cand]
        has_ev = xs != 0
        # 前窗：变动数（下标 [t-d-W+1, t-1]）= 0
        pre_ok = (cs[cand, j] - cs[cand - d - W + 1, j]) == 0
        # 后窗：非事件行 [t, t+d+W-1] 内无变动；事件行 e 之后 [e+1, t+d+W-1] 内无变动，且 e ≤ t+d+W-1
        post_end = cand + d + W - 1
        post_non = (cs[post_end + 1, j] - cs[cand, j]) == 0
        e_safe = np.where(has_ev, es, cand)
        post_ev = (es <= post_end) & ((cs[post_end + 1, j] - cs[np.minimum(e_safe + 1, post_end + 1), j]) == 0)
        post_ok = np.where(has_ev, post_ev, post_non)
        # 无效值：[t-d-W, t+d+W-1] 内无无效点
        bad_ok = (cbad[cand + d + W, j] - cbad[cand - d - W, j]) == 0
        ok = pre_ok & post_ok & bad_ok
        # 非事件行不能落在"变动点"上（dP[t] 为真而没有对应过渡）——post_non 已排除（区间包含 t）
        cand, xs, has_ev = cand[ok], xs[ok], has_ev[ok]
        if len(cand) == 0:
            continue
        post_m = (cu[cand + d + W, j] - cu[cand + d, j]) / W
        pre_m = (cu[cand - d, j] - cu[cand - d - W, j]) / W
        Jv = np.log(post_m + spec.eps) - np.log(pre_m + spec.eps)
        A_post = cy[cand + d + W, j] - cy[cand + d, j]
        A_pre = cy[cand - d, j] - cy[cand - d - W, j]
        z = int(fm.st_zone[j])
        zc = zone_pos.get(z)
        row = {
            "station": np.full(len(cand), j), "zone": np.full(len(cand), z), "t": cand, "J": Jv, "x": xs,
            "Apre": A_pre, "Apost": A_post,
            "tou": np.full(len(cand), types[j] == 2), "group": np.full(len(cand), groups_by_zone.get(z, -1)),
            "ctx": ctx_all[cand], "slot": slot[cand], "day": day[cand],
        }
        if zc is not None:
            base = zc * K
            post_S = (cS[cand + d + W, base:base + K] - cS[cand + d, base:base + K]) / W
            pre_S = (cS[cand - d, base:base + K] - cS[cand - d - W, base:base + K]) / W
            dS = post_S - pre_S
        else:
            dS = np.zeros((len(cand), K))
        for k in range(K):
            row[f"S{k + 1}"] = dS[:, k]
        frames.append(pd.DataFrame(row))
    panel = pd.concat(frames, ignore_index=True)
    panel["dh"] = panel["day"] * STEPS_PER_DAY + panel["slot"]
    panel["zh"] = panel["station"] * STEPS_PER_DAY + panel["slot"]
    # 与前一天同一时刻的跳变不同（新切换）
    xkey = panel.set_index(["station", "t"]).x
    prev = xkey.reindex(pd.MultiIndex.from_arrays([panel.station, panel.t - STEPS_PER_DAY])).to_numpy()
    prev = np.nan_to_num(prev, nan=0.0)
    panel["novel"] = np.abs(panel.x.to_numpy() - prev) > spec.jump_threshold
    info = {"n_transitions_train": int(len(tr)), "n_tou_stations": int(len(tou_j)), "n_fixed_controls": int(len(ctrl)),
            "station_types": {"fixed": int((types == 0).sum()), "weak": int((types == 1).sum()), "TOU": int((types == 2).sum())}}
    return panel, info


_COMMON_CACHE: dict = {}


def COMMON_SLOTS(tr: pd.DataFrame, slot: np.ndarray, fm: FiveMin, types: np.ndarray, spec: Spec5) -> np.ndarray:
    """固定电价对照站点使用的时刻槽：分时站点里至少有 20 次过渡的槽。"""
    key = (id(tr), spec.min_slot_events)
    if key not in _COMMON_CACHE:
        sub = tr[types[tr.j.to_numpy()] == 2]
        vals, cnts = np.unique(slot[sub.s.to_numpy()], return_counts=True)
        _COMMON_CACHE[key] = vals[cnts >= 20]
    return _COMMON_CACHE[key]


# ============================================================ 估计（聚类可选）
def fe_ols_multi(df: pd.DataFrame, y: str, xs: list, fe: list, clusters=("zone",)) -> dict:
    """吸收固定效应的 OLS；clusters 含 1 个变量 → CR1；含 2 个 → 双向聚类（CGM：V1 + V2 − V12，取正半定部分）。"""
    codes = [pd.factorize(df[f])[0] for f in fe]
    M = _demean(df[[y] + xs].to_numpy(), codes)
    yv, X = M[:, 0], M[:, 1:]
    keep = np.diag(X.T @ X) > 1e-12
    b = np.full(len(xs), np.nan)
    se = np.full(len(xs), np.nan)
    if keep.any():
        Xk = X[:, keep]
        A = np.linalg.pinv(Xk.T @ Xk)
        bk = A @ (Xk.T @ yv)
        u = yv - Xk @ bk

        def meat(g):
            g = pd.factorize(g)[0]
            G = g.max() + 1
            sc = np.zeros((G, Xk.shape[1]))
            np.add.at(sc, g, Xk * u[:, None])
            return (sc.T @ sc) * (G / max(G - 1, 1)), G

        if len(clusters) == 1:
            m, G = meat(df[clusters[0]].to_numpy())
        else:
            m1, G1 = meat(df[clusters[0]].to_numpy())
            m2, G2 = meat(df[clusters[1]].to_numpy())
            m12, _ = meat((df[clusters[0]].astype(np.int64) * 100000 + df[clusters[1]].astype(np.int64)).to_numpy())
            m = m1 + m2 - m12
            w, v = np.linalg.eigh((m + m.T) / 2)
            m = (v * np.clip(w, 0, None)) @ v.T
            G = min(G1, G2)
        V = A @ m @ A
        b[keep], se[keep] = bk, np.sqrt(np.clip(np.diag(V), 0, None))
    else:
        G = 0
    return {"coef": dict(zip(xs, b)), "se": dict(zip(xs, se)), "n": int(len(df)), "n_clusters": int(G)}


# ============================================================ PPML（条件泊松 = 二项 logit，吸收固定效应）
def _demean_w(M: np.ndarray, codes: list, w: np.ndarray, tol: float = 1e-8, max_iter: int = 400) -> np.ndarray:
    """加权吸收固定效应（交替投影）。"""
    M = M.astype(np.float64).copy()
    sums = [np.bincount(c, weights=w).astype(np.float64) for c in codes]
    for _ in range(max_iter if len(codes) > 1 else 1):
        delta = 0.0
        for c, sw in zip(codes, sums):
            means = np.stack([np.bincount(c, weights=w * M[:, j], minlength=len(sw)) for j in range(M.shape[1])], axis=1)
            means = np.divide(means, sw[:, None], out=np.zeros_like(means), where=sw[:, None] > 0)
            adj = means[c]
            M -= adj
            delta = max(delta, float(np.abs(adj).max()) if adj.size else 0.0)
        if delta < tol:
            break
    return M


def ppml_pair(df: pd.DataFrame, xs: list, fe: list, clusters=("zone",), pseudo: float = 0.1, max_iter: int = 30) -> dict:
    """窗口后/前的计数比：条件泊松 ⇔ 后窗份额的二项 logit，logit = Σ β x + FE。系数 = 半弹性口径的 log(μ_post/μ_pre) 对 x 的斜率。

    pseudo：给两个窗口计数各加的桩·小时数（防止一侧为 0 时 logit 发散；只影响计数很小的行）。
    返回 coef、se（按 clusters 聚类；双向聚类用 CGM）、n、迭代次数。
    """
    a1 = df["Apost"].to_numpy() + pseudo
    a0 = df["Apre"].to_numpy() + pseudo
    n = a1 + a0
    y = a1 / n
    X = df[xs].to_numpy(dtype=np.float64)
    # 去掉只有 1 个观测的 FE 单元（不携带信息），迭代到没有
    keep = np.ones(len(df), dtype=bool)
    for _ in range(10):
        changed = False
        for f in fe:
            cnt = df.loc[keep, f].map(df.loc[keep, f].value_counts())
            bad = keep.copy()
            bad[keep] = cnt.to_numpy() < 2
            if bad.any():
                keep &= ~bad
                changed = True
        if not changed:
            break
    d = df.loc[keep]
    y, n, X = y[keep], n[keep], X[keep]
    codes = [pd.factorize(d[f])[0] for f in fe]
    eta = np.zeros(len(d))
    b = np.zeros(len(xs))
    it = 0
    for it in range(1, max_iter + 1):
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        w = n * mu * (1 - mu)
        z = eta + (y - mu) / (mu * (1 - mu))
        M = _demean_w(np.column_stack([z, X]), codes, w)
        zt, Xt = M[:, 0], M[:, 1:]
        A = Xt.T @ (w[:, None] * Xt)
        b_new = np.linalg.pinv(A) @ (Xt.T @ (w * zt))
        # 线性预测量：demean 后的 z 减去 Xt b 是 FE 之外的残差，FE 部分 = (z − X b) 的加权 FE 拟合；用 z − resid 还原 η
        res = zt - Xt @ b_new
        eta_new = z - res                                     # = FE 部分 + X b
        if np.abs(b_new - b).max() < 1e-7 and it > 1:
            b, eta = b_new, eta_new
            break
        b, eta = b_new, eta_new
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-8, 1 - 1e-8)
    w = n * mu * (1 - mu)
    z = eta + (y - mu) / (mu * (1 - mu))
    M = _demean_w(np.column_stack([z, X]), codes, w)
    zt, Xt = M[:, 0], M[:, 1:]
    A = np.linalg.pinv(Xt.T @ (w[:, None] * Xt))
    score_i = Xt * (w * (zt - Xt @ b))[:, None]

    def meat(g):
        g = pd.factorize(g)[0]
        G = g.max() + 1
        sc = np.zeros((G, len(xs)))
        np.add.at(sc, g, score_i)
        return (sc.T @ sc) * (G / max(G - 1, 1)), G

    if len(clusters) == 1:
        m, G = meat(d[clusters[0]].to_numpy())
    else:
        m1, G1 = meat(d[clusters[0]].to_numpy())
        m2, G2 = meat(d[clusters[1]].to_numpy())
        m12, _ = meat((d[clusters[0]].astype(np.int64) * 100000 + d[clusters[1]].astype(np.int64)).to_numpy())
        m = m1 + m2 - m12
        ww, vv = np.linalg.eigh((m + m.T) / 2)
        m = (vv * np.clip(ww, 0, None)) @ vv.T
        G = min(G1, G2)
    V = A @ m @ A
    return {"coef": dict(zip(xs, b)), "se": dict(zip(xs, np.sqrt(np.clip(np.diag(V), 0, None)))), "n": int(len(d)),
            "n_clusters": int(G), "iters": it}


# ============================================================ 事件研究（5 分钟）
def add_path_counts(panel: pd.DataFrame, fm: FiveMin, invalid: np.ndarray, ks, width: int = 3, outcome: str = "duration") -> pd.DataFrame:
    """给面板加 A_k 列：以 t 为起点、第 k 步开始的 width 步窗口内的计数（桩·小时）。k < 0 为切换前。无效点按 0 计（面板行本身已保证 [t-W, t+W) 内无无效点）。"""
    Y = {"duration": fm.dur, "occupancy": fm.occ, "volume": fm.vol}[outcome].astype(np.float64)
    Y = np.where(invalid, 0.0, np.clip(Y, 0, None))
    cy = _csum(Y)
    st = panel["station"].to_numpy()
    t = panel["t"].to_numpy()
    out = panel.copy()
    for k in ks:
        out[f"A{k}"] = cy[t + k + width, st] - cy[t + k, st]
    return out


def event_study_ppml(panel: pd.DataFrame, ks, fe: list, width: int = 3, base_len: int = 36, pseudo: float = 0.1,
                     clusters=("zone",)) -> pd.DataFrame:
    """每个 k：后窗 = A_k（width 步），前窗 = 基线（Apre，base_len 步）按长度折算；logit 斜率 = log(μ_k / μ_base) 对 x 的斜率。"""
    rows = []
    scale = width / base_len
    for k in ks:
        d = panel.copy()
        d["Apost"] = d[f"A{k}"]
        d["Apre"] = d["Apre"] * scale
        r = ppml_pair(d, ["x"], fe, clusters=clusters, pseudo=pseudo * scale)
        rows.append({"k": k, "minutes": k * 5, "beta": r["coef"]["x"], "se": r["se"]["x"], "n": r["n"]})
    return pd.DataFrame(rows)


# ============================================================ 三层级格子估计（PPML 版，供合并规则使用）
def estimate_cells_ppml(df: pd.DataFrame, n_groups: int, n_ctx: int = 4, design: str = "D", exposures: bool = True,
                        clusters=("zone",), pseudo: float = 0.1) -> pd.DataFrame:
    """与 switch_did.estimate_cells 同一输出格式：格子（功能区 × 情境）、情境、全市三层的自身弹性（同一回归内、固定效应共用）。"""
    from .switch_did import DESIGNS
    dsg = DESIGNS[design]
    sub = (df[df.tou] if dsg["tou_only"] else df).copy()
    ex = [c for c in sub.columns if c.startswith("S") and c[1:].isdigit()] if exposures else []
    rows = []

    def run(level, keys, masks):
        cols = []
        for key, msk in zip(keys, masks):
            name = f"x_{level}_{key}"
            sub[name] = np.where(msk, sub.x.to_numpy(), 0.0)
            cols.append(name)
        r = ppml_pair(sub, cols + ex, dsg["fe"], clusters=clusters, pseudo=pseudo)
        for key, msk, name in zip(keys, masks, cols):
            ev = msk & (sub.x.to_numpy() != 0)
            rows.append({"level": level, "key": key, "beta": r["coef"][name], "se": r["se"][name],
                         "n_events": int(ev.sum()), "n_zones": int(np.unique(sub.zone.to_numpy()[ev]).size)})
        sub.drop(columns=cols, inplace=True)
        return r

    g, c = sub.group.to_numpy(), sub.ctx.to_numpy()
    run("cell", [(gi, ci) for gi in range(n_groups) for ci in range(n_ctx)],
        [(g == gi) & (c == ci) for gi in range(n_groups) for ci in range(n_ctx)])
    run("context", list(range(n_ctx)), [c == ci for ci in range(n_ctx)])
    r_city = run("city", ["all"], [np.ones(len(sub), dtype=bool)])
    out = pd.DataFrame(rows)
    out.attrs["cross"] = {k: (r_city["coef"][k], r_city["se"][k]) for k in ex}
    return out
