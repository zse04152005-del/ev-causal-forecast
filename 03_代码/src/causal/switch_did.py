"""分时电价切换点的准实验估计（设计文档 6.2–6.4）。

对每个（小区 i，小时 t）构造一个"窗口观测"：
    J_{i,t} = log(ȳ_{[t, t+W-1]} + ε) − log(ȳ_{[t-W, t-1]} + ε)          （窗口均值的对数差）
    x_{i,t} = 本区 t 时刻的对数价格跳变（|x| ≤ 阈值记为 0）
    S^(k)_{i,t} = 第 k 环带内邻区价格跳变的平均                              （与模型的溢出特征同一口径）
条件：本区价格在切换前后各 W 小时内不变（干净窗口）；窗口内没有冻结值；可选排除节假日前后。

四种设计：
    A 分时 + 固定小区，日期×时刻固定效应          B = A + 小区×时刻固定效应
    C 仅分时小区，日期×时刻固定效应（主设计）       D = C + 小区×时刻固定效应
标准误按小区聚类（CR1）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import calendar as cal

DESIGNS = {
    "A": {"tou_only": False, "fe": ["dh"]},
    "B": {"tou_only": False, "fe": ["dh", "zh"]},
    "C": {"tou_only": True, "fe": ["dh"]},
    "D": {"tou_only": True, "fe": ["dh", "zh"]},
}


@dataclass
class PanelSpec:
    window: int = 3               # 切换前后各 W 小时
    jump_threshold: float = 0.01
    eps: float = 0.5              # 平滑常数（与 Y 同单位，默认 0.5 桩·小时）
    exclude_holiday_pm1: bool = True


def _rolling_mean(Y: np.ndarray, W: int) -> np.ndarray:
    """M[t] = mean(Y[t : t+W])，末尾不足 W 的位置为 nan。"""
    c = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Y, axis=0)])
    out = np.full(Y.shape, np.nan)
    out[: Y.shape[0] - W + 1] = (c[W:] - c[:-W]) / W
    return out


def _rolling_all(mask: np.ndarray, W: int) -> np.ndarray:
    """A[t] = mask[t : t+W] 全为 True。"""
    c = np.vstack([np.zeros((1, mask.shape[1])), np.cumsum(mask.astype(np.int64), axis=0)])
    out = np.zeros(mask.shape, dtype=bool)
    out[: mask.shape[0] - W + 1] = (c[W:] - c[:-W]) == W
    return out


def build_panel(Y: np.ndarray, lp: np.ndarray, pricing: np.ndarray, groups: np.ndarray, time, members: np.ndarray,
                valid: np.ndarray | None = None, t_range: tuple[int, int] | None = None,
                spec: PanelSpec | None = None, day_start: int = 7, night_start: int = 21) -> pd.DataFrame:
    """构造窗口观测面板（不含弱变动小区）。

    Y [T, N] 需求（正数，通常为充电时长，桩·小时）；lp [T, N] 对数价格；pricing [N]（0 固定 / 1 弱 / 2 分时）；
    members [K, N, N] 环带成员；valid [T, N] 未冻结；t_range = (lo, hi)：窗口必须完全落在 [lo, hi] 内。
    """
    spec = spec or PanelSpec()
    T, N = Y.shape
    W = spec.window
    valid = np.ones((T, N), dtype=bool) if valid is None else valid
    lo, hi = (0, T - 1) if t_range is None else t_range

    x = np.zeros_like(lp)
    x[1:] = lp[1:] - lp[:-1]
    x[np.abs(x) <= spec.jump_threshold] = 0.0
    tou = pricing == 2
    x_tou = np.where(tou[None, :], x, 0.0)

    flat = np.zeros((T, N), dtype=bool)
    flat[1:] = np.abs(lp[1:] - lp[:-1]) < 1e-4                 # flat[t]：t-1→t 价格不变
    # 干净窗口：t-W..t-1 之间不变（flat[t-W+1..t-1]），t..t+W-1 之间不变（flat[t+1..t+W-1]）
    pre_flat = np.zeros((T, N), dtype=bool)
    post_flat = np.zeros((T, N), dtype=bool)
    fW = _rolling_all(flat, W - 1) if W > 1 else np.ones((T, N), dtype=bool)
    pre_flat[W:] = fW[1: T - W + 1]                            # flat[t-W+1 .. t-1]
    post_flat[: T - W + 1] = np.roll(fW, -1, axis=0)[: T - W + 1] if W > 1 else True
    ok_valid = _rolling_all(valid, 2 * W)                      # valid[t-W .. t+W-1]
    ok = np.zeros((T, N), dtype=bool)
    ok[W:] = ok_valid[: T - W]

    m = _rolling_mean(Y, W)
    post = np.full((T, N), np.nan)
    pre = np.full((T, N), np.nan)
    post[:] = m
    pre[W:] = m[: T - W]
    J = np.log(post + spec.eps) - np.log(pre + spec.eps)

    tt = np.arange(T)
    in_range = (tt - W >= lo) & (tt + W - 1 <= hi)
    keep = pre_flat & post_flat & ok & in_range[:, None] & np.isfinite(J)
    keep &= (pricing != 1)[None, :]
    if spec.exclude_holiday_pm1:
        keep &= ~cal.holiday_window(time, 1)[:, None]

    K = members.shape[0]
    S = np.zeros((T, N, K))
    for k in range(K):
        n = members[k].sum(axis=1)
        S[:, :, k] = np.divide(x_tou @ members[k].T, n[None, :], out=np.zeros((T, N)), where=n[None, :] > 0)

    ti, zi = np.nonzero(keep)
    t_idx = pd.DatetimeIndex(time)
    hour = np.asarray(t_idx.hour)[ti]
    day = np.asarray((t_idx.normalize() - t_idx.normalize()[0]).days)[ti]
    ctx = cal.context_index(t_idx, day_start, night_start)[ti]
    x_lag = np.zeros_like(x)
    x_lag[24:] = x[:-24]
    df = pd.DataFrame({
        "zone": zi, "t": ti, "J": J[ti, zi], "x": x[ti, zi], "tou": tou[zi], "group": groups[zi],
        "ctx": ctx, "hour": hour, "day": day,
        "novel": np.abs(x[ti, zi] - x_lag[ti, zi]) > spec.jump_threshold,   # 与前一天同一时刻的跳变不同
        "Apre": pre[ti, zi] * W, "Apost": post[ti, zi] * W,                 # 窗口计数（PPML 用，与 5 分钟面板同名）
    })
    for k in range(K):
        df[f"S{k + 1}"] = S[ti, zi, k]
    df["dh"] = df["day"] * 24 + df["hour"]
    df["zh"] = df["zone"] * 24 + df["hour"]
    return df


def _demean(M: np.ndarray, codes: list, tol: float = 1e-9, max_iter: int = 500) -> np.ndarray:
    M = M.astype(np.float64).copy()
    if not codes:
        return M - M.mean(axis=0, keepdims=True)
    cnts = [np.bincount(c).astype(np.float64) for c in codes]
    for _ in range(max_iter if len(codes) > 1 else 1):
        delta = 0.0
        for c, n in zip(codes, cnts):
            means = np.stack([np.bincount(c, weights=M[:, j], minlength=len(n)) for j in range(M.shape[1])], axis=1)
            means = np.divide(means, n[:, None], out=np.zeros_like(means), where=n[:, None] > 0)
            adj = means[c]
            M -= adj
            delta = max(delta, float(np.abs(adj).max()) if adj.size else 0.0)
        if delta < tol:
            break
    return M


def fe_ols(df: pd.DataFrame, y: str, xs: list, fe: list, cluster: str = "zone") -> dict:
    """吸收固定效应的 OLS，按 cluster 聚类（CR1）标准误。"""
    codes = [pd.factorize(df[f])[0] for f in fe]
    M = _demean(df[[y] + xs].to_numpy(), codes)
    yv, X = M[:, 0], M[:, 1:]
    XtX = X.T @ X
    keep = np.diag(XtX) > 1e-12                    # 没有变动的回归量（例如空格子）不估计
    b = np.full(len(xs), np.nan)
    se = np.full(len(xs), np.nan)
    if keep.any():
        Xk = X[:, keep]
        A = np.linalg.pinv(Xk.T @ Xk)
        bk = A @ (Xk.T @ yv)
        u = yv - Xk @ bk
        g = pd.factorize(df[cluster])[0]
        G = g.max() + 1
        scores = np.zeros((G, Xk.shape[1]))
        np.add.at(scores, g, Xk * u[:, None])
        V = A @ (scores.T @ scores) @ A * (G / max(G - 1, 1))
        b[keep], se[keep] = bk, np.sqrt(np.clip(np.diag(V), 0, None))
    resid_share = float((X[:, 0] ** 2).sum() / max(((df[xs[0]] - df[xs[0]].mean()) ** 2).sum(), 1e-12))
    return {"coef": dict(zip(xs, b)), "se": dict(zip(xs, se)), "n": int(len(df)),
            "n_clusters": int(pd.Series(df[cluster]).nunique()), "resid_var_share_x0": resid_share}


def exposure_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("S") and c[1:].isdigit()]


def estimate_design(df: pd.DataFrame, design: str = "C", exposures: bool = True) -> dict:
    d = DESIGNS[design]
    sub = df[df.tou] if d["tou_only"] else df
    xs = ["x"] + (exposure_cols(sub) if exposures else [])
    r = fe_ols(sub, "J", xs, d["fe"])
    r["design"] = design
    r["n_events"] = int((sub.x != 0).sum())
    r["n_event_zones"] = int(sub.loc[sub.x != 0, "zone"].nunique())
    return r


def estimate_cells(df: pd.DataFrame, n_groups: int, n_ctx: int = 4, design: str = "C", exposures: bool = True) -> pd.DataFrame:
    """三个层级的自身弹性：格子（功能区×情境）、情境、全市；同一回归内估计，固定效应共用。"""
    d = DESIGNS[design]
    sub = (df[df.tou] if d["tou_only"] else df).copy()
    ex = exposure_cols(sub) if exposures else []
    rows = []

    def run(level: str, keys: list, masks: list):
        cols = []
        for key, msk in zip(keys, masks):
            name = f"x_{level}_{key}"
            sub[name] = np.where(msk, sub.x.to_numpy(), 0.0)
            cols.append(name)
        r = fe_ols(sub, "J", cols + ex, d["fe"])
        for key, msk, name in zip(keys, masks, cols):
            ev = msk & (sub.x.to_numpy() != 0)
            rows.append({"level": level, "key": key, "beta": r["coef"][name], "se": r["se"][name],
                         "n_events": int(ev.sum()), "n_zones": int(sub.zone.to_numpy()[ev].size and
                                                                    np.unique(sub.zone.to_numpy()[ev]).size)})
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


def event_path(Y: np.ndarray, lp: np.ndarray, pricing: np.ndarray, time, valid: np.ndarray, t_range: tuple,
               pre: int = 4, post: int = 6, eps: float = 0.5, jump_threshold: float = 0.01,
               exclude_holiday_pm1: bool = True) -> pd.DataFrame:
    """事件研究：β_k = 对数需求相对 t-1 的变化对价格跳变的回归系数（仅分时小区，日期×时刻固定效应）。"""
    T, N = Y.shape
    x = np.zeros_like(lp)
    x[1:] = lp[1:] - lp[:-1]
    x[np.abs(x) <= jump_threshold] = 0.0
    lo, hi = t_range
    tou = np.flatnonzero(pricing == 2)
    hol = cal.holiday_window(time, 1) if exclude_holiday_pm1 else np.zeros(T, dtype=bool)
    t_idx = pd.DatetimeIndex(time)
    hour = np.asarray(t_idx.hour)
    day = np.asarray((t_idx.normalize() - t_idx.normalize()[0]).days)
    rows = []
    for t in range(max(lo + pre + 1, 1), hi - post + 1):
        if hol[t]:
            continue
        seg = lp[t - pre - 1: t + post, :][:, tou]
        d = np.abs(np.diff(seg, axis=0)) > 1e-4
        d[pre, :] = False                                     # 允许 t-1→t 的切换
        good = ~d.any(axis=0) & valid[t - pre - 1: t + post, tou].all(axis=0)
        for j in np.flatnonzero(good):
            z = tou[j]
            base = np.log(Y[t - 1, z] + eps)
            path = np.log(Y[t - pre - 1: t + post, z] + eps) - base
            rows.append((z, t, x[t, z], hour[t], day[t], *path))
    ks = list(range(-pre - 1, post))
    df = pd.DataFrame(rows, columns=["zone", "t", "x", "hour", "day"] + [f"k{k}" for k in ks])
    df["dh"] = df["day"] * 24 + df["hour"]
    out = []
    for k in ks:
        if k == -1:
            out.append({"k": k, "beta": 0.0, "se": 0.0})
            continue
        r = fe_ols(df, f"k{k}", ["x"], ["dh"])
        out.append({"k": k, "beta": r["coef"]["x"], "se": r["se"]["x"]})
    res = pd.DataFrame(out)
    res.attrs["n_events"] = int((df.x != 0).sum())
    return res
