"""价格特征：定价类型、价格偏离 Δℓ、环带暴露 S^(k)、负荷转移特征 U、切换事件表。"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICING_CODES = {"fixed": 0, "weak": 1, "TOU": 2}


def price_matrix(e_price: np.ndarray, s_price: np.ndarray, kind: str = "total") -> np.ndarray:
    if kind == "total":
        return e_price + s_price
    if kind == "electricity":
        return e_price.copy()
    raise ValueError(f"未知价格口径：{kind}")


def classify_pricing(p: np.ndarray, time, strict_share: float = 0.5, decimals: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """返回 (share [N], label [N])。share = 有日内价格变动的天数占比。

    label：'TOU'（share ≥ strict_share）、'weak'（0 < share < strict_share）、'fixed'（share = 0）。
    与数据核查 check2.py 一致：90 / 27 / 158（按总价）。
    """
    day = pd.DatetimeIndex(time).normalize()
    df = pd.DataFrame(np.round(p, decimals))
    varies = df.groupby(np.asarray(day)).nunique() > 1
    share = varies.mean(axis=0).to_numpy()
    label = np.where(share >= strict_share, "TOU", np.where(share > 0, "weak", "fixed"))
    return share, label


def reference_log_price(lp: np.ndarray, train_end: int) -> np.ndarray:
    """参考价格 ℓ̄_i：训练期（下标 0..train_end）的对数价格均值。"""
    return lp[: train_end + 1].mean(axis=0)


def ring_members(dist_km: np.ndarray, rings) -> np.ndarray:
    """[K, N, N]：members[k, i, j] = 1 表示 j 在 i 的第 k 个环带内（不含 i 自身）。"""
    N = dist_km.shape[0]
    d = dist_km.copy()
    d[np.arange(N), np.arange(N)] = np.inf
    return np.stack([((d >= lo) & (d < hi)).astype(np.float64) for lo, hi in rings])


def ring_exposure(dl: np.ndarray, members: np.ndarray) -> np.ndarray:
    """[T, N, K]：S[t, i, k] = 第 k 环带内邻区 Δℓ 的平均（环带为空时为 0）。"""
    T, N = dl.shape
    K = members.shape[0]
    out = np.zeros((T, N, K), dtype=np.float64)
    for k in range(K):
        n = members[k].sum(axis=1)
        s = dl @ members[k].T
        out[:, :, k] = np.divide(s, n[None, :], out=np.zeros_like(s), where=n[None, :] > 0)
    return out


def shift_feature(dl: np.ndarray) -> np.ndarray:
    """U_t = Δℓ_{t+1} − Δℓ_t（下一小时价格变化；最后一小时取 0）。"""
    u = np.zeros_like(dl)
    u[:-1] = dl[1:] - dl[:-1]
    return u


def switch_events(lp: np.ndarray, threshold: float = 0.01, flat_hours: int = 3) -> pd.DataFrame:
    """所有 |Δlog p| > threshold 的切换：zone_idx、t（新价格的第一个小时）、dlp、clean（前后各 flat_hours 小时价格不变）。"""
    d = np.zeros_like(lp)
    d[1:] = lp[1:] - lp[:-1]
    t_idx, z_idx = np.nonzero(np.abs(d) > threshold)
    T = lp.shape[0]
    clean = np.zeros(len(t_idx), dtype=bool)
    for n, (t, z) in enumerate(zip(t_idx, z_idx)):
        if t - flat_hours >= 0 and t + flat_hours <= T:
            pre = lp[t - flat_hours: t, z]
            post = lp[t: t + flat_hours, z]
            clean[n] = np.ptp(pre) < 1e-4 and np.ptp(post) < 1e-4
    return pd.DataFrame({"zone_idx": z_idx, "t": t_idx, "dlp": d[t_idx, z_idx], "clean": clean})
