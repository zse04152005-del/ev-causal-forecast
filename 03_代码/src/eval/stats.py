"""显著性检验：Diebold–Mariano（HLN 小样本修正）与按小区的 Wilcoxon 符号秩检验。"""
from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(loss1: np.ndarray, loss2: np.ndarray, h: int = 1) -> dict:
    """loss1、loss2：按预测起点排列的损失序列（已对小区取平均）。d = loss1 − loss2，负值表示模型 1 更好。

    方差用 h−1 阶自协方差（Newey–West 矩形核），再做 Harvey–Leybourne–Newbold 修正，t(n−1) 分布。
    """
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm": np.nan, "p": np.nan, "n": n, "mean_diff": float(np.mean(d)) if n else np.nan}
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm": np.nan, "p": np.nan, "n": n, "mean_diff": float(dbar)}
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm": float(stat), "p": float(p), "n": n, "mean_diff": float(dbar)}


def wilcoxon_zones(err1: np.ndarray, err2: np.ndarray) -> dict:
    """按小区的平均误差成对比较。"""
    a, b = np.asarray(err1, float), np.asarray(err2, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.allclose(a[m], b[m]):
        return {"stat": np.nan, "p": np.nan, "n": int(m.sum()), "share_better": np.nan}
    r = stats.wilcoxon(a[m], b[m])
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n": int(m.sum()),
            "share_better": float(np.mean(a[m] < b[m]))}
