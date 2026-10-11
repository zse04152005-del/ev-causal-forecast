"""方案 C（研究大纲 3.3.4）：逐小时小区面板上的补充估计——面板 PPML 与双重机器学习（DML）。

面板 PPML：E[y_{it}] = exp(β·ℓ_{it} + α_{i,how} + λ_{d,h})，ℓ 为对数总价，how = 一周中的小时（0..167），
    α = 小区 × 周内小时固定效应，λ = 日期 × 时刻固定效应；β 直接是价格弹性。按小区聚类。
DML（部分线性）：log(y/桩数 + ε) = θ·ℓ + g(X) + u，ℓ = m(X) + v；g、m 用梯度提升（按小区分组的交叉拟合），
    θ = Σ ṽ ũ / Σ ṽ²，按小区聚类的稳健标准误。X = 时刻、星期、节假日、气象、静态特征、小区训练期均值（目标编码）。
两者都只用训练期与未冻结的点。它们与切换点设计的区别：识别来自"同一小区、同一周内小时、不同周"的价格变动，
不利用切换时刻的窗口结构，所以只作补充与对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .switch_5min import _demean_w


def ppml_levels(y: np.ndarray, X: np.ndarray, fe_codes: list, cluster: np.ndarray, max_iter: int = 40, tol: float = 1e-7,
                names: list | None = None) -> dict:
    """吸收固定效应的 Poisson 伪极大似然（IRLS）；y ≥ 0，可含 0；聚类稳健标准误（CR1）。"""
    n, p = X.shape
    mu = np.full(n, max(y.mean(), 1e-6))
    eta = np.log(mu)
    b = np.zeros(p)
    for it in range(1, max_iter + 1):
        z = eta + (y - mu) / mu
        M = _demean_w(np.column_stack([z, X]), fe_codes, mu)
        zt, Xt = M[:, 0], M[:, 1:]
        b_new = np.linalg.solve(Xt.T @ (mu[:, None] * Xt), Xt.T @ (mu * zt))
        res = zt - Xt @ b_new
        eta_new = z - res
        mu_new = np.exp(np.clip(eta_new, -30, 30))
        done = np.abs(b_new - b).max() < tol and it > 1
        b, eta, mu = b_new, eta_new, mu_new
        if done:
            break
    z = eta + (y - mu) / mu
    M = _demean_w(np.column_stack([z, X]), fe_codes, mu)
    zt, Xt = M[:, 0], M[:, 1:]
    A = np.linalg.inv(Xt.T @ (mu[:, None] * Xt))
    score_i = Xt * (mu * (zt - Xt @ b))[:, None]
    g = pd.factorize(cluster)[0]
    G = g.max() + 1
    sc = np.zeros((G, p))
    np.add.at(sc, g, score_i)
    V = A @ ((sc.T @ sc) * (G / max(G - 1, 1))) @ A
    names = names or [f"x{k}" for k in range(p)]
    return {"coef": dict(zip(names, b)), "se": dict(zip(names, np.sqrt(np.diag(V)))), "n": int(n), "n_clusters": int(G), "iters": it}


def dml_plr(y: np.ndarray, d: np.ndarray, X: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0,
            max_iter: int = 200) -> dict:
    """部分线性 DML：按 groups（小区）分组的 K 折交叉拟合，梯度提升学 E[y|X] 与 E[d|X]。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    uy = np.zeros(len(y))
    vd = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        gy = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, random_state=seed).fit(X[tr], y[tr])
        gd = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, random_state=seed).fit(X[tr], d[tr])
        uy[te] = y[te] - gy.predict(X[te])
        vd[te] = d[te] - gd.predict(X[te])
    theta = float((vd * uy).sum() / (vd * vd).sum())
    psi = vd * (uy - theta * vd)
    g = pd.factorize(groups)[0]
    G = g.max() + 1
    sc = np.zeros(G)
    np.add.at(sc, g, psi)
    se = float(np.sqrt((sc ** 2).sum() * (G / max(G - 1, 1))) / (vd * vd).sum())
    return {"theta": theta, "se": se, "n": int(len(y)), "n_groups": int(G), "r2_d": float(1 - (vd ** 2).sum() / ((d - d.mean()) ** 2).sum())}
