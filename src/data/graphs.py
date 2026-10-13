"""图结构：对称化邻接、距离高斯核、POI 相似度，均加自环后行归一化（设计文档 5.1）。"""
from __future__ import annotations

import numpy as np


def symmetric_adjacency(adj: np.ndarray) -> np.ndarray:
    """adj.csv 不对称且含自环：取 A ∨ Aᵀ 并去掉自环（4 个孤立小区靠其他图获得空间信息）。"""
    a = ((adj + adj.T) > 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    return a


def distance_kernel(dist_km: np.ndarray, sigma: float | None = None, eps: float = 0.1) -> np.ndarray:
    d = dist_km.copy()
    off = d[~np.eye(len(d), dtype=bool)]
    sigma = float(np.std(off)) if sigma is None else sigma
    k = np.exp(-(d / sigma) ** 2)
    k[k < eps] = 0.0
    np.fill_diagonal(k, 0.0)
    return k


def poi_similarity(features: np.ndarray, topk: int = 10) -> np.ndarray:
    """features [N, F]（三类 POI 密度取对数）；余弦相似度，每行保留前 topk 个邻居。"""
    x = features - features.mean(axis=0, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    out = np.zeros_like(s)
    k = min(topk, len(s) - 1)
    idx = np.argsort(-s, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    out[rows, idx] = np.clip(s[rows, idx], 0.0, None)
    return out


def row_normalize(a: np.ndarray, self_loop: bool = True) -> np.ndarray:
    a = a + np.eye(len(a)) if self_loop else a.copy()
    rs = a.sum(axis=1, keepdims=True)
    return np.divide(a, rs, out=np.zeros_like(a), where=rs > 0)


def build_supports(adj: np.ndarray, dist_km: np.ndarray, poi_feat: np.ndarray, use_adj: bool = True,
                   use_dist: bool = True, use_poi: bool = True, poi_topk: int = 10, dist_eps: float = 0.1) -> list:
    sup = []
    if use_adj:
        sup.append(row_normalize(symmetric_adjacency(adj)))
    if use_dist:
        sup.append(row_normalize(distance_kernel(dist_km, eps=dist_eps)))
    if use_poi:
        sup.append(row_normalize(poi_similarity(poi_feat, poi_topk)))
    return [s.astype(np.float32) for s in sup]


# ---------------------------------------------------------------- 深度基线用的图矩阵（GCN、STGCN、ASTGCN）
def sym_norm_adj(adj: np.ndarray, self_loop: bool = True) -> np.ndarray:
    """D^{-1/2}(A + I)D^{-1/2}（GCN，Kipf & Welling 2017）。"""
    a = np.asarray(adj, dtype=np.float64)
    if self_loop:
        a = a + np.eye(len(a))
    d = a.sum(axis=1)
    inv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    return (inv[:, None] * a * inv[None, :]).astype(np.float32)


def scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """L̃ = 2L/λmax − I，L = I − D^{-1/2}AD^{-1/2}（孤立节点的 L 行取单位行）。"""
    a = np.asarray(adj, dtype=np.float64)
    n = len(a)
    lap = np.eye(n) - sym_norm_adj(a, self_loop=False).astype(np.float64)
    lam = float(np.linalg.eigvalsh((lap + lap.T) / 2).max())
    lam = lam if lam > 1e-6 else 2.0
    return (2.0 * lap / lam - np.eye(n)).astype(np.float32)


def cheb_polynomials(l_tilde: np.ndarray, K: int) -> np.ndarray:
    """切比雪夫多项式 T_0..T_{K-1}，返回 [K, N, N]。"""
    n = len(l_tilde)
    polys = [np.eye(n, dtype=np.float64), l_tilde.astype(np.float64)]
    for _ in range(2, K):
        polys.append(2 * l_tilde @ polys[-1] - polys[-2])
    return np.stack(polys[:K]).astype(np.float32)
