"""POI 功能区聚类（设计文档 9.5）：三类 POI 密度的对数 + 商住类占比，K-means。

组编号按 POI 总密度从低到高排序：0 低密度、1 居住混合、2 高密度商业。
正式实验使用 assets/zone_static.csv 中保存的组别（跨机器、跨 sklearn 版本保持一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def functional_groups(st: pd.DataFrame, k: int = 3, seed: int = 0, n_init: int = 50) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    area = st.area_km2.to_numpy()
    counts = st[["poi_business_res", "poi_food", "poi_life"]].to_numpy()
    dens = np.log1p(counts / area[:, None])
    tot = counts.sum(axis=1)
    share = np.divide(counts[:, 0], tot, out=np.full(len(tot), np.nan), where=tot > 0)
    share = np.where(np.isnan(share), np.nanmedian(share), share)
    x = StandardScaler().fit_transform(np.column_stack([dens, share]))
    lab = KMeans(k, n_init=n_init, random_state=seed).fit_predict(x)
    order = np.argsort([dens[lab == g].sum(axis=1).mean() for g in range(k)])
    remap = {int(o): i for i, o in enumerate(order)}
    return np.array([remap[int(g)] for g in lab], dtype=np.int64)
