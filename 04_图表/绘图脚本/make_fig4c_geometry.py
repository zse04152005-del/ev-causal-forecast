"""图 4(c) 的地图素材：从 UrbanEV 小区边界提取示例小区周围的真实几何，存成小文件，绘图脚本只读这个文件。

用法（任选其一）：
    python make_fig4c_geometry.py                       # 在 04_图表/绘图脚本/ 下运行，自动找 03_代码 与 02_数据
    python make_fig4c_geometry.py --code <03_代码> --data <urbanev_github/data> --zone 525

输出 assets/fig4c_zone<id>.json：
- 以中心小区质心为原点的局部平面坐标（km；x 向东、y 向北；经度 × 111.32·cos(纬度)，纬度 × 110.57）
- 窗口内每个多边形：小区编号、是否在 275 个研究小区内、所属环带（按质心距离，与 src/data/prices.ring_members 一致）、
  定价类型（与 src/data/prices.classify_pricing 一致：严格分时 / 弱变动 / 固定）
- 质心圆点放在 distance.csv 给出的精确距离上（沿投影方向），保证图上圆点落在哪个环带与模型完全一致
- 全市 491 个多边形的简化轮廓（定位小图用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def find_root(start: str) -> str:
    p = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "03_代码")):
            return p
        p = os.path.dirname(p)
    return ""


def simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas–Peucker 折线简化（闭合环：首尾点保留）。"""
    if len(pts) < 4:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        L = np.hypot(*seg)
        mid = pts[a + 1: b]
        if L < 1e-12:
            d = np.hypot(*(mid - p).T)
        else:
            d = np.abs(seg[0] * (mid[:, 1] - p[1]) - seg[1] * (mid[:, 0] - p[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = find_root(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=os.path.join(root, "03_代码") if root else None)
    ap.add_argument("--data", default=os.path.join(root, "02_数据", "raw", "urbanev_github", "data") if root else None)
    ap.add_argument("--zone", type=int, default=525)
    ap.add_argument("--window", type=float, default=7.6, help="窗口半宽（km）")
    ap.add_argument("--out", default=os.path.join(here, "assets"))
    a = ap.parse_args()
    if not a.code or not a.data:
        sys.exit("找不到 03_代码 或数据目录，请用 --code、--data 指定")
    sys.path.insert(0, a.code)
    import pandas as pd

    from src.data._shp import merc_to_ll, read_dbf, read_shp
    from src.data.prices import classify_pricing
    from src.data.urbanev import load_urbanev

    U = load_urbanev(a.data)
    zones = U.zones
    _, label = classify_pricing(U["e_price"] + U["s_price"], U.time)
    st = pd.read_csv(os.path.join(a.code, "assets", "zone_static.csv"), index_col=0)
    st.index = st.index.astype(int)
    st = st.loc[zones]
    i = int(np.flatnonzero(zones == a.zone)[0])
    lon0, lat0 = float(st.lon.iloc[i]), float(st.lat.iloc[i])
    kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57

    def local(lon, lat):
        return np.c_[(np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky]

    d = U.dist_km[i].astype(float).copy()
    d[i] = np.inf
    rings = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    zi = {int(z): k for k, z in enumerate(zones)}
    recs = read_dbf(os.path.join(a.data, "SZ_districts", "SZ_districts.dbf"))
    shapes = read_shp(os.path.join(a.data, "SZ_districts", "SZ_districts.shp"))
    polys, city = [], []
    for rec, parts in zip(recs, shapes):
        if not parts:
            continue
        taz = int(rec["TAZID"])
        xy_parts = []
        for ring in parts:
            lon, lat = merc_to_ll(ring[:, 0], ring[:, 1])
            xy_parts.append(local(lon, lat))
        city.append([np.round(simplify(p, 0.12), 3).tolist() for p in xy_parts])
        allp = np.vstack(xy_parts)
        W = a.window
        if not (allp[:, 0].min() < W and allp[:, 0].max() > -W and allp[:, 1].min() < W and allp[:, 1].max() > -W):
            continue                                          # 外接矩形与窗口不相交
        e = {"id": taz, "in_data": taz in zi, "parts": [np.round(simplify(p, 0.004), 3).tolist() for p in xy_parts]}
        if taz in zi:
            j = zi[taz]
            e["pricing"] = str(label[j])
            if j == i:
                e["ring"] = 0
                e["dot"] = [0.0, 0.0]
                e["dist_km"] = 0.0
            else:
                c = local(st.lon.iloc[j], st.lat.iloc[j])[0]
                r = float(d[j])
                ang = float(np.arctan2(c[1], c[0]))
                e["dist_km"] = round(r, 4)
                e["ring"] = next((k + 1 for k, (lo, hi) in enumerate(rings) if lo <= r < hi), None)
                e["dot"] = [round(r * np.cos(ang), 4), round(r * np.sin(ang), 4)]
                e["dot_shift_km"] = round(float(np.hypot(*(np.array(e["dot"]) - c))), 4)
        polys.append(e)
    members = [[e["id"] for e in polys if e.get("ring") == k + 1] for k in range(3)]
    # 与 ring_members 核对：窗口内的环带成员必须与按全体小区计算的完全相同
    for k, (lo, hi) in enumerate(rings):
        ref = sorted(int(zones[j]) for j in np.flatnonzero((d >= lo) & (d < hi)))
        if sorted(members[k]) != ref:
            sys.exit(f"环带 {k + 1} 的成员与 ring_members 不一致：{sorted(members[k])} vs {ref}")
    out = {
        "centre": a.zone, "group": int(st.group.iloc[i]), "pricing": str(label[i]), "lon0": lon0, "lat0": lat0,
        "km_per_deg": [kx, ky], "rings_km": rings, "window_km": a.window,
        "ring_counts": {f"ring{k + 1}": {c: int(sum(1 for e in polys if e.get("ring") == k + 1 and e["pricing"] == c))
                                         for c in ("TOU", "weak", "fixed")} for k in range(3)},
        "zones": polys, "city": city,
        "source": "UrbanEV SZ_districts + distance.csv + e_price/s_price（总价）",
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"fig4c_zone{a.zone}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    shift = [e["dot_shift_km"] for e in polys if "dot_shift_km" in e]
    print(f"已写入 {path}（{os.path.getsize(path) / 1024:.0f} KB）；窗口内多边形 {len(polys)} 个；"
          f"环带成员 {[len(m) for m in members]}；圆点相对质心的最大移动 {max(shift):.3f} km")
    print(out["ring_counts"])


if __name__ == "__main__":
    main()
