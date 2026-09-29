import sys, numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from shp import *
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
recs = read_dbf(f"{D}/SZ_districts/SZ_districts.dbf"); shapes = read_shp(f"{D}/SZ_districts/SZ_districts.shp")
print("polygons", len(shapes), "records", len(recs))
att = pd.DataFrame(recs); att["TAZID"] = att.TAZID.astype(int); att["ZONE"] = att.ZONE.astype(int)
occ = pd.read_csv(f"{D}/occupancy.csv", nrows=1); Z = [int(c) for c in occ.columns[1:]]
print("UrbanEV zones matched by TAZID field:", len(set(Z) & set(att.TAZID)), "| by ZONE field:", len(set(Z) & set(att.ZONE)))
# geometry check: polygon centroid (mercator->ll) vs X,Y attribute
cx = []; 
for parts in shapes:
    r = max(parts, key=lambda q: abs(ring_area(q))); a = ring_area(r); x, y = r[:, 0], r[:, 1]
    c = (x * np.roll(y, -1) - np.roll(x, -1) * y); gx = ((x + np.roll(x, -1)) * c).sum() / (6 * a); gy = ((y + np.roll(y, -1)) * c).sum() / (6 * a)
    cx.append(merc_to_ll(gx, gy))
cx = np.array(cx); off = np.c_[cx[:, 0] - att.X, cx[:, 1] - att.Y]
print("attribute X,Y minus geometric centroid (deg): median", np.median(off, 0).round(5), "p90 abs", np.quantile(np.abs(off), .9, 0).round(5))
# stations: GCJ-02 -> WGS84 -> mercator -> polygon, compare with inf.TAZID
inf = pd.read_csv(f"{D}/inf.csv")
for label, (lo, la) in {"raw GCJ coords": (inf.longitude.values, inf.latitude.values), "converted to WGS84": gcj_to_wgs(inf.longitude.values, inf.latitude.values)}.items():
    mx, my = ll_to_merc(lo, la); k = assign_points(shapes, mx, my)
    got_t = np.where(k >= 0, att.TAZID.values[k], -1); got_z = np.where(k >= 0, att.ZONE.values[k], -1)
    print(f"stations ({label}): inside a polygon {np.mean(k>=0):.3f}; polygon TAZID == inf.TAZID {np.mean(got_t==inf.TAZID.values):.3f}; polygon ZONE == inf.TAZID {np.mean(got_z==inf.TAZID.values):.3f}")
json.dump({"n_poly": len(shapes)}, open("check8_tmp.json", "w"))
