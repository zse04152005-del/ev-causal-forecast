import sys, numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from shp import *
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
recs = read_dbf(f"{D}/SZ_districts/SZ_districts.dbf"); shapes = read_shp(f"{D}/SZ_districts/SZ_districts.shp")
att = pd.DataFrame(recs); att["TAZID"] = att.TAZID.astype(int)
poi = pd.read_csv(f"{D}/poi.csv"); print("POIs", len(poi), poi.primary_types.value_counts().to_dict())
rng = np.random.default_rng(0); sm = poi.sample(60000, random_state=0)
res = {}
for label, (lo, la) in {"POI as WGS84 (no shift)": (sm.longitude.values, sm.latitude.values), "POI shifted WGS84->GCJ": wgs_to_gcj(sm.longitude.values, sm.latitude.values)}.items():
    mx, my = ll_to_merc(lo, la); k = assign_points(shapes, mx, my); res[label] = k
    print(f"{label}: inside any polygon {np.mean(k>=0):.4f}")
# near-coast diagnostic: POIs whose assignment differs between frames
a, b = res.values(); print("assignment differs between frames: %.3f of sampled POIs" % np.mean(a != b))
