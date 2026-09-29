import sys, numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from shp import *
from sklearn.cluster import KMeans; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import silhouette_score
from _paths import DATA as D   # 数据路径与输出目录见 _paths.py
recs = read_dbf(f"{D}/SZ_districts/SZ_districts.dbf"); shapes = read_shp(f"{D}/SZ_districts/SZ_districts.shp")
att = pd.DataFrame(recs); att["TAZID"] = att.TAZID.astype(int)
Z = [int(c) for c in pd.read_csv(f"{D}/occupancy.csv", nrows=1).columns[1:]]
poi = pd.read_csv(f"{D}/poi.csv")
lo, la = wgs_to_gcj(poi.longitude.values, poi.latitude.values)      # same frame as the dataset's station->zone assignment
mx, my = ll_to_merc(lo, la); k = assign_points(shapes, mx, my)
poi["TAZID"] = np.where(k >= 0, att.TAZID.values[np.maximum(k, 0)], -1)
print("POIs assigned to any polygon: %.4f; to one of the 275 zones: %.4f" % (np.mean(k >= 0), poi.TAZID.isin(Z).mean()))
cnt = poi[poi.TAZID.isin(Z)].groupby(["TAZID", "primary_types"]).size().unstack(fill_value=0).reindex(Z, fill_value=0)
cnt.columns = ["poi_business_res", "poi_food", "poi_life"] if list(cnt.columns) == ["business and residential", "food and beverage services", "lifestyle services"] else cnt.columns
a = att.set_index("TAZID").reindex(Z)
st = pd.DataFrame(index=Z); st.index.name = "zone"
inf0 = pd.read_csv(f"{D}/inf.csv"); st["area_km2"] = (inf0.groupby("TAZID").area.first() / 1e6).reindex(Z).values
st["road_km"] = a.LENG_ROAD.values
cen = {}
for kidx, parts in enumerate(shapes):
    r = max(parts, key=lambda q: abs(ring_area(q))); ar = ring_area(r); x, y = r[:, 0], r[:, 1]
    c = (x * np.roll(y, -1) - np.roll(x, -1) * y); cen[att.TAZID.values[kidx]] = merc_to_ll(((x + np.roll(x, -1)) * c).sum() / (6 * ar), ((y + np.roll(y, -1)) * c).sum() / (6 * ar))
st["lon"] = [cen[z][0] for z in Z]; st["lat"] = [cen[z][1] for z in Z]
st = st.join(cnt)
inf = pd.read_csv(f"{D}/inf.csv"); st["piles"] = inf.groupby("TAZID").charge_count.sum().reindex(Z).values; st["stations"] = inf.groupby("TAZID").size().reindex(Z).values
dur = pd.read_csv(f"{D}/duration.csv").drop(columns="time"); vol = pd.read_csv(f"{D}/volume.csv").drop(columns="time")
st["kw_per_pile_hour"] = (vol.sum() / dur.sum()).values
frac = pd.read_csv("zone_tou_share.csv", index_col=0).iloc[:, 0]; frac.index = frac.index.astype(int)
st["tou_share_days"] = frac.reindex(Z).values; st["pricing"] = np.where(st.tou_share_days >= .5, "TOU", np.where(st.tou_share_days > 0, "weak", "fixed"))
print(st.describe().T[["mean", "50%", "min", "max"]].round(2))
print("area check vs inf.area (m2): corr %.3f" % np.corrcoef(st.area_km2, inf.groupby("TAZID").area.first().reindex(Z) / 1e6)[0, 1])
# functional clustering on POI structure only
dens = np.log1p(st[["poi_business_res", "poi_food", "poi_life"]].div(st.area_km2, axis=0))
tot = st[["poi_business_res", "poi_food", "poi_life"]].sum(1)
share_br = st.poi_business_res / tot.replace(0, np.nan)
F = pd.concat([dens, share_br.fillna(share_br.median()).rename("share_br")], axis=1)
X = StandardScaler().fit_transform(F)
for kk in range(2, 7):
    lab = KMeans(kk, n_init=20, random_state=0).fit_predict(X); print("k=%d silhouette %.3f sizes %s" % (kk, silhouette_score(X, lab), np.bincount(lab).tolist()))
km = KMeans(3, n_init=50, random_state=0).fit(X); lab = km.labels_
# order clusters by total POI density (low -> high)
order = np.argsort([dens[lab == g].sum(1).mean() for g in range(3)]); rel = {o: i for i, o in enumerate(order)}
st["group"] = [rel[g] for g in lab]
summ = st.groupby("group").agg(n=("piles", "size"), piles_med=("piles", "median"), kw=("kw_per_pile_hour", "median"),
    dens_br=("area_km2", lambda s: 0.0), n_tou=("pricing", lambda s: (s == "TOU").sum()), n_fixed=("pricing", lambda s: (s == "fixed").sum()))
for g in range(3):
    m = st.group == g
    summ.loc[g, "dens_br"] = (st.poi_business_res[m] / st.area_km2[m]).median()
    summ.loc[g, "dens_food"] = (st.poi_food[m] / st.area_km2[m]).median(); summ.loc[g, "dens_life"] = (st.poi_life[m] / st.area_km2[m]).median()
    summ.loc[g, "share_br"] = share_br[m].median(); summ.loc[g, "area_med"] = st.area_km2[m].median()
print(summ.round(2).to_string())
st.to_csv("zone_static.csv")
