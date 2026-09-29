"""阶段 6.3：逐小时面板 PPML 与 DML（补充估计，只用训练期）→ panel_supp.csv。

用法：python scripts/estimate_panel_supp.py --d1 <UrbanEV 逐小时数据目录> --out <输出目录>
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.causal.panel_models import dml_plr, ppml_levels
from src.data import calendar as cal
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--d1", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides) + ([f"paths.data_dir={a.d1}"] if a.d1 else []))
root = project_root(cfg.paths.get("root"))
out = ensure_dir(a.out or os.path.join(root, "05_实验结果", "因果估计"))
P = prepare(cfg)
te = P.split.train_end + 1
T, N = te, P.N
t_idx = pd.DatetimeIndex(P.time[:te])
how = (np.asarray(t_idx.dayofweek) * 24 + np.asarray(t_idx.hour))
date = np.asarray((t_idx.normalize() - t_idx.normalize()[0]).days)
hour = np.asarray(t_idx.hour)
hol = cal.holiday_window(t_idx, 1)
valid = (~P.frozen[:te]) & (~hol[:, None])
dur = P.raw["duration"][:te]
lp = P.lp[:te]
cap = P.capacity
rows = []
t0 = time.time()


def sample(mask_zone):
    tt, zz = np.nonzero(valid & mask_zone[None, :])
    return tt, zz


for name, mz in (("TOU zones", P.pricing == 2), ("TOU + fixed", P.pricing != 1)):
    tt, zz = sample(mz)
    y = dur[tt, zz]
    x = lp[tt, zz][:, None]
    fe = [pd.factorize(zz * 168 + how[tt])[0], pd.factorize(date[tt] * 24 + hour[tt])[0]]
    r = ppml_levels(y, x, fe, zz, names=["lp"])
    rows.append({"method": "PPML (zone×how, date×hour FE)", "sample": name, "beta": r["coef"]["lp"], "se": r["se"]["lp"], "n": r["n"],
                 "n_clusters": r["n_clusters"]})
    print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)

# DML：TOU 小区，目标 log(利用率 + 0.01)，特征含小区训练期目标编码
tt, zz = sample(P.pricing == 2)
u = dur[tt, zz] / cap[zz]
y = np.log(u + 0.01)
d = lp[tt, zz]
zmean = pd.Series(y).groupby(zz).transform("mean").to_numpy()
wx = P.weather[:te][tt]
X = np.column_stack([hour[tt], np.asarray(t_idx.dayofweek)[tt], hol[tt], wx, P.static[zz], P.groups[zz], zmean])
for name, dd in (("log price", d),):
    r = dml_plr(y, dd, X, zz)
    rows.append({"method": "DML (gradient boosting, 5-fold by zone)", "sample": "TOU zones", "beta": r["theta"], "se": r["se"],
                 "n": r["n"], "n_clusters": r["n_groups"], "r2_price_on_X": r["r2_d"]})
    print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "panel_supp.csv"), index=False)
print("done")
