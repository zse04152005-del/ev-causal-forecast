"""设计 D 与设计 C 之差的小区聚类自助法检验（设计文档 16.2：主设计 D，报告 C 与 D 之差是否显著）。

对小区（聚类单位）有放回地重抽 B 次；每次重抽后（重复抽到的小区当作新的站点重新编号）分别估计 C、D 的 β，
差 = D − C；输出差的均值、自助标准误、百分位区间，以及 C、D 的自助标准误（与聚类稳健标准误对照）。
用法：python scripts/boot_cd_diff.py --data-dir ... --d1 ... --out ... [--reps 100 --window 36]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.causal import switch_5min as s5
from src.causal.switch_did import DESIGNS
from src.data.dataset import prepare
from src.utils.config import load_config
from src.utils.paths import ensure_dir, project_root

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("--data-dir", default=None)
ap.add_argument("--d1", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--reps", type=int, default=100)
ap.add_argument("--window", type=int, default=36)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, list(a.overrides) + ([f"paths.data_dir={a.d1}"] if a.d1 else []))
root = project_root(cfg.paths.get("root"))
out = ensure_dir(a.out or os.path.join(root, "05_实验结果", "因果估计", "5min"))
P = prepare(cfg)
fm = s5.load_fivemin(a.data_dir or os.path.join(root, "02_数据", "interim", "fiveMin"))
zpos = {int(z): i for i, z in enumerate(P.zones)}
S_zone = s5.zone_ring_exposure(fm.zone_P, P.ring_members, P.ref_lp)
inv = s5.invalid_mask(fm, P.frozen, np.array([zpos[int(z)] for z in fm.st_zone]))
T_HI = (P.split.train_end + 1) * 12 - 1
spec = s5.Spec5(window=a.window, jump_threshold=cfg.causal.jump_threshold)
pn, _ = s5.build_panel_5min(fm, {int(z): int(g) for z, g in zip(P.zones, P.groups)}, zpos, S_zone, inv, T_HI, spec)
zones = np.unique(pn.zone)
by_zone = {z: g for z, g in pn.groupby("zone")}
rng = np.random.default_rng(a.seed)
rows = []
t0 = time.time()
for b in range(a.reps):
    draw = rng.choice(zones, size=len(zones), replace=True)
    parts = []
    for i, z in enumerate(draw):
        g = by_zone[z].copy()
        g["zone"] = i                                         # 重复抽到的小区当作不同的聚类
        g["station"] = g["station"] * 1000 + i
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    d["zh"] = d["station"] * s5.STEPS_PER_DAY + d["slot"]
    rc = s5.ppml_pair(d, ["x"], DESIGNS["C"]["fe"])["coef"]["x"]
    rd = s5.ppml_pair(d, ["x"], DESIGNS["D"]["fe"])["coef"]["x"]
    rows.append({"rep": b, "C": rc, "D": rd, "D_minus_C": rd - rc})
    if b % 10 == 9:
        pd.DataFrame(rows).to_csv(os.path.join(out, "boot_cd_diff.csv"), index=False)
        print(f"rep {b + 1}/{a.reps}  {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "boot_cd_diff.csv"), index=False)
full = {k: s5.ppml_pair(pn, ["x"], DESIGNS[k]["fe"])["coef"]["x"] for k in "CD"}
summ = {"window_min": a.window * 5, "reps": a.reps, "beta_C": full["C"], "beta_D": full["D"], "diff_D_minus_C": full["D"] - full["C"],
        "boot_se_C": df.C.std(), "boot_se_D": df.D.std(), "boot_se_diff": df.D_minus_C.std(),
        "diff_ci95": [float(df.D_minus_C.quantile(0.025)), float(df.D_minus_C.quantile(0.975))],
        "share_diff_ge0": float((df.D_minus_C >= 0).mean())}
import json  # noqa: E402
json.dump(summ, open(os.path.join(out, "boot_cd_diff_summary.json"), "w"), indent=1)
print(summ)
