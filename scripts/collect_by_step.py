"""把每次运行的逐步长指标（metrics_test.csv）和保形校准结果（conformal_test.csv）汇成两张长表。

用法（在仓库根目录下）：
    python scripts/collect_by_step.py --runs <runs 目录> --out results/comparisons
输出：metrics_by_step.csv（run、group、seed、step、各指标）、conformal_by_step.csv（run、group、seed、step、method、PICP、MPIW、IS）。
分步长精度图（fig_horizon）和消融表用这两张表。
"""
import argparse
import os
import re

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
M, C = [], []
for r in sorted(os.listdir(a.runs)):
    m = re.match(r"(.+)_s(\d+)$", r)
    g, s = (m.group(1), int(m.group(2))) if m else (r, 0)
    for name, acc in (("metrics_test.csv", M), ("conformal_test.csv", C)):
        f = os.path.join(a.runs, r, name)
        if os.path.exists(f):
            d = pd.read_csv(f)
            d.insert(0, "seed", s); d.insert(0, "group", g); d.insert(0, "run", r)
            acc.append(d)
os.makedirs(a.out, exist_ok=True)
pd.concat(M).to_csv(os.path.join(a.out, "metrics_by_step.csv"), index=False)
pd.concat(C).to_csv(os.path.join(a.out, "conformal_by_step.csv"), index=False)
print(len(M), "runs with metrics;", len(C), "with conformal")
