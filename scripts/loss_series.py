"""把逐点预测（predictions_test.npz，每个约 24 MB）压缩成做显著性检验用的损失序列。

每个模型组（如 v2_main_A0，对应运行 v2_main_A0_s0 … s4）：先算每个种子的损失，再对种子取平均。
输出键：`<组>|mae_o`[起点, 步长]、`|mae_z`[小区]、`|crps_o`、`|crps_z`、`|seed_mae`[种子]，以及 `steps`。
只统计质量掩码有效的点；CRPS 用分位数近似（2 × 各分位 pinball 损失的平均），与 src/eval 的口径一致。

用法（在仓库根目录下）：
    python scripts/loss_series.py --runs <runs 目录> --out results/comparisons/dm/loss_a.npz 组1 组2 …
内存：一次只读一个 npz（约 1 GB），4 GB 内存可跑。之后用 scripts/significance_tests.py 做检验。
"""
import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--sub", type=int, default=0, help="also write <group>@<sub> keys averaged over the first <sub> seeds found")
ap.add_argument("groups", nargs="+")
a = ap.parse_args()
P = prepare(load_config(_bootstrap.DEFAULT_CFG))
out, st = {}, None
for g in a.groups:
    Lo, Lz, Co, Cz = [], [], [], []
    for s in range(a.seeds):
        f = os.path.join(a.runs, f"{g}_s{s}", "predictions_test.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        o, st, q = z["origins"], z["steps"], z["quantiles"]
        fidx = o[:, None] + st[None, :]
        y, v = P.y[fidx].astype(np.float32), P.valid[fidx]
        yq = z["yq"].astype(np.float32)
        ae = np.abs(yq[..., int(np.argmin(np.abs(q - 0.5)))] - y) * v
        d = y[..., None] - yq
        pin = np.maximum(q * d, (q - 1) * d).mean(-1) * 2 * v
        nv_o, nv_z = np.maximum(v.sum(2), 1), np.maximum(v.sum((0, 1)), 1)
        Lo.append(ae.sum(2) / nv_o); Lz.append(ae.sum((0, 1)) / nv_z)
        Co.append(pin.sum(2) / nv_o); Cz.append(pin.sum((0, 1)) / nv_z)
        del yq, d, pin, ae
    if Lo:
        out[g + "|mae_o"], out[g + "|mae_z"] = np.mean(Lo, 0), np.mean(Lz, 0)
        out[g + "|crps_o"], out[g + "|crps_z"] = np.mean(Co, 0), np.mean(Cz, 0)
        out[g + "|seed_mae"] = np.array([l.mean() for l in Lo])
        if a.sub and len(Lo) > a.sub:                              # reference averaged over the same number of seeds as a group
            k = f"{g}@{a.sub}"
            out[k + "|mae_o"], out[k + "|mae_z"] = np.mean(Lo[:a.sub], 0), np.mean(Lz[:a.sub], 0)
            out[k + "|crps_o"], out[k + "|crps_z"] = np.mean(Co[:a.sub], 0), np.mean(Cz[:a.sub], 0)
            out[k + "|seed_mae"] = np.array([l.mean() for l in Lo[:a.sub]])
        print(g, len(Lo), round(float(np.mean(Lo)), 5), round(float(np.mean(Co)), 5), flush=True)
    else:
        print(g, "没有找到 predictions_test.npz", flush=True)
out["steps"] = st
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, **out)
