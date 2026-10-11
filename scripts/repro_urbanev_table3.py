"""按 UrbanEV 数据集论文（Li et al. 2025）的官方协议复现表 3 中 LO 与 FCNN 两行（只用 numpy，不需要 torch）。

目的：确认我们手里的数据版本与论文一致（LO 是确定性的，应逐位吻合），
以及官方 FCNN（单层线性，12 个滞后，各小区共享权重）的量级。

官方协议（code/main.py、utils.py、train.py）：
- 目标：小区占用率 occupancy / 桩数（不标准化）
- 6 折滚动：第 k 折用前 k 个月，80% 训练、10% 验证、10% 测试
- seq_len 12，直接预测第 pred_len 步（3、6、9、12）
- FCNN：Linear(12 → 1)，Adam(lr 1e-3, weight_decay 1e-5)，MSE，batch 32（打乱、丢弃不足一批），20 轮，取验证损失最低的一轮
- 指标：MAE、RMSE、MAPE（真值 ≤ 0.02 的点把真值和预测都改成 |·| + 0.02）、RAE

用法：python scripts/repro_urbanev_table3.py --d1 <UrbanEV/data> [--out 结果目录] [--seeds 3]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PAPER = {  # 论文表 3（×10⁻²；MAPE 为 %）
    ("LO", "MAE"): [4.91, 7.26, 9.17, 9.98], ("LO", "RMSE"): [9.75, 12.52, 14.65, 15.45],
    ("LO", "MAPE"): [25.39, 39.07, 50.92, 56.70], ("LO", "RAE"): [36.62, 54.05, 68.08, 74.05],
    ("FCNN", "MAE"): [6.11, 7.33, 7.54, 6.62], ("FCNN", "RMSE"): [9.47, 10.74, 10.95, 9.79],
    ("FCNN", "MAPE"): [40.59, 50.12, 52.67, 46.22], ("FCNN", "RAE"): [45.62, 54.62, 56.02, 49.08],
}
HORIZONS = (3, 6, 9, 12)


def load_occ(d1: str, denom: str = "sum") -> tuple[np.ndarray, pd.DatetimeIndex]:
    """denom = "sum"：小区桩数 = 站点桩数求和；"official"：照搬官方 dict(zip(TAZID, charge_count))，
    在现行站点级 inf.csv 上只取到每个小区最后一个站点的桩数（论文发布时 inf.csv 可能是小区级）。"""
    inf = pd.read_csv(os.path.join(d1, "inf.csv"))
    occ = pd.read_csv(os.path.join(d1, "occupancy.csv"), index_col=0)
    if denom == "sum":
        cc = inf.groupby(inf["TAZID"].astype(str))["charge_count"].sum().to_dict()
    else:
        cc = dict(zip(inf["TAZID"].astype(str), inf["charge_count"]))
    for c in occ.columns:
        occ[c] = occ[c] / cc[c]
    return occ.values.astype(np.float64), pd.to_datetime(occ.index)


def split(time: pd.DatetimeIndex, fold: int):
    months = list(time.month.unique())
    n = int(time.month.isin(months[:fold]).sum())
    tr = int(n * 0.8)
    va = int(tr + n * 0.1)
    return tr, va, n


def windows(x: np.ndarray, L: int, h: int):
    """官方 create_rnn_data：X[i] = x[i:i+L]，y[i] = x[i+L+h-1]，i < len-L-h。"""
    n = len(x) - L - h
    idx = np.arange(n)[:, None] + np.arange(L)[None, :]
    return x[idx], x[np.arange(n) + L + h - 1]              # [n, L, N], [n, N]


def metrics(pred: np.ndarray, real: np.ndarray) -> dict:
    eps = 2e-2
    r, p = real.copy(), pred.copy()
    m = r <= eps
    r[m] = np.abs(r[m]) + eps
    p[m] = np.abs(p[m]) + eps
    mape = np.mean(np.abs(p - r) / np.maximum(np.abs(r), np.finfo(np.float64).eps))
    return {"MAE": np.mean(np.abs(pred - real)), "RMSE": np.sqrt(np.mean((pred - real) ** 2)), "MAPE": mape,
            "RAE": np.sum(np.abs(p - r)) / np.sum(np.abs(r.mean() - r))}


def lo(train_valid: np.ndarray, test: np.ndarray, h: int) -> np.ndarray:
    pred = np.empty_like(test)
    pred[:h] = train_valid[-h:]
    pred[h:] = test[:-h]
    return pred


def fcnn(Xtr, ytr, Xva, yva, Xte, seed, epochs=20, bs=32, lr=1e-3, wd=1e-5):
    """Linear(L→1)，各小区共享；样本单位为 (窗口, 小区)，但批次按窗口取（与官方 DataLoader 一致）。"""
    rng = np.random.default_rng(seed)
    L = Xtr.shape[1]
    bound = 1 / np.sqrt(L)
    w, b = rng.uniform(-bound, bound, L), rng.uniform(-bound, bound)
    mw, vw, mb, vb, t = np.zeros(L), np.zeros(L), 0.0, 0.0, 0
    b1, b2, e = 0.9, 0.999, 1e-8
    best, best_wb = np.inf, (w.copy(), b)
    Xtr_ = Xtr.transpose(0, 2, 1)                               # [n, N, L]
    Xva_ = Xva.transpose(0, 2, 1)
    for _ in range(epochs):
        order = rng.permutation(len(Xtr_))
        for s in range(0, len(order) - bs + 1, bs):
            j = order[s:s + bs]
            x, y = Xtr_[j], ytr[j]
            r = x @ w + b - y                                    # [bs, N]
            gw = 2 * np.einsum("bn,bnl->l", r, x) / r.size + wd * w       # torch Adam 的 weight_decay 加在梯度上
            gb = 2 * r.mean() + wd * b
            t += 1
            mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw ** 2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
            w = w - lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + e)
            b = b - lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + e)
        vl = np.mean((Xva_ @ w + b - yva) ** 2)
        if vl < best:
            best, best_wb = vl, (w.copy(), b)
    w, b = best_wb
    return Xte.transpose(0, 2, 1) @ w + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--denom", default="sum", choices=["sum", "official"])
    a = ap.parse_args()
    occ, time = load_occ(a.d1, a.denom)
    rows = []
    for fold in range(1, 7):
        tr, va, n = split(time, fold)
        train, valid, test = occ[:tr], occ[tr:va], occ[va:n]
        for h in HORIZONS:
            L = 12
            tv = np.vstack([train, valid, test[:L + h]])
            p = lo(tv, test[L + h:], h)
            rows.append({"model": "LO", "fold": fold, "h": h, "seed": -1, **metrics(p, test[L + h:])})
            Xtr, ytr = windows(train, L, h)
            Xva, yva = windows(valid, L, h)
            Xte, yte = windows(test, L, h)
            for s in range(a.seeds):
                p = fcnn(Xtr, ytr, Xva, yva, Xte, seed=s)
                rows.append({"model": "FCNN", "fold": fold, "h": h, "seed": s, **metrics(p, yte)})
    df = pd.DataFrame(rows)
    per = df.groupby(["model", "h", "fold"])[["MAE", "RMSE", "MAPE", "RAE"]].mean().groupby(["model", "h"]).mean()
    out = []
    for (m, h), r in per.iterrows():
        k = HORIZONS.index(h)
        for met in ("MAE", "RMSE", "MAPE", "RAE"):
            ours = r[met] * 100
            out.append({"model": m, "h": h, "metric": met, "ours": round(ours, 2), "paper": PAPER[(m, met)][k]})
    res = pd.DataFrame(out)
    piv = res.pivot_table(index=["model", "metric"], columns="h", values=["ours", "paper"])
    print(piv.round(2).to_string())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        df.to_csv(os.path.join(a.out, f"urbanev_table3_repro_runs_{a.denom}.csv"), index=False)
        res.to_csv(os.path.join(a.out, f"urbanev_table3_repro_{a.denom}.csv"), index=False)


if __name__ == "__main__":
    main()
