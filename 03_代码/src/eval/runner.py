"""统一的"预测 → 指标 → 校准 → 保存"流程。PyTorch 模型与 numpy 基线共用，输出格式完全相同。

每次运行在 05_实验结果/<run_name>/ 下写出：
- config.yaml                     本次配置
- metrics_test.csv                测试期各报告步长指标（valid 掩码）；metrics_test_unmasked.csv 为不掩码版本
- metrics_val.csv                 验证期指标
- conformal_test.csv              各报告步长：原始分位数区间 / 滚动分割保形 / ACI 的覆盖率与宽度
- predictions_test.npz            测试期预测（float16，步长 1–6 与报告步长，全部分位数），供 DM 检验与切换点一致性使用
- summary.json                    关键数字汇总
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..conformal.aci import aci_calibrate, zone_scale
from ..data.dataset import Prepared
from ..data.windows import WindowDataset
from .metrics import interval_metrics, metrics_table

STORE_STEPS = [1, 2, 3, 4, 5, 6, 9, 12, 24]


def targets(W: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
    f = W.origins[:, None] + np.arange(1, W.H + 1)[None, :]
    return W.P.y[f], W.P.valid[f]


def conformal_table(P: Prepared, W_val: WindowDataset, W_test: WindowDataset, yq_val: np.ndarray, yq_test: np.ndarray,
                    quantiles, steps, ccfg, stride: int = 1) -> pd.DataFrame:
    q = list(np.round(quantiles, 4))
    li, ui = q.index(ccfg.lower_q), q.index(ccfg.upper_q)
    tr = P.split.train_end
    sigma = zone_scale(P.y[: tr + 1], P.valid[: tr + 1])
    y_v, v_v = targets(W_val)
    y_t, v_t = targets(W_test)
    rows = []
    nv = len(W_val)
    for s in steps:
        lo = np.concatenate([yq_val[:, s - 1, :, li], yq_test[:, s - 1, :, li]])
        hi = np.concatenate([yq_val[:, s - 1, :, ui], yq_test[:, s - 1, :, ui]])
        yy = np.concatenate([y_v[:, s - 1], y_t[:, s - 1]])
        vv = np.concatenate([v_v[:, s - 1], v_t[:, s - 1]])
        raw = interval_metrics(lo[nv:], hi[nv:], yy[nv:], vv[nv:], ccfg.alpha)
        rows.append({"step": s, "method": "quantile_raw", **raw})
        for name, adaptive in (("split_rolling", False), ("aci", True)):
            r = aci_calibrate(lo, hi, yy, vv, sigma, ccfg.alpha, ccfg.gamma, ccfg.window, h=s, stride=stride,
                              adaptive=adaptive)
            m = interval_metrics(r["lo"][nv:], r["hi"][nv:], yy[nv:], vv[nv:], ccfg.alpha)
            rows.append({"step": s, "method": name, **m, "alpha_end": float(r["alpha"][-1])})
    return pd.DataFrame(rows)


def save_run(out_dir: str, P: Prepared, cfg, W_val: WindowDataset, W_test: WindowDataset, yq_val: np.ndarray,
             yq_test: np.ndarray, extra: dict | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    quantiles = list(cfg.model.quantiles)
    steps = [s for s in cfg.data.report_steps if s <= W_test.H]
    y_v, v_v = targets(W_val)
    y_t, v_t = targets(W_test)
    mt = metrics_table(yq_test, y_t, v_t, quantiles, steps)
    mt.to_csv(os.path.join(out_dir, "metrics_test.csv"), index=False)
    metrics_table(yq_test, y_t, np.isfinite(y_t), quantiles, steps).to_csv(
        os.path.join(out_dir, "metrics_test_unmasked.csv"), index=False)
    mv = metrics_table(yq_val, y_v, v_v, quantiles, steps)
    mv.to_csv(os.path.join(out_dir, "metrics_val.csv"), index=False)
    ct = conformal_table(P, W_val, W_test, yq_val, yq_test, quantiles, steps, cfg.conformal,
                         stride=int(cfg.train.get("eval_stride", 1)))
    ct.to_csv(os.path.join(out_dir, "conformal_test.csv"), index=False)
    ss = [s for s in STORE_STEPS if s <= W_test.H]
    np.savez_compressed(os.path.join(out_dir, "predictions_test.npz"), origins=W_test.origins, steps=np.array(ss),
                        quantiles=np.array(quantiles), yq=yq_test[:, [s - 1 for s in ss]].astype(np.float16),
                        zones=P.zones)
    avg = mt[mt.step == "avg"].iloc[0].to_dict()
    aci = ct[ct.method == "aci"]
    summary = {
        "target": P.target, "n_test_origins": int(len(W_test)), "test_valid_share": float(v_t.mean()),
        "test_MAE": avg["MAE"], "test_RMSE": avg["RMSE"], "test_WAPE": avg["WAPE"], "test_CRPS_q": avg["CRPS_q"],
        "test_PICP90_raw": avg.get("PICP90"), "test_PICP90_aci_mean": float(aci.PICP.mean()) if len(aci) else None,
        "val_MAE": float(mv[mv.step == "avg"].MAE.iloc[0]), **(extra or {}),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    return summary
