"""反事实评价（设计文档 3.2、10、11.4）。

- switch_consistency：在测试期用同一个估计量（设计 C，窗口算子）分别作用于观测序列和模型预测序列，
  比较两者的切换点跳变 —— 检验模型隐含的价格响应是否与数据一致（口径对齐，原则 P5）
- plugin_counterfactual：推论 1 的"插补基线" P0：任意模型的事实预测 × exp(β̂ (Δℓ' − Δℓ) + δ̂·(S' − S))
- cf_error_decomposition：反事实对数误差 = 事实误差 + (β̂ − β)(Δℓ' − Δℓ)
"""
from __future__ import annotations

import numpy as np

from ..causal.switch_did import PanelSpec, build_panel, estimate_design
from ..data.dataset import Prepared


def _target_scale(P: Prepared) -> np.ndarray:
    return P.capacity if P.target in ("utilization", "occupancy") else np.ones(P.N)


def switch_consistency(P: Prepared, origins: np.ndarray, y_med: np.ndarray, spec: PanelSpec | None = None,
                       design: str = "C", part: str = "test", subset: str = "all") -> dict:
    """y_med [n_origins, H, N]：模型的中位数预测（目标单位），H ≥ 2W。

    subset="novel"：只保留"与前一天同一时刻跳变不同"的观测（新出现或消失的切换）。
    重复出现的切换会被只会照抄日周期的模型复现（命题 1），新切换才能区分模型是否真的理解价格。
    """
    spec = spec or PanelSpec()
    W = spec.window
    if y_med.shape[1] < 2 * W:
        raise ValueError("预测步长不足 2W，无法构造切换前后窗口")
    scale = _target_scale(P)
    Y_obs = P.y.astype(np.float64) * scale[None, :]
    panel = build_panel(Y_obs, P.lp, P.pricing, P.groups, P.time, P.ring_members, valid=~P.frozen,
                        t_range=P.split.part_range(part), spec=spec)
    pos = np.full(P.T, -1, dtype=np.int64)
    pos[origins] = np.arange(len(origins))
    k = pos[np.clip(panel.t.to_numpy() - W - 1, 0, None)]
    has = (panel.t.to_numpy() - W - 1 >= 0) & (k >= 0)
    if subset == "novel":
        has &= panel.novel.to_numpy()
    panel = panel[has].copy()
    k = k[has]
    z = panel.zone.to_numpy()
    pre = np.mean([y_med[k, h, z] for h in range(W)], axis=0) * scale[z]
    post = np.mean([y_med[k, h, z] for h in range(W, 2 * W)], axis=0) * scale[z]
    pred = panel.copy()
    pred["J"] = np.log(post + spec.eps) - np.log(pre + spec.eps)
    r_obs = estimate_design(panel, design)
    r_mod = estimate_design(pred, design)
    bo, so = r_obs["coef"]["x"], r_obs["se"]["x"]
    bm, sm = r_mod["coef"]["x"], r_mod["se"]["x"]
    return {"design": design, "subset": subset, "beta_obs": bo, "se_obs": so, "beta_model": bm, "se_model": sm,
            "diff": bm - bo, "z_diff": (bm - bo) / so if so and np.isfinite(so) else np.nan,
            "n_events": r_obs["n_events"], "n_obs": r_obs["n"]}


def plugin_counterfactual(yq_fact: np.ndarray, dl_fact: np.ndarray, dl_cf: np.ndarray, beta_nc: np.ndarray,
                          ctx: np.ndarray, spill_fact: np.ndarray | None = None, spill_cf: np.ndarray | None = None,
                          delta: np.ndarray | None = None, clip_max: float | None = 1.0) -> np.ndarray:
    """yq_fact [n, H, N, Q]；dl_* [n, H, N]；beta_nc [N, C]；ctx [n, H]。"""
    beta = beta_nc.T[ctx]                                         # [n, H, N]
    eta = beta * (dl_cf - dl_fact)
    if delta is not None and spill_fact is not None:
        eta = eta + ((spill_cf - spill_fact) * delta).sum(-1)
    out = yq_fact * np.exp(eta)[..., None]
    return np.minimum(out, clip_max) if clip_max is not None else out


def cf_error_decomposition(log_pred_fact: np.ndarray, log_y_fact: np.ndarray, beta_hat, beta_true,
                           dprice: np.ndarray) -> dict:
    """推论 1：反事实对数误差 = 事实对数误差 + (β̂ − β)·Δprice（各项取绝对值的平均）。"""
    fact = log_pred_fact - log_y_fact
    causal = (np.asarray(beta_hat) - np.asarray(beta_true)) * dprice
    total = fact + causal
    return {"total": float(np.nanmean(np.abs(total))), "factual": float(np.nanmean(np.abs(fact))),
            "causal": float(np.nanmean(np.abs(causal)))}
