"""E-SS 半合成反事实基准的完整流程（设计文档 11.3；大纲 7.2）。不依赖 PyTorch。

train.py 在 `ess.enabled=true` 时：
1. setup_ess：生成半合成数据（semisynthetic.make_semisynthetic）→ 只含所选小区的 Prepared；
   在合成数据的**训练期**上用与真实数据相同的识别流程（小时面板、设计 D、PPML）估计锚定值，写出 ess_anchor.json
   （ess.anchor：cells = 格子 + 合并规则；pooled = 全市一个值，与主实验一致；oracle = 真值，作上界）
2. 照常训练、预测
3. evaluate_ess：测试期上，对分时（处理组）小区施加若干反事实价格路径，比较
   - 模型直接替换价格得到的反事实预测（CPA-STGNN 的 price_override）
   - 插补基线 P0：模型事实预测 × exp(β̂_anchor (Δℓ' − Δℓ))（推论 1）
   与已知的真实反事实；并比较模型内的弹性与真值
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..causal.anchors import merge_levels, save_json
from ..causal.switch_5min import estimate_cells_ppml
from ..causal.switch_did import PanelSpec, build_panel
from ..data.calendar import CONTEXT_NAMES
from ..data.dataset import Prepared
from .counterfactual import plugin_counterfactual
from .semisynthetic import SemiSynth, make_semisynthetic, to_prepared


@dataclass
class ESSRun:
    S: SemiSynth
    Q: Prepared
    anchor_path: str
    scenarios: dict = field(default_factory=dict)      # 名称 → 反事实对数价格 [T, n]
    est: pd.DataFrame | None = None


def scenarios(S: SemiSynth, names=("flat", "spread1.5")) -> dict:
    """反事实价格路径（只改处理组小区）：flat = 取消分时（价格固定在训练期均值）；spreadX = 峰谷差放大 X 倍。"""
    out = {}
    tr = S.treated[None, :]
    for nm in names:
        if nm == "flat":
            lp = np.where(tr, S.ref_lp[None, :], S.lp)
        elif nm.startswith("spread"):
            k = float(nm[len("spread"):])
            lp = np.where(tr, S.ref_lp[None, :] + k * (S.lp - S.ref_lp[None, :]), S.lp)
        else:
            raise ValueError(f"未知反事实情景：{nm}")
        out[nm] = lp
    return out


def ess_anchor_json(Q: Prepared, S: SemiSynth, cfg) -> tuple[dict, pd.DataFrame | None]:
    e, ac, cc = cfg.ess, cfg.anchor, cfg.causal
    base = {"version": "ess", "target": cfg.data.target, "price": cfg.data.price, "estimand": "window_hourly_ppml",
            "cross": [{"ring": f"ring{k + 1}", "delta": 0.0, "se": 0.2} for k in range(Q.K)],
            "shift": {"gamma": 0.0, "se": 1.0}, "lag_weights": None}
    policy = str(e.get("anchor", "cells"))
    if policy == "oracle":
        own = [{"group": int(g), "context": "all", "beta": float(S.beta[Q.groups == g][0]), "se": 0.05, "level": "oracle"}
               for g in np.unique(Q.groups)]
        return dict(base, source="ess_oracle", design="oracle", own=own), None
    spec = PanelSpec(window=int(ac.window_hours), jump_threshold=cc.jump_threshold, eps=cc.eps_pile_hours,
                     exclude_holiday_pm1=cc.exclude_holiday_pm1)
    panel = build_panel(Q.raw["duration"], Q.lp, Q.pricing, Q.groups, Q.time, Q.ring_members, valid=~Q.frozen,
                        t_range=(0, Q.split.train_end), spec=spec, day_start=cfg.data.context.day_start,
                        night_start=cfg.data.context.night_start)
    panel = panel[np.isfinite(panel.Apre) & np.isfinite(panel.Apost)]
    design = str(e.get("design", "D"))
    est = estimate_cells_ppml(panel, Q.G, 4, design=design, exposures=False)
    if policy == "pooled":
        city = est[est.level == "city"].iloc[0]
        own = [{"group": "all", "context": "all", "beta": float(min(city.beta, 0.0)), "se": float(city.se),
                "n_zones": int(city.n_zones), "n_events": int(city.n_events), "level": "city"}]
    elif policy == "cells":
        own = merge_levels(est, Q.G, 4, ac.min_zones, ac.max_se, require_negative=True)
    else:
        raise ValueError(f"未知 ess.anchor：{policy}（cells | pooled | oracle）")
    return dict(base, source=f"ess_{policy}", design=design, own=own, own_levels=est.to_dict(orient="records")), est


def setup_ess(P: Prepared, cfg, out_dir: str) -> ESSRun:
    e = cfg.ess
    clip = cfg.model.get("clip_max")
    S = make_semisynthetic(P, rho=float(e.rho), betas_by_group=tuple(e.betas_by_group), delta=tuple(e.delta),
                           amp=float(e.amp), high_hours=int(e.high_hours), frac_treated=float(e.frac_treated),
                           day_shift_prob=float(e.day_shift_prob), shift_hours=int(e.shift_hours),
                           flat_day_prob=float(e.flat_day_prob), max_frozen=float(e.max_frozen), seed=int(e.seed),
                           clip=None if clip is None else float(clip))
    Q = to_prepared(P, S)
    js, est = ess_anchor_json(Q, S, cfg)
    path = os.path.join(out_dir, "ess_anchor.json")
    save_json(js, path)
    return ESSRun(S=S, Q=Q, anchor_path=path, scenarios=scenarios(S, tuple(e.get("scenarios", ("flat", "spread1.5")))),
                  est=est)


def _slope(r: np.ndarray, d: np.ndarray) -> float:
    m = np.isfinite(r) & np.isfinite(d) & (np.abs(d) > 1e-9)
    return float((r[m] * d[m]).sum() / max((d[m] ** 2).sum(), 1e-12))


def evaluate_ess(E: ESSRun, origins: np.ndarray, H: int, yq_fact: np.ndarray, cf_preds: dict, quantiles,
                 model_beta_nc: np.ndarray | None, anchor_beta_gc: np.ndarray, clip: float | None = 1.0) -> dict:
    """yq_fact [n, H, N, Q]；cf_preds：情景名 → 模型在该价格路径下的预测 [n, H, N, Q]（None = 模型不能替换价格）；
    model_beta_nc [N, C]：模型内的（长期）弹性；anchor_beta_gc [G, C]：锚定值（插补基线 P0 用）。"""
    S, Q = E.S, E.Q
    qi = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    fut = origins[:, None] + np.arange(1, H + 1)[None, :]                   # [n, H]
    valid = Q.valid[fut]                                                     # [n, H, N]
    tr = np.broadcast_to(S.treated[None, None, :], valid.shape)
    m_tr = valid & tr
    y_fact = S.y[fut]
    m_eff = m_tr & (y_fact > 0.01)                                         # 效应（对数比）只在需求不接近 0 的点上比较
    p_fact = yq_fact[..., qi]
    ctx = Q.ctx[fut]
    beta_anchor_nc = anchor_beta_gc[Q.groups]                                # [N, C]
    dl_fact = (S.lp - S.ref_lp[None, :])[fut]
    eps = 1e-3
    rep = {"n_zones": int(len(S.idx)), "n_treated": int(S.treated.sum()), "rho": S.rho,
           "fact_MAE_treated": float(np.abs(p_fact - y_fact)[m_tr].mean()),
           "fact_MAE_all": float(np.abs(p_fact - y_fact)[valid].mean()), "scenarios": {}}
    for nm, lp_alt in E.scenarios.items():
        y_cf = S.true_outcome(lp_alt, clip=clip)[fut]
        dl_cf = (lp_alt - S.ref_lp[None, :])[fut]
        d = dl_cf - dl_fact
        true_r = np.log(y_cf + eps) - np.log(y_fact + eps)
        out = {"true_mean_effect": float(true_r[m_eff].mean()),
               "true_implied_beta": _slope(true_r[m_eff], d[m_eff])}
        cands = {"plugin_P0": plugin_counterfactual(yq_fact, dl_fact, dl_cf, beta_anchor_nc, ctx, clip_max=clip)[..., qi]}
        if cf_preds.get(nm) is not None:
            cands["model"] = cf_preds[nm][..., qi]
        else:
            cands["model"] = p_fact                                          # 价格盲：反事实 = 事实预测
        for k, p_cf in cands.items():
            r = np.log(p_cf + eps) - np.log(p_fact + eps)
            out[k] = {"CF_MAE": float(np.abs(p_cf - y_cf)[m_tr].mean()),
                      "effect_MAE": float(np.abs(r - true_r)[m_eff].mean()),
                      "implied_beta": _slope(r[m_eff], d[m_eff]),
                      "mean_effect": float(r[m_eff].mean())}
        rep["scenarios"][nm] = out
    if model_beta_nc is not None:
        bz = np.asarray(model_beta_nc).mean(axis=1)                          # 各小区平均（跨情境）
        t = S.treated
        rep["elasticity"] = {"MAE_treated": float(np.abs(bz[t] - S.beta[t]).mean()),
                             "sign_correct_treated": float((bz[t] < 0).mean()),
                             "by_group": {int(g): {"true": float(S.beta[Q.groups == g][0]),
                                                   "model": float(bz[t & (Q.groups == g)].mean()) if (t & (Q.groups == g)).any() else None}
                                          for g in np.unique(Q.groups)}}
    rep["anchor_by_group_ctx"] = {int(g): {CONTEXT_NAMES[c]: float(anchor_beta_gc[g, c]) for c in range(anchor_beta_gc.shape[1])}
                                  for g in range(anchor_beta_gc.shape[0])}
    return rep
