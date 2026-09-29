"""训练并评价 CPA-STGNN（需要 PyTorch）。

用法：
    python scripts/train.py                                   # 默认配置（设计文档 v1.1）
    python scripts/train.py --config configs/experiment/smoke.yaml
    python scripts/train.py model.backbone=staeformer anchor.mode=soft train.seed=1
输出：05_实验结果/runs/<run_name>/（格式见 src/eval/runner.py），另加 model.pt、train_log.csv、price_params.json
"""
import argparse
import datetime
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import _bootstrap  # noqa: E402,F401
from src.causal.anchors import load_anchors  # noqa: E402
from src.causal.switch_did import PanelSpec  # noqa: E402
from src.data.dataset import prepare  # noqa: E402
from src.data.windows import WindowDataset  # noqa: E402
from src.eval.counterfactual import switch_consistency  # noqa: E402
from src.eval.runner import save_run  # noqa: E402
from src.models.cpastgnn import CPASTGNN  # noqa: E402
from src.models.losses import anchor_loss, hier_loss, pinball_loss  # noqa: E402
from src.models.torch_utils import pick_device, predict_array, to_tensors  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.paths import project_root, resolve  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def predict(model, W: WindowDataset, bs: int, device) -> np.ndarray:
    return predict_array(model, W, bs, device, key="y_q")


@torch.no_grad()
def val_loss(model, W: WindowDataset, bs: int, device, quantiles) -> float:
    model.eval()
    num = den = 0.0
    for b in W.iterate(bs):
        tb = to_tensors(b, device)
        w = float(tb["valid_fut"].sum())
        num += float(pinball_loss(model(tb)["y_q"], tb["y_fut"], tb["valid_fut"], quantiles)) * w
        den += w
    return num / max(den, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
    ap.add_argument("--no-consistency", action="store_true", help="跳过测试期切换点一致性检验")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    root = project_root(cfg.paths.get("root"))
    tc, mc, ac = cfg.train, cfg.model, cfg.anchor
    run = tc.get("run_name") or (f"{mc.backbone}_{ac.mode}_{cfg.data.target}_s{tc.seed}_"
                                 f"{datetime.datetime.now():%m%d_%H%M%S}")
    out_dir = os.path.join(resolve(cfg.paths.results_dir, root), "runs", run)
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger(f"train.{run}", os.path.join(out_dir, "train.log"))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    set_seed(int(tc.seed))
    device = pick_device(tc.device)
    log.info(f"运行 {run}，设备 {device}，torch {torch.__version__}")

    t_start = time.time()
    P = prepare(cfg)
    lags = int(mc.get("price_lags", 0) or 0)
    kw = dict(L=cfg.data.L, H=cfg.data.H, future_weather=cfg.data.future_weather, price_lags=lags)
    W_tr, W_va = WindowDataset(P, "train", **kw), WindowDataset(P, "val", **kw)
    W_te = WindowDataset(P, "test", stride=int(tc.eval_stride), **kw)
    log.info(f"数据：T={P.T} N={P.N} 训练/验证/测试起点 {len(W_tr)}/{len(W_va)}/{len(W_te)}；{P.meta['split']}")

    anchors = load_anchors(resolve(ac.file, root), G=P.G, K=P.K)
    placeholder = anchors.is_placeholder                  # 先记下：改写 source 后 is_placeholder 会失效
    if ac.get("fixed_beta") is not None:                  # E-ID 剖面损失：把 β 固定在给定值
        anchors.beta[:] = float(ac.fixed_beta)
        anchors.source = f"{anchors.source}|fixed_beta={ac.fixed_beta}"
    if cfg.data.target == "volume" and mc.get("clip_max") is not None:
        log.warning("目标为电量（kWh），取消 clip_max 上限")
        mc["clip_max"] = None
    if placeholder:
        if not ac.get("allow_placeholder", True):
            raise SystemExit("锚定文件是占位值，而配置不允许使用占位锚定")
        log.warning("正在使用占位锚定（β̂=−0.45）：结果只能用于调试，不能写进论文")
    quantiles = list(mc.quantiles)
    tr = P.split.train_end
    init_q = np.quantile(P.y[: tr + 1][P.valid[: tr + 1]], quantiles)
    model = CPASTGNN(mc, N=P.N, L=cfg.data.L, H=cfg.data.H, K=P.K, cov_hist_dim=12, cov_fut_dim=W_tr.n_cov_fut,
                     static=P.static, supports=P.supports, groups=P.groups, anchors=anchors, anchor_mode=ac.mode,
                     graph_cfg=cfg.graph, init_quantiles=init_q, init_beta=ac.get("init_beta"),
                     anchor_window=int(ac.window_hours)).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"模型参数 {n_par:,}；锚定方式 {ac.mode}；锚定来源 {anchors.source}")
    # 价格参数（θ、ξ、δ、γ）不加权重衰减：否则 β 会被拉向 −softplus(0) = −0.69，污染 E-ID 实验；ξ 的收缩由层级损失负责
    price_ids = {id(p) for m in (model.price, model.spill, model.shift) if m is not None for p in m.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in price_ids], "weight_decay": float(tc.weight_decay)},
              {"params": [p for p in model.parameters() if id(p) in price_ids], "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=float(tc.lr))
    wf = model.wf                                          # 有滞后核时 < 1（口径对齐）
    tou = P.tou_mask()
    rng = np.random.default_rng(int(tc.seed))

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(int(tc.epochs)):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in W_tr.iterate(int(tc.batch_size), shuffle=True, rng=rng, max_batches=tc.get("max_train_batches")):
            tb = to_tensors(b, device)
            out = model(tb)
            loss = pinball_loss(out["y_q"], tb["y_fut"], tb["valid_fut"], quantiles)
            if ac.mode == "soft":
                loss = loss + float(ac.lambda_a) * anchor_loss(model, anchors, tou, P.groups, wf)
            if ac.mode in ("soft", "free"):
                loss = loss + float(ac.lambda_h) * hier_loss(model)
            opt.zero_grad()
            loss.backward()
            if tc.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.grad_clip))
            opt.step()
            tot += loss.item()
            nb += 1
        vl = val_loss(model, W_va, int(tc.batch_size), device, quantiles)
        pp = model.price_parameters()
        bmean = float(pp["beta"][tou].mean()) if "beta" in pp and tou.any() else float("nan")
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_pinball": vl, "beta_tou_mean": bmean,
                     "seconds": time.time() - t0})
        log.info(f"第 {ep + 1:3d} 轮  训练 {tot / max(nb, 1):.5f}  验证 pinball {vl:.5f}  分时小区平均 β {bmean:+.3f}"
                 f"  {time.time() - t0:.1f}s")
        if not math.isfinite(vl):
            log.warning("验证损失不是有限值（可能数值发散）")
        if math.isfinite(vl) and vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc.patience):
                log.info(f"验证损失 {tc.patience} 轮未改善，提前停止")
                break
    if best_state is None:
        log.warning("没有得到有限的验证损失，使用最后一轮的参数")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    import pandas as pd
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)

    yq_va = predict(model, W_va, int(tc.batch_size), device)
    yq_te = predict(model, W_te, int(tc.batch_size), device)
    pp = model.price_parameters()
    price_report = {"anchor_source": anchors.source, "anchor_placeholder": placeholder, "mode": ac.mode,
                    "window_factor": wf}
    if "beta" in pp:
        bt = pp["beta"]
        price_report["beta_tou_by_group_ctx"] = [[float(bt[tou & (P.groups == g), c].mean()) if (tou & (P.groups == g)).any()
                                                  else None for c in range(bt.shape[1])] for g in range(P.G)]
    if "delta" in pp:
        price_report["delta"] = [float(v) for v in pp["delta"]]
    if "gamma" in pp:
        price_report["gamma"] = pp["gamma"]
    extra = {"model": "cpastgnn", "backbone": mc.backbone, "anchor_mode": ac.mode, "anchor_source": anchors.source,
             "anchor_placeholder": placeholder, "n_params": n_par, "epochs_run": len(hist),
             "best_val_pinball": best, "seconds": time.time() - t_start, "device": str(device)}
    if not a.no_consistency and cfg.data.H >= 6 and int(tc.eval_stride) == 1:
        q = list(np.round(quantiles, 4))
        spec = PanelSpec(window=int(ac.window_hours), jump_threshold=cfg.causal.jump_threshold,
                         eps=cfg.causal.eps_pile_hours, exclude_holiday_pm1=cfg.causal.exclude_holiday_pm1)
        for sub in ("all", "novel"):
            try:
                r = switch_consistency(P, W_te.origins, yq_te[..., q.index(0.5)], spec, subset=sub)
                price_report[f"switch_consistency_{sub}"] = r
                log.info(f"切换点一致性（{sub}）：观测 β {r['beta_obs']:+.3f}（{r['se_obs']:.3f}）  "
                         f"模型 β {r['beta_model']:+.3f}（{r['se_model']:.3f}）")
            except Exception as e:  # noqa: BLE001
                log.warning(f"切换点一致性检验失败：{e}")
    with open(os.path.join(out_dir, "price_params.json"), "w", encoding="utf-8") as f:
        json.dump(price_report, f, ensure_ascii=False, indent=1, default=float)
    s = save_run(out_dir, P, cfg, W_va, W_te, yq_va, yq_te, extra)
    log.info(f"测试期 MAE {s['test_MAE']:.4f}  WAPE {s['test_WAPE']:.3f}  CRPS {s['test_CRPS_q']:.4f}  "
             f"PICP90 原始 {s['test_PICP90_raw']:.3f} / ACI {s['test_PICP90_aci_mean']:.3f}；输出 {out_dir}")


if __name__ == "__main__":
    main()
