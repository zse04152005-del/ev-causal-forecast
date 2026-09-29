"""训练并评价深度基线（需要 PyTorch）。输出格式与 train.py 完全相同，可直接用 evaluate.py 做 DM / Wilcoxon 比较。

用法（在 03_代码 目录下）：
    python scripts/train_baseline.py baseline.name=stgcn                          # 价格盲 STGCN
    python scripts/train_baseline.py baseline.name=agcrn baseline.price_input=hist_fut   # 价格作输入（对照 P3）
    python scripts/train_baseline.py baseline.name=piast data.target=occupancy baseline.piast.prior=-1.48   # E-PS
    python scripts/train_baseline.py baseline.name=pag data.target=occupancy "baseline.pag.laws=[-3.0]"     # E-PS
    python scripts/train_baseline.py --config configs/experiment/smoke.yaml baseline.name=fcnn
可选基线：fcnn、lstm、gcn、gcnlstm（UrbanEV 官方）、stgcn、astgcn、agcrn、pag、piast。
输出：05_实验结果/runs/<run_name>/，另有 model.pt、train_log.csv、train.log、price_params.json（价格响应读出、切换点一致性；
PIAST 另有各小区 λ）。
"""
import argparse
import copy
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import _bootstrap  # noqa: E402,F401
from src.causal.switch_did import PanelSpec  # noqa: E402
from src.data.dataset import prepare  # noqa: E402
from src.data.windows import WindowDataset  # noqa: E402
from src.eval.counterfactual import switch_consistency  # noqa: E402
from src.eval.price_readout import arc_elasticity  # noqa: E402
from src.eval.runner import save_run, targets  # noqa: E402
from src.models.baselines import PAG, PIAST, build_baseline, fomaml_pretrain, readout_keys_of  # noqa: E402
from src.models.losses import masked_mse, pinball_loss  # noqa: E402
from src.models.point_intervals import point_to_quantiles, residual_quantiles  # noqa: E402
from src.models.torch_utils import pick_device, predict_array, to_tensors  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.paths import project_root, resolve  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

READOUT_PCT = 0.05


@torch.no_grad()
def val_loss(model, W: WindowDataset, bs: int, device, quantiles) -> float:
    model.eval()
    num = den = 0.0
    for b in W.iterate(bs):
        tb = to_tensors(b, device)
        out = model(tb)
        w = float(tb["valid_fut"].sum())
        if "y_q" in out:
            val = pinball_loss(out["y_q"], tb["y_fut"], tb["valid_fut"], quantiles)
        else:
            val = masked_mse(out["y"], tb["y_fut"], tb["valid_fut"])
        num += float(val) * w
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
    tc, bc = cfg.train, cfg.baseline
    name = bc.name
    if name == "piast":
        tag = f"prior{bc.piast.prior}"
    elif name == "pag":
        tag = f"{bc.pag.pretrain}_laws{'_'.join(str(v) for v in bc.pag.laws)}"
    else:
        tag = bc.price_input
    run = tc.get("run_name") or f"baseline_{name}_{tag}_{cfg.data.target}_s{tc.seed}"
    out_dir = os.path.join(resolve(cfg.paths.results_dir, root), "runs", run)
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger(f"baseline.{run}", os.path.join(out_dir, "train.log"))
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    set_seed(int(tc.seed))
    device = pick_device(tc.device)
    if name == "piast" and device.type == "mps":
        log.warning("PIAST 的物理约束需要对 LSTM 二阶求导，MPS 后端不保证支持，改用 CPU")
        device = torch.device("cpu")
    log.info(f"运行 {run}，基线 {name}，设备 {device}，torch {torch.__version__}")

    t_start = time.time()
    P = prepare(cfg)
    kw = dict(L=cfg.data.L, H=cfg.data.H, future_weather=cfg.data.future_weather)
    W_tr, W_va = WindowDataset(P, "train", **kw), WindowDataset(P, "val", **kw)
    W_te = WindowDataset(P, "test", stride=int(tc.eval_stride), **kw)
    log.info(f"数据：T={P.T} N={P.N} 目标 {P.target} 训练/验证/测试起点 {len(W_tr)}/{len(W_va)}/{len(W_te)}")
    quantiles = list(cfg.model.quantiles)
    tr = P.split.train_end
    init_q = np.quantile(P.y[: tr + 1][P.valid[: tr + 1]], quantiles)
    model = build_baseline(cfg, P, W_tr.n_cov_fut, init_q, seed=int(tc.seed)).to(device)
    is_piast, is_pag = isinstance(model, PIAST), isinstance(model, PAG)
    point = bool(getattr(model, "point", False))
    keys = readout_keys_of(model)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    price_desc = "预测时刻价格（PIAST）" if is_piast else ("历史价格（PAG）" if is_pag else bc.price_input)
    log.info(f"模型参数 {n_par:,}；价格输入 {price_desc}；损失 {'MSE（点预测）' if point else 'pinball（分位数）'}")
    rng = np.random.default_rng(int(tc.seed))
    clip = None if cfg.data.target == "volume" else cfg.model.get("clip_max")
    pre_hist = []

    if is_piast:
        pc = bc.piast
        opt = torch.optim.Adam(model.parameters(), lr=float(pc.lr))
        e1, e2, e3 = (int(e) for e in pc.epochs)
        w1, w2 = (float(w) for w in pc.prior_weights)
        phases = [("阶段1 强物理约束", e1, w1), ("阶段2 弱物理约束", e2, w2), ("阶段3 只用预测损失", e3, None)]
        clamp = pc.get("clamp")
        lo, hi = (None, None) if clamp is None else (float(clamp[0]), float(clamp[1]))
        prior = float(pc.prior)
        log.info(f"PIAST：注入先验 {prior}，λ 截断 {clamp}，三阶段轮数 {pc.epochs}，"
                 f"复现发布代码问题 {bool(pc.released_code_quirks)}")
    elif is_pag:
        gc = bc.pag
        if gc.pretrain == "paper":
            half = len(W_tr) // 2
            W_sup, W_qry = copy.copy(W_tr), copy.copy(W_tr)
            W_sup.origins, W_qry.origins = W_tr.origins[:half], W_tr.origins[half:]
            log.info(f"PAG：论文版元学习预训练，先验 {list(gc.laws)}，{gc.pretrain_epochs} 轮，支持集/查询集 "
                     f"{len(W_sup)}/{len(W_qry)} 个窗口")
            pre_hist = fomaml_pretrain(model, W_sup, W_qry, [float(v) for v in gc.laws], int(gc.pretrain_epochs),
                                       device, rng, P.adj, clip, int(tc.batch_size), outer_lr=float(gc.outer_lr),
                                       inner_lr=float(gc.lr), weight_decay=float(gc.weight_decay),
                                       gamma_den=float(gc.gamma_den), max_batches=tc.get("max_train_batches"), log=log)
        elif gc.pretrain == "none":
            log.info("PAG：不做预训练（发布代码实际行为 / 论文中的 PAG-）")
        else:
            raise ValueError(f"baseline.pag.pretrain 只能是 paper 或 none，收到 {gc.pretrain}")
        opt = torch.optim.Adam(model.parameters(), lr=float(gc.lr), weight_decay=float(gc.weight_decay))
        phases = [("微调", int(tc.epochs), None)]
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(tc.lr), weight_decay=float(tc.weight_decay))
        phases = [("训练", int(tc.epochs), None)]

    hist, best, best_state = [], float("inf"), None
    for pi, (pname, n_ep, w_prior) in enumerate(phases):
        last_phase = pi == len(phases) - 1
        bad = 0
        for ep in range(n_ep):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            for b in W_tr.iterate(int(tc.batch_size), shuffle=True, rng=rng, max_batches=tc.get("max_train_batches")):
                tb = to_tensors(b, device)
                if is_piast and w_prior is not None:
                    y, con1 = model.physics(tb)
                    loss = (masked_mse(y, tb["y_fut"], tb["valid_fut"]) + (con1 ** 2).mean()
                            + w_prior * ((model.lambda_1 - prior) ** 2).mean())
                elif point:
                    loss = masked_mse(model(tb)["y"], tb["y_fut"], tb["valid_fut"])
                else:
                    loss = pinball_loss(model(tb)["y_q"], tb["y_fut"], tb["valid_fut"], quantiles)
                opt.zero_grad()
                loss.backward()
                if tc.get("grad_clip") and not (is_piast or is_pag):     # 两者的原代码都没有梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.grad_clip))
                opt.step()
                if is_piast:
                    model.clamp_lambda(lo, hi)
                tot += loss.item()
                nb += 1
            vl = val_loss(model, W_va, int(tc.batch_size), device, quantiles)
            row = {"phase": pname, "epoch": len(hist) + 1, "train_loss": tot / max(nb, 1), "val_loss": vl,
                   "seconds": time.time() - t0}
            if is_piast:
                lam = model.lambda_1.detach().cpu().numpy()
                row["lambda_mean"] = float(lam.mean())
                row["lambda_tou_mean"] = float(lam[P.tou_mask()].mean()) if P.tou_mask().any() else float("nan")
            hist.append(row)
            log.info(f"{pname} 第 {ep + 1:3d} 轮  训练 {row['train_loss']:.5f}  验证 {vl:.5f}"
                     + (f"  λ 均值 {row['lambda_mean']:+.3f}" if is_piast else "") + f"  {row['seconds']:.1f}s")
            if not math.isfinite(vl):
                log.warning("验证损失不是有限值（可能数值发散）")
            if not last_phase:
                continue
            if math.isfinite(vl) and vl < best - 1e-9:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(tc.patience):
                    log.info(f"验证损失 {tc.patience} 轮未改善，提前停止")
                    break
    if best_state is None:
        log.warning("没有得到有限的验证损失（或最后阶段轮数为 0），使用最后一轮的参数")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)

    bs = int(tc.batch_size)
    q = list(np.round(quantiles, 4))
    if point:
        pv = predict_array(model, W_va, bs, device, key="y")
        pt = predict_array(model, W_te, bs, device, key="y")
        y_v, v_v = targets(W_va)
        rq = residual_quantiles(pv, y_v, v_v, quantiles)
        yq_va, yq_te = point_to_quantiles(pv, rq, clip), point_to_quantiles(pt, rq, clip)
        med_te = pt
    else:
        yq_va = predict_array(model, W_va, bs, device, key="y_q")
        yq_te = predict_array(model, W_te, bs, device, key="y_q")
        med_te = yq_te[..., q.index(0.5)]

    tou = P.tou_mask()
    price_in = "target_time_price" if is_piast else ("history_price" if is_pag else bc.price_input)
    report = {"baseline": name, "price_input": price_in, "readout_keys": list(keys)}
    if keys:
        up = torch.as_tensor(np.where(tou, math.log1p(READOUT_PCT), 0.0), dtype=torch.float32, device=device)
        key = "y" if point else "y_q"
        y_up = predict_array(model, W_te, bs, device, key=key,
                             price_override_fn=lambda tb: {k: tb[k] + up for k in keys})
        y_up = y_up if point else y_up[..., q.index(0.5)]
        r = arc_elasticity(med_te, y_up, READOUT_PCT, tou)
        report["arc_elasticity_tou"] = {k: v for k, v in r.items() if k != "by_zone"}
        report["arc_elasticity_by_zone"] = r["by_zone"]
        log.info(f"价格响应读出（分时小区 {'+'.join(keys)} 价格 +{READOUT_PCT:.0%}）：弧弹性均值 {r['mean']:+.3f}，"
                 f"中位数 {r['median']:+.3f}，"
                 f"为负的比例 {r['share_negative']:.2f}")
    if is_piast:
        lam = model.lambda_1.detach().cpu().numpy()
        report.update({"prior": prior, "clamp": clamp, "released_code_quirks": bool(bc.piast.released_code_quirks),
                       "lambda_mean_all": float(lam.mean()),
                       "lambda_mean_tou": float(lam[tou].mean()) if tou.any() else None,
                       "lambda_by_zone": lam.astype(float).tolist()})
        log.info(f"PIAST 读出 λ：全部小区均值 {lam.mean():+.3f}；分时小区均值 "
                 f"{(lam[tou].mean() if tou.any() else float('nan')):+.3f}（注入先验 {prior}）")
    if is_pag:
        report.update({"pretrain": bc.pag.pretrain, "laws": [float(v) for v in bc.pag.laws],
                       "released_code_quirks": bool(bc.pag.released_code_quirks), "pretrain_query_loss": pre_hist})
    if not a.no_consistency and cfg.data.H >= 6 and int(tc.eval_stride) == 1:
        spec = PanelSpec(window=int(cfg.anchor.window_hours), jump_threshold=cfg.causal.jump_threshold,
                         eps=cfg.causal.eps_pile_hours, exclude_holiday_pm1=cfg.causal.exclude_holiday_pm1)
        for sub in ("all", "novel"):
            try:
                r = switch_consistency(P, W_te.origins, np.maximum(med_te, 0.0), spec, subset=sub)
                report[f"switch_consistency_{sub}"] = r
                log.info(f"切换点一致性（{sub}）：观测 β {r['beta_obs']:+.3f}（{r['se_obs']:.3f}）  "
                         f"模型 β {r['beta_model']:+.3f}（{r['se_model']:.3f}）")
            except Exception as e:  # noqa: BLE001
                log.warning(f"切换点一致性检验失败：{e}")
    with open(os.path.join(out_dir, "price_params.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    extra = {"model": name, "price_input": price_in, "loss": "mse" if point else "pinball",
             "n_params": n_par, "epochs_run": len(hist), "best_val_loss": best, "seconds": time.time() - t_start,
             "device": str(device)}
    if is_piast:
        extra.update({"piast_prior": prior, "piast_lambda_mean_tou": report["lambda_mean_tou"]})
    if is_pag:
        extra.update({"pag_pretrain": bc.pag.pretrain, "pag_laws": report["laws"]})
    if "arc_elasticity_tou" in report:
        extra["arc_elasticity_tou_mean"] = report["arc_elasticity_tou"]["mean"]
    s = save_run(out_dir, P, cfg, W_va, W_te, yq_va, yq_te, extra)
    log.info(f"测试期 MAE {s['test_MAE']:.4f}  WAPE {s['test_WAPE']:.3f}  CRPS {s['test_CRPS_q']:.4f}  "
             f"PICP90 原始 {s['test_PICP90_raw']:.3f} / ACI {s['test_PICP90_aci_mean']:.3f}；输出 {out_dir}")


if __name__ == "__main__":
    main()
