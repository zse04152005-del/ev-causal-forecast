"""训练脚本共用的 PyTorch 小工具：设备选择、批数据转张量、批量预测。"""
from __future__ import annotations

import numpy as np
import torch

LONG_KEYS = {"t0", "ctx_fut", "tod_hist", "dow_hist"}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensors(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k in LONG_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        elif v.dtype == bool:
            out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


@torch.no_grad()
def predict_array(model, W, bs: int, device, key: str = "y_q", price_override_fn=None) -> np.ndarray:
    """对窗口数据集逐批预测，拼成 numpy 数组。price_override_fn(tb) → dict，用于反事实与价格响应读出。"""
    model.eval()
    outs = []
    for b in W.iterate(bs):
        tb = to_tensors(b, device)
        ov = price_override_fn(tb) if price_override_fn is not None else None
        outs.append(model(tb, price_override=ov)[key].float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def price_override_from_path(P, lp_alt: np.ndarray, H: int, lags: int, device) -> callable:
    """反事实价格路径 lp_alt [T, N] → predict_array 用的 price_override_fn（按批内的预测起点 t0 取未来价格特征）。"""
    F = P.price_features(lp_alt)

    def fn(tb: dict) -> dict:
        t0 = tb["t0"].detach().cpu().numpy()
        fut = t0[:, None] + np.arange(1, H + 1)[None, :]
        out = {k + "_fut": torch.as_tensor(F[k][fut], dtype=torch.float32, device=device) for k in ("dl", "spill", "shift")}
        if lags > 0:
            ext = t0[:, None] + np.arange(1 - lags, H + 1)[None, :]
            out["dl_fut_ext"] = torch.as_tensor(F["dl"][ext], dtype=torch.float32, device=device)
        return out
    return fn

