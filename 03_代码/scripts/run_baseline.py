"""运行不依赖 PyTorch 的基线（naive_last / naive_seasonal / profile），输出格式与 train.py 相同。

用法：python scripts/run_baseline.py model.name=profile [其他覆盖项 ...]
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.data.windows import WindowDataset
from src.eval.runner import save_run
from src.models.naive import NaiveForecaster
from src.utils.config import load_config, save_config
from src.utils.paths import project_root, resolve

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
t = time.time()
P = prepare(cfg)
kw = dict(L=cfg.data.L, H=cfg.data.H, future_weather=cfg.data.future_weather)
W_tr, W_va = WindowDataset(P, "train", **kw), WindowDataset(P, "val", **kw)
W_te = WindowDataset(P, "test", stride=int(cfg.train.eval_stride), **kw)
m = NaiveForecaster(cfg.model.name, cfg.model.quantiles, cfg.model.get("clip_max")).fit(P, W_tr)
run = cfg.train.get("run_name") or f"baseline_{cfg.model.name}_{cfg.data.target}"
out = os.path.join(resolve(cfg.paths.results_dir, project_root(cfg.paths.get("root"))), "runs", run)
os.makedirs(out, exist_ok=True)
save_config(cfg, os.path.join(out, "config.yaml"))
s = save_run(out, P, cfg, W_va, W_te, m.predict(P, W_va), m.predict(P, W_te), {"model": cfg.model.name,
                                                                                  "seconds": time.time() - t})
print(f"{run}: 测试期 MAE {s['test_MAE']:.4f}  WAPE {s['test_WAPE']:.3f}  CRPS {s['test_CRPS_q']:.4f}  "
      f"PICP90 原始 {s['test_PICP90_raw']:.3f} / ACI {s['test_PICP90_aci_mean']:.3f}  → {out}")
