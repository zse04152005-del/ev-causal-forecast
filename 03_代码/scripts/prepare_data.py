"""检查数据并打印建模数据摘要；assets/zone_static.csv 缺失时重新生成。

用法：python scripts/prepare_data.py [--config configs/default.yaml] [覆盖项 ...]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from src.data.dataset import prepare
from src.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=_bootstrap.DEFAULT_CFG)
ap.add_argument("overrides", nargs="*")
a = ap.parse_args()
cfg = load_config(a.config, a.overrides)
P = prepare(cfg)
print(json.dumps({"T": P.T, "N": P.N, "K": P.K, "G": P.G, "target": P.target, **P.meta}, ensure_ascii=False, indent=1))
