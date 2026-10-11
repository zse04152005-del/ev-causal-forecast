"""测试公用：定位项目根目录、按需加载真实数据（数据不在时跳过相关测试）。"""
from __future__ import annotations

import os
import sys
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import project_root, resolve  # noqa: E402

DEFAULT_CFG = os.path.join(CODE_DIR, "configs", "default.yaml")

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = unittest.skipUnless(HAS_TORCH, "未安装 PyTorch（云端跳过，在 Mac / Windows 上运行）")


def data_dir() -> str:
    cfg = load_config(DEFAULT_CFG)
    return resolve(cfg.paths.data_dir, project_root(cfg.paths.get("root")))


requires_data = unittest.skipUnless(os.path.isdir(data_dir()), "找不到 UrbanEV 数据目录")

_CACHE = {}


def prepared(overrides=()):
    key = tuple(overrides)
    if key not in _CACHE:
        from src.data.dataset import prepare
        _CACHE[key] = prepare(load_config(DEFAULT_CFG, list(overrides)))
    return _CACHE[key]
