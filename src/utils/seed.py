from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """统一设置随机种子（random、numpy，以及可用时的 torch）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
