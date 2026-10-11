"""项目路径约定。

项目根目录 = 仓库根目录（含 src/、data/、results/ 的那一级）。
可用环境变量 CPA_PROJECT_ROOT 或配置 paths.root 覆盖。
注意：这里用 abspath 而不是 realpath，以免符号链接改变根目录。
"""
from __future__ import annotations

import os

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def project_root(cfg_root: str | None = None) -> str:
    if cfg_root:
        return os.path.abspath(cfg_root)
    env = os.environ.get("CPA_PROJECT_ROOT")
    if env:
        return os.path.abspath(env)
    return CODE_DIR


def resolve(path: str, root: str) -> str:
    """相对路径按项目根目录解析；绝对路径原样返回。"""
    return path if os.path.isabs(path) else os.path.join(root, path)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
