"""核对 02_数据/raw/urbanev_github/ 与 Mac 上使用的数据是否逐字节相同（Windows 上 clone 数据后先运行一次）。

用法（在 03_代码 目录下）：python scripts/verify_data.py
注意：Windows 上的 git 可能把 CSV 的换行改成 CRLF，导致校验不通过；clone 前先执行
      git config --global core.autocrlf false   （或 clone 后用 git -c core.autocrlf=false checkout .）
"""
import hashlib
import os
import sys

import _bootstrap  # noqa: F401
from src.utils.paths import project_root

root = os.path.join(project_root(None), "02_数据", "raw", "urbanev_github")
sums = os.path.join(_bootstrap.CODE_DIR, "assets", "urbanev_github_SHA256SUMS.txt")
bad = 0
with open(sums, encoding="utf-8") as f:
    for line in f:
        want, rel = line.split()
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"缺失   {rel}")
            bad += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as g:
            for chunk in iter(lambda: g.read(1 << 20), b""):
                h.update(chunk)
        ok = h.hexdigest() == want
        bad += not ok
        print(f"{'一致  ' if ok else '不一致'} {rel}")
print("全部一致，可以开始实验" if bad == 0 else f"{bad} 个文件有问题：请改用网盘/U 盘拷贝 Mac 上的 02_数据/raw/urbanev_github/")
sys.exit(1 if bad else 0)
