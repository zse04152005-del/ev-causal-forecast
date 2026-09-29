#!/bin/bash
# 按依赖顺序重跑全部数据核查（约 2 分钟）。结果写入 02_数据/processed/audit/
# 依赖：python3，numpy，pandas，scipy，scikit-learn，matplotlib（不需要 GIS 库和 PyTorch）
set -e
cd "$(dirname "$0")"
for s in check1 check2 check3 check4 check5 check6 check7 check8 check9 check10 check11 check12 fig_audit; do
  echo "== $s"; python3 $s.py
done
