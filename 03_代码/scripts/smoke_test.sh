#!/bin/bash
# 冒烟测试（Mac / Linux）：检查环境 → 全部单元测试（含 PyTorch 部分）→ 两次极小训练 → 日志写入项目文件夹
# 用法（在 03_代码 目录下，先激活装有 PyTorch 的环境）：bash scripts/smoke_test.sh
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
ROOT="$(cd .. && pwd)"
LOG_DIR="$ROOT/05_实验结果/云端小实验"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"
{
  echo "== 环境"
  "$PY" --version
  "$PY" -c "import torch, numpy, pandas, scipy, sklearn, yaml, matplotlib; print('torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| mps', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
  uname -a
  echo "== 单元测试"
  "$PY" -m unittest discover -s tests -t . -v
  echo "== 极小训练 1：Graph WaveNet + 截断反馈"
  "$PY" scripts/train.py --config configs/experiment/smoke.yaml
  echo "== 极小训练 2：STAEformer + 软锚定"
  "$PY" scripts/train.py --config configs/experiment/smoke.yaml model.backbone=staeformer anchor.mode=soft train.run_name=smoke_staeformer_soft
  echo "== 极小训练 3：自由弹性（E-ID 用）+ 负荷转移 + 滞后核"
  "$PY" scripts/train.py --config configs/experiment/smoke.yaml anchor.mode=free anchor.init_beta=-1.0 model.use_shift=true model.price_lags=2 train.run_name=smoke_free_lags
  echo "== 结束"
} 2>&1 | tee "$LOG"
echo ""
echo "日志已保存：$LOG"
