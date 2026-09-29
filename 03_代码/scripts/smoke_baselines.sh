#!/bin/bash
# 深度基线冒烟测试（Mac / Linux）：基线单元测试 → 9 个基线各跑一次极小训练 → PAG、PIAST 的发布代码模式 → 日志写入项目文件夹
# 用法（在 03_代码 目录下，先激活装有 PyTorch 的环境）：bash scripts/smoke_baselines.sh      约 5–10 分钟
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
ROOT="$(cd .. && pwd)"
LOG_DIR="$ROOT/05_实验结果/云端小实验"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_baselines_$(date +%Y%m%d_%H%M%S).log"
SMOKE=(--config configs/experiment/smoke.yaml baseline.D=16 "baseline.stgcn.blocks=[[16,8,16],[16,8,16]]"
       baseline.astgcn.chev_filter=16 baseline.astgcn.time_filter=16 baseline.agcrn.hidden=16)
{
  echo "== 环境"
  "$PY" -c "import torch; print('torch', torch.__version__, '| mps', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
  echo "== 单元测试（基线 + 模型）"
  "$PY" -m unittest tests.test_baselines tests.test_models -v
  for name in fcnn lstm gcn gcnlstm stgcn astgcn agcrn; do
    echo "== 极小训练：${name}（价格盲）"
    "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=$name train.run_name=smoke_baseline_$name
  done
  echo "== 极小训练：stgcn（价格作输入，对照 P3）"
  "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=stgcn baseline.price_input=hist_fut train.run_name=smoke_baseline_stgcn_price
  echo "== 极小训练：PAG（论文版元学习预训练 1 轮，占用率口径）"
  "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=pag data.target=occupancy baseline.pag.pretrain_epochs=1 train.run_name=smoke_baseline_pag
  echo "== 极小训练：PAG（发布代码行为：不预训练）"
  "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=pag data.target=occupancy baseline.pag.pretrain=none baseline.pag.released_code_quirks=true train.run_name=smoke_baseline_pag_released
  echo "== 极小训练：PIAST（修正版，占用率口径，先验 −0.45）"
  "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=piast data.target=occupancy "baseline.piast.epochs=[1,1,1]" train.run_name=smoke_baseline_piast
  echo "== 极小训练：PIAST（发布代码模式，先验 −1.48，不截断 λ）"
  "$PY" scripts/train_baseline.py "${SMOKE[@]}" baseline.name=piast data.target=occupancy "baseline.piast.epochs=[1,1,1]" baseline.piast.released_code_quirks=true baseline.piast.prior=-1.48 baseline.piast.clamp=null train.run_name=smoke_baseline_piast_quirks
  echo "== 结束"
} 2>&1 | tee "$LOG"
echo ""
echo "日志已保存：$LOG"
