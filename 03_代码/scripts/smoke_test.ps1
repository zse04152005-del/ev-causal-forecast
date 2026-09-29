# 冒烟测试（Windows PowerShell）：在 03_代码 目录下运行  powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = Split-Path -Parent (Get-Location)
$logDir = Join-Path $root "05_实验结果\云端小实验"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("smoke_test_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$env:PYTHONIOENCODING = "utf-8"
& {
  "== 环境"
  python --version
  python -c "import torch, numpy, pandas, scipy, sklearn, yaml, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
  "== 单元测试"
  python -m unittest discover -s tests -t . -v
  "== 极小训练 1：Graph WaveNet + 截断反馈（CUDA）"
  python scripts/train.py --config configs/experiment/smoke.yaml train.device=cuda
  "== 极小训练 2：STAEformer + 软锚定"
  python scripts/train.py --config configs/experiment/smoke.yaml model.backbone=staeformer anchor.mode=soft train.device=cuda train.run_name=smoke_staeformer_soft
  "== 结束"
} *>&1 | Tee-Object -FilePath $log
Write-Host "日志已保存：$log"
