# 深度基线冒烟测试（Windows PowerShell）：在 03_代码 目录下运行  powershell -ExecutionPolicy Bypass -File scripts\smoke_baselines.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = Split-Path -Parent (Get-Location)
$logDir = Join-Path $root "05_实验结果\云端小实验"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("smoke_baselines_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
# 编码：让 PowerShell 按 UTF-8 解读 Python 的输出（否则中文变乱码），日志写成 UTF-8（Tee-Object 在 5.1 版写 UTF-16，Git 会当成二进制）
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$writer = New-Object System.IO.StreamWriter($log, $false, $utf8)
$smoke = @("--config", "configs/experiment/smoke.yaml", "train.device=cuda", "baseline.D=16", "baseline.stgcn.blocks=[[16,8,16],[16,8,16]]",
           "baseline.astgcn.chev_filter=16", "baseline.astgcn.time_filter=16", "baseline.agcrn.hidden=16")
& {
  "== 环境"
  python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
  "== 单元测试（基线 + 模型）"
  python -m unittest tests.test_baselines tests.test_models -v
  foreach ($name in @("fcnn", "lstm", "gcn", "gcnlstm", "stgcn", "astgcn", "agcrn")) {
    "== 极小训练：${name}（价格盲）"
    python scripts/train_baseline.py @smoke "baseline.name=$name" "train.run_name=smoke_baseline_$name"
  }
  "== 极小训练：stgcn（价格作输入）"
  python scripts/train_baseline.py @smoke baseline.name=stgcn baseline.price_input=hist_fut train.run_name=smoke_baseline_stgcn_price
  "== 极小训练：PAG（论文版元学习预训练 1 轮）"
  python scripts/train_baseline.py @smoke baseline.name=pag data.target=occupancy baseline.pag.pretrain_epochs=1 train.run_name=smoke_baseline_pag
  "== 极小训练：PAG（发布代码行为：不预训练）"
  python scripts/train_baseline.py @smoke baseline.name=pag data.target=occupancy baseline.pag.pretrain=none baseline.pag.released_code_quirks=true train.run_name=smoke_baseline_pag_released
  "== 极小训练：PIAST（修正版，CUDA 上验证二阶求导）"
  python scripts/train_baseline.py @smoke baseline.name=piast data.target=occupancy "baseline.piast.epochs=[1,1,1]" train.run_name=smoke_baseline_piast
  "== 极小训练：PIAST（发布代码模式，先验 −1.48，不截断 λ）"
  python scripts/train_baseline.py @smoke baseline.name=piast data.target=occupancy "baseline.piast.epochs=[1,1,1]" baseline.piast.released_code_quirks=true baseline.piast.prior=-1.48 baseline.piast.clamp=null train.run_name=smoke_baseline_piast_quirks
  "== 结束"
# Python 写到 stderr 的警告在 5.1 版会被包成红色的 NativeCommandError（并不是报错）；这里统一转成普通文本
} *>&1 | ForEach-Object { $line = "$_"; Write-Host $line; $writer.WriteLine($line); $writer.Flush() }
$writer.Close()
Write-Host "日志已保存：$log"
