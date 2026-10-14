# 第二轮审稿后的预测基线补强（2026-10-13）。在仓库根目录下运行：
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase smoke   # 先跑：STID、DLinear、无图消融各 1 次极短训练
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase tune    # 同预算调参，约 15 小时
#   python scripts\select_round2.py                                                            # 选出每个基线的最优超参数
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase seeds   # 10 个种子，约 16 小时
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase pag     # PAG 预训练检查 5 次 + 利用率口径 5 个种子，约 2 小时
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase t2      # 定理 T2 组合：符号约束弹性从 0 学起，10 个种子，约 2 小时
#   powershell -ExecutionPolicy Bypass -File scripts\run_round2_baselines.ps1 -Phase gbdt    # 梯度提升树（只用 CPU，也可以在 Mac 上跑）
# 调参：每个深度基线 12 组（D {32, 64} × 学习率 {5e-4, 1e-3, 2e-3} × dropout {0.1, 0.3}，种子 0，按验证集 pinball 选），
#       与 CPA-STGNN 第一轮的调参预算相同；STAEformer 4 组（学习率 {5e-4, 1e-3} × dropout {0.1, 0.3}，批大小 8）。
#       Graph WaveNet（价格盲）与 CPA-STGNN 共用骨干，沿用 final_main.yaml 的调参结果。
# 种子：所有模型 0–9；CPA-STGNN 与 Graph WaveNet 已有 0–4（v2_main_A0、v2_gwn_blind），只补 5–9。
# 无图消融：CPA-STGNN 去掉自适应图（graph.adaptive=false），只剩时间卷积，10 个种子。
# 断点续跑：运行目录里已有 summary.json 就跳过。
param([string]$Phase = "smoke")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path
$runs = Join-Path $root "results\runs"
$logDir = Join-Path $root "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$cfg = @("--config", "configs/experiment/final_main.yaml")
$script:writer = $null

function Log([string]$s) { Write-Host $s; if ($script:writer) { $script:writer.WriteLine($s); $script:writer.Flush() } }

function RunScript([string]$script, [string]$name, [string[]]$extra) {
  if (Test-Path (Join-Path $runs "$name\summary.json")) { Log "跳过（已完成）：$name"; return }
  Log ("== {0}  开始 {1:HH:mm:ss}" -f $name, (Get-Date))
  $t0 = Get-Date
  & python $script @extra "train.run_name=$name" 2>&1 | ForEach-Object { Log "$_" }
  $ok = Test-Path (Join-Path $runs "$name\summary.json")
  Log ("== {0}  {1}，用时 {2:N1} 分钟" -f $name, $(if ($ok) { "完成" } else { "失败（没有 summary.json）" }), ((Get-Date) - $t0).TotalMinutes)
}
function Base([string]$name, [string[]]$over) { RunScript "scripts/train_baseline.py" $name ($cfg + $over) }
function Main([string]$name, [string[]]$over) { RunScript "scripts/train.py" $name ($cfg + $over) }

$baselines = @("fcnn", "lstm", "gcn", "gcnlstm", "stgcn", "astgcn", "agcrn", "stid", "dlinear")

foreach ($ph in $Phase.Split(",")) {
  $ph = $ph.Trim()
  $logFile = Join-Path $logDir ("run_round2_{0}_{1:yyyyMMdd_HHmmss}.log" -f $ph, (Get-Date))
  $script:writer = New-Object System.IO.StreamWriter($logFile, $true, $utf8)
  Log "阶段 $ph 开始，日志：$logFile"
  switch ($ph) {
    "smoke" {
      foreach ($b in "stid", "dlinear") { Base "r2_smoke_$b" @("baseline.name=$b", "train.epochs=2", "train.max_train_batches=20") }
      Main "r2_smoke_nograph" @("graph.adaptive=false", "train.epochs=2", "train.max_train_batches=20")
    }
    "tune" {
      foreach ($b in $baselines) { foreach ($D in "32", "64") { foreach ($lr in "5e-4", "1e-3", "2e-3") { foreach ($do in "0.1", "0.3") {
        Base "r2_tune_${b}_D${D}_lr${lr}_do${do}_s0" @("baseline.name=$b", "baseline.D=$D", "train.lr=$lr", "baseline.dropout=$do", "train.seed=0")
      } } } }
      foreach ($lr in "5e-4", "1e-3") { foreach ($do in "0.1", "0.3") {
        Main "r2_tune_staeformer_lr${lr}_do${do}_s0" @("model.use_price=false", "model.backbone=staeformer", "train.batch_size=8", "train.lr=$lr", "model.staeformer.dropout=$do", "train.seed=0")
      } }
    }
    "seeds" {
      $bestPath = Join-Path $root "configs\experiment\round2_best.json"
      if (-not (Test-Path $bestPath)) { Log "缺少 $bestPath，请先运行 python scripts\select_round2.py"; break }
      $best = Get-Content $bestPath -Raw -Encoding UTF8 | ConvertFrom-Json
      foreach ($s in 0..9) {
        foreach ($b in $baselines) { Base "r2_${b}_s$s" (@("baseline.name=$b", "train.seed=$s") + [string[]]$best.$b) }
        Main "r2_staeformer_s$s" (@("model.use_price=false", "model.backbone=staeformer", "train.batch_size=8", "train.seed=$s") + [string[]]$best.staeformer)
        Main "r2_nograph_s$s" @("graph.adaptive=false", "train.seed=$s")
        if ($s -ge 5) {
          Main "v2_main_A0_s$s" @("train.seed=$s")
          Main "v2_gwn_blind_s$s" @("model.use_price=false", "train.seed=$s")
        }
      }
    }
    "pag" {
      # 预训练后、微调前的隐含弹性（检查先验是否注入；与 eps_pag_paper 的设置相同，占用率口径，每次约 12 分钟）
      foreach ($p in "-0.1", "-0.45", "-0.76", "-1.48", "-3.0") { Base "r2_pagcheck${p}" @("baseline.name=pag", "data.target=occupancy", "baseline.pag.laws=[$p]") }
      # PAG 用利用率作目标、同协议训练，进表 2（5 个种子）
      foreach ($s in 0..4) { Base "r2_pag_util_s$s" @("baseline.name=pag", "train.seed=$s") }
    }
    "t2" {
      # 定理 T2 组合所需的"学习弹性"：符号约束（投影）、从 0 起步，与主模型同配置、同种子（与半合成基准的 Bproj0 相同）
      foreach ($s in 0..9) { Main "r2_proj0_s$s" @("anchor.mode=proj", "anchor.init_beta=0.0", "train.seed=$s") }
    }
    "gbdt" {
      foreach ($s in 0..9) { RunScript "scripts/run_baseline.py" "r2_gbdt_s$s" @("model.name=gbdt", "train.seed=$s") }
    }
    default { Log "未知阶段：$ph（可选 smoke, tune, seeds, pag, t2, gbdt）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close(); $script:writer = $null
}
