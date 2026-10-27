# 多骨干锚定（投稿前补充，2026-10-27）：把锚定价格项接到 STID 和 LSTM 上，检验锚定与骨干网络无关。在仓库根目录下运行：
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbones.ps1 -Phase smoke     # 先跑：3 次极短训练，约 2 分钟
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbones.ps1 -Phase anchor    # STID、LSTM 加锚定价格项，各 10 个种子，约 1.5 小时
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbones.ps1 -Phase learn     # 可选：符号约束弹性从 0 学起（自适应锚定用），各 10 个种子，约 1.5 小时
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbones.ps1 -Phase dm        # DM 检验的损失序列（读 predictions_test.npz）
#   powershell -ExecutionPolicy Bypass -File scripts\run_backbones.ps1 -Phase summary   # 汇总表（只用 CPU）
# 也可以一次跑完：-Phase anchor,learn,dm,summary
#
# 设计：与 CPA-STGNN（v2_main_A0）对 Graph WaveNet（v2_gwn_blind）的比较完全对应。
#   - 价格盲版本就是表 2 里已有的 r2_stid_s0–9、r2_lstm_s0–9（目录已存在则跳过；不存在则按同样设置补跑）
#   - 锚定版本 r2_stid_anchor、r2_lstm_anchor：同一套调参结果（configs/experiment/round2_best.json）、同样的种子，
#     编码器与输出头仍然价格盲，只在输出头之后乘 exp(β̂ Δℓ)，β̂ = −0.21（configs/anchor/fivemin_main.json），不接收梯度。
#     截断锚定不增加参数、不消耗随机数，所以同一种子下两个版本的初始权重完全相同，差异只来自价格项。
#   - learn 阶段 r2_stid_learn0、r2_lstm_learn0：baseline.anchor=proj、anchor.init_beta=0（与 r2_proj0 相同），
#     用来在每个骨干上重复第 4.5 节的自适应锚定（ŵ 与 β_ŵ）。
# 输出：results/comparisons/backbones/（loss_bb.npz、learned_elasticity_bb.csv、backbones_seeds.csv、
#       backbones_summary.csv、tab_backbones.tex）。传回 Mac 时带上这个文件夹和 results/runs 下新增运行的 summary.json、price_params.json。
# 断点续跑：运行目录里已有 summary.json 就跳过。
param([string]$Phase = "smoke")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path
$runs = Join-Path $root "results\runs"
$logDir = Join-Path $root "results\logs"
$outDir = Join-Path $root "results\comparisons\backbones"
New-Item -ItemType Directory -Force -Path $logDir, $outDir | Out-Null
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

$backbones = @("stid", "lstm")
$bestPath = Join-Path $root "configs\experiment\round2_best.json"
if (-not (Test-Path $bestPath)) { Write-Host "缺少 $bestPath（第二轮调参结果）"; exit 1 }
$best = Get-Content $bestPath -Raw -Encoding UTF8 | ConvertFrom-Json

foreach ($ph in $Phase.Split(",")) {
  $ph = $ph.Trim()
  $logFile = Join-Path $logDir ("run_backbones_{0}_{1:yyyyMMdd_HHmmss}.log" -f $ph, (Get-Date))
  $script:writer = New-Object System.IO.StreamWriter($logFile, $true, $utf8)
  Log "阶段 $ph 开始，日志：$logFile"
  switch ($ph) {
    "smoke" {
      foreach ($b in $backbones) {
        Base "bb_smoke_${b}_anchor" (@("baseline.name=$b", "baseline.anchor=cut", "train.epochs=2", "train.max_train_batches=20") + [string[]]$best.$b)
      }
      Base "bb_smoke_stid_learn0" (@("baseline.name=stid", "baseline.anchor=proj", "anchor.init_beta=0.0", "train.epochs=2", "train.max_train_batches=20") + [string[]]$best.stid)
      Log "检查：日志里锚定版本的“价格响应读出”应为 −0.213，learn0 的平均 β 应 ≤ 0。冒烟运行目录 bb_smoke_* 之后可以删除。"
    }
    "anchor" {
      foreach ($s in 0..9) {
        foreach ($b in $backbones) {
          Base "r2_${b}_s$s" (@("baseline.name=$b", "train.seed=$s") + [string[]]$best.$b)                       # 价格盲：已有则跳过
          Base "r2_${b}_anchor_s$s" (@("baseline.name=$b", "baseline.anchor=cut", "train.seed=$s") + [string[]]$best.$b)
        }
      }
    }
    "learn" {
      foreach ($s in 0..9) {
        foreach ($b in $backbones) {
          Base "r2_${b}_learn0_s$s" (@("baseline.name=$b", "baseline.anchor=proj", "anchor.init_beta=0.0", "train.seed=$s") + [string[]]$best.$b)
        }
      }
    }
    "dm" {
      # 价格盲版本的损失序列在 loss_r2.npz 里已有；这里连同锚定版本一起重算，保证两边来自同一批运行
      $groups = @()
      foreach ($b in $backbones) { $groups += @("r2_$b", "r2_${b}_anchor") }
      $out = Join-Path $outDir "loss_bb.npz"
      Log "python scripts/loss_series.py --runs $runs --seeds 10 --out $out $($groups -join ' ')"
      & python scripts/loss_series.py --runs $runs --seeds 10 --out $out @groups 2>&1 | ForEach-Object { Log $_ }
    }
    "summary" {
      $learnDirs = @(Get-ChildItem $runs -Directory | Where-Object { $_.Name -match '^r2_(stid|lstm)_learn0_s\d+$' } | ForEach-Object { $_.FullName })
      $le = Join-Path $outDir "learned_elasticity_bb.csv"
      if ($learnDirs.Count -gt 0) {
        & python scripts/learned_elasticity.py --out $le @learnDirs 2>&1 | ForEach-Object { Log $_ }
      } else { Log "没有 learn 阶段的运行，跳过自适应锚定" }
      $dm = @("results/comparisons/dm/loss_r2.npz", "results/comparisons/dm/loss_r2m.npz", (Join-Path $outDir "loss_bb.npz"))
      & python scripts/summarize_backbones.py --runs $runs --dm @dm --le results/comparisons/learned_elasticity.csv $le --out $outDir 2>&1 | ForEach-Object { Log $_ }
    }
    default { Log "未知阶段：$ph（可选 smoke, anchor, learn, dm, summary）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close(); $script:writer = $null
}
