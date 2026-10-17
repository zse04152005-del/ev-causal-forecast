# 主设置（只用自适应图）的主模型、消融、可识别性与 STAEformer 骨干。半合成基准见 run_ess_v3.ps1。
# 在仓库根目录下运行：
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_v2.ps1
# 默认依次跑全部阶段；也可以用 -Phase 指定（逗号连写），例如 -Phase main,eid
#   main       v2 主设置 A0，5 个种子
#   ablations  A1 A2 A3 A6 A13 A14 各 3 个种子；价格盲（= A5，兼作基线）5 个种子
#   eid        可识别性：free 模式 4 个初始值 × 3 个种子；剖面损失 9 个固定 β
#   sens       附录：滞后核，3 个种子
#   stae       STAEformer 骨干（批大小 8）：价格盲 5 个种子 + A10 3 个种子；最慢，放最后
# 所有运行名带前缀 v2_。第一轮里继续使用的结果（不依赖主设置）：
#   价格作输入的 base_stgcn_price_*、base_agcrn_price_*、eps_*、
#   main_A0_*（= 消融"预定义图 + 自适应图"）、abl_A11_predefined_*（= 消融"只用预定义图"）
# 断点续跑：运行目录里已有 summary.json 就跳过；中断后重跑同一命令即可。
param([string]$Phase = "main,ablations,eid,sens,stae")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = (Get-Location).Path                      # 仓库根目录（results\ 在这里）
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

function Run([string]$name, [string[]]$extra) {
  if (Test-Path (Join-Path $runs "$name\summary.json")) { Log "跳过（已完成）：$name"; return }
  Log ("== {0}  开始 {1:HH:mm:ss}" -f $name, (Get-Date))
  $t0 = Get-Date
  & python "scripts/train.py" @extra "train.run_name=$name" 2>&1 | ForEach-Object { Log "$_" }
  $ok = Test-Path (Join-Path $runs "$name\summary.json")
  Log ("== {0}  {1}，用时 {2:N1} 分钟" -f $name, $(if ($ok) { "完成" } else { "失败（没有 summary.json）" }), ((Get-Date) - $t0).TotalMinutes)
}

function Main([string]$name, [string[]]$over) { Run $name ($cfg + $over) }

foreach ($ph in $Phase.Split(",")) {
  $ph = $ph.Trim()
  $log = Join-Path $logDir ("run_final_v2_" + $ph + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
  $script:writer = New-Object System.IO.StreamWriter($log, $false, $utf8)
  Log "第二轮 阶段 $ph；日志 $log"
  switch ($ph) {
    "main" { foreach ($s in 0..4) { Main "v2_main_A0_s$s" @("train.seed=$s") } }
    "ablations" {
      foreach ($s in 0..4) { Main "v2_gwn_blind_s$s" @("train.seed=$s", "model.use_price=false") }
      foreach ($s in 0..2) {
        Main "v2_abl_A1_free_s$s" @("train.seed=$s", "anchor.mode=free")
        Main "v2_abl_A2_none_s$s" @("train.seed=$s", "anchor.mode=none")
        foreach ($b in "-0.45", "-0.76", "-1.48") { Main "v2_abl_A3_prior${b}_s$s" @("train.seed=$s", "anchor.fixed_beta=$b") }
        Main "v2_abl_A6_priceinput_s$s" @("train.seed=$s", "model.price_input=true", "model.use_price=false")
        foreach ($la in "0.01", "0.1", "1", "10") { Main "v2_abl_A13_soft${la}_s$s" @("train.seed=$s", "anchor.mode=soft", "anchor.lambda_a=$la") }
        Main "v2_abl_A14_nomask_s$s" @("train.seed=$s", "data.quality_mask=false")
      }
    }
    "eid" {
      foreach ($s in 0..2) { foreach ($b in "-0.05", "-0.3", "-1.0", "-2.0") { Main "v2_eid_init${b}_s$s" @("train.seed=$s", "anchor.mode=free", "anchor.init_beta=$b") } }
      foreach ($b in "-0.05", "-0.1", "-0.2", "-0.4", "-0.8", "-1.2", "-1.6", "-2.0", "-3.0") { Main "v2_eid_profile${b}_s0" @("train.seed=0", "anchor.fixed_beta=$b") }
    }
    "sens" { foreach ($s in 0..2) { Run "v2_sens_lag_s$s" @("--config", "configs/experiment/sens_lag.yaml", "train.seed=$s") } }
    "stae" {
      foreach ($s in 0..4) { Main "v2_staeformer_blind_s$s" @("train.seed=$s", "model.use_price=false", "model.backbone=staeformer", "train.batch_size=8") }
      foreach ($s in 0..2) { Main "v2_abl_A10_staeformer_s$s" @("train.seed=$s", "model.backbone=staeformer", "train.batch_size=8") }
    }
    default { Log "未知阶段：$ph（可选 main, ablations, eid, sens, stae）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close()
  $script:writer = $null
}
