# 第二轮最终实验（2026-10-04 决策后）：主设置改为"只用自适应图"；E-SS 增加"弱可识别"设置与"从 −0.05 起步的自由学习"。
# 在 03_代码 目录下运行：
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_v2.ps1
# 默认依次跑全部阶段；也可以用 -Phase 指定（逗号连写），例如 -Phase main,ess
#   main       v2 主设置 A0，5 个种子
#   ablations  A1 A2 A3 A6 A13 A14 各 3 个种子；价格盲（= A5，兼作基线）5 个种子
#   eid        可识别性：free 模式 4 个初始值 × 3 个种子；剖面损失 9 个固定 β
#   sens       附录：滞后核，3 个种子
#   ess        E-SS：{strong, weak} × ρ {0, 0.5, 1} × 7 种模型 × 3 个种子，另加真值锚定上界 2 次（共 128 次，每次约 4 分钟）
#   stae       STAEformer 骨干（批大小 8）：价格盲 5 个种子 + A10 3 个种子；最慢，放最后
# 所有运行名带前缀 v2_，与第一轮结果并存。第一轮里不依赖主设置的结果继续使用：
#   深度基线 base_*（fcnn、lstm、gcn、gcnlstm、stgcn、astgcn、agcrn 及价格作输入版本）、eps_*、
#   main_A0_*（= 消融"预定义图 + 自适应图"）、abl_A11_predefined_*（= 消融"只用预定义图"）
# 断点续跑：运行目录里已有 summary.json 就跳过；中断后重跑同一命令即可。
param([string]$Phase = "main,ablations,eid,sens,ess,stae")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$root = Split-Path -Parent (Get-Location)
$runs = Join-Path $root "05_实验结果\runs"
$logDir = Join-Path $root "05_实验结果\云端小实验"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$cfg = @("--config", "configs/experiment/final_main.yaml")
$essCfg = @("--config", "configs/experiment/ess.yaml", "--no-consistency")
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
    "ess" {
      # strong：ess.yaml 的默认合成设置（对数价差 ±0.15 的幅度参数，20% 的日子时刻表平移、20% 的日子不执行分时）——弹性可以从预测损失里学出来
      # weak：接近真实数据（幅度 0.05，2% 的日子平移、1% 的日子不执行分时）——固定效应之后几乎没有可识别的价格变动
      $settings = [ordered]@{
        "strong" = @();
        "weak"   = @("ess.amp=0.05", "ess.day_shift_prob=0.02", "ess.flat_day_prob=0.01")
      }
      $variants = [ordered]@{
        "A0cells"  = @("anchor.mode=cut", "ess.anchor=cells");
        "A0pooled" = @("anchor.mode=cut", "ess.anchor=pooled");
        "A1free"   = @("anchor.mode=free");
        "A1free0"  = @("anchor.mode=free", "anchor.init_beta=-0.05");
        "A2none0"  = @("anchor.mode=none", "anchor.init_beta=-0.05");
        "A5blind"  = @("model.use_price=false");
        "A6input"  = @("model.use_price=false", "model.price_input=true")
      }
      foreach ($st in $settings.Keys) {
        foreach ($rho in "0.5", "1", "0") {
          foreach ($s in 0..2) {
            foreach ($k in $variants.Keys) {
              Run "v2_ess_${st}_rho${rho}_${k}_s$s" ($essCfg + @("ess.rho=$rho", "ess.seed=$s", "train.seed=$s") + $settings[$st] + $variants[$k])
            }
          }
        }
        Run "v2_ess_${st}_rho0.5_A0oracle_s0" ($essCfg + @("ess.rho=0.5", "ess.seed=0", "train.seed=0", "anchor.mode=cut", "ess.anchor=oracle") + $settings[$st])
      }
    }
    "stae" {
      foreach ($s in 0..4) { Main "v2_staeformer_blind_s$s" @("train.seed=$s", "model.use_price=false", "model.backbone=staeformer", "train.batch_size=8") }
      foreach ($s in 0..2) { Main "v2_abl_A10_staeformer_s$s" @("train.seed=$s", "model.backbone=staeformer", "train.batch_size=8") }
    }
    default { Log "未知阶段：$ph（可选 main, ablations, eid, sens, ess, stae）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close()
  $script:writer = $null
}
