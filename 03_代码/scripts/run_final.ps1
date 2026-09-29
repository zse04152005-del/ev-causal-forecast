# 最终实验批量运行（Windows PowerShell，RTX 4060）：在 03_代码 目录下运行
#   powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Phase pilot
# 阶段（-Phase，可用逗号连写，例如 -Phase pilot,tune）：
#   pilot      主模型 1 次（种子 0），测每轮耗时、检查结果是否合理（约 20–40 分钟）
#   tune       Graph WaveNet 小网格：D {32,64} × 学习率 {5e-4,1e-3,2e-3} × dropout {0.1,0.3}，种子 0（12 次）
#   ---- 以下阶段等 Claude 根据 tune 的结果把最优超参数写进 final_main.yaml 之后再跑 ----
#   main       A0 完整 CPA-STGNN，5 个种子
#   baselines  9 个深度基线（价格盲）× 5 个种子；STGCN、AGCRN 价格作输入（P3）× 5 个种子
#   ablations  消融 A1 A2 A3 A5 A6 A10 A11 A13 A14，各 3 个种子
#   eid        可识别性：free 模式 4 个初始值 × 3 个种子；剖面损失 9 个固定 β
#   eps        先验敏感性：PAG（论文版 / 发布代码版）与 PIAST（截断 / 不截断）× 5 个先验，占用率口径
#   sens       附录敏感性：滞后核（sens_lag.yaml），3 个种子
# 断点续跑：某个运行目录里已有 summary.json 就跳过；中断后重跑同一命令即可。
# 日志：05_实验结果\云端小实验\run_final_<阶段>_<时间>.log（UTF-8）
param([string]$Phase = "pilot")
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
$script:writer = $null

function Log([string]$s) { Write-Host $s; if ($script:writer) { $script:writer.WriteLine($s); $script:writer.Flush() } }

function Run([string]$name, [string]$py, [string[]]$extra) {
  if (Test-Path (Join-Path $runs "$name\summary.json")) { Log "跳过（已完成）：$name"; return }
  Log ("== {0}  开始 {1:HH:mm:ss}" -f $name, (Get-Date))
  $t0 = Get-Date
  & python $py @extra "train.run_name=$name" 2>&1 | ForEach-Object { Log "$_" }
  $ok = Test-Path (Join-Path $runs "$name\summary.json")
  Log ("== {0}  {1}，用时 {2:N1} 分钟" -f $name, $(if ($ok) { "完成" } else { "失败（没有 summary.json）" }), ((Get-Date) - $t0).TotalMinutes)
}

function Main([string]$name, [string[]]$over) { Run $name "scripts/train.py" ($cfg + $over) }
function Base([string]$name, [string[]]$over) { Run $name "scripts/train_baseline.py" ($cfg + $over) }

foreach ($ph in $Phase.Split(",")) {
  $ph = $ph.Trim()
  $log = Join-Path $logDir ("run_final_" + $ph + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
  $script:writer = New-Object System.IO.StreamWriter($log, $false, $utf8)
  Log "阶段 $ph；日志 $log"
  switch ($ph) {
    "pilot" { Main "pilot_main_s0" @("train.seed=0") }
    "tune" {
      foreach ($D in 32, 64) { foreach ($lr in "5e-4", "1e-3", "2e-3") { foreach ($do in 0.1, 0.3) {
        Run "tune_gwn_D${D}_lr${lr}_do${do}" "scripts/train.py" ($cfg + @("--no-consistency", "train.seed=0", "model.D=$D", "train.lr=$lr", "model.dropout=$do"))
      } } }
    }
    "main" { foreach ($s in 0..4) { Main "main_A0_s$s" @("train.seed=$s") } }
    "baselines" {
      foreach ($s in 0..4) {
        foreach ($b in "fcnn", "lstm", "gcn", "gcnlstm", "stgcn", "astgcn", "agcrn") { Base "base_${b}_s$s" @("baseline.name=$b", "train.seed=$s") }
        Main "base_gwn_blind_s$s" @("train.seed=$s", "model.use_price=false")
        Main "base_staeformer_blind_s$s" @("train.seed=$s", "model.use_price=false", "model.backbone=staeformer")
        foreach ($b in "stgcn", "agcrn") { Base "base_${b}_price_s$s" @("baseline.name=$b", "baseline.price_input=hist_fut", "train.seed=$s") }
      }
    }
    "ablations" {
      foreach ($s in 0..2) {
        Main "abl_A1_free_s$s" @("train.seed=$s", "anchor.mode=free")
        Main "abl_A2_none_s$s" @("train.seed=$s", "anchor.mode=none")
        foreach ($b in "-0.45", "-0.76", "-1.48") { Main "abl_A3_prior${b}_s$s" @("train.seed=$s", "anchor.fixed_beta=$b") }
        Main "abl_A5_noprice_s$s" @("train.seed=$s", "model.use_price=false")
        Main "abl_A6_priceinput_s$s" @("train.seed=$s", "model.price_input=true")
        Main "abl_A10_staeformer_s$s" @("train.seed=$s", "model.backbone=staeformer")
        Main "abl_A11_predefined_s$s" @("train.seed=$s", "graph.adaptive=false")
        Main "abl_A11_adaptive_s$s" @("train.seed=$s", "graph.use_adj=false", "graph.use_dist=false", "graph.use_poi=false")
        foreach ($la in "0.01", "0.1", "1", "10") { Main "abl_A13_soft${la}_s$s" @("train.seed=$s", "anchor.mode=soft", "anchor.lambda_a=$la") }
        Main "abl_A14_nomask_s$s" @("train.seed=$s", "data.quality_mask=false")
      }
    }
    "eid" {
      foreach ($s in 0..2) { foreach ($b in "-0.05", "-0.3", "-1.0", "-2.0") { Main "eid_init${b}_s$s" @("train.seed=$s", "anchor.mode=free", "anchor.init_beta=$b") } }
      foreach ($b in "-0.05", "-0.1", "-0.2", "-0.4", "-0.8", "-1.2", "-1.6", "-2.0", "-3.0") { Main "eid_profile${b}_s0" @("train.seed=0", "anchor.fixed_beta=$b") }
    }
    "eps" {
      foreach ($p in "-0.1", "-0.45", "-0.76", "-1.48", "-3.0") {
        Base "eps_pag_paper${p}" @("baseline.name=pag", "data.target=occupancy", "baseline.pag.laws=[$p]")
        Base "eps_pag_released${p}" @("baseline.name=pag", "data.target=occupancy", "baseline.pag.laws=[$p]", "baseline.pag.pretrain=none", "baseline.pag.released_code_quirks=true")
        Base "eps_piast_clamp${p}" @("baseline.name=piast", "data.target=occupancy", "baseline.piast.prior=$p")
        Base "eps_piast_noclamp${p}" @("baseline.name=piast", "data.target=occupancy", "baseline.piast.prior=$p", "baseline.piast.clamp=null")
      }
    }
    "sens" { foreach ($s in 0..2) { Run "sens_lag_s$s" "scripts/train.py" @("--config", "configs/experiment/sens_lag.yaml", "train.seed=$s") } }
    default { Log "未知阶段：$ph（可选 pilot, tune, main, baselines, ablations, eid, eps, sens）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close()
  $script:writer = $null
}
