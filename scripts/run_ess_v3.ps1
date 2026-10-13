# 半合成基准 v3（第二轮审稿后，2026-10-13）。在仓库根目录下运行：
#   powershell -ExecutionPolicy Bypass -File scripts\run_ess_v3.ps1 -Phase smoke      # 先跑这一步：2 次运行，看耗时与是否报错
#   powershell -ExecutionPolicy Bypass -File scripts\run_ess_v3.ps1 -Phase main       # 正式：500 次，约 33 小时，可分几个晚上，断点续跑
#   powershell -ExecutionPolicy Bypass -File scripts\run_ess_v3.ps1 -Phase oracle,endo # 补充：真值锚定上界 20 次 + 内生偏离 50 次
# 设计：
#   价格变动  rich（幅度 0.15，20% 的日子平移、20% 的日子不执行分时）/ scarce（0.05、2%、1%，接近真实数据）
#   ρ        −1、−0.5、0、0.5、1（负值 = 需求高时电价低，深圳属于这一侧）
#   真实弹性  所有小区同为 −0.2（接近实证估计）或 −1.0
#   模型      A0pooled 截断反馈 + 全市锚定；Bnone0 无约束，从 0 起步；Bproj0 符号约束（投影实现），从 0 起步；
#             A5blind 价格盲；A6input 价格作普通输入
#   种子      0–4；种子放在最外层循环，跑完一个种子就得到全部格子的一组结果
# 收缩、T2 最优组合等事后方法不需要训练，由 scripts\summarize_ess_v3.py 从各运行的 ess_report.json 计算。
# 断点续跑：运行目录里已有 summary.json 就跳过；中断后重跑同一命令即可。
param([string]$Phase = "main")
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

$settings = [ordered]@{
  "rich"   = @();
  "scarce" = @("ess.amp=0.05", "ess.day_shift_prob=0.02", "ess.flat_day_prob=0.01")
}
$arms = [ordered]@{
  "A0pooled" = @("anchor.mode=cut", "ess.anchor=pooled");
  "Bnone0"   = @("anchor.mode=none", "anchor.init_beta=0.0");
  "Bproj0"   = @("anchor.mode=proj", "anchor.init_beta=0.0");
  "A5blind"  = @("model.use_price=false");
  "A6input"  = @("model.use_price=false", "model.price_input=true")
}
$rhos = @("-1", "-0.5", "0", "0.5", "1")
$betas = @("-0.2", "-1.0")

function Cell([string]$st, [string]$b, [string]$rho, [string]$arm, [int]$s, [string[]]$more, [string]$tag) {
  $name = "v3_ess_${st}_b${b}_rho${rho}${tag}_${arm}_s$s"
  Run $name ($essCfg + @("ess.rho=$rho", "ess.beta_true=$b", "ess.seed=$s", "train.seed=$s") + $settings[$st] + $more)
}

foreach ($ph in $Phase.Split(",")) {
  $ph = $ph.Trim()
  $logFile = Join-Path $logDir ("run_ess_v3_{0}_{1:yyyyMMdd_HHmmss}.log" -f $ph, (Get-Date))
  $script:writer = New-Object System.IO.StreamWriter($logFile, $true, $utf8)
  Log "阶段 $ph 开始，日志：$logFile"
  switch ($ph) {
    "smoke" {
      Cell "scarce" "-0.2" "-1" "A0pooled" 0 $arms["A0pooled"] ""
      Cell "scarce" "-0.2" "-1" "Bproj0" 0 $arms["Bproj0"] ""
    }
    "main" {
      foreach ($s in 0..4) { foreach ($st in $settings.Keys) { foreach ($b in $betas) { foreach ($rho in $rhos) {
        foreach ($k in $arms.Keys) { Cell $st $b $rho $k $s $arms[$k] "" }
      } } } }
    }
    "oracle" {
      foreach ($st in $settings.Keys) { foreach ($b in $betas) { foreach ($rho in $rhos) {
        Cell $st $b $rho "A0oracle" 0 @("anchor.mode=cut", "ess.anchor=oracle") ""
      } } }
    }
    "endo" {
      foreach ($s in 0..4) { foreach ($rho in "-1", "1") { foreach ($k in $arms.Keys) {
        Cell "scarce" "-0.2" $rho $k $s ($arms[$k] + @("ess.endo=1.0")) "_endo1"
      } } }
    }
    default { Log "未知阶段：$ph（可选 smoke, main, oracle, endo）" }
  }
  Log "阶段 $ph 结束"
  $script:writer.Close(); $script:writer = $null
}
