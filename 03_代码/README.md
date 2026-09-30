# 03_代码：CPA-STGNN 代码仓库

对应《00_项目管理/模型设计文档_v1.1.md》（以下简称"设计文档"）。阶段 3 初版，2026-09-27。

## 1. 环境

**Mac（Apple Silicon，冒烟测试与小实验）**

```bash
conda create -n cpa python=3.11 -y
conda activate cpa
cd ~/Desktop/EV充电因果预测sci论文/03_代码
pip install -r requirements.txt
bash scripts/smoke_test.sh          # 约 5–10 分钟，日志写到 05_实验结果/云端小实验/
```

**Windows + RTX 4060（最终实验）**

```powershell
conda create -n cpa python=3.11 -y
conda activate cpa
# PyTorch：到 https://pytorch.org/get-started/locally/ 选 Windows / Pip / CUDA 12.x，复制命令（只保留 torch），例如
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 必须输出 True
pip install -r requirements_windows_cuda.txt
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
powershell -ExecutionPolicy Bypass -File scripts\smoke_baselines.ps1
```

代码与文档通过私有 GitHub 仓库在 Mac 与 Windows 之间同步；数据（`02_数据/`，约 1.1 GB）不入库：当前实验只需要
`02_数据/raw/urbanev_github/data/`，在 Windows 上 `git clone -c core.autocrlf=false --depth 1 https://github.com/IntelligentSystemsLab/UrbanEV 02_数据/raw/urbanev_github`
即可得到同一份数据（再用 `python scripts/verify_data.py` 核对与 Mac 上逐字节相同）；阶段 6 需要的两个大压缩包用网盘或 U 盘拷贝。`.ps1` 脚本带 UTF-8 BOM，Windows PowerShell 5.1 才能正确读取中文路径。

**云端（Claude 工作区）**：没有 PyTorch，只运行数据管线、因果估计、numpy 基线、评价与仿真（`requirements_cloud.txt`）。

数据放在 `02_数据/raw/urbanev_github/data/`（项目根目录由 `03_代码` 的上一级自动确定，也可以设环境变量 `CPA_PROJECT_ROOT`）。

## 2. 常用命令（在 03_代码 目录下）

| 目的 | 命令 | 需要 PyTorch |
|---|---|---|
| 核对数据与 Mac 上逐字节相同 | `python scripts/verify_data.py` | 否 |
| 检查数据、打印摘要 | `python scripts/prepare_data.py` | 否 |
| 用训练期切换事件估计锚定值 | `python scripts/estimate_anchors.py` → `configs/anchor/hourly_train.json` | 否 |
| 5 分钟站点级因果估计（规格表、事件研究、安慰剂、格子锚定） | `python scripts/estimate_causal_5min.py --stages all`（数据在 `02_数据/interim/fiveMin/`）→ `configs/anchor/fivemin_train*.json`、`fivemin_main.json`（主实验锚定）、`05_实验结果/因果估计/5min/` | 否 |
| 5 分钟半合成验证（设计 C、D 在已知真值下的偏误、存量稀释、反推到达弹性） | `python scripts/semisynthetic_5min.py --means 30,60 --seeds 3 --tag m30_60`，再 `python scripts/summarize_semisynth_5min.py` | 否 |
| C 与 D 之差的自助法检验 | `python scripts/boot_cd_diff.py --reps 100` | 否 |
| 逐小时面板 PPML 与 DML（补充估计） | `python scripts/estimate_panel_supp.py` | 否 |
| 按官方协议复现 UrbanEV 论文表 3 的 LO 与 FCNN（数据版本核对） | `python scripts/repro_urbanev_table3.py --d1 <UrbanEV/data> --out 05_实验结果/基线/urbanev_repro` | 否 |
| 探索性分析（阶段 5.2） | `python scripts/eda.py` → `05_实验结果/EDA/`；作图 `python ../04_图表/绘图脚本/fig_eda.py` | 否 |
| numpy 基线 | `python scripts/run_baseline.py model.name=naive_seasonal`（或 `naive_last`、`profile`） | 否 |
| 训练 CPA-STGNN | `python scripts/train.py [覆盖项]` | 是 |
| 深度基线（9 个） | `python scripts/train_baseline.py baseline.name=stgcn`（见第 4 节） | 是 |
| 比较多次运行（DM、Wilcoxon） | `python scripts/evaluate.py --runs 运行1 运行2 ...` | 否 |
| 半合成识别预实验（E-SS） | `python scripts/pilot_semisynthetic.py` | 否 |
| E-SS 完整流程（合成数据上识别 → 训练 → 反事实评价） | `python scripts/train.py --config configs/experiment/ess.yaml ess.rho=0.5 ess.anchor=cells`，汇总 `python scripts/summarize_ess.py` | 是 |
| 汇总一批运行（调参选型、结果总表） | `python scripts/summarize_runs.py --prefix tune_ --sort best_val_pinball` | 否 |
| 全部单元测试 | `python -m unittest discover -s tests -t .` | 部分 |
| **最终实验批量运行（Windows）** | `powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Phase pilot,tune`，详见 `WINDOWS_RUNBOOK.md` | 是 |
| 基线冒烟测试 | `bash scripts/smoke_baselines.sh`（Windows：`scripts\smoke_baselines.ps1`） | 是 |

覆盖项写法：`model.D=64 anchor.mode=soft train.seed=3 data.target=occupancy`。

## 3. 设计决策与配置开关

| 设计文档决策（D1、D7、D8、D9 已于 2026-09-28 确认） | 配置项 | 默认 |
|---|---|---|
| D1 目标变量 | `data.target: utilization \| occupancy \| volume` | utilization（时长 ÷ 桩数） |
| D2 价格口径 | `data.price: total \| electricity` | total |
| D3 负荷转移 / 滞后核 | `model.use_shift`、`model.price_lags` | false / 0 |
| D4 锚定层级与合并 | `anchor.min_zones`、`anchor.max_se` | 10 / 0.5 |
| D5 骨干网络 | `model.backbone: graph_wavenet \| staeformer` | graph_wavenet |
| D6 窗口 | `data.L`、`data.H` | 24 / 24 |
| D7 锚定方式 | `anchor.mode: cut \| soft \| free \| none` | cut（截断反馈） |
| D8 数据质量掩码 | `data.quality_mask` | true |
| D9 识别主设计 | `causal.design: C \| D` | D（小区内跨日变动 + 小区×时刻固定效应） |
| 锚定文件 | `anchor.file` | 占位锚定 `placeholder.json`（β̂ = −0.45，评价脚本 `--paper` 会拒绝） |

消融：A5 `model.use_price=false`；A6 `model.price_input=true`；A7 `model.use_spill=false`；A10 `model.backbone=staeformer`；A11 `graph.use_adj/use_dist/use_poi/adaptive`；A13 `anchor.mode=soft anchor.lambda_a=...`；A14 `data.quality_mask=false`。
E-ID：`anchor.mode=free anchor.init_beta=-1.0`（初始化漂移），`anchor.fixed_beta=-0.4`（剖面损失）。

## 4. 深度基线（`src/models/baselines/`，阶段 3.5）

统一协议：与 CPA-STGNN 同切分、同窗口、同质量掩码、同一套未来日历与天气输入、同一个非交叉分位数头与 pinball 损失，只替换编码器。

| `baseline.name` | 来源 | 价格进入方式 |
|---|---|---|
| `fcnn`、`lstm`、`gcn`、`gcnlstm` | UrbanEV 官方基线（结构移植） | `baseline.price_input`：`none` / `hist` / `hist_fut` |
| `stgcn`、`astgcn`、`agcrn` | 各自官方实现 | 同上 |
| `pag` | ST-EVCDP 仓库（PAG 官方代码）+ 论文版元学习预训练 | 历史价格；`baseline.pag.laws`、`pretrain: paper \| none` |
| `piast` | PIAST 官方代码 | 预测时刻价格；`baseline.piast.prior`、`clamp` |

常用命令：

```bash
python scripts/train_baseline.py baseline.name=agcrn                                    # 价格盲
python scripts/train_baseline.py baseline.name=agcrn baseline.price_input=hist_fut      # 价格作输入（对照 P3）
python scripts/train_baseline.py baseline.name=pag data.target=occupancy "baseline.pag.laws=[-3.0]"      # E-PS
python scripts/train_baseline.py baseline.name=piast data.target=occupancy baseline.piast.prior=-3.0 baseline.piast.clamp=null
```

`price_params.json` 里有价格响应读出（分时小区价格上调 5% 的弧弹性）与切换点一致性；PIAST 另有各小区 λ。
PAG、PIAST 发布代码中的实现问题及我们的处理见设计文档 16.8；`released_code_quirks=true` 可按发布代码复现。
PIAST 的物理约束需要对 LSTM 二阶求导，在 Mac 上会自动改用 CPU。

## 5. 目录

```
configs/            default.yaml；experiment/（smoke、final_main）；anchor/（placeholder、hourly_train、fivemin_train、fivemin_train_pooled、fivemin_main = 主实验用）
assets/             zone_static.csv：小区静态特征与功能区（固定保存，跨机器一致）
src/data/           读取、日历（假日与调休）、质量掩码、价格特征、图、分区与 POI、切分、标准化、窗口
src/causal/         switch_did（小时数据上的设计 A–D、暴露映射、事件研究）、switch_5min（站点级 5 分钟：价格过渡、干净窗口、PPML、聚类、事件研究）、
                    panel_models（面板 PPML、DML）、anchors（锚定文件、合并规则、口径对齐）
src/models/         backbones（Graph WaveNet、STAEformer）、heads（分位数头、价格响应/溢出/转移）、cpastgnn、losses、naive、
                    baselines（9 个深度基线）、pseudo_samples（PAG 伪样本）、point_intervals、torch_utils
src/conformal/      aci（带掩码的自适应保形校准）
src/eval/           metrics、stats（DM、Wilcoxon）、counterfactual（切换点一致性、插补基线 P0）、semisynthetic（E-SS）、runner、
                    price_readout（价格响应读出）
src/simulate/       scenarios（S1–S5、利用率→负荷、参数抽样）
scripts/            prepare_data、estimate_anchors、estimate_causal_5min、boot_cd_diff、estimate_panel_supp、run_baseline、train、evaluate、pilot_semisynthetic、
                    train_baseline、smoke_test、smoke_baselines、audit/
tests/              test_data、test_causal、test_causal_5min、test_eval（numpy）；test_models、test_baselines（PyTorch 为主）
```

## 6. 每次运行的输出（`05_实验结果/runs/<运行名>/`）

`config.yaml`、`metrics_test.csv`（掩码）、`metrics_test_unmasked.csv`、`metrics_val.csv`、`conformal_test.csv`、`predictions_test.npz`、`summary.json`；PyTorch 模型另有 `model.pt`、`train_log.csv`、`train.log`、`price_params.json`（弹性表、δ、切换点一致性）。

## 7. 测试状态（2026-09-27）

- 云端：数据、因果、评价共 29 个测试通过（含与数据核查冻结掩码逐点一致、半合成数据上设计 D 在内生时刻表下仍能还原弹性）
- Mac（torch 2.14.0，Apple Silicon）：41 个测试全部通过，含 `tests/test_models.py` 的 12 个 PyTorch 测试（形状、分位数不交叉、价格盲、反事实替换、推论 1 比例关系、符号与单调约束、消融开关、滞后核与口径对齐、带滞后核的反事实替换、训练能降低损失、掩码损失、真实批次前向）；3 个极小训练全部跑通。日志：`05_实验结果/云端小实验/smoke_test_20260927_190735.log`
- 深度基线（2026-09-28，Mac）：`tests/test_baselines.py` 16 个测试全部通过；`scripts/smoke_baselines.sh` 的 13 个极小训练全部跑通
- Windows + RTX 4060（2026-09-29，torch 2.11.0+cu128）：57 个测试通过（1 个跳过：Windows 上没有 `02_数据/processed/audit/frozen_mask.csv`），2 个极小训练在 CUDA 上跑通，结果与 Mac CPU 相同到小数点后 6 位；FCNN 全量训练完成（`05_实验结果/runs/full_fcnn/`）。`smoke_test.ps1`、`smoke_baselines.ps1` 已改为 UTF-8 输出与日志（此前日志为 UTF-16、终端中文乱码、警告显示为红色 NativeCommandError）；`smoke_baselines.ps1` 已在 Windows 上运行（28 个测试通过，13 个极小训练跑通）
