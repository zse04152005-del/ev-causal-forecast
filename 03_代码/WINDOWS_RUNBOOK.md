# Windows 最终实验运行手册（RTX 4060）

> 对应工作顺序大纲阶段 7.5 与阶段 8。所有命令在 Anaconda Prompt（`conda activate cpa`）里运行。

## 1. 每次开始前

```bat
cd /d D:\EV充电因果预测sci论文
git pull
conda activate cpa
cd 03_代码
python -c "import torch; print(torch.cuda.is_available())"
```

- 电源接通；Windows 设置 → 电源 → 睡眠改为"从不"（夜间批量运行不能休眠）
- 数据：`02_数据\raw\urbanev_github\data\` 已就位（`python scripts\verify_data.py` 核对与 Mac 逐字节相同）

## 2. 分阶段运行（`scripts\run_final.ps1`）

```bat
powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Phase pilot,tune
```

| 阶段 | 内容 | 次数 | 前提 |
|---|---|---|---|
| `pilot` | 主模型（`final_main.yaml`，锚定 −0.21）种子 0：测每轮耗时，检查结果是否合理 | 1 | — |
| `tune` | Graph WaveNet：D {32, 64} × 学习率 {5e-4, 1e-3, 2e-3} × dropout {0.1, 0.3}，按验证集 pinball 选 | 12 | — |
| `main` | A0 完整 CPA-STGNN | 5 | Claude 把 tune 选出的超参数写进 `final_main.yaml` 之后 |
| `baselines` | 7 个深度基线 + Graph WaveNet / STAEformer 价格盲 + STGCN、AGCRN 价格作输入 | 55 | 同上 |
| `ablations` | A1 A2 A3 A5 A6 A10 A11 A13 A14 | 3 × 15 | 同上 |
| `eid` | 可识别性：free 模式 4 个初始值 × 3 种子；剖面损失 9 个固定 β | 21 | 同上 |
| `eps` | 先验敏感性：PAG 论文版 / 发布代码版、PIAST 截断 / 不截断 × 5 个先验（占用率口径） | 20 | 同上；PIAST 最慢 |
| `sens` | 附录：滞后核敏感性（`sens_lag.yaml`，长期弹性约 −0.36） | 3 | 同上 |

- **断点续跑**：运行目录里已有 `summary.json` 就跳过；中断（关机、报错）后重跑同一条命令即可
- **日志**：`05_实验结果\云端小实验\run_final_<阶段>_<时间>.log`（UTF-8）；每次运行的结果在 `05_实验结果\runs\<运行名>\`
- 某次运行失败时，日志里会写"失败（没有 summary.json）"，其余运行照常继续；把日志推送上来由 Claude 排查

## 3. 结果回传

```bat
cd /d D:\EV充电因果预测sci论文
git add -A
git commit -m "run_final: <阶段>"
git pull --rebase
git push
```

`model.pt` 等权重文件不入库；**`predictions_test.npz` 也不再入库**（每个约 24 MB，一百多次运行会超过 GitHub 仓库容量）。指标、`summary.json`、`price_params.json`、训练日志照常推送；需要逐点预测做 DM 检验和作图时，把 `05_实验结果\runs` 整个文件夹用 U 盘或网盘拷到 Mac 桌面的项目文件夹里，Claude 直接读取。

## 4. 常见问题

- 终端出现红色 `NativeCommandError`：旧版脚本的显示问题，拉取最新代码后不会再出现；它不是报错
- 显存不足（`CUDA out of memory`）：在该阶段的命令后加覆盖项不方便，直接告诉 Claude，改 `train.batch_size`
- PIAST 很慢：它对 24 个步长逐样本求二阶导，`eps` 阶段放在最后跑
