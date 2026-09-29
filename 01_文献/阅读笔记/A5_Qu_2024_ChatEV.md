# A5 Qu et al. (2024) ChatEV：用大语言模型预测充电需求

- **引用**：Qu, H., Li, H., You, L., Zhu, R., Yan, J., Santi, P., Ratti, C., Yuen, C. (2024). ChatEV: Predicting electric vehicle charging demand as natural language processing. *Transportation Research Part D*, 136, 104470. https://doi.org/10.1016/j.trd.2024.104470
- **等级**：领域 SCI
- **笔记**：Claude 初稿（2026-09-27），待你精读核对
- **重要程度**：★★★（可选基线；了解该团队方向）

---

## 1. 研究问题

充电数据稀缺、异构特征难以对齐、模型难以泛化到新区域。论文把预测任务改写成"文本到文本"，用预训练语言模型做预测，重点解决少样本和零样本场景。

## 2. 数据

ST-EVCDP（与 A3、A4 相同）：深圳 247 个小区，30 天，5 分钟；需求为占用。零样本设定下，随机取 20%–80% 的小区作为"未见区域"，不参与微调。已见区域按时间 6:2:2 切分。

## 3. 方法

1. **提示词改写**：把小区特征（历史占用、价格、天气、道路密度、POI、快慢充类型、邻区平均需求与价格）写成自然语言描述，加入"角色扮演"式指令
2. **多区域对齐微调**：以 Sentence-T5 为主干，用 MAML 式元学习微调，损失为负对数似然
3. 设置：回看 12 步、预测 6 步（30 分钟）；AdamW，batch 48，学习率 0.001，最多 200 轮，早停 10 轮；两张 RTX 3090

## 4. 价格与因果的处理方式

价格只是提示词中的一个描述性字段，**没有任何因果或结构化处理**。

## 5. 基线

ARIMA、Lasso、FCNN、LSTM、GCN-LSTM、STGCN、HSTGCN（F1）、PIAST（Kuang et al. 2024，Applied Energy）、PromptCast、LLMTIME。指标：RMSE、MAE。

## 6. 主要结果

在全量、少样本、零样本三种设定下均优于基线，少样本和零样本优势更明显（具体数值见原文表格，需要时再查）。

## 7. 与我们的关系

| 方面 | 说明 |
|---|---|
| 作为基线 | 可选。主干是语言模型，在 4060（8 GB）上微调 Sentence-T5 可行，但成本较高，**建议放在扩展实验，不作为主基线** |
| 作为相关工作 | 代表"预测精度与泛化"路线，与我们"因果可信性"路线形成对比 |
| 可借鉴 | 零样本"未见区域"的评价设定，可用于我们的跨区域 / 跨城市扩展实验 |
| 局限（原文自述） | 可解释性不足，列为未来工作 |

## 8. 代码

https://github.com/Quhaoh233/ChatEV

## 9. 需要补读的文献（出自本文基线）

- **PIAST**：Kuang, H., Qu, H., Deng, K., Li, J. (2024). A physics-informed graph learning approach for citywide electric vehicle charging demand prediction and pricing. *Applied Energy*, 363, 123059。**同团队在我们首选目标期刊上的论文，涉及"预测 + 定价"，必须找来精读并对比**（需学校账号）
