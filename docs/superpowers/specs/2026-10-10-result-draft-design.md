# RESULT.md 结果初稿设计

日期：2026-10-10

## 目标与位置

在 `formal_run_001/RESULT.md` 撰写中文论文结果部分初稿。正文使用平实、直观、严谨的学术语言，所有定量陈述来自 `formal_run_001` 中的正式 CSV、拟合摘要和反事实诊断文件，直接嵌入已有正式 PNG 图像，不重新制图。

## 结果证据链

正文采用“训练与行为 → 认知模型拟合 → actor反事实重放”的顺序：

1. 网络训练效果；
2. 先验吸收与正式冲突条件中的行为更新；
3. M1/M2/M3 拟合质量和模型比较；
4. M3 学习率、gate 与拟合 belief 的时序；
5. 冻结 actor 反事实策略有效更新率；
6. M3 与 direct-actor 结果的一致和不一致；
7. 反事实方法的数值验证与解释边界。

## 条件命名

- 强证据条件族：`mac1A`、`mac1B`，正文简称 `mac1`。
- 弱证据条件族：`mac2A`、`mac2B`，正文简称 `mac2`。
- 后缀 A/B 仍用于区分各条件族内不同 cue/true-probability 组合，不再用作强弱证据标签。
- `mac1A/mac1B` 使用 reversal 表述；`mac2A/mac2B` 的真实概率为0.5，使用 settling 表述。

## 章节与图像映射

### 1. 网络训练与策略形成

数据：`analysis/behaviour_analysis/metrics/formal_training_summary_h16_b200.csv`。

图：`analysis/behaviour_analysis/figures/formal_training_curves_h16_b200.png`。

报告训练/验证收益、策略熵和 checkpoint 选择所需信息；不把训练损失解释为认知机制。

### 2. 先验读取与冲突条件行为

数据：`prior_injection_h16.csv`、`prior_injection_slopes_h16.csv`、`formal_behavior_conflict_only_h16_b200.csv`、`reversal_metrics_h16.csv`。

图：

- `formal_prior_injection_h16.png`；
- `formal_behavior_trajectories_h16_b200.png`；
- `formal_reversal_metrics_h16.png`。

先报告 opening P(bet) 随 cue 的单调变化，再报告 `mac1` 从先验一侧向真实概率支持的策略方向反转。`mac2` 只报告向中性行为区域的 settling，不强行定义反转潜伏期或唯一最优行动。

### 3. 模型拟合与比较

数据：`analysis/fitting_NN/metrics/fit_summary.csv`、`ic_comparison.csv`、`ic_pairwise.csv`。

图：`analysis/fitting_NN/figures/model_comparison_ic.png`。

报告参数层 R-hat、ESS、divergence、treedepth、LOOIC、WAIC、差值标准误和 Pareto-k。正文将 M3 相对 M1/M2 的预测改善与 PSIS 可靠性限制同时呈现，不把存在高 Pareto-k 的精确差值作强机制解释。

### 4. M3学习率时序

数据：`lr_sequence_mac1A.csv`、`lr_sequence_mac1B.csv`、`lr_sequence_mac2A.csv`、`lr_sequence_mac2B.csv` 和各组 `key_params_*.csv`。

图：`analysis/fitting_NN/figures/lr_sequence_condA.png`。

正文说明：`mac1A/mac1B` 的正负学习率发生交叉；`mac2A/mac2B` 不出现稳定交叉。学习率曲线与 gate、拟合 belief 联合描述，不把群体中位轨迹写成每个网络都相同。

### 5. Direct-actor反事实重放

数据：`analysis/counterfactoral_lr/outputs/result/metrics/actor_pbet_lr_trial_summary.csv`、`actor_pbet_lr_window_summary.csv`、`actor_pbet_lr_denom_sensitivity.csv`、`actor_pbet_lr_diagnostics.csv`。

图：

- `analysis/counterfactoral_lr/outputs/result/figures/actor_pbet_lr_timeseries.png`；
- `analysis/counterfactoral_lr/outputs/result/figures/actor_pbet_lr_diagnostics.png`。

正式方法称为 direct-actor counterfactual replay。它直接调用冻结 actor，不使用 p-bet decoder 计算学习率。正文报告策略有效更新率的时序形态、未归一化概率变化和分母有效比例。

### 6. 两类学习率证据的比较

比较方向、相对排序和交叉时序，不直接比较绝对值。预期表述为：`mac1` 的反馈权重重组在两种方法中方向相符；`mac2B` 均保持正反馈率较高；`mac2A` 的 actor 轨迹存在局部接近或短暂交叉，而 M3 群体中位轨迹不交叉，因此整体为部分一致而非完全复制。

### 7. 反事实方法验证

将验证分为两层：

- 计算实现：自然分支 hidden-state replay、磁盘 P(bet) 复现、增益恒等式、分母阈值敏感性和原始 delta-p；
- 构念效度：所得量是 policy-level effective update rate，不等同于内部 latent-belief learning rate。

只有第一层可使用“数值验证充分”或“实现检查通过”的表述；第二层保留解释限制。

## 定量报告规则

- 网络 seed 是跨网络统计单位，episode 和 trial 不作为独立网络样本。
- 优先报告均值及95% CI，或文件中已经定义的中位数及IQR；不混用统计口径。
- 图中可直接读取但 CSV 中没有对应字段的数值不写成精确统计量。
- 不从旧 `正式版网络代码/RESULTS_DRAFT.md` 复制数值。
- 不将描述性差异写成显著性差异，除非正式输出包含对应检验。
- M3 后验轨迹、actor反事实轨迹和行为概率的尺度不同，正文不将绝对量级直接等同。

## RESULT.md版式

- 一级标题：`# 结果`。
- 使用编号二级标题对应证据链。
- 每张图以相对 Markdown 路径嵌入，并在图下提供中文图题和必要的读图说明。
- 正文保持论文结果语气，不写运行命令、代码审计过程或 README 式文件导航。
- 文末加入“当前结果的解释范围”小节，集中说明模型比较、反事实指标和群体汇总的限制。

## 验收标准

- 三项用户指定结果均有正式数据、定量描述和对应图像。
- 强证据 `mac1` 与弱证据 `mac2` 的交叉差异表述与正式 M3 图一致。
- direct-actor replay 与 p-bet decoder 不混称。
- 对模型拟合和反事实方法的稳健性既给出支持证据，也明确限制。
- 所有图片链接和数据来源路径存在。
- 不改写任何 CSV、拟合文件或图像。
