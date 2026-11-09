# 正式运行数据

`outputs/runs/` 保存 hidden size 16、200 次更新、30 个 seed 的正式训练和冻结测试。`outputs/outputs_prerun/` 保存正式批次之前的预测试：`arms/` 中有 h16/h32 × 200/600 次更新、每臂三个 seed 的 `best.pt` 和训练历史；`metrics/` 与 `figures/` 提供现有正交测试和其他预测试的汇总结果。按[预测试复现指南](../scripts/plotting/README.md)，可从这些 checkpoint 重新生成未随项目附带的逐 episode 冻结测试与正交 cell 文件。

## `runs/` 的三个部分

| 路径 | 阶段 | 内容 |
|---|---|---|
| `arms/h16_b200/runs/formal_osla_seed*/` | 正式训练 | `best.pt`、`final.pt`、`training_history.csv`、`run_metadata.json` |
| `frozen_tests/h16_b200/conflict_only/formal_osla_seed*/<group>/` | 正式测试 | 每个网络×条件的逐试次行为和 hidden state |
| `consistent_rollout/cue*/h16_seed*/custom/formal_osla_seed*/mac1A/` | 训练后辅助测试 | cue 与真实概率一致的单轮数据，供先验分析和 belief decoder 校准 |

`arms` 是训练臂的简称；`h16_b200` 表示 hidden size 16、最多 200 次参数更新。`frozen_tests` 从 `best.pt` 加载网络并固定权重。`consistent_rollout` 单独生成，用于分析网络对一致环境概率的表征。

一个冻结测试 cell 包含 `trials.csv`、`boundary_events.csv`、`hidden.npz` 和 `boundary_hidden.npz`。测试调用的 `manifest.json` 保存在 `frozen_tests/h16_b200/`，同目录的 `manifest_provenance.json` 记录 checkpoint SHA-256 等来源信息；`runs/frozen_test_checks.json` 记录结构检查。manifest 中的 checkpoint 路径是生成时的绝对路径，核对当前文件时应使用 `arms/h16_b200/runs/` 中的 `best.pt` 及其哈希。逐试次 `choice_probability` 为 actor 的投注概率，`action` 为相应的采样动作。

## 运行入口

[总指南](../README.md)提供独立目录中的单网络与 30-seed 命令。训练模块是 `scripts.training.prerun_batch`，冻结测试模块是 `scripts.testing.run_prerun_tests`；`outputs/scripts/` 保存正式批次的驱动脚本和参数。运行分析时，把 `outputs/runs/` 作为正式数据根目录：

- [行为与隐状态分析](../analysis/formal_behav_analysis/README.md)读取训练历史、冲突测试和一致条件 rollout。
- [模型拟合](../analysis/fitting_NN/README.md)从冲突测试提取网络选择。
- [反事实分析](../analysis/counterfactoral_lr/README.md)同时读取 checkpoint、逐试次记录和 hidden state。

批量计算会写入同名训练或测试文件；建议在独立工作目录运行，再与这里提供的结果核对。
