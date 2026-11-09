# 分析代码与结果

各子目录收录相应分析的代码、指标和图表。运行命令默认从 `formal_run_001` 根目录执行；模型拟合的 R 命令以 [拟合目录](fitting_NN/README.md) 为工作目录。

| 目录 | 分析内容 | 主要结果 |
|---|---|---|
| [prerun_analysis](prerun_analysis/README.md) | 预测试训练曲线、投注行为及先验×环境正交测试 | `metrics/`、`figures/` |
| [formal_behav_analysis](formal_behav_analysis/README.md) | 正式训练摘要、冻结测试行为、反转及隐状态解码 | `metrics/`、`figures/` |
| [fitting_NN](fitting_NN/README.md) | M1–M3 Stan 拟合、模型比较和学习率时序 | `data/`、`fits/`、`metrics/`、`figures/` |
| [counterfactoral_lr](counterfactoral_lr/README.md) | 冻结 actor 的反事实单步重放 | `outputs/result/` |

`analysis/__init__.py` 标记 Python 包。正式行为分析中的 `loading.py` 也供反事实分析读取冻结测试数据。
