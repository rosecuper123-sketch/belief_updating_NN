# 预测试分析

本目录收录正式批次确定前的网络训练与测试分析，以及先验线索×真实概率正交测试。汇总表位于 `metrics/`，图像位于 `figures/`，相应脚本位于 `code/`。

| 脚本 | 所需输入 | 主要结果 |
|---|---|---|
| `code/plot_learning_curves.py` | 预测试训练历史 | `learning_curves_summary.csv`、figA1–A3 |
| `code/plot_pbet_structure.py` | 预测试冻结测试 | `pbet_by_structure_*.csv`、figB1–B2 |
| `code/test_prior_injection.py` | 单轮一致/冲突测试 | `prior_injection_single_round.csv` 与图 |
| `code/diag_round_structure.py` | 预测试 checkpoint | 轮次结构诊断图与运行时原始数据 |
| `code/test_orthogonal_prior_grid.py` | 预测试 checkpoint | 9×9 正交网格轨迹与热图 |
| `code/plot_orthogonal_by_capacity.py` | 正交网格逐试次汇总 | 按 hidden size 汇总的指标和图 |

`metrics/orthogonal_grid_trajectories.csv` 提供每个网络×条件×试次的汇总轨迹；其余指标和图表可直接在本目录查看。完整的预测试 checkpoint、训练历史及一份结果副本位于 [outputs/outputs_prerun](../../outputs/outputs_prerun/)：`arms/` 是四个训练臂（h16/h32 × 200/600 次更新），每臂各有三个 seed 的 `best.pt` 和 `training_history.csv`；`metrics/` 和 `figures/` 可供直接查阅。逐 episode 的 `frozen_tests/` 与正交网格原始 cell 文件可从这些 checkpoint 重新生成。

## 使用方法

从 `formal_run_001` 根目录运行，可先查看各脚本参数：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -B -m analysis.prerun_analysis.code.test_prior_injection --help
& $PY -B -m analysis.prerun_analysis.code.test_orthogonal_prior_grid --help
& $PY -B -m analysis.prerun_analysis.code.plot_orthogonal_by_capacity --help
```

从随附 checkpoint 重跑时，先在独立目录复制 `arms/`；此后所有命令均传入同一个 `--output-root`。训练曲线直接读取训练历史；冻结测试生成 `frozen_tests/` 后，投注概率图和单轮先验分析才能读取逐试次文件；正交测试直接读取 h16_b200 与 h32_b200 的 checkpoint。完整命令见[绘图与正交测试说明](../../scripts/plotting/README.md)。正交网格每个 cell 独立启动一轮 120 个反馈试次；同一真实概率下的不同 cue 使用匹配的 outcome 序列。

新生成的指标、PNG/PDF 图和原始数据分别位于 `<output-root>/metrics/`、`<output-root>/figures/`、`<output-root>/frozen_tests/` 或 `<output-root>/diagnostics/`。请使用全新输出目录，以保留项目中提供的结果。
