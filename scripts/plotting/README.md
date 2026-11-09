# 预测试绘图与正交测试

这里的“预测试”指正式 30-seed 批次之前的网络容量与更新预算扫描，以及训练好网络后的先验线索×真实概率正交测试；它不指正式冲突测试。现有 checkpoint、训练历史、汇总表和图像位于 [outputs/outputs_prerun](../../outputs/outputs_prerun/)；分析目录也收录了一份[结果与脚本](../../analysis/prerun_analysis/README.md)。

| 脚本 | 输入 | 功能 |
|---|---|---|
| `plot_learning_curves.py` | 预测试 `arms/` 的训练历史 | hidden size 和更新预算的训练曲线 |
| `plot_pbet_structure.py` | 预测试 `frozen_tests/` | 各测试结构下的投注概率 |
| `test_orthogonal_prior_grid.py` | 预测试 `h16_b200`、`h32_b200` checkpoint | 9×9 cue×真实概率单轮测试 |
| `plot_orthogonal_by_capacity.py` | 正交测试逐试次汇总 | 分 hidden size 的轨迹和指标 |
| `plot_decoded_belief.py` | 正式冲突测试与线性 decoder | hidden-state 读出轨迹 |

正交测试的 cue 和真实概率各取 0.1–0.9。每个单轮 cell 从零 hidden state 开始；相同真实概率下各 cue 的 outcome 序列逐试次匹配。四个测试端点 cue（0.1、0.2、0.8、0.9）超出训练 cue 网格。

## 从随附结果重跑

从项目根目录运行，先按[总指南](../../README.md)安装依赖并设置单线程环境变量。下面使用新目录，避免改动随附结果。若虚拟环境建在项目上一级，把 `$PY` 改为 `"..\.venv\Scripts\python.exe"`。复制 `arms/` 后即可重绘训练曲线；复制现有正交轨迹后也可直接重绘容量分析图：

```powershell
$PY = ".\.venv\Scripts\python.exe"
$WORK = ".\review_prerun"
New-Item -ItemType Directory -Path $WORK | Out-Null
Copy-Item -LiteralPath ".\outputs\outputs_prerun\arms" -Destination "$WORK\arms" -Recurse
New-Item -ItemType Directory -Path "$WORK\metrics" | Out-Null
Copy-Item -LiteralPath ".\outputs\outputs_prerun\metrics\orthogonal_grid_trajectories.csv" -Destination "$WORK\metrics\orthogonal_grid_trajectories.csv"
& $PY -B -m scripts.plotting.plot_learning_curves --output-root $WORK
& $PY -B -m scripts.plotting.plot_orthogonal_by_capacity --output-root $WORK
```

若要从 checkpoint 重新生成逐试次数据及完整分析，请使用另一个全新目录（同样先复制 `arms/`，此时无需复制轨迹 CSV），按顺序运行：

```powershell
$WORK = ".\review_prerun_full"
New-Item -ItemType Directory -Path $WORK | Out-Null
Copy-Item -LiteralPath ".\outputs\outputs_prerun\arms" -Destination "$WORK\arms" -Recurse
& $PY -B -m scripts.testing.run_prerun_tests `
  --output-root $WORK --arms h16_b200 h16_b600 h32_b200 h32_b600 `
  --protocols human_default legacy_3plus1 conflict_only --episodes 200 --seed 20261107
& $PY -B -m scripts.plotting.plot_pbet_structure --output-root $WORK --budget 200
& $PY -B -m scripts.plotting.plot_pbet_structure --output-root $WORK --budget 600
& $PY -B -m analysis.prerun_analysis.code.test_prior_injection --output-root $WORK
& $PY -B -m scripts.plotting.test_orthogonal_prior_grid `
  --output-root $WORK --episodes 50
& $PY -B -m scripts.plotting.plot_orthogonal_by_capacity `
  --output-root $WORK
```

正交测试覆盖 9×9 个 cue×真实概率条件、6 个冻结网络、每 cell 50 个 episode，生成 486 个原始 cell，耗时和磁盘占用明显高于绘图。其输出写入 `$WORK\diagnostics\orthogonal_grid\`，冻结测试写入 `$WORK\frozen_tests\`；全部新图和汇总表统一写入 `$WORK\figures\` 与 `$WORK\metrics\`。已有汇总结果可先直接查看，无需等待完整重跑。请勿在不同 episode 数的运行间复用同一个工作目录；正交脚本会复用已存在的 cell 文件。
