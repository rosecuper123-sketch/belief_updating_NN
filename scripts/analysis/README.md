# 正式训练与行为分析入口

本目录的脚本读取 `outputs/runs/` 中的训练历史和冻结测试数据。各项结果集中在 [formal_behav_analysis](../../analysis/formal_behav_analysis/README.md)，其中给出完整运行顺序、参数和输出文件。

| 入口 | 作用 |
|---|---|
| `plot_formal_training_curves.py` | 训练轨迹与 checkpoint 摘要 |
| `plot_formal_behavior.py` | 冲突条件的投注概率与行为指标 |
| `collect_consistent_rollout.py` | 为先验和 decoder 分析生成一致条件单轮数据 |
| `analyze_prior_injection.py` | 一致条件下的先验响应 |
| `train_belief_decoder.py` | 在一致条件 hidden state 上拟合线性读出 |
| `analyze_reversal.py` | 反转、settling 与 decoder 应用 |

示例：从 `formal_run_001` 根目录，将训练与行为摘要写入新的工作目录。

```powershell
$PY = ".\.venv\Scripts\python.exe"
$RUNS = ".\outputs\runs"
$OUT = ".\review_analysis\formal_behavior"
& $PY -B -m scripts.analysis.plot_formal_training_curves `
  --out-root $RUNS --analysis-root $OUT --hidden-size 16 --budget 200
& $PY -B -m scripts.analysis.plot_formal_behavior `
  --out-root $RUNS --analysis-root $OUT `
  --hidden-size 16 --budget 200 --protocol conflict_only
```

先验分析和 decoder 需要 `consistent_rollout/`；生成步骤见上述分析目录的 README。
