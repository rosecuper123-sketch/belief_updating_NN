# 正式训练与行为分析

本目录整理 30 个正式网络的训练轨迹、冲突条件投注行为、先验响应、反转指标和 hidden-state belief decoder。分析读取 [正式运行数据](../../outputs/README.md)，不更新网络权重。

## 目录与输入

`code/` 保存各分析脚本；`loading.py` 读取冻结测试文件，`belief_decoder.py` 实现线性读出。`metrics/` 保存表格与 decoder，`figures/` 保存图像。相应命令行入口在 `scripts/analysis/`，解码轨迹绘图入口在 `scripts/plotting/plot_decoded_belief.py`。

输入根目录为 `outputs/runs/`，其中：

- `arms/h16_b200/runs/` 提供训练历史和 `best.pt`；
- `frozen_tests/h16_b200/conflict_only/` 提供四个正式冲突条件的行为和 hidden state；
- `consistent_rollout/` 提供 cue 与真实概率一致的单轮数据，供先验分析和 decoder 校准。

## 运行顺序

以下命令从 `formal_run_001` 根目录执行。示例把指标和图像写入新的 `review_analysis/formal_behavior/`；请先依照[总指南](../../README.md)配置 Python 环境。

```powershell
$PY = ".\.venv\Scripts\python.exe"
$RUNS = ".\outputs\runs"
$OUT = ".\review_analysis\formal_behavior"

& $PY -B -m scripts.analysis.plot_formal_training_curves `
  --out-root $RUNS --analysis-root $OUT --hidden-size 16 --budget 200
& $PY -B -m scripts.analysis.plot_formal_behavior `
  --out-root $RUNS --analysis-root $OUT `
  --hidden-size 16 --budget 200 --protocol conflict_only
& $PY -B -m scripts.analysis.analyze_prior_injection `
  --out-root $RUNS --analysis-root $OUT --hidden-size 16
& $PY -B -m scripts.analysis.train_belief_decoder `
  --out-root $RUNS --frozen-root $RUNS --analysis-root $OUT `
  --hidden-size 16 --budget 200
& $PY -B -m scripts.analysis.analyze_reversal `
  --out-root $RUNS --analysis-root $OUT --hidden-size 16 --budget 200
```

`train_belief_decoder` 在一致条件 rollout 的 hidden state 上按 episode 划分训练与保留集，为每个网络拟合一个 decoder，再将其应用于冲突条件。若需重建 `consistent_rollout/`，在包含 `arms/` 的工作副本中先运行：

```powershell
$COPY_RUNS = ".\review_batch"
& $PY -B -m scripts.analysis.collect_consistent_rollout `
  --out-root $COPY_RUNS `
  --hidden-size 16 --budget 200 --episodes 200 --skip-existing
```

## 结果

| 路径 | 内容 |
|---|---|
| `metrics/formal_training_summary_h16_b200.csv` | 更新过程与权重选择摘要 |
| `metrics/formal_behavior_conflict_only_h16_b200.csv` | 冲突条件的投注行为 |
| `metrics/prior_injection_h16.csv`、`prior_injection_slopes_h16.csv` | 一致条件下的先验响应 |
| `metrics/validated_decoders/` | 运行 `train_belief_decoder` 后生成的线性读出文件与诊断 |
| `metrics/reversal_metrics_h16.csv` | 反转和收敛指标 |
| `figures/` | 对应训练、行为、先验与反转图像 |

`choice_probability` 是 actor 的投注概率，`action` 是据此采样的选择。opening window 为试次 1–5，late window 为试次 101–120；真实概率为 0.5 时，crossing 指标按脚本定义可为空。decoder 读出的是 hidden state 中可线性提取的环境概率信息，解释范围见 [核心代码](belief_decoder.py)。
