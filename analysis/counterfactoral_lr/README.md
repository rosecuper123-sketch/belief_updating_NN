# 冻结策略的反事实更新率

本分析从同一个反馈前 hidden state 出发，分别将下一步输入中的 outcome 设为 win 和 loss，再使用冻结 actor 读取两条分支的投注概率。主程序为 `counterfactual_actor_pbet_lr.py`；`replay_gate.py` 检查单步重放能否重现已保存的状态，`pbet_stats.py` 负责聚合与统计。结果写入 `outputs/result/metrics/` 和 `outputs/result/figures/`。

## 输入与指标

输入根目录为 `outputs/runs/`：`arms/h16_b200/runs/` 提供 `best.pt`，`frozen_tests/h16_b200/conflict_only/` 提供逐试次行为和 hidden state。默认分析每个 cell 中预先划定的 held-out episodes；`--all-episodes` 用于另行分析全部 episode。

令当前投注概率为 `p = actor(h_t)`，两个反事实下一状态的概率分别为 `p_win`、`p_loss`。程序计算

```text
alpha_win  = (p_win - p) / (1 - p)
alpha_loss = (p - p_loss) / p
```

这两个量描述策略概率尺度上的有效更新。主分析仅在 `0.01 ≤ p ≤ 0.99` 时报告比率，并输出阈值 0.005、0.01、0.02、0.05 下的敏感性结果。原始概率变化仍保留；比率不裁剪到 `[0, 1]`。每轮最后一个试次没有同轮的下一 actor state，因此更新率计算到试次 119。

## 运行

从 `formal_run_001` 根目录运行，按[总指南](../../README.md)设置 Python 环境和单线程变量：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -B -m analysis.counterfactoral_lr.counterfactual_actor_pbet_lr `
  --batch-runs .\outputs\runs `
  --analysis-root .\review_analysis\counterfactual `
  --hidden-size 16 --budget 200 --protocol conflict_only `
  --denom-eps 0.01
```

使用独立的 `--analysis-root` 保存新结果。改变 episode 范围或分母阈值时，也应分别保存输出和命令。

## 结果与检查

程序生成 `actor_pbet_lr_episode.csv`、按 seed 和 trial 聚合的 CSV、时间窗摘要、分母敏感性表、诊断表、图像和运行报告。当前目录提供的汇总文件可在 `outputs/result/metrics/` 查看。诊断包括自然 outcome 分支对已保存下一状态的重放误差、actor 概率与 `trials.csv` 的一致性，以及两分支增益的代数核对。

2026-11-09 的独立稳健性复查见 [报告](outputs/robustness_20261109/ROBUSTNESS_REPORT.md)和[复现命令](outputs/robustness_20261109/REPRODUCE.md)。复查保存了原结果缺少的逐 episode 表、全部 episode 对照、seed 层面的 bootstrap 与留一分析，以及按 pbet 区间的诊断。
