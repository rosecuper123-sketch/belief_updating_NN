# 正式网络实验：代码与复现指南

本项目提供 GRU 网络的训练、冻结测试，以及预测试、行为分析、认知模型拟合和反事实更新率分析。以下路径均相对于 `formal_run_001` 根目录。

## 目录

| 路径 | 内容 |
|---|---|
| `formal_osla/` | 任务协议、GRU、训练、测试和数据导出 |
| `scripts/` | 单网络训练、冻结测试与 Python 分析入口 |
| `outputs/runs/` | 正式批次的 checkpoint、逐试次测试数据及一致条件 rollout |
| `outputs/outputs_prerun/` | 预测试 checkpoint、训练历史、汇总表和图像 |
| `outputs/scripts/` | 正式批次的参数与驱动脚本 |
| `analysis/` | 按主题整理的分析代码、拟合对象、指标和图像 |

正式批次包含 30 个独立 seed（20260804–20260833），采用 hidden size 16 的 GRU，每个训练最多 200 次更新。网络以任务收益为训练信号；训练和验证采用 cue 与真实概率一致的环境。冻结测试采用四个冲突条件，每个网络×条件运行 200 个 episode。本项目以每个 episode 的 120 个反馈试次作为与人类实验冲突阶段对齐的分析单元。模型权重在测试时固定，轮内适应由循环状态实现。任务与 checkpoint 细节见 [核心模块说明](formal_osla/README.md)。

## 软件环境

建议使用 Python 3.12。从本目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

如果已有位于项目上级目录的 Python 环境，可改用 `$PY = "..\.venv\Scripts\python.exe"`。Stan 拟合还需要 R、CmdStan 和 C++ 编译工具，配置见 [模型拟合说明](analysis/fitting_NN/README.md)。

## 快速检查与单网络示例

```powershell
& $PY -B -m scripts.testing.selfcheck
& $PY -B -m scripts.training.train_formal `
  --output-root .\review_single --seed 20260804 `
  --hidden-size 16 --updates 200 --batch-size 64
& $PY -B -m scripts.testing.run_frozen_tests `
  --checkpoint .\review_single\runs\formal_osla_seed20260804\best.pt `
  --out-root .\review_single\frozen_tests `
  --protocol conflict_only --episodes 200 --seed 20261107
```

`selfcheck` 进行小规模环境、训练和导出检查。后两步分别生成一个网络的训练记录和四个冲突条件的冻结测试；示例使用独立输出目录，保留随项目提供的正式结果。

## 正式批次

下面的通用入口使用与正式批次相同的网络规模、更新预算、seed 和测试协议。完整训练与测试可能耗时较长，请将输出写入独立目录。

```powershell
$SEEDS = 20260804..20260833
& $PY -B -m scripts.training.prerun_batch `
  --output-root .\review_batch --hidden-sizes 16 --budgets 200 --seeds $SEEDS
& $PY -B -m scripts.testing.run_prerun_tests `
  --output-root .\review_batch --arms h16_b200 `
  --protocols conflict_only --episodes 200 --seed 20261107
```

正式结果位于 `outputs/runs/`：[目录说明](outputs/README.md)解释 `arms/`、`frozen_tests/` 和 `consistent_rollout/` 的含义与文件层级。训练选择 `best.pt` 供冻结测试使用。

## 分析入口

建议从 [分析目录索引](analysis/README.md)进入各流程：

1. [预测试](analysis/prerun_analysis/README.md)：随附 checkpoint 的训练曲线、投注概率与先验×环境正交网格；[复现命令](scripts/plotting/README.md)可在独立目录重新生成原始测试数据和图表。
2. [正式行为](analysis/formal_behav_analysis/README.md)：训练摘要、反转行为和隐状态解码。
3. [认知模型拟合](analysis/fitting_NN/README.md)：M1–M3 Stan 拟合、模型比较及学习率时序。
4. [反事实更新率](analysis/counterfactoral_lr/README.md)：冻结 actor 的 win/loss 单步重放。

各分析的输入、运行顺序和输出位置以对应 README 为准。重新生成指标或图像时，使用新的输出目录并保存所用命令、软件版本与随机种子。
