# 网络与任务实现

`formal_osla/` 实现 one-shot cue 输入的 GRU 策略、环境、训练和冻结测试。命令行入口集中在 [scripts](../scripts/README.md)；本目录提供这些入口调用的核心函数。

## 模块

| 文件 | 作用 |
|---|---|
| `config.py` | 网络、环境、训练和验证参数 |
| `model.py` | `GRUCell` 与投注概率输出层 |
| `protocols.py` | 训练条件、测试条件和 outcome 生成 |
| `rollout.py` | episode 推进、反馈、奖励与策略梯度 |
| `train.py` | 更新循环、验证、权重选择与训练记录 |
| `test_runner.py` | 加载 checkpoint 并运行冻结测试 |
| `export.py` | 导出逐试次、round boundary 和 hidden-state 数据 |

## 输入与时序

网络接收七项输入：前一次 action、前一次可见 outcome、前一次 token 变化、标准化累计 token、试次位置、round boundary 标记，以及该 round 开始时一次性呈现的 cue logit。每个 episode 的 hidden state 从全零开始；同一 episode 的 round 之间连续传递 hidden state。当前 trial 的 outcome 在选择后才用于下一次状态更新。

训练与验证均从 cue 和真实概率一致的条件采样。训练信号是任务收益；正式冲突条件在网络训练后用于冻结测试。测试时仍保存随机采样的 `action` 和采样前的 `choice_probability`。

## 权重选择

`train.py` 默认每 10 次更新进行一次验证。验证分数提高时保存 `best.pt`；达到 `patience` 次连续未提高的验证后提前结束，或在更新预算耗尽时结束。正式配置的 `lambda_anchor=0`，权重选择指标为验证集平均收益。`final.pt` 保存训练结束时恢复的最佳验证权重；正式测试使用 `best.pt`。

## 输出文件

训练目录 `<output-root>/runs/formal_osla_seed<seed>/` 包含 `best.pt`、`final.pt`、`training_history.csv` 和 `run_metadata.json`。冻结测试的每个 network×group 目录包含：

| 文件 | 内容 |
|---|---|
| `trials.csv` | action、投注概率、outcome、reward 及试次信息 |
| `boundary_events.csv` | round 开始时的 cue 和边界事件 |
| `hidden.npz` | 逐试次 recurrent state |
| `boundary_hidden.npz` | boundary 前后状态 |

测试根目录的 `manifest.json` 记录 checkpoint、seed 和各条件输出路径。正式批次的实际目录见 [运行数据说明](../outputs/README.md)。

## 基本检查

从项目根目录运行：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -B -m scripts.testing.selfcheck
& $PY -B -m scripts.training.train_formal --help
& $PY -B -m scripts.testing.run_frozen_tests --help
```

单网络训练与冻结测试的完整示例见[总指南](../README.md)。
