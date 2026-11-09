# 冻结测试与自检

测试脚本加载已训练的 `best.pt`，导出逐试次行为、投注概率和 hidden state。主要入口如下：

| 入口 | 用途 |
|---|---|
| `scripts.testing.selfcheck` | 小规模检查环境、训练、boundary 和导出流程 |
| `scripts.testing.run_frozen_tests` | 对指定 checkpoint 运行冻结测试 |
| `scripts.testing.run_prerun_tests` | 遍历训练臂进行批量测试或结构检查 |

从 `formal_run_001` 根目录运行：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -B -m scripts.testing.selfcheck
& $PY -B -m scripts.testing.run_frozen_tests `
  --checkpoint .\review_single\runs\formal_osla_seed20260804\best.pt `
  --out-root .\review_single\frozen_tests `
  --protocol conflict_only --episodes 200 --seed 20261107
```

对 [训练入口](../training/README.md)产生的 30-seed 训练臂进行正式测试：

```powershell
& $PY -B -m scripts.testing.run_prerun_tests `
  --output-root .\review_batch --arms h16_b200 `
  --protocols conflict_only --episodes 200 --seed 20261107
```

预测试网络的 checkpoint 已随项目放在 `outputs/outputs_prerun/arms/`。如需重跑预测试，请先将该 `arms/` 复制到一个新工作目录，再按[预测试指南](../plotting/README.md)运行三个协议和正交网格；无需重新训练网络。`--check-only` 对现有测试文件执行结构检查，并会更新检查记录，请在工作副本中使用。逐试次 `action` 是从 `choice_probability` 采样的选择。数据层级和 manifest 位置见[运行数据说明](../../outputs/README.md)。
