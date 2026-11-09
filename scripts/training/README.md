# 训练入口

`train_formal.py` 训练单个 GRU；`prerun_batch.py` 遍历给定的 hidden size、更新预算和 seed。两者调用 [formal_osla](../../formal_osla/README.md) 中相同的训练实现。训练输出位于指定根目录的 `runs/`，批量运行再按训练臂整理到 `arms/`。

| 入口 | 关键参数 | 输出目录 |
|---|---|---|
| `scripts.training.train_formal` | `--seed`、`--hidden-size`、`--updates`、`--output-root` | `<output-root>/runs/formal_osla_seed<seed>/` |
| `scripts.training.prerun_batch` | `--seeds`、`--hidden-sizes`、`--budgets`、`--output-root` | `<output-root>/arms/h<h>_b<budget>/runs/` |

从 `formal_run_001` 根目录运行，先按[总指南](../../README.md)准备 Python 环境与单线程环境变量：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -B -m scripts.training.train_formal `
  --output-root .\review_single --seed 20260804 `
  --hidden-size 16 --updates 200 --batch-size 64
```

完整正式批次可用 `$SEEDS = 20260804..20260833`，再运行：

```powershell
& $PY -B -m scripts.training.prerun_batch `
  --output-root .\review_batch --hidden-sizes 16 --budgets 200 --seeds $SEEDS
```

`best.pt` 由验证集选择，正式冻结测试使用该文件。预测试容量扫描采用 hidden size 16/32、预算 200/600、seed 20260928–20260930；其 `best.pt` 和训练历史已位于 `outputs/outputs_prerun/arms/`。复核预测试的训练后测试与图表可以直接使用这些权重，无需先重训；若要复核训练本身，再按上述入口设置对应参数，并写入独立输出目录。
