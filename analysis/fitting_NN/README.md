# 网络行为的 Stan 模型拟合

本目录用层级贝叶斯模型拟合冻结网络在四个 `conflict_only` 条件中的二元 `action`。每个条件从 30 个网络各选取一条 120 试次 episode；所选 episode 记录在 `metrics/m3_sampling_manifest.csv`。拟合只读取网络行为，不更新网络参数。

## 模型与文件

| 模型 | 学习率设定 | Stan 文件 |
|---|---|---|
| M1 | 单一固定学习率 | `stan/M1.1.stan` |
| M2 | 固定的正、负结果学习率 | `stan/M2.1.stan` |
| M3 | 随试次变化的学习率 | `stan/M3.1.stan` |

`stan/M3.1_multi_episode.stan` 与 `scripts/fit_multi_episode.R` 用于单独的多 episode 扩展。主分析的输入长表在 `metrics/formal_osla__h16_b200__*_m3_subjects.csv`，Stan 数据在 `data/*.rds`，拟合对象在 `fits/`，模型比较和学习率时序在 `metrics/`，图像在 `figures/`。

长表中 `choice` 对应网络 `action`，`reward` 对应环境 outcome（0/1）；任务收益另存于冻结测试 `trials.csv` 的 `reward` 字段。固定抽样的 manifest、长表和 RDS 应成套使用，以复核现有结果。

## 环境

下面的命令从 `formal_run_001/analysis/fitting_NN` 执行。需要 Python 环境、R 4.5.x、R 包 `cmdstanr`、`posterior`、`loo`、`jsonlite`，以及 CmdStan 2.38 和 C++ 编译工具。将 `M31_R_LIB` 设为包含上述 R 包的 library；CmdStan 编译和临时文件使用纯 ASCII 路径。

```powershell
$PY = "..\..\.venv\Scripts\python.exe"
$RSCRIPT = "C:\Program Files\R\R-4.5.3\bin\Rscript.exe"
$env:M31_R_LIB = "D:\path\to\Rlib"
$env:CMDSTAN = "C:\Users\<user>\.cmdstan\cmdstan-2.38.0"
$env:M31_RUNTIME_ROOT = "D:\m31_runtime_h16"

& $RSCRIPT -e ".libPaths(c(Sys.getenv('M31_R_LIB'),.libPaths())); p<-c('cmdstanr','posterior','loo','jsonlite'); print(sapply(p, requireNamespace, quietly=TRUE))"
& $RSCRIPT -e ".libPaths(c(Sys.getenv('M31_R_LIB'),.libPaths())); cmdstanr::set_cmdstan_path(Sys.getenv('CMDSTAN')); cmdstanr::cmdstan_version()"
```

这些路径需要按本机安装位置填写。可先对现有拟合文件做只读检查：

```powershell
& $PY -B scripts\write_fit_provenance.py --check
```

## 从固定输入重新拟合

`data/*.rds` 已提供。若需从同目录 `metrics/` 中的长表重新生成，可先运行 `& $RSCRIPT scripts\build_rds.R`。建议在项目工作副本中执行以下拟合与绘图命令，保留随附产物。

```powershell
& $RSCRIPT scripts\fit_model.R --model M1 --group mac1A --mode smoke

$MODELS = "M1", "M2", "M3"
$GROUPS = "mac1A", "mac1B", "mac2A", "mac2B"
foreach ($m in $MODELS) {
  foreach ($g in $GROUPS) {
    & $RSCRIPT scripts\fit_model.R --model $m --group $g --mode final
  }
}

& $RSCRIPT scripts\compare_loo.R --mode final
& $RSCRIPT scripts\compare_ic.R --mode final
foreach ($g in $GROUPS) {
  & $RSCRIPT scripts\extract_lr_sequence.R `
    --fit ".\fits\M3_condA_${g}_final\fit.rds" `
    --data ".\data\formal_osla__h16_b200__${g}.rds" --group $g
}
& $PY -B scripts\audit_diagnostics.py
& $PY -B scripts\summarise_fits.py
& $PY -B scripts\plot_lr_sequence.py
& $PY -B scripts\plot_model_comparison.py --mode final
```

`smoke` 用较短采样检查工具链；`final` 的链数、迭代数和采样参数由 `fit_model.R` 定义。正式拟合共 3 模型×4 条件。运行后应检查 `fits/<model>_condA_<group>_final/diagnostic_summary.csv`、`metrics/ic_comparison.csv`、`metrics/ic_pairwise.csv` 和 `metrics/lr_sequence_*.csv`。

## 重新抽样

如需从 `outputs/runs/frozen_tests/` 另选 episode，使用 `data/extract_m3_subjects.py --runs ..\..\outputs\runs --out-dir <新目录> --sampling-seed <种子>`，并保存新生成的 manifest。新抽样应使用独立的数据、拟合和结果目录；主分析的固定输入见本目录现有文件。
