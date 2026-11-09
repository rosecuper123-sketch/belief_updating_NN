# formal_run_001 复现审计（2026-10-10）

## 结论

核心 Python 训练、冻结测试、行为分析和 actor 反事实学习率管线可运行，并能复现已归档的主要数值结果。Stan 源码通过 CmdStan 2.38 的语法/语义检查，12 个已有 fit 的 hash 与 provenance 记录匹配，但当前 R 4.5.3 library 缺少必需包，因此本次没有重跑 MCMC 或直接读取 `fit.rds`。

## 已执行的检查

| 检查 | 结果 |
|---|---|
| 44 个主 Python 文件语法编译 | 通过 |
| 22 个 CLI 入口 `--help` | 通过 |
| `scripts.testing.selfcheck` | 通过 |
| 2-update 小型训练 + 4 groups × 3 episodes frozen test | 通过 |
| 训练曲线、冲突行为、prior injection 重算 | CSV 逐值一致 |
| 30 个 belief decoder 重新拟合 + reversal | 通过；行为字段一致 |
| actor 反事实学习率完整重算 | 通过；最大汇总差异 `1.17e-12` |
| 6 个 R 脚本 parse | 通过 |
| 4 个 Stan 文件 `stanc --warn-pedantic` | exit 0；只有 reduce-sum partial function 的保守赋值警告 |
| 12 个 fit 的 36 个 SHA-256 条目 | 全部匹配 |
| 参数摘要诊断 | 12/12 参数层 `Rhat <= 1.05`，0 divergences，0 max-treedepth hits |

## 审计中修复的问题

1. `counterfactual_actor_pbet_lr.py` 仍导入已移动的 `analysis.decode.pbet_stats`，已改为 `analysis.counterfactoral_lr.pbet_stats`。
2. 该脚本的默认输出路径指向旧目录，已改为当前 `analysis/counterfactoral_lr/outputs/result`。
3. reversal 对 true probability 0.50 的中性条件仍调用无定义的 normalized return score，会产生 NumPy warning；现在显式记为 `NaN`，不改变结果含义。
4. 原总 README 中的路径和“本目录不包含正式产物”说明已过时，且文本存在乱码；已按当前目录重写。

## 已知限制和未解决项

1. 当前 R 4.5.3 没有安装 `cmdstanr`、`posterior`、`loo` 和 `jsonlite`。安装这些包前，R 拟合、LOO/WAIC 和 `fit.rds` 读取命令不能在该 R library 中运行。
2. 初次审计发现打包目录缺少 mac2B 候选 draw manifest；已从原项目补入 20261002/20261003/20261004 三份清单。`write_fit_provenance.py --check` 现可独立识别当前 mac2B draw 为 20261003。对应的约 3 GB 探索性 redraw fit 没有复制，它们不是当前 12 个主拟合的一部分。
3. 部分历史 Python/R 注释和终端提示含有已经写入文件的 mojibake。这不影响本次计算，但若作为公开代码附件，仍建议做一次独立的编码清理，并在清理后重跑语法检查。
4. mac2B 使用经收敛筛选的 draw 20261003 与 `adapt_delta=0.99`；这一组与其他三组的抽样过程不同，报告时必须保留该限制。

## 移植完整性补充检查

以原项目 `正式版网络代码` 为参照后，发现并修正了四类路径问题：正式批处理驱动不再回调原项目；驱动使用当前 Python interpreter；反事实与 Stan subject extractor 的默认输入指向移植包内的 `outputs/output_formal/runs`；行为分析的默认输出指向 `analysis/behaviour_analysis`。详细清单见 `docs/PORTABILITY_INVENTORY.md`。

## 临时产物

本次审计的训练、测试、decoder、行为分析、counterfactual 和 Stan 前端产物均写入工作区内独立的 `.audit_tmp_formal_run_001_20261010` 目录。数值比对完成后按用户要求删除，不保留大型 episode-level 中间表。
