# formal_run_001 移植完整性清单

## 环境清单

- `requirements.txt` 列出网络训练、测试和 Python 分析所需的运行依赖。
- 建议使用 Python 3.11 或 3.12；2026-10-10 实际核验环境的精确版本记录在顶层 README 中。
- R/CmdStan 依赖单独记录在 `analysis/fitting_NN/README.md`，因为这些依赖不能通过 pip 安装。

参照源：原项目的 `正式版网络代码`。检查日期：2026-10-10。

## 主链结论

`formal_run_001` 已包含独立重跑主训练、主冻结测试、行为分析、belief decoder、M1/M2/M3 主拟合和 actor 反事实学习率所需的项目内代码与正式输入。外部仍需要 Python/R packages、CmdStan 和可写的 ASCII runtime 目录。

## 与源项目的对照

| 部分 | 对照结果 |
|---|---|
| `formal_osla` 核心 8 文件 | 与原项目 SHA-256 逐文件一致 |
| `analysis/{loading,belief_decoder,replay_gate}.py` | 与原项目一致 |
| M1/M2/M3/M3-multi Stan | 4 个文件与原项目一致 |
| 主 R 拟合/诊断脚本 | 与原项目一致；`fit_multi_episode.R` 仅修正默认 frozen-test 路径 |
| 两个 Stan 绘图脚本 | 仅将原项目的扁平 `prerun_style` import 改为移植包的 `scripts.shared` import |
| `scripts/` | 从原项目扁平结构重组为 training/testing/analysis/plotting/shared；主入口已通过 import 和 CLI 检查 |
| 正式产物 | 30 checkpoints、30 histories、120 conflict cells、540 consistent calibration cells |
| Stan fits | 12 个主 `fit.rds` 及其 summary 存在，36 项 provenance hash 匹配 |

## 审计后修正的移植路径

1. `outputs/output_formal/scripts/{train_batch,run_tests}.py` 改为调用包内脚本，不再指向原项目。
2. 两个驱动使用 `sys.executable`，不再假设固定的兄弟 `.venv`。
3. `counterfactual_actor_pbet_lr.py` 的默认 runs 改为包内 `outputs/output_formal/runs`。
4. `extract_m3_subjects.py` 的默认 runs 改为包内正式 runs。
5. 六个行为/decoder 脚本的默认 analysis root 改为 `analysis/behaviour_analysis`。
6. `fit_multi_episode.R` 的默认 frozen root 改为包内正式 frozen tests。
7. 补入 20261002/20261003/20261004 三份候选 sampling manifest；provenance check 现可识别 mac2B draw 20261003。

## 有意不列入主链的内容

- 原项目 `formal_osla/analysis/` 的旧认知拟合工具在移植包 `_archive/analysis/formal_osla_analysis/` 中保留，不是当前 actor 主分析的依赖。
- 原项目中已被取代的 hidden-state 学习率估计、一次性诊断和 ablation 脚本在 `_archive/` 中保留或未列入主包。
- `episode_sampling_variability.py`、`plot_mac2b_diagnosis.py`、`run_redraw_fits.py` 和 `report_redraw_fits.py` 是 mac2B 收敛问题的历史探索工具，其文档前提仍是“mac2B 未收敛”的旧状态。当前主 fit 已替换为 draw 20261003 @ `adapt_delta=0.99`，因此未将这四个旧探索脚本作为主链复现入口。
- 约 3 GB 的 redraw exploratory fits、约 2 GB 的 high-adapt 中间 fit 和 multi-episode 试验产物没有复制。它们不是当前 12 个主 fit；对应的主结论、当前 fit 和 provenance 已包含。

## 外部依赖，不是移植缺文件

- Python virtual environment 或等价 dependencies；
- R library：`cmdstanr`、`posterior`、`loo`、`jsonlite`；
- CmdStan 2.38 和 C++ toolchain；
- `M31_RUNTIME_ROOT` 指向的可写 ASCII 路径。
