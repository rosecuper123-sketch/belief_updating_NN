# Python 命令入口

从 `formal_run_001` 根目录运行 `python -m scripts.<子目录>.<模块>`。各子目录负责一段明确的流程：

| 子目录 | 用途 |
|---|---|
| [training](training/README.md) | 单网络或多 seed 训练，生成 checkpoint 与训练历史 |
| [testing](testing/README.md) | 自检、冻结测试与测试产物检查 |
| [analysis](analysis/README.md) | 正式训练和行为分析、隐状态解码 |
| [plotting](plotting/README.md) | 预测试图表和先验×真实概率正交测试 |
| [shared](shared/README.md) | 绘图样式等共用函数 |

现有正式数据在 `outputs/runs/`；预测试 checkpoint、训练历史、汇总表和图像在 `outputs/outputs_prerun/`。运行命令时显式传入输入和输出根目录，便于将新结果与提供的结果分开。预测试复现步骤见[绘图与正交测试指南](plotting/README.md)，完整项目入口见[总指南](../README.md)。
