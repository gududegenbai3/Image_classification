# 模型接入说明

本目录存放参与者提交的模型代码。模型域还包括根目录的 `configs/`、`checkpoints/` 和 `results/`。
完整协作规范见 [AGENTS.md](../AGENTS.md)，项目介绍见 [README.md](../README.md)。

## 命名与文件边界

每个提交模型的实现放在一个独立文件中：`<预训练模型名>_<GitHub用户名>.py`。
例如 GitHub 用户 `alice` 提交 DINOv2-Small，代码文件为 `dinov2_small_alice.py`，对应 `model_key` 为 `dinov2_small_alice`。
预训练模型名转为小写，空格、斜杠和连字符替换为下划线；GitHub 用户名保留原值，含连字符的文件通过文件路径加载。
示例用户名不代表已有参与者；不知道提交者用户名时先确认身份。

同一模型的配置放在 `configs/<model_key>.json`，权重放在 `checkpoints/<model_key>/<run_id>/`，评分放在 `results/<model_key>/<run_id>/`。
不同实验使用不同 `run_id`，已有运行目录不得覆盖。

## 函数接口

模型文件须提供 `build_model(config)`、`load_weights(model, checkpoint_path, *, device)`、`predict(model, inputs)`。
分别返回构建好的模型、已加载参数的模型和 `[batch_size, num_classes]` 的 logits。
配置由外部读取后传入，权重通过路径传入，输入为调用方已预处理的批量图像；导入模块时不得自动训练、评估或下载权重。
数据、训练调度与统一评分由公共流程负责，详细契约见根目录 `AGENTS.md`。

## 交付内容

除单文件模型代码、独立 JSON 配置和权重外，每次训练提交配置快照、类别映射、逐轮记录及包含 accuracy、平均交叉熵损失和原始混淆矩阵的评估 JSON。
记录预训练来源与版本、代码版本、依赖和运行环境；不得用示例分数充当实验结果。

当前已接入 `dinov2_small_gududegenbai3.py`，配置为 `configs/dinov2_small_gududegenbai3.json`，公共入口为 `python -m pipeline`。
模型通过本地配置构建，预训练权重由公共 `prepare` 命令单独准备，`load_weights` 负责加载本地预训练目录或新格式运行权重。
冻结主干时，即使训练流程调用 `model.train()`，主干仍保持 eval 模式；`predict` 返回 logits，梯度及训练/评估上下文由公共流程控制。
公共流程将配置中的预训练路径解析为绝对路径后传入模型；配置快照保留仓库相对路径。
新格式权重包含完整 `model_state_dict`、模型身份、类别映射及配置，公共评估先校验再调用模型加载接口。
原 `dinov2_linear_probe` 的历史成果保持原路径与内容，不自动迁移或改写。
