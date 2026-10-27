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
当前也已接入 `dinov3_vitl16_JackChen070421.py`，配置为 `configs/dinov3_vitl16_JackChen070421.json`；其 `final.pt` 仅保存分类头，冻结主干按配置中的固定 revision 从预训练缓存恢复。
模型通过本地配置构建，预训练权重由公共 `prepare` 命令单独准备，`load_weights` 负责加载本地预训练目录或新格式运行权重。
冻结主干时，即使训练流程调用 `model.train()`，主干仍保持 eval 模式；`predict` 返回 logits，梯度及训练/评估上下文由公共流程控制。
公共流程将配置中的预训练路径解析为绝对路径后传入模型；配置快照保留仓库相对路径。
新格式权重包含完整 `model_state_dict`、模型身份、类别映射及配置，公共评估先校验再调用模型加载接口。
原 `dinov2_linear_probe` 的历史成果保持原路径与内容，不自动迁移或改写。

## DINOv3 ViT-L/16：yjqsh233

`dinov3_vitl16_yjqsh233.py` 实现同一组函数接口。`build_model` 从本地架构配置构建网络；`load_weights` 验证预训练权重 SHA256，加载外部主干及紧凑运行参数；`predict` 返回 FP32 分类头 logits。

`n_last_blocks=1` 拼接最终 CLS 与排除 register tokens 后的 patch 均值；设为 4 时拼接最后四层经过最终 LayerNorm 的 CLS，再拼接末层 patch 均值。`trainable_last_blocks` 控制可训练尾层，冻结部分及 RoPE 坐标增强保持 eval；配置同时决定 CUDA FP16 autocast 和历史 FP16 特征缓存的舍入行为。

请按某次运行的 `results/dinov3_vitl16_yjqsh233/<run_id>/config_snapshot.json` 构建模型，先准备其中固定 revision 的本地预训练文件，再显式加载对应 `final.pt`。分类头、特征统计与实际训练的尾层均在参数文件中，不能只加载分类头来代替微调候选。

提交说明见 [实验记录](../results/dinov3_vitl16_yjqsh233/README.md)。公共 `train` 的优化器及预处理与历史实验不同，基础配置中的历史字段用于记录，不会改变公共训练器的行为。


## ConvNeXt V2：gududegenbai3

`convnextv2_base_gududegenbai3.py` 提供标准 build_model/load_weights/predict 接口。构建时不加载权重，显式加载本地主干目录时校验 SHA256 并移除原 ImageNet 分类层，保持特征主干冻结。运行权重仅保存新的线性分类头，重载时要求模型身份、配置、类别映射及主干 SHA 一致。
缓存特征训练由 `pipeline/convnext.py` 组织，入口为 `python -m pipeline.convnext`，配置为根目录同名 JSON，操作步骤见 [训练与实验记录](../results/convnextv2_base_gududegenbai3/README.md)。已归档 AutoDL 运行 `convnext_20261025_143320`，原图与镜像训练、原始 test 10,000 张 accuracy 88.44%。历史主干依赖、配置及评分保持原样，原训练源码文本位于该运行的 provenance/。

## henry-liyingjie

本次提交 DINOv2 最后六层微调和 DINOv3 最后一层微调两个检查点，接口及分片加载方式见根 README 对应章节。固定权重复评入口为 python -m pipeline.henry_submission，配置不能传给通用训练入口。final.pt 必须配合目录内全部 weights-*.pt 使用。

DINOv2 的历史 test Top-1 为 91.24%，混淆矩阵与逐样本预测缺失；经仓库所有者确认作为证据不完整的 archived 记录接收，不追加复评。DINOv3 test Top-1 为 93.66%，普通交叉熵为 0.2868556800834438，已提交混淆矩阵；逐样本预测及原训练源码 provenance 未随本次提交。

新增 tests/test_henry_submission.py 覆盖真实记录与分片清单一致性及离线微型模型加载检查，不执行正式模型复评或训练。
