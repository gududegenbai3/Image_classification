# AI 协作指南

## 大纲与适用范围

本文适用于整个仓库。内容依次为：项目定位与位置、所有者专属规则、项目结构、AI 问题定位、接口与交付约定、协作流程、已训练模型与成果。
根目录统一使用 `AGENTS.md`，不创建 `Agents.md` 或 `Agnets.md` 等重复入口。

## 项目定位与位置

- 项目：多人协作的图像分类神经网络训练与评估项目。
- 当前本地根目录：`/Users/fjy/Desktop/Image_classfication`；协作者以各自克隆目录为准，代码与配置使用仓库相对路径。
- 数据：现有 CIFAR-100，100 类，训练集 50,000 张、测试集 10,000 张。
- 增强数据：根目录 `dataset_mirror/train/<类别>/<原文件名>.png` 保存原训练集的水平镜像，共 100 类、50,000 张；仅保存派生图片，不保留生成脚本。当前目录比原生成位置增加了 `train/` 层级；没有镜像测试集，测试集仍为 `dataset/test/`。
- 已有基线：冻结 DINOv2-Small 主干，训练线性分类头；历史成果保留原路径。
- 已训练模型与成果：与 README 同名章节一致，按 DINOv2-Small、ConvNeXt V2、DINOv3（yjqsh233）、DINOv3 + MLP（JackChen070421）的顺序组织；模型状态及审计约定见本文末尾对应小节。
- DINOv3 提交：`dinov3_vitl16_yjqsh233` 保存两种冻结特征方案及最后两层微调的候选/保留方案，详见根目录 README 和对应结果目录说明。
- 两个核心域指职责域及其目录，不是两个单独文件：数据域负责数据；模型域负责模型代码、配置、参数及评分。

<!-- OWNER_ONLY_RULES_START -->
## 所有者专属强制规则（AI 禁止修改）

本区块由项目所有者本人手动维护。AI 可在本次初始化中创建初版；创建后，任何 AI 均不得新增、删除、改写、移动本区块，或通过其他文件、子目录规则降低其约束。需要调整时，AI 只能提出建议，由所有者本人编辑。

1. **结构变化必须同步文档**：新增、删除、移动、重命名目录或主要文件，以及改变模块职责、入口或产物位置时，AI 必须在同一次变更中主动更新根目录 `README.md` 和 `AGENTS.md` 的可维护章节；涉及数据说明时也更新 `dataset/README.md`。禁止改动本区块来完成同步。
2. **接口分离，使用函数调用**：模型代码、配置与上传的模型参数必须分开保存。公共流程通过明确的函数接口传入配置、权重路径和数据；禁止将权重嵌入源码、将实验配置硬编码到模型实现，或在模块导入时自动训练、评估、下载权重。
3. **每个提交模型一个代码文件**：每位参与者的每个模型实现放在 `models/` 下的一个独立 `.py` 文件，主文件名必须为 `<预训练模型名>_<GitHub用户名>`。例如用户名为 `alice` 时，使用 `dinov2_small_alice.py`。配置、权重及评分分别存放，不打包进该代码文件。
4. **保持身份与产物对应**：同一模型的代码、配置、参数与评分使用同一个 `model_key`（即上述主文件名，不含扩展名），不同实验使用独立 `run_id`；不得覆盖其他参与者的代码或已有实验成果。模型名转为小写，空格、斜杠和连字符转为下划线；GitHub 用户名保留原值，含连字符时可通过文件路径加载，不能冒用其他用户名。
5. **保护真实数据和结果**：AI 不得擅自删除或重划分原始数据、修改标签、覆盖历史权重及评估记录，或编造 accuracy、交叉熵损失、混淆矩阵。未运行的实验必须明确标注为未运行；新的实验结果写入新的运行目录。

以上规则为仓库内协作规范的最高约束。与具体任务冲突时，AI 应说明冲突并提供不违反规则的方案；规则变更由所有者本人完成。
<!-- OWNER_ONLY_RULES_END -->

## 项目结构（AI 随结构变化维护）

<!-- PROJECT_STRUCTURE_START -->
```text
./
├── .githooks/
│   └── pre-commit
├── .github/
│   └── workflows/
│       └── …
├── .gitignore
├── AGENTS.md
├── README.md
├── checkpoints/
│   ├── convnextv2_base_gududegenbai3/
│   │   └── …
│   ├── dinov2_linear_probe/
│   │   └── …
│   ├── dinov2_small_henry-liyingjie/
│   │   └── …
│   ├── dinov3_vitl16_JackChen070421/
│   │   └── …
│   ├── dinov3_vitl16_henry-liyingjie/
│   │   └── …
│   └── dinov3_vitl16_yjqsh233/
│       └── …
├── configs/
│   ├── convnextv2_base_environment.json
│   ├── convnextv2_base_gududegenbai3.json
│   ├── dinov2_linear_probe.json
│   ├── dinov2_small_gududegenbai3.json
│   ├── dinov2_small_henry-liyingjie.json
│   ├── dinov3_vitl16_JackChen070421.json
│   ├── dinov3_vitl16_henry-liyingjie.json
│   └── dinov3_vitl16_yjqsh233.json
├── dataset/
│   ├── README.md
│   ├── test/
│   │   └── …
│   └── train/
│       └── …
├── dataset_mirror/
│   └── train/
│       └── …
├── models/
│   ├── README.md
│   ├── convnextv2_base_gududegenbai3.py
│   ├── dinov2_small_gududegenbai3.py
│   ├── dinov2_small_henry-liyingjie.py
│   ├── dinov3_vitl16_JackChen070421.py
│   ├── dinov3_vitl16_henry-liyingjie.py
│   └── dinov3_vitl16_yjqsh233.py
├── pipeline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── convnext.py
│   ├── data.py
│   ├── henry_submission.py
│   ├── metrics.py
│   ├── model_api.py
│   └── runner.py
├── requirements.txt
├── results/
│   ├── convnextv2_base_gududegenbai3/
│   │   └── …
│   ├── dinov2_linear_probe/
│   │   └── …
│   ├── dinov2_small_henry-liyingjie/
│   │   └── …
│   ├── dinov3_vitl16_JackChen070421/
│   │   └── …
│   ├── dinov3_vitl16_henry-liyingjie/
│   │   └── …
│   └── dinov3_vitl16_yjqsh233/
│       └── …
├── tests/
│   ├── test_convnext_evidence.py
│   ├── test_convnext_training.py
│   ├── test_dinov3_evidence.py
│   ├── test_dinov3_submission.py
│   ├── test_henry_submission.py
│   └── test_workflow.py
└── tools/
    ├── check_convnext_environment.py
    └── sync_docs.py
```
<!-- PROJECT_STRUCTURE_END -->

目录树按 Git 项目文件生成，不列出被忽略的缓存。ConvNeXt 临时工作目录、下载包解压目录及迁移辅助备份已清理，正式入口保留 `pipeline/convnext.py`。清理时仅删除已核对的重复辅助备份及可再生字节码、系统元数据；不得将 `results/` 中用于哈希审计的 `provenance/`、正式配置或权重视为多余文件删除。

数据域由 `dataset/` 与 `dataset_mirror/` 组成，镜像训练图片的读取根目录为 `dataset_mirror/train/`，不是 `dataset_mirror/`。模型域由 `models/`、`configs/`、`checkpoints/`、`results/` 共同组成。`pipeline/` 是连接两个域的公共流程；`tools/`、`.githooks/`、`.github/`、`tests/` 提供文档同步和验证。保留现有平级目录，避免破坏已有成果引用。

逐文件职责见 `README.md` 的“各文件的作用”章节；运行依赖由根目录 `requirements.txt` 声明，修改直接使用的第三方依赖时同步维护该清单。

## AI 面对问题时的位置与处理方式

- 数据读取、类别或预处理问题：先看 `pipeline/data.py`、`dataset/README.md`、`dataset/train/`、`dataset_mirror/train/`、`dataset/test/` 及相应配置；核对类别映射，保持训练与测试一致。
- 模型结构、分类头或前向推理问题：在 `models/<model_key>.py` 定位；不得把数据清洗与运行配置塞进模型文件。
- 超参数、预训练版本、冻结策略、设备与随机种子问题：在 `configs/` 定位；运行时通过函数参数传入。
- 参数加载、恢复训练问题：在对应 `checkpoints/<model_key>/<run_id>/` 定位，核对模型配置和类别映射。
- accuracy、交叉熵损失、混淆矩阵或模型对比问题：在对应 `results/<model_key>/<run_id>/` 定位，先核对数据划分、样本数和指标定义。
- 结构、入口与协作约定问题：读取本文件及 `README.md`；只维护非保护区。模型实验统一查阅 README 的“已训练模型与成果”及本文件对应小节，不将章节移动或合并视为模型文件、入口或实验结果被删除。
- 缺少 GitHub 用户名时：可先完成公共说明与接口工作；提交个人模型前向提交者确认，禁止从本地用户名推断身份。

## 接口分离与 function calling 约定

通用入口为 `python -m pipeline`，通过 `pipeline/model_api.py` 调用下列函数；ConvNeXt 使用 `python -m pipeline.convnext` 编排并调用相同模型函数接口。这里的 function calling 指程序通过明确参数调用函数。

```python
build_model(config) -> model
load_weights(model, checkpoint_path, *, device) -> model
predict(model, inputs) -> logits
```

- 每个模型文件提供上述三个函数；`config` 为调用方读取 JSON 后传入的配置字典，`checkpoint_path` 为外部传入的权重路径。
- `build_model` 只负责构建网络；预训练参数由调用方准备，再经 `load_weights` 加载。分类头单独保存时，配置与加载逻辑必须说明所需主干及版本。
- `inputs` 为已按配置预处理的批量图像张量，`logits` 为形状 `[batch_size, num_classes]` 的未归一化分类分数，列顺序必须与类别映射一致。
- 数据读取、训练调度与指标计算分别由 `pipeline/data.py`、`pipeline/runner.py`、`pipeline/metrics.py` 负责。调用方会把预训练目录解析为绝对路径后传给模型，但配置快照保留仓库相对路径。
- 配置至少记录：`model_key`、`run_id`、预训练来源及版本、数据路径、类别映射、预处理、随机种子、训练超参数、权重与输出路径。
- 权重文件只存参数与必要的恢复状态，不以完整模型对象替代独立源码及配置。

## 模型评分与交付

新提交的路径约定（占位符须替换为真实值）：

```text
models/<model_key>.py
configs/<model_key>.json
checkpoints/<model_key>/<run_id>/final.pt
results/<model_key>/<run_id>/config_snapshot.json
results/<model_key>/<run_id>/class_to_idx.json
results/<model_key>/<run_id>/metrics.json
results/<model_key>/<run_id>/history.json
```

- `metrics.json` 必须包含 `model_key`、`run_id`、`split`、`samples`、`accuracy`、`cross_entropy_loss`、`confusion_matrix`。accuracy 使用 `[0, 1]` 比例；交叉熵为全体评估样本的平均值。
- 混淆矩阵为 `num_classes × num_classes` 的原始整数计数，行是真实类别，列是预测类别，顺序按 `class_to_idx.json` 的索引排列；可另外输出图片，但不能只提交图片。
- 混淆矩阵总和必须等于 `samples`，对角线之和除以 `samples` 应与 accuracy 一致。评估使用固定权重和确定的预处理，并关闭训练行为及梯度计算。
- `config_snapshot.json` 保存该次运行实际使用的配置，`history.json` 保存逐轮训练记录；`environment.json` 记录代码版本、模型源码哈希、输入权重与依赖环境，`status.json` 区分运行中、失败与完成状态。
- 当前基线沿用历史字段 `top1_accuracy`、`loss`，分别对应新约定的 `accuracy`、`cross_entropy_loss`；不为统一命名回写历史结果。混淆矩阵已包含在其 `metrics.json` 中。

## AI 协作流程与当前状态

1. 动手前读取本文件、`README.md` 和当前任务相关文件，检查已有改动；不要覆盖协作者工作。
2. 依照问题定位到数据域或模型域，按命名和接口约定修改；不为文档任务擅自启动训练。
3. 根据改动进行必要验证；结构变化时同步两份根目录文档的可维护内容，再检查路径与说明一致。
4. 交付时说明改动、验证情况与尚未实现的部分，不将接口约定描述成可运行功能。

当前已接入 DINOv2-Small、ConvNeXt V2 及两位贡献者的 DINOv3 模型，均在 `models/` 提供独立代码文件，并使用 `configs/` 下同名 JSON 配置。DINO 系列使用以下通用命令；ConvNeXt 的独立编排命令见本文“ConvNeXt V2：gududegenbai3”小节：

```sh
python -m pipeline validate --config configs/dinov2_small_gududegenbai3.json
python -m pipeline prepare --config configs/dinov2_small_gududegenbai3.json
python -m pipeline train --config configs/dinov2_small_gududegenbai3.json --run-id run01
python -m pipeline evaluate --config configs/dinov2_small_gududegenbai3.json --run-id eval01 --checkpoint checkpoints/dinov2_small_gududegenbai3/run01/final.pt
python -m pipeline evaluate --config configs/dinov3_vitl16_JackChen070421.json --run-id eval01 --checkpoint checkpoints/dinov3_vitl16_JackChen070421/run01/final.pt
```

- `validate` 不导入模型、不依赖 PyTorch，检查配置、路径、类别映射和接口声明；实际函数签名在运行时检查。
- `prepare` 是唯一主动联网下载预训练参数的命令，固定版本保存在 `.cache/`；`build_model`、训练和评估均只读取本地文件。
- `train` 使用 AdamW 与交叉熵，冻结主干时只更新分类头；保存完整 `state_dict`，再加载保存的权重完成一次测试集评估。不会缓存全量特征，计算成本与历史缓存方案不同。
- `evaluate` 必须显式传入新格式权重，核对模型身份、配置及类别映射。已有运行目录拒绝覆盖。`train --checkpoint` 表示从权重继续优化，优化器重新建立，不是完整断点恢复。
- 所有运行都保存配置快照、映射、环境与状态；失败目录保留用于排查，下次使用新 `run_id`。
- 数据加载以配置中的 `data.train_dir` 为准；镜像集单独读取时指向 `dataset_mirror/train`。公共流程不自动联合两组训练集，也不解释 `data.augmentation` 等历史说明字段。JackChen070421 的配置及实验记录可能保留整理前路径描述，应区分历史来源说明与当前实际目录；禁止为整理结构而回写历史快照或评分。
- 历史 `dinov2_linear_probe` 权重与快照保持原样，不保证兼容新的权重封装。根目录旧配置的数据路径已修正，新流程使用同名规范化配置；旧配置不作为公共命令入口。

结构文档由 `python3 tools/sync_docs.py` 同步生成标记内的目录树；其他职责及接口说明仍由 AI 随变更维护。`--check` 只校验，`--staged` 以 Git 暂存区为准。
`.githooks/pre-commit` 自动同步暂存结构，若文档出现未暂存变更则阻止提交，要求先检查并暂存；新克隆需执行 `git config core.hooksPath .githooks`。CI 校验目录树、测试与语法。
同步器只修改 `PROJECT_STRUCTURE` 标记内的内容，并核对所有者专属区原文不变。

## 已训练模型与成果

本节与 README 的“已训练模型与成果”保持模型顺序和状态一致。README 集中展示实验说明，AI 在此核对入口、产物和证据边界；逐次实验以对应 results/ 下的原始记录为准。章节重新组织不改变已有代码、模型身份或运行目录。

### DINOv2-Small + 线性分类头

冻结 Meta DINOv2-Small 主干，仅训练 `Linear(384, 100)`，原始 50,000 张训练图训练 10 轮。原始 test 10,000 张 Top-1 为 84.96%、Top-5 为 97.34%、平均交叉熵约 0.5173；最后一轮训练准确率 93.316% 不能作为测试分数。

历史分类头为 `checkpoints/dinov2_linear_probe/formal/run01/final.pt`，配置、训练记录和评分为 `results/dinov2_linear_probe/formal/run01/`。历史字段 `top1_accuracy`、`loss` 及原路径保持不变，不为命名统一回写成果，也不从本地用户名推断历史贡献者身份。预测仍需同版本官方主干；历史参数封装不保证兼容新公共入口。

### ConvNeXt V2：gududegenbai3

ConvNeXt V2 原临时目录 `convenext/` 已完成职责归位：模型单文件为 `models/convnextv2_base_gududegenbai3.py`，训练配置为 `configs/convnextv2_base_gududegenbai3.json`，缓存特征编排为 `pipeline/convnext.py`，入口为 `python -m pipeline.convnext`（validate/prepare/train）。环境检查为 `tools/check_convnext_environment.py`，环境目标配置为 `configs/convnextv2_base_environment.json`，直接依赖统一由根目录 `requirements.txt` 声明。不再使用或重新创建临时目录。

云端使用 AutoDL 镜像自带 Python 3.12，不创建或激活额外环境。RTX PRO 6000 Blackwell 使用支持其架构的 CUDA 构建；已完成运行使用 PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、timm 1.0.22。prepare 显式下载并固定 Hugging Face commit SHA 与文件 SHA256，缓存位置继续为 `.cache/<model_key>/`；train 只读取本地已准备权重。目录迁移不改变既有缓存布局。

ConvNeXt 流程缓存原图/镜像的固定特征，按原图身份分层划分：45,000 原图及对应镜像共 90,000 张用于选择训练，5,000 原图用于验证，其镜像不参与选择训练。验证选定轮数后重新初始化头，在全部 100,000 张特征上拟合，再重新加载保存的分类头，对原始 10,000 张测试图评分一次。不得把镜像与原图随机拆分到两侧。只缓存特征，不改动真实图片或物理划分。失败运行保留状态，重试使用新 run_id，不自动覆盖或续训。当前没有关闭镜像的配置或命令参数，无镜像对照尚未实现或运行。

此配置格式供 `python -m pipeline.convnext` 使用，不直接传入通用 `python -m pipeline`。权重只保存分类头及模型身份/配置/类别映射，依赖固定外部主干。输出遵守根目录 checkpoints/results 命名；快照记录解析后的版本及预处理。特征缓存位于 `.cache/features/<model_key>/<run_id>/`，不提交 Git。

已从用户下载包归档真实运行 `convnext_20261025_143320`：模型与结果分别移入 `checkpoints/convnextv2_base_gududegenbai3/` 和 `results/convnextv2_base_gududegenbai3/`，保留原 run_id 及全部原始文件字节。验证选择 5 轮，原始 test 10,000 张 accuracy 为 0.8844、平均交叉熵为 0.38219886627197264。`provenance/` 保存原源码文本、基础配置、预训练来源及逐文件迁移哈希；不得为了反映新入口而回写历史快照、环境哈希或评分。说明见 `results/convnextv2_base_gududegenbai3/README.md`；真实证据由 `tests/test_convnext_evidence.py` 校验。此次整理未重新执行真实训练或评估。

从项目根目录运行：

```sh
python -m pipeline.convnext validate
python -m pipeline.convnext prepare
python tools/check_convnext_environment.py
python -m pipeline.convnext train --run-id mirror_run02
```

`pipeline/convnext.py` 负责模型之外的数据检查、特征提取、训练调度、指标和运行记录；模型结构及参数加载仍通过模型单文件函数调用。`python -m pipeline` 的 DINO 接口和原有配置保持兼容，不将 ConvNeXt 配置误传给该通用入口。

### DINOv3 ViT-L/16：yjqsh233

贡献者已确认 GitHub 用户名为 `yjqsh233`。模型、基础配置、参数和评分均使用 `dinov3_vitl16_yjqsh233`；四个独立运行目录对应三组实验，禁止将微调第 3 轮的验证集评分写成测试集评分。

- 模型适配器仅在显式 `load_weights` 时加载参数。紧凑 `final.pt` 依赖配置指定版本及 SHA256 的 Meta 预训练主干；冻结方案只含分类头与特征归一化统计，微调候选另含最后两层参数。
- `results/dinov3_vitl16_yjqsh233/README.md` 解释选择、重拟合、预处理及历史复评过程；`dataset_identity.json` 记录全部 60,000 张 PNG 与原实验图片的字节核对。
- 每次运行提供固定权重评分、逐样本预测、原始整数混淆矩阵、配置、类别映射、训练历史、划分、环境与状态。`provenance/` 保留原实验源码文本及其哈希，供审计；模型加载仍只有 `models/` 下的单文件入口。
- 公共流程使用 HF 预处理和 AdamW，历史实验使用 PIL resize、缓存特征 SGD 或分组 AdamW；不要把公共 `train` 命令称为历史训练的精确复现。历史评分的评估路径以每个 `metrics.json` 为准。
- `tests/test_dinov3_submission.py` 检查微型模型接口；`tests/test_dinov3_evidence.py` 核对真实评分、混淆矩阵、逐样本预测及参数哈希。DINOv3 许可位于 `checkpoints/dinov3_vitl16_yjqsh233/LICENSE.md`。

### DINOv3 ViT-L/16 + MLP 分类头（JackChen070421）

README 将该模型分为 v1 无增强和 v2 水平镜像两组实验。模型为 `models/dinov3_vitl16_JackChen070421.py`，配置为 `configs/dinov3_vitl16_JackChen070421.json`，冻结主干使用固定 revision，分类头采用 `2048 → 512 → 100` MLP。运行权重仅保存分类头，推理仍依赖同版本外部主干。

- v1：参数为 `checkpoints/dinov3_vitl16_JackChen070421/run01/final.pt`，公共复评为 `results/dinov3_vitl16_JackChen070421/eval01/`，测试 Top-1 为 93.24%，平均交叉熵约 0.3465。
- v2：参数为 `checkpoints/dinov3_vitl16_JackChen070421/run02_aug/final.pt`，公共复评为 `results/dinov3_vitl16_JackChen070421/eval02_aug/`，测试 Top-1 为 93.53%，平均交叉熵约 0.3936。

公共评分使用原始 test 10,000 张、单次前向且无 TTA；README 中本地 TTA 分数单独说明，不替换公共评分。v2 的原图与镜像可能跨训练/验证两侧，验证准确率 99.27% 存在同图泄漏，不能作为无泄漏验证结果；不回写历史划分或指标，也不把该划分用于新的增强实验。

历史快照可能记录整理前的镜像路径，当前读取根目录为 `dataset_mirror/train`。通用入口不会解释历史 `data.augmentation` 字段或自动合并两组数据。复评必须使用新 run_id，已有 eval01、eval02_aug 目录保持原样。

### DINOv2 / DINOv3 微调：henry-liyingjie

本次提交两个模型、各一个最佳检查点。原训练在 RTX 5070 Laptop GPU 上完成，按类别使用 seed 42 将原始训练集划为 45,000 / 5,000，随机裁剪、水平翻转及颜色扰动只用于训练，各轮按验证准确率选择权重；本次整理和检查不重新训练模型。

- DINOv2-Small：最后六层微调，last6_ls01_epoch16；原始 test 10,000 张 Top-1 为 91.24%，原记录交叉熵为 0.48103854219913483。该实验未保存混淆矩阵及逐样本预测，status.json 保持 archived，confusion_matrix 保持 null。经仓库所有者确认，本次作为证据不完整的历史评分归档接收，不追加复评，也不将其视为已满足完整评分证据要求。
- DINOv3 ViT-L/16：最后一层微调，last1_ls005_epoch18；原始 test 10,000 张 Top-1 为 93.66%，FP32、无 TTA、普通交叉熵为 0.2868556800834438。已提交 100×100 原始整数混淆矩阵，计数总和为 10,000、对角线为 9,366。历史训练 test_loss 使用 0.05 标签平滑，不与此次普通交叉熵混用。

本 PR 未提交 last2_ls0_epoch10 的参数或评分目录，不将该次实验列为本次交付成果。本次也未提交逐样本预测文件或原训练源码 provenance，不能声称这些材料已经随包交付；已有指标、类别映射、训练历史、实际配置、环境及状态记录均保持原始内容。此次接受 DINOv2 的归档例外不改变其他新模型的完整评分交付要求。

模型实现为 models/dinov2_small_henry-liyingjie.py 和 models/dinov3_vitl16_henry-liyingjie.py，均提供 build_model/load_weights/predict；同名 configs 保存默认实验，各运行的 config_snapshot.json 保存实际配置。pipeline/henry_submission.py 是历史 PIL 直接缩放的固定权重复评入口，这些配置不传入通用 python -m pipeline，也不声称通用训练可精确重现历史过程。

final.pt 是元数据与分片清单，需连同其目录内全部 weights-*.pt 使用。各分片是同一个检查点的原始 FP32 参数，不是逐轮权重；加载逐片核对 SHA256。DINOv2 保存完整模型参数，无外部主干依赖；DINOv3 只保存实际训练过的末层与分类头，依赖配置所列固定 revision 和 SHA256 的预训练主干。build_model 不联网或加载权重。

原参数及结果位于：
- checkpoints/dinov2_small_henry-liyingjie/last6_ls01_epoch16/ 与 results/dinov2_small_henry-liyingjie/last6_ls01_epoch16/。
- checkpoints/dinov3_vitl16_henry-liyingjie/last1_ls005_epoch18/ 与 results/dinov3_vitl16_henry-liyingjie/last1_ls005_epoch18/。

后续仅在有明确要求时对新输出目录复评；输出目录必须不存在，默认 test 路径为 dataset/test。DINOv3 还需自行准备获授权的固定主干，并通过 --pretrained-dir 传入；DINOv2 不需要该参数。当前不启动这两者的复评。

tests/test_henry_submission.py 核对已提交权重的分片清单、文件哈希、配置与类别映射、DINOv3 混淆矩阵、DINOv2 证据不完整状态，以及微型离线模型的冻结行为、完整/部分参数重载和损坏分片拒绝。微型测试不读取真实图片或产生正式分数。原训练使用 timm 1.0.30、Transformers 4.57.6，实际复评环境见 environment.json。
