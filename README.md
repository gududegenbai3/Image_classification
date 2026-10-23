# 图像分类与预训练模型迁移学习

## 项目目的

本项目由多人协作，使用公开预训练视觉模型提取图像特征，通过训练新的分类头完成 CIFAR-100 图像分类，并在此基础上逐步探索更先进的模型。

目前包含冻结主干的 DINOv2、DINOv3 线性分类实验，以及 DINOv3 最后两层微调的验证记录。不同实验的训练和选择过程分别保存。

## 协作入口与项目位置

本项目是多人协作的图像分类神经网络训练项目，按**数据域**和**模型域**划分职责。
AI 开始工作前必须读取 [AGENTS.md](AGENTS.md)；其中“所有者专属强制规则”初版建立后仅由项目所有者本人手动维护，AI 禁止修改。
当前本地目录为 `/Users/fjy/Desktop/Image_classfication`；其他协作者以自己的克隆目录为准，程序与配置使用仓库相对路径。

## 项目结构

数据域负责数据来源、训练/测试图片及类别一致性；模型域负责代码、配置、参数和评分。两个域指职责划分，目录组织如下：

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
│   ├── dinov2_linear_probe/
│   │   └── …
│   └── dinov3_vitl16_yjqsh233/
│       └── …
├── configs/
│   ├── dinov2_linear_probe.json
│   ├── dinov2_small_gududegenbai3.json
│   └── dinov3_vitl16_yjqsh233.json
├── dataset/
│   ├── README.md
│   ├── test/
│   │   └── …
│   └── train/
│       └── …
├── models/
│   ├── README.md
│   ├── dinov2_small_gududegenbai3.py
│   └── dinov3_vitl16_yjqsh233.py
├── pipeline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model_api.py
│   └── runner.py
├── requirements.txt
├── results/
│   ├── dinov2_linear_probe/
│   │   └── …
│   └── dinov3_vitl16_yjqsh233/
│       └── …
├── tests/
│   ├── test_dinov3_evidence.py
│   ├── test_dinov3_submission.py
│   └── test_workflow.py
└── tools/
    └── sync_docs.py
```
<!-- PROJECT_STRUCTURE_END -->

模型域保留现有平级目录，不搬动历史权重和结果。模型接入详见 [models/README.md](models/README.md)。

## 各文件的作用

以下路径均相对于项目根目录。源码和配置逐文件说明；大量图片和每次运行产生的同类文件按路径模式说明，尖括号表示需要替换的名称。

### 根目录与协作说明

- `README.md`：面向团队成员的项目入口，说明项目目标、文件职责、运行命令、协作方式和已有成果。
- `AGENTS.md`：面向 AI 的操作规范，规定问题定位、命名、接口、评分及文档同步方式；其中所有者专属规则区只能由项目所有者本人手动修改。
- `requirements.txt`：Python 运行依赖清单。PyTorch 负责训练与推理，Transformers 提供 DINOv2、DINOv3 和图像预处理，Hugging Face Hub 负责显式下载预训练文件，Pillow 读取图片，safetensors 加载预训练张量参数。通过 `python -m pip install -r requirements.txt` 安装；这是版本范围约束，不是完整依赖锁文件。
- `.gitignore`：忽略虚拟环境、预训练缓存、Python 字节码和系统生成文件，避免把这些本地产物新增到版本管理中。

### 数据与模型文件

- `dataset/README.md`：记录数据来源、引用信息、图片目录形式及本项目的数据读取约定。
- `dataset/train/<类别>/<图片文件>`：训练图片，类别目录名作为标签来源；训练时读取，不修改原始文件。
- `dataset/test/<类别>/<图片文件>`：测试图片，使用与训练集一致的类别索引，供统一评估使用。
- `models/README.md`：说明参与者如何命名和提交模型、实现哪些函数、分别保存哪些产物。
- `models/dinov2_small_gududegenbai3.py`：该参与者的单文件模型实现，组合 DINOv2 主干和线性分类头，提供构建模型、加载参数和输出 logits 的函数；配置与权重从外部传入。
- `configs/dinov2_small_gududegenbai3.json`：新公共流程使用的配置，记录模型身份、运行编号、数据路径、预训练版本、本地缓存位置、冻结策略、训练超参数和输出目录。
- `configs/dinov2_linear_probe.json`：历史基线的参考配置，保留旧实验的工作流和参数格式；数据路径已修正，但不直接作为新公共命令的配置入口。
- `models/dinov3_vitl16_yjqsh233.py`：DINOv3 ViT-L/16 单文件适配器，支持末层 CLS 与 patch 均值、最后四层 CLS 与 patch 均值，以及最后两层微调；构建和权重加载分离。
- `configs/dinov3_vitl16_yjqsh233.json`：DINOv3 基础配置；各已提交实验以各自 `config_snapshot.json` 为准，公共 AdamW 流程不会自动重现历史 SGD 缓存训练。

### 公共训练与评估代码

- `pipeline/__init__.py`：标识公共流程的 Python 包，供其他模块导入；自身不启动任务。
- `pipeline/__main__.py`：命令行入口，使 `python -m pipeline` 可用；解析 `validate`、`prepare`、`train`、`evaluate` 及其参数并分派任务。
- `pipeline/config.py`：读取并校验 JSON 配置，检查模型命名、用户名、运行编号、超参数、固定预训练版本和输出路径，并将仓库相对路径安全解析为本地路径。
- `pipeline/data.py`：扫描类别目录，生成和校验类别索引，读取图片并转为 RGB，再交给图像预处理器生成模型输入。
- `pipeline/model_api.py`：根据文件路径加载参与者的模型代码，检查 `build_model`、`load_weights`、`predict` 的函数签名；支持 GitHub 用户名含连字符的文件名。
- `pipeline/metrics.py`：累计评估样本数、交叉熵总和及混淆矩阵，计算准确率和按样本数加权的平均损失。
- `pipeline/runner.py`：组织数据加载、设备选择、随机种子、公共训练循环、权重保存与重新加载、统一评估及运行记录；同时校验权重身份和类别映射，并拒绝覆盖已有运行目录。

### 文档同步与自动检查

- `tools/sync_docs.py`：根据工作目录或 Git 暂存区生成项目目录树，仅更新 README 和 AGENTS 的结构标记区；支持只检查是否同步，并保护所有者专属规则区原文。
- `.githooks/pre-commit`：本地 Git 提交检查，自动按暂存区同步目录树；如果文档存在未暂存修改，则阻止提交并提示先检查、暂存。需设置 `core.hooksPath` 才会启用。
- `.github/workflows/checks.yml`：GitHub CI 配置，在推送或拉取请求时检查目录文档、测试与 Python 语法，并在安装依赖的任务中运行离线微型 DINOv2 集成测试。
- `tests/test_workflow.py`：验证指标计算、模型身份与接口、路径限制、类别映射、防覆盖、文档同步，以及微型网络的训练、权重保存和复评流程。
- `tests/test_dinov3_submission.py`：以本地微型 DINOv3 检查特征拼接、紧凑权重加载和冻结行为。
- `tests/test_dinov3_evidence.py`：检查提交的类别映射、逐样本预测、混淆矩阵、评分和权重哈希是否一致。

### 新运行产生的文件

下列文件由新公共流程按 `model_key` 和 `run_id` 隔离保存；文档列出它们的用途，不表示已经完成了新的全量训练。

- `checkpoints/<model_key>/<run_id>/final.pt`：公共训练保存完整 `state_dict`；本次 DINOv3 历史提交保存分类头、归一化统计及实际训练的主干层，加载时需要固定版本的外部预训练主干。两种格式均记录身份、类别映射和配置，不包含优化器状态。
- `results/<model_key>/<run_id>/config_snapshot.json`：这次运行实际使用的配置快照，包括命令行覆盖后的运行编号。
- `results/<model_key>/<run_id>/class_to_idx.json`：类别名称到整数索引的映射，用于解释输出列和混淆矩阵，并校验权重与数据是否匹配。
- `results/<model_key>/<run_id>/history.json`：逐轮训练准确率、损失和耗时；单独评估时为空列表。
- `results/<model_key>/<run_id>/metrics.json`：相应 `split` 的评分，包括 accuracy、平均交叉熵、原始混淆矩阵、样本数及被评估权重的路径和哈希；验证集评分不能当作测试集评分。
- `results/<model_key>/<run_id>/environment.json`：Python、系统、设备、依赖版本、Git 版本状态、模型源码哈希及输入权重来源等复现信息。
- `results/<model_key>/<run_id>/status.json`：运行状态，区分运行中、完成和失败；发生异常时记录错误信息。

### 已有基线的历史成果文件

以下文件保留原内容和原路径，独立于新运行格式：

- `checkpoints/dinov2_linear_probe/formal/run01/final.pt`：历史 DINOv2 实验的分类头权重，预测仍需配套的预训练主干；其封装不保证兼容新公共入口。
- `results/dinov2_linear_probe/pretrained_source.json`：历史实验所用预训练模型的来源、版本、相关库版本和预处理参数。
- `results/dinov2_linear_probe/formal/run01/config_snapshot.json`：历史正式训练当时的完整配置，保留当时的数据路径，不作为当前路径说明。
- `results/dinov2_linear_probe/formal/run01/history.json`：历史 10 轮训练的准确率、交叉熵损失及耗时。
- `results/dinov2_linear_probe/formal/run01/metrics.json`：历史测试集的 Top-1、Top-5、损失、各类别准确率及 100×100 混淆矩阵。
- `results/dinov2_linear_probe/formal/run01/processor/preprocessor_config.json`：历史图像预处理配置，包括缩放、中心裁剪和归一化参数。

### 本地辅助文件

- `.cache/` 下的文件：由准备步骤下载的预训练模型配置、图像预处理配置、safetensors 权重及下载元信息，不属于原始图片数据。
- `.venv/` 下的文件：本地隔离的 Python 解释器与已安装依赖，由创建虚拟环境和安装依赖的命令产生。
- `__pycache__/`、`*.pyc`：Python 自动生成的字节码缓存，无需手动编辑。
- `.DS_Store`：macOS 自动生成的文件夹显示设置，与训练逻辑无关。
- `.git/` 下的文件：Git 管理提交历史、暂存区及本地仓库配置的元数据，使用 Git 命令维护。

## 多人协作与强制规范

- 项目结构变化时，AI 必须在同一次变更中主动更新 `README.md` 和 `AGENTS.md` 的可维护章节；涉及数据说明时同步 `dataset/README.md`。目录树通过 `tools/sync_docs.py` 自动同步，提交 hook 自动更新并检查暂存状态，CI 再次校验；章节说明由 AI 同步维护。
- 每个参与者提交的每个模型代码放在一个独立文件：`models/<预训练模型名>_<GitHub用户名>.py`。例如 `dinov2_small_alice.py`，对应 `model_key=dinov2_small_alice`。模型名转为小写，空格、斜杠和连字符转为下划线；GitHub 用户名保留原值，含连字符时通过文件路径加载。
- 模型代码、配置和上传参数必须分离。使用明确的函数调用接口（function calling）传入配置、权重路径和数据；导入模型文件不得自动训练、评估或下载权重。
- 同一个 `model_key` 关联代码、配置、权重和评分，每次实验使用独立 `run_id`；不得覆盖他人代码、原始数据或历史实验产物。
- 只有所有者本人可修改 `AGENTS.md` 的受保护规则区；AI 随结构变更更新其余章节，不能修改规则以绕过约束。

## 模型接口与提交方式

每个模型文件统一提供以下函数，详细输入输出见 [AGENTS.md](AGENTS.md)：

```python
build_model(config) -> model
load_weights(model, checkpoint_path, *, device) -> model
predict(model, inputs) -> logits
```

配置由调用方读取 JSON 后传入，权重由路径加载；`inputs` 为已预处理的批量图像，`logits` 为 `[batch_size, num_classes]` 的未归一化分类分数。
数据读取、训练调度与指标计算由公共流程负责。已由 `pipeline/` 实现公共流程，当前模型代码为 `models/dinov2_small_gududegenbai3.py`。

新模型的交付路径约定如下，尖括号为待替换的占位符：

```text
models/<model_key>.py
configs/<model_key>.json
checkpoints/<model_key>/<run_id>/final.pt
results/<model_key>/<run_id>/config_snapshot.json
results/<model_key>/<run_id>/class_to_idx.json
results/<model_key>/<run_id>/metrics.json
results/<model_key>/<run_id>/history.json
```

配置和运行快照记录模型来源及版本、数据路径与类别映射、预处理、随机种子、训练超参数和输出路径；`environment.json` 同时记录代码版本、模型源码哈希、输入权重和依赖环境，`status.json` 记录运行中、失败或完成状态。

## 模型分数规范

每次评估的 `metrics.json` 必须包含 `model_key`、`run_id`、`split`、`samples` 及以下三项：

- **accuracy**：Top-1 正确率，文件中使用 `[0, 1]` 比例，展示时可转为百分比。
- **cross_entropy_loss**：全部评估样本的平均交叉熵损失，使用原始 logits 计算。
- **confusion_matrix**：按类别索引排列的原始整数计数矩阵，行是真实类别、列是预测类别；同时交付 `class_to_idx.json`，可另附矩阵图片。

混淆矩阵总和须等于评估样本数，对角线之和除以样本数须与 accuracy 一致。训练与测试指标分别注明，未运行的实验不填写虚构分数。
现有基线使用历史字段 `top1_accuracy` 和 `loss`，分别对应上述 accuracy 与交叉熵损失；保留原记录，不回写历史文件。

## 已有基线的训练过程

1. **数据准备**：使用 CIFAR-100，共 100 个类别、60,000 张 32×32 RGB 图片；沿用已有的 50,000 张训练集和 10,000 张测试集，不重新划分。
2. **模型选型**：选择 Meta 官方公开的 DINOv2-Small 作为预训练特征提取主干，先完成基线模型，再考虑进阶模型。
3. **特征提取**：使用模型配套的图像预处理，冻结主干参数，提取并缓存每张图片的 384 维 CLS 特征。
4. **分类头训练**：添加 100 类线性分类层，使用 AdamW 和交叉熵损失训练；主干参数保持不变。
5. **评估与交付**：完成测试集评估，保存分类头权重、配置、类别映射与训练记录，由团队统一整理和上传。

## 已训练模型与成果

当前已完成 **DINOv2-Small + 线性分类头** 的训练与评估。

- 预训练模型：[Meta DINOv2-Small](https://huggingface.co/facebook/dinov2-small)。
- 训练方式：冻结主干，仅训练 `Linear(384, 100)` 分类头，共 38,500 个可训练参数。
- 训练数据：50,000 张图片，训练 10 轮。
- 最后一轮训练准确率：**93.316%**。
- 最后一轮训练交叉熵：**0.2245**。
- 测试数据：完整的 10,000 张测试图片。
- 测试 Top-1 准确率：**84.96%**。
- 测试 Top-5 准确率：**97.34%**。
- 测试交叉熵：**0.5173**。

上述为当前实验结果。正式分类头权重位于 `checkpoints/dinov2_linear_probe/formal/run01/final.pt`，配置和评估记录位于 `results/dinov2_linear_probe/formal/run01/`。预测时需加载同一版本的官方 DINOv2 主干。后续实验沿用统一的测试集记录成果。

评估文件 [metrics.json](results/dinov2_linear_probe/formal/run01/metrics.json) 已包含 100×100 混淆矩阵；训练记录见 [history.json](results/dinov2_linear_probe/formal/run01/history.json)。历史实验未按新命名约定记录 GitHub 用户名，保留原名，不推断贡献者身份；新提交按上述约定执行。

## DINOv3 ViT-L/16：yjqsh233

模型为 Meta `facebook/dinov3-vitl16-pretrain-lvd1689m`，固定版本 `ea8dc2863c51be0a264bab82070e3e8836b02d51`。原实验在单张 Tesla T4 16GB、Python 3.12、PyTorch 2.8.0+cu128 上完成。全部 60,000 张原实验图片与仓库 PNG 的 Git blob 哈希一致。

| 运行目录（`results/dinov3_vitl16_yjqsh233/` 下） | 数据划分 | Top-1 | 平均交叉熵 |
| --- | --- | --- | --- |
| `linear_seed42_20261023`：末层 CLS + patch 均值 | 测试 10,000 | 93.22% | 0.2277118073 |
| `last4_seed42_20261023`：末四层 CLS + patch 均值 | 测试 10,000 | 92.94% | 0.2320456329 |
| `partialft_epoch3_val_20261023`：最后两层微调第 3 轮候选 | 验证 5,000 | 93.30% | 0.2616679520 |
| `partialft_selected0_20261023`：验证选择保留第 0 轮 | 测试 10,000 | 93.22% | 0.2277120284 |

微调第 1–3 轮均未超过第 0 轮的 93.34% 验证准确率。第 3 轮候选没有测试集评分；保留方案的测试分数来自原主干与完整训练集重新拟合的分类头，不能归因于微调提升。各目录含真实 100×100 混淆矩阵、逐样本预测、配置、历史、划分、环境和参数哈希。

详细协议、权重依赖、复核方式及精度差异见 [实验记录](results/dinov3_vitl16_yjqsh233/README.md)。DINOv3 紧凑参数保存在同名 `checkpoints/` 子目录，需先准备固定版本的外部主干；[DINOv3 License](checkpoints/dinov3_vitl16_yjqsh233/LICENSE.md) 随参数提供。已有 DINOv2 成果继续保留。

## 环境与使用

建议使用 Python 3.13。先在项目根目录创建隔离环境并安装依赖：

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git config core.hooksPath .githooks
```

先校验数据与命名，再显式下载固定版本预训练文件，最后训练：

```sh
python -m pipeline validate --config configs/dinov2_small_gududegenbai3.json
python -m pipeline prepare --config configs/dinov2_small_gududegenbai3.json
python -m pipeline train --config configs/dinov2_small_gududegenbai3.json --run-id run01
```

`validate` 使用标准库即可运行，不导入模型，检查配置、接口声明和训练/测试类别映射；不代表执行过网络推理。
`prepare` 是显式联网步骤，缓存固定版本的模型配置、预处理及 safetensors 权重；训练与评估只读取本地文件。
训练由 AdamW 更新可训练参数，使用交叉熵损失；冻结 DINOv2 主干时只更新分类头，主干保持 eval 模式。
当前公共训练逐批处理图片，不缓存全量特征，计算成本与历史基线的缓存方案不同；不会自动启动完整数据集训练。

训练后用新的运行编号复评已保存权重：

```sh
python -m pipeline evaluate --config configs/dinov2_small_gududegenbai3.json --run-id eval01 --checkpoint checkpoints/dinov2_small_gududegenbai3/run01/final.pt
```

同一个 `run_id` 的产物目录存在时会拒绝写入。评估核对权重的模型身份、模型配置、预处理及类别映射，使用保存的完整 `state_dict` 计算三项统一指标。
`train --checkpoint <路径>` 可从新格式权重开始新一轮训练，但优化器会重新建立，不提供完整断点恢复。
历史权重及快照保持原样，其封装不保证兼容新入口。`configs/dinov2_linear_probe.json` 的数据路径已修正，但保留历史配置格式；新运行使用 `configs/dinov2_small_gududegenbai3.json`。

## 文档同步与验证

```sh
python3 tools/sync_docs.py
python3 tools/sync_docs.py --check
python -m unittest discover -s tests -v
```

同步工具只更新 `README.md` 和 `AGENTS.md` 的目录树标记区，保留所有者专属规则原文。职责、接口等说明仍应随结构变化同步编辑。
本地提交 hook 按暂存区自动更新目录树，若有文档未暂存编辑则要求检查并暂存后重新提交，不会替你暂存其他内容。每个克隆都需设置 `core.hooksPath`；GitHub CI 会检查同步结果、测试和语法。
未安装依赖时，标准库测试可运行，离线 DINOv2 集成测试会明确跳过；安装依赖后，集成测试在临时目录中以微型网络和合成图片验证训练、保存、复评与防覆盖，不下载正式预训练权重、不修改真实数据或历史评分。

实现参考：[PyTorch 参数保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、[交叉熵接口](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[Transformers DINOv2](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/dinov2)。
