# ConvNeXt V2 分类头：训练入口与 AutoDL 实验记录

模型、配置、权重和评分统一使用 `convnextv2_base_gududegenbai3`。原临时目录 `convenext/` 已归位；从项目根目录使用 `python -m pipeline.convnext`，不再执行 `convenext/train_head.py`。迁移未改变训练协议、模型源码、缓存布局或历史评分。

## 已完成的真实运行

运行编号：`convnext_20261025_143320`。用户从 AutoDL 下载的原始文件已移入仓库根目录对应位置：

- 参数：`checkpoints/convnextv2_base_gududegenbai3/convnext_20261025_143320/final.pt`。
- 评分与训练记录：本目录下 `convnext_20261025_143320/`。
- 测试集：原始 `dataset/test`，10,000 张，100 类。
- 测试 accuracy：**0.8844（88.44%）**。
- 平均交叉熵：**0.38219886627197264**。
- 验证选定轮数：**5**；全量重新拟合 5 轮，最后一轮训练 accuracy 为 0.93676，不能当作测试分数。
- 实际环境：Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、timm 1.0.22、CUDA 12.8、NVIDIA RTX PRO 6000 Blackwell Server Edition。

以上数字来自原始 metrics.json/history.json；此次归档只核对证据，未重新训练或计算新的模型分数。

主干来源为 [timm/convnextv2_base.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384/tree/bea9dc139d5981aff6ce4d6d9690086e45ad8b56)，固定 commit `bea9dc139d5981aff6ce4d6d9690086e45ad8b56`。主干 SHA256 为 `972bcbb8d9965eca7a2f72d9592f54dfc890f9e6aef920cea4269963f46f2f42`；分类头 SHA256 为 `35f5199ea9b581cfbe5b76e59ffeb93d969f2580c70ffcbd12bca0da2449dc60`。主干许可及来源见该版本官方模型卡。

## 文件职责与原始证据

- `models/convnextv2_base_gududegenbai3.py`：唯一模型实现，提供 build_model/load_weights/predict，冻结主干，显式加载本地权重。
- `configs/convnextv2_base_gududegenbai3.json`：独立实验配置；与下载包的原基础配置相同。
- `pipeline/convnext.py`：数据检查、显式准备主干、特征缓存、验证选择、分类头重拟合及测试评分。
- `configs/convnextv2_base_environment.json`：AutoDL 目标环境及合成输入检查的配置。
- `tools/check_convnext_environment.py`：读取根目录 requirements.txt 和环境配置进行依赖及 GPU 自检，不下载主干、不读取真实图片。
- 根目录 `requirements.txt`：统一运行依赖，包含 timm、torchvision、tqdm、packaging；该入口不直接使用的 scikit-learn 不再单独要求。

本次运行的 `provenance/` 保存 `convnextv2_base_gududegenbai3.py.txt`、原 `convenext/train_head.py.txt`、原环境脚本及操作说明、基础 `config.json`、预训练来源 `pretrained.json`。Python 源码以文本留存供审计，不作为第二个模型实现入口。原 environment.json 的源码哈希指向这些原始文本；迁移后的训练模块仅更新入口说明，后续新运行会记录当前源码哈希。

`provenance/migration_manifest.json` 记录下载包原始文件路径、归档路径及 SHA256。已保存的 final.pt、config_snapshot.json、environment.json、metrics.json、history.json、selection_history.json、samples.json、split.json、predictions.json、class_to_idx.json、status.json 均按字节保留，没有为新目录说明回写历史记录。

## 数据与训练协议

数据目录相对于项目根目录：

```text
dataset/train/<类别>/<文件名>.png         # 50,000 原图
dataset_mirror/train/<类别>/<文件名>.png  # 50,000 对应镜像
dataset/test/<类别>/<文件名>.png          # 10,000 原始测试图
```

固定 384×384、bicubic、squash 预处理和 ImageNet 均值/标准差，以本次 config_snapshot.json 为准。主干输出 1024 维特征，默认 GPU BF16 前向、FP32 特征存储，仅训练 Linear(1024,100)。

1. 按原图身份、seed 42、类别分层划分 45,000 训练原图、5,000 验证原图。
2. 选择训练使用训练原图及其镜像，共 90,000 张；验证只用 5,000 原图，其镜像不进入选择训练。
3. AdamW、学习率 0.001、weight decay 0.0001、分类头 batch 1024，最多 20 轮。按验证 accuracy 选轮数；相同则选择交叉熵更小的轮次，仍相同保留较早者。
4. 重新初始化分类头及优化器，在全部 100,000 张训练特征上拟合选定轮数。
5. 保存、重新加载分类头，再提取原始 test 特征并评分一次；测试分数不用于选择训练轮数。

当前入口固定使用镜像，没有 `--no-mirror` 等开关。无镜像对照尚未实现或运行，因此不能仅凭本次分数断言镜像带来了提升。不改动原始图片或物理划分。

## AutoDL 操作

沿用镜像自带 Python 3.12，不创建或激活新环境。以下命令在项目根目录执行；AutoDL 可使用 `/root/miniconda3/bin/python` 替换 python。

上传根目录 `pipeline/`、上述模型源码、两份 ConvNeXt 配置、环境检查脚本、requirements.txt；数据按上面布局准备。已在本地下载模型时，必须额外上传整个 `.cache/convnextv2_base_gududegenbai3/`，包括 pretrained.json 和固定 commit 子目录。该缓存被 Git 忽略，不会随代码提交自动上传。

```sh
cd /root/autodl-tmp/Image_classfication
python -m pip install -r requirements.txt
python tools/check_convnext_environment.py
python -m pipeline.convnext validate
python -m pipeline.convnext prepare
```

validate 使用标准库，只检查数据与类别映射，不训练、不联网。环境自检使用随机网络与合成输入；通过不等于真实模型评分。prepare 复用并校验已有主干；首次没有缓存时，才显式联网解析 commit SHA、下载并记录配置/权重 SHA256，之后不自动追踪 main。

创建新的运行编号并后台训练：

```sh
RUN_ID=convnext_$(date +%Y%m%d_%H%M%S)
mkdir -p logs
nohup python -u -m pipeline.convnext train --run-id "$RUN_ID" > "logs/$RUN_ID.log" 2>&1 &
tail -f "logs/$RUN_ID.log"
```

Ctrl+C 只停止 tail 查看，不会终止 nohup 训练。训练期间保持实例运行。已有运行目录一律拒绝覆盖，失败后保留目录并使用新的 run_id；没有自动续训或完整断点恢复。当前没有独立 evaluate 子命令，train 内包含保存后单次测试评分。

不要将该配置传给通用 `python -m pipeline`，两套编排的配置格式不同；原 DINO 公共入口保持不变。

## 输出和备份

```text
checkpoints/<model_key>/<run_id>/final.pt
results/<model_key>/<run_id>/
    config_snapshot.json
    class_to_idx.json
    samples.json
    split.json
    selection_history.json
    history.json
    environment.json
    metrics.json
    predictions.json
    status.json
```

仅 `status.json` 为 complete 才代表完整训练和测试结束。metrics.json 中包含 accuracy、平均交叉熵及按类别索引排列的原始整数混淆矩阵；predictions.json 的顺序对应 samples.json 的 test 列表。split.json 的索引对应 original 列表。

final.pt 是 convnext_head_v1 格式，只包含分类头参数、模型身份、类别映射、实际配置与所选轮数。它需要相同版本、相同 SHA256 的外部主干和相同预处理才能推理；不包含主干或优化器恢复状态。

特征位于 `.cache/features/<model_key>/<run_id>/`，不提交 Git；原图/镜像/test 特征默认约 450MB，不包含训练张量副本等 RAM 开销。关闭实例前备份本次 checkpoints/results、源码、配置及主干来源；保留本地已下载主干，不必再次下载数据图片和特征缓存。

## 离线验证

```sh
python -m unittest discover -s tests -p 'test_convnext*.py' -v
```

test_convnext_evidence.py 无需 GPU 或 PyTorch，核对真实预测、混淆矩阵、准确率、原图分组、验证所选轮数及参数/源码/迁移文件哈希。没有 logits 时不能从预测标签重新计算交叉熵，原交叉熵仍取自云端记录。

test_convnext_training.py 的集成部分需要 PyTorch、torchvision、timm、Pillow、safetensors，使用临时目录、微型随机模型和合成图片验证训练、保存、重载及防覆盖；不下载正式主干、不修改真实图片或已有运行，分数不作为 CIFAR-100 成果。缺依赖时会明确跳过。
