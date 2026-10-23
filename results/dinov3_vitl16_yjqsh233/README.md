# DINOv3 ViT-L/16 — yjqsh233

本提交整理三个已经完成的实验，不启动新的训练。为区分微调候选与最终保留方案，按四个 `run_id` 保存固定权重证据。原有 DINOv2 数据与结果保持原样。

## 模型与数据

- 来源：Meta `facebook/dinov3-vitl16-pretrain-lvd1689m`，revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`。
- 原始 `model.safetensors`：1,212,559,808 字节；SHA256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`。
- 原始 `config.json` SHA256 `135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5`；`preprocessor_config.json` SHA256 `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7`。
- 官方 CIFAR-100：50,000 训练、10,000 测试，100 个类别；官方 Python 归档 MD5 `eb9058c3a382ffc7106e4002c42a8d85`。全部 60,000 张图片按原文件名编码 PNG 后与上游 Git blob 哈希逐张一致，见 `dataset_identity.json`。
- seed 42，在官方训练集中分层取 45,000/5,000（每类 450/50）用于超参数选择。`split.json` 的索引对应官方 Python 归档顺序，不能直接解释成仓库文件名字典序。
- 历史确定性预处理：RGB、PIL bilinear 直接 resize 224×224、除以 255、ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`。评估无随机增强。

## 实验与结果

| run_id | 权重与评估 | 样本 | Top-1 | 平均交叉熵 |
| --- | --- | ---: | ---: | ---: |
| `linear_seed42_20261023` | 末层 CLS + patch 均值，测试集 | 10,000 | 93.22% | 0.22771180725097656 |
| `last4_seed42_20261023` | 末四层 CLS + patch 均值，测试集 | 10,000 | 92.94% | 0.2320456329345703 |
| `partialft_epoch3_val_20261023` | 最后两层微调第 3 轮候选，验证集 | 5,000 | 93.30% | 0.2616679520100355 |
| `partialft_selected0_20261023` | 保留第 0 轮，原主干 + 全训练集分类头，测试集 | 10,000 | 93.22% | 0.22771202836036683 |

前两种方案冻结主干，以 CUDA FP16 autocast 提取特征并保存 FP16 缓存；计算 FP32 特征均值/标准差，训练 FP32 线性头。末层方案为 2,048 维；末四层方案为最后四个 block 经最终 LayerNorm 的 CLS 与最后 patch 均值，共 5,120 维。patch 均值不包含 CLS 和 4 个 register tokens。特征提取 batch 16，分类头 SGD、momentum 0.9、batch 1024。验证选择末层方案 lr 0.01、weight decay 0、23 轮（验证 93.34%）；末四层方案 lr 0.01、weight decay 0.001、14 轮（验证 93.60%）。选好参数后分别在 50,000 张训练图片上重新拟合分类头。

微调只更新最后两个 block 与分类头（25,399,396 个参数），其余 277,935,104 个参数冻结，最终 LayerNorm 和 RoPE 坐标增强保持 eval。分类头先在 45,000 张选择训练集上以末层方案参数预热 23 轮，归一化统计也只使用该 45,000 张；第 0 轮选择不使用 50,000 张拟合的头，防止验证泄漏。随后 AdamW 3 轮，主干 lr 1e-5、分类头 lr 1e-3、weight decay 0.01，batch 64、梯度累积 1、clip 1、FP16 autocast/GradScaler；训练采用 RandomCrop(32,padding=4) 和水平翻转再 resize。

验证准确率依次为第 0 轮 93.34%、第 1 轮 92.36%、第 2 轮 93.24%、第 3 轮 93.30%。没有候选超过第 0 轮，因此第 3 轮保存为未选中的候选，未在测试集评分。最终保留原主干，使用完整训练集重新拟合的原分类头测试，额外微调轮数为 0。该测试分数不代表微调改善。

## 评分证据

每个运行目录提供 `config_snapshot.json`、`class_to_idx.json`、`history.json`、`split.json`、`environment.json`、`status.json`、`metrics.json` 和 `predictions.json`。后两者的类别索引均对应 `class_to_idx.json`；混淆矩阵行是真实类别、列是预测类别，所有元素为原始整数。

冻结方案用保存的原始测试特征缓存和固定分类头重新计算 logits、交叉熵与混淆矩阵（评分 batch 2048），得到与原记录完全相同的 accuracy 和 loss。微调候选与保留方案用原始图片及历史 PIL 预处理在线复评，batch 64，关闭梯度、所有训练行为和随机评估增强。适配器与原实现对两张实际图片的特征或 logits 最大绝对差均为 0；这是实现抽查，不是全数据等价证明。

候选第 3 轮原历史 loss 为 `0.26166679520100355`，本次固定权重复评为 `0.2616679520100355`，差约 `1.16e-6`；accuracy 相同。两个值分别保存在原始 `history.json` 和新 `metrics.json`，不改写历史。FP16/CUDA 数值路径和计算库版本会影响最后几位；跨设备评分应重新记录。

`provenance/` 保存原实验及本次导出脚本的源码文本、SHA256 清单、导出日志和原始附加记录。它们用于审计；不作为新的模型入口，也不声称历史训练由仓库公共 runner 执行。原实验额外依赖 torchvision 0.23.0+cu128、NumPy 2.3.2，其他原环境和源码哈希见各运行环境记录及 provenance。

## 权重加载与复核

参数路径为 `checkpoints/dinov3_vitl16_yjqsh233/<run_id>/final.pt`，格式 `dinov3_external_backbone_partial_v1`。冻结方案只存分类头及特征统计，微调候选另存实际训练过的最后两层；不重复上传 1.2GB 公共预训练主干或优化器。候选的 PyTorch ZIP 容器进行了无损 DEFLATE 压缩，压缩前后每个张量已用 `torch.equal` 检查完全相等，`torch.load(weights_only=True)` 可直接加载。参数 SHA256 对应最终交付文件。

先安装根目录依赖，再显式准备固定版本的 Meta 主干（遵守模型访问要求）：

```sh
python -m pipeline prepare --config configs/dinov3_vitl16_yjqsh233.json
```

也可把已有同版本的 `config.json`、`preprocessor_config.json`、`model.safetensors` 放到配置指定的本地目录。完整主干 SHA256 必须一致。模型由 Meta DINOv3 派生，许可见 `checkpoints/dinov3_vitl16_yjqsh233/LICENSE.md`。

调用时读取对应运行的配置快照，把 `model.pretrained_dir` 解析为本地绝对路径，依次调用 `build_model(config)`、`load_weights(model, checkpoint_path, device=...)` 和 `predict(model, preprocessed_inputs)`。输入由调用方按上述历史预处理构建。基础配置只对应末层冻结方案，不能用于最后四层或微调候选的参数。

仓库公共 `train` 固定使用 AdamW 和 HF processor，不会解释历史 SGD、缓存特征或分组学习率字段；直接运行它会产生新的实验。公共 `evaluate` 使用 HF processor，且只读取测试集，不能重现微调候选的验证集协议。复核上述历史分数应遵循每个 `metrics.json` 的缓存/在线路径、`split.json` 和 provenance 中的原实现，另用新运行目录保存复核结果。

```sh
python -m unittest discover -s tests -v
python tools/sync_docs.py --check --staged
```

微型 DINOv3 测试离线检查特征与参数接口；证据测试重建每份混淆矩阵并检查 accuracy、样本数、参数及源码哈希。这些检查不代替重新训练，也不声称已在其他 GPU 上复现实验。
