# CIFAR-100 dataset by classes folder
From: https://www.cs.toronto.edu/~kriz/cifar.html

## BibTeX
```sh
@article{krizhevsky2009learning,
    title="Learning Multiple Layers of Features from Tiny Images",
    author="Alex {Krizhevsky}",
    journal="",
    notes="Sourced from Microsoft Academic - https://academic.microsoft.com/paper/3118608800",
    year="2009"
}
```
 
## File Tree
```txt
dataset
├── train
│   ├── class_001
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
│   ├── class_002
|   |   ├── 1.jpg
|   |   ├── 2.jpg
|   |   └── ...
│   └── ...
└── test
    ├── class_001
    |    ├── 1.jpg
    |    ├── 2.jpg
    |    └── ...
    ├── class_002
    |    ├── 1.jpg
    |    ├── 2.jpg
    |    └── ...
    └── ...
```

## 国内下载

飞桨社区：https://aistudio.baidu.com/aistudio/datasetdetail/85769


## 本项目的数据域约定

本仓库使用现有 `dataset/train/<类别>/` 和 `dataset/test/<类别>/`，已核对为 100 类、50,000 张训练图片、10,000 张测试图片。
公共读取逻辑位于 `pipeline/data.py`，按训练目录类别名排序生成索引，要求测试集的类别集合和索引完全一致。
原始图片和划分保持不变，不自动重命名、重划分或修改标签；读取时转换为 RGB，再交给本地官方预处理器。
支持 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp`；类别下图片按一层目录读取，空类别或类别集合不一致会报错。
每次运行把映射写入 `results/<model_key>/<run_id>/class_to_idx.json`，评估加载权重时核对映射，避免类别顺序错位。

无需安装训练依赖即可检查配置、路径及类别对应关系：

```sh
python3 -m pipeline validate --config configs/dinov2_small_gududegenbai3.json
```

本地预训练文件由单独的 `prepare` 命令准备，不写入原始数据目录。完整训练与评估步骤见根目录 [README.md](../README.md)。

## DINOv3 提交的数据核对

`yjqsh233` 的历史实验读取官方 CIFAR-100 Python 归档（MD5 `eb9058c3a382ffc7106e4002c42a8d85`）。本次提交按原始文件名将全部图片编码成 PNG，逐张核对仓库 Git blob SHA1：训练 50,000 张和测试 10,000 张全部字节一致。证据见 `results/dinov3_vitl16_yjqsh233/dataset_identity.json`；提交不修改原始图片或标签。

超参数选择在官方训练集中按 seed 42 分层选取 45,000/5,000，具体原归档索引保存在各运行的 `split.json`。冻结分类头选定超参数后在完整 50,000 张训练集上重新拟合。最后两层微调使用 45,000 张训练、5,000 张验证；未选中第 1–3 轮，保留第 0 轮，并用完整训练集分类头进行测试。其测试分数不能视为微调提升。

历史评估采用 PIL 双线性直接缩放至 224×224，再按 ImageNet 均值/标准差归一化。它与公共 HF processor 的数值路径可能有差别；复核历史评分时还需区分保存的 FP16 特征缓存与在线图像推理。类别映射与仓库按类别名排序的 100 类索引一致。
## 水平镜像增强集

根目录的 `dataset_mirror/train/<类别>/<原文件名>.png` 为本训练集的水平镜像副本：100 个类别，每类 500 张，共 50,000 张 32×32 PNG 图片。类别目录和图片文件名与 `dataset/train/` 一一对应，标签不变。
生成后已逐张核对像素为原图的左右翻转，并核对原图文件哈希未变；`dataset/test/` 未参与变换。
该目录只保存增强图片，生成过程未留下脚本；整理后增加了 `train/` 层级，与原始数据集的训练层级一致，不含 `test/`。原训练集和镜像集分别保存，测试仍使用 `dataset/test/`。
公共流程只读取配置中的 `data.train_dir`；单独使用镜像集时应指向 `dataset_mirror/train`，指向上一级会把 `train` 误作类别。原图与镜像的联合加载不由当前公共流程自动完成；配置中的历史增强说明也不会触发合并。历史实验快照中的旧目录描述保持原样。
