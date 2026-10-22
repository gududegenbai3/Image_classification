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
