# ConvNeXt V2 分类头训练：AutoDL / Python 3.12

沿用镜像自带 `/root/miniconda3/bin/python`（Python 3.12），不创建或激活额外环境。目录实际名是 `convenext/`。本目录保存专用训练入口，模型实现和训练配置分别位于根目录 models、configs；不使用现有公共 pipeline 的训练入口。

目标模型：[timm/convnextv2_base.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384)，官方 384×384 预处理，1024 维特征，冻结整个主干，仅训练 Linear(1024,100)。当前仅交付训练代码并做离线微型网络验证，未执行真实云端训练或产生正式分数。

## 1. 上传代码

数据已上传的情况下，不必重新上传 110,000 张图片。把本地代码包 `convnext-training-code.tar.gz` 上传至服务器 `/root/autodl-tmp/`，在服务器终端执行：

```sh
cd /root/autodl-tmp/Image_classfication
tar -xzf /root/autodl-tmp/convnext-training-code.tar.gz -C .
```

代码包包含 convenext 下的训练/环境文件、`models/convnextv2_base_gududegenbai3.py`、`configs/convnextv2_base_gududegenbai3.json`，不含数据、权重或整个公共 pipeline。项目根目录名按实际上传位置调整；直接运行脚本时根路径按脚本位置推导，与当前终端目录无关。

## 2. 确认依赖和数据

已配置好的 Python 3.12 镜像可以直接使用。若需要补依赖，在已工作的 GPU 环境中执行 `python -m pip install -r convenext/requirements.txt`；不为本实验另建环境。Blackwell 需要匹配架构的 CUDA wheel，推荐已验证的 PyTorch 2.8/cu128 与 torchvision 0.23；不要仅因基础镜像版本与推荐组合不同就重装。

```sh
/root/miniconda3/bin/python -m pip check
/root/miniconda3/bin/python convenext/train_head.py validate
```

预期输出原图 50,000、镜像 50,000、test 10,000、classes 100。数据应为：

```text
dataset/train/<类别>/<文件名>.png
dataset_mirror/train/<类别>/<相同文件名>.png
dataset/test/<类别>/<文件名>.png
```

`validate` 只做标准库检查，不下载模型、不启动训练。`check_environment.py` 另用于 GPU/合成输入检查；它按 Python 3.12 和依赖兼容范围校验，配置中的 torch 等精确字符串是推荐参考值。

## 3. 显式准备预训练权重

```sh
/root/miniconda3/bin/python -u convenext/train_head.py prepare
```

该步骤需要服务器连接 Hugging Face。基础配置 revision 为 null 时，只在第一次 prepare 查询当前 commit SHA，然后按该 SHA 下载 `config.json`、`model.safetensors` 等，并记录权重/配置 SHA256 到 `.cache/convnextv2_base_gududegenbai3/pretrained.json`。后续 prepare 校验并复用固定快照，不自动追踪 main 更新。

下载失败时保留下载缓存，可以重试 prepare；未经成功 prepare，train 会明确报错，不隐式联网下载。不要在聊天中粘贴账号凭据。此模型的许可信息以官方模型卡为准。

## 4. 启动训练

```sh
/root/miniconda3/bin/python -u convenext/train_head.py train --run-id run01
```

运行目录存在时拒绝覆盖，包括失败运行；重试用新的 `--run-id run02`。省略 run-id 时自动生成带 UTC 时间和微秒的唯一编号。本版不提供完整断点恢复；失败后新编号会重新提取特征。

长时间任务可在服务器上先进入 `screen -S convnext`（需已安装 screen），再运行训练命令；Ctrl+A 后按 D 可离开会话，重新连接后执行 `screen -r convnext`。不要在前台训练未结束时直接关掉终端或实例。

脚本依次执行：

1. 检查类别、相对文件名、样本数和权重 SHA256，冻结主干，使用 timm 的确定性官方预处理。
2. 原图和镜像分别提取 FP32 存储的特征（GPU 前向默认 BF16），缓存到本次运行独立目录；不同运行不混用缓存。
3. 按原图身份分层划分：45,000 原图及其对应镜像共 90,000 张参与选择训练，5,000 原图验证，验证原图的镜像不进入选择训练。
4. 最多训练 20 轮，通过验证准确率选轮数；准确率相同则比较验证交叉熵，仍相同保留较早轮数。优化器为 AdamW，学习率 0.001、weight decay 0.0001，分类头 batch 1024。
5. 重置分类头与优化器，在全部 100,000 张特征上训练选定轮数。
6. 保存并重新加载分类头，然后才提取原始 test 特征并评分一次，保存准确率、平均交叉熵和原始 100×100 混淆矩阵。

原图与镜像不会跨训练/验证泄漏；测试分数不用于选轮数。不会重命名、移动、重新保存图片，也不会覆盖已有成果。

## 5. 查看结果与归档

```text
checkpoints/convnextv2_base_gududegenbai3/<run_id>/final.pt
results/convnextv2_base_gududegenbai3/<run_id>/
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

- `final.pt` 只包含分类头参数和身份、类别映射、配置、选定轮数，不保存优化器或复制完整预训练主干；格式为 convnext_head_v1。推理仍需要相同 SHA 的主干和相同预处理。
- `selection_history.json` 为验证选择阶段的逐轮记录，`history.json` 为全量重拟合记录。
- `config_snapshot.json` 固定实际 revision、主干 SHA、预处理和运行参数；`environment.json` 记录实际库版本、GPU、源码哈希等。
- `samples.json` 固定三个数据源的文件顺序；`split.json` 的索引以 original 文件列表为准；预测顺序以 test 文件列表为准。
- `status.json` 为 running/failed/complete。失败目录保留错误信息；仅 status 为 complete 才表示整个交付流程结束。
- `.cache/features/<model_key>/<run_id>/` 下 original.pt、mirror.pt、test.pt 为固定特征缓存，默认共约 450MB 的 FP32 张量（不含额外开销），不提交 Git。RAM 还要容纳合并张量和选择训练副本，不能按缓存大小估计全部主机内存。

下载本次 checkpoints/results 目录、训练源码、配置和必要的预训练来源记录后，再结束实例。保存系统镜像不能代替数据盘中的成果备份。

## 参数调整和边界

所有实验超参数只修改 `configs/convnextv2_base_gududegenbai3.json`，用新 run-id 运行。特征 batch 默认 16；显存不足可调小该值。CPU/内存压力大可降低 workers，分类头 batch 与图像 batch 独立。运行时打印每轮选择训练、验证和全量拟合记录，提取阶段显示进度条。

不要把此 JSON 直接传给 `python -m pipeline`，两套编排的配置格式不同。本适配器提供标准 build_model/load_weights/predict 函数；临时入口不豁免团队的模型命名和产物分离规范。

## 测试

```sh
python -m unittest discover -s tests -p test_convnext_training.py -v
```

测试在临时目录中以随机初始化的微型 ConvNeXt 和合成图片验证分组、特征缓存、分类头训练、权重重载、指标一致性及防覆盖，不下载模型、不读写真实实验成果。本地测试通过不代表已经在 AutoDL GPU 上完成正式训练。
