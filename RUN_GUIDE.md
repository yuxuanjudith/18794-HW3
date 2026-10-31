# HW3 Problems 1–4：本机运行与复现

## 已采用的配置

- 数据：PASCAL VOC2012 官方 segmentation train（1464 张）/ val（1449 张）。
- 模型：DeepLabV3+、ResNet-50 ImageNet-1K V2 分类权重初始化、21 类、output stride 16。
- 两个独立实验：交叉熵；训练集类别频率加权的 Focal Loss（gamma=2）。
- 每个实验 5000 次优化更新，batch size 8，513×513 随机裁剪。
- SGD：head lr=0.02，backbone lr=0.002，momentum=0.9，weight decay=1e-4。
- 每 1000 次更新，两个学习率都乘以 0.1。
- FP16 AMP 训练节省显存；每个 epoch 使用全部验证集、原始分辨率、FP32 评估。
- 每次实验从 ImageNet 骨干与同种子随机 head 独立开始，不加载另一次分割训练的权重。

## 题目与代码对应

| 题目 | 主要代码 | 产出 |
|---|---|---|
| Problem 1 | `datasets/voc.py`、`dataset_stats.py`、`visualize.py samples` | 图像/标签读取、类别统计、样例图与解释 |
| Problem 2 | `network/deeplab.py`、`network/modeling.py` | ASPP、DeepLabV3 和 DeepLabV3+ 分割头 |
| Problem 3.1–3.3 | `main.py` | 分组 SGD、StepLR、交叉熵 5000 次训练和每轮验证 |
| Problem 3.4 | `utils/loss.py`、`class_weights.npy` | 类别加权 Focal Loss 与独立训练 |
| Problem 3.5 | `finish_report.py`、`visualize.py compare` | 最佳模型复评、指标表、四列预测对比、PDF |

完整模型是 **DeepLabV3+**；它内部的骨干 **ResNet50** 提取图像特征，
ASPP 与解码器将特征转成逐像素的分类结果。**V1/V2 指 ImageNet 预训练参数版本**，
切换权重后，完整模型和骨干的名称、结构都保持不变。PDF 指定了 DeepLabV3+、
ResNet50 和 ImageNet 预训练，没有指定权重必须使用 V1 或 V2；本报告明确记录实际版本。

## 环境

本机解释器：项目根目录 `.venv/Scripts/python.exe`。该环境复用了本机已有的
PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128，安装独立 NumPy wheel
以避免 Anaconda MKL 与 PyTorch OpenMP 冲突；没有修改 Anaconda 的全局包。
精确版本见 `environment.json` 和 `requirements.txt`。

在新的兼容 Python 环境中：

```powershell
python -m pip install -r requirements.txt
```

运行以下命令前进入本文件所在的代码目录。
本机可以把 `python` 替换为 `../../.venv/Scripts/python.exe`。

## 数据

下载并解压到 `datasets/data`：

```powershell
curl.exe -L -C - -o datasets/data/VOCtrainval_11-May-2012.tar https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar
tar -xf datasets/data/VOCtrainval_11-May-2012.tar -C datasets/data
```

归档 MD5 应为 `6cd6e144f989b92b3379bac3b3de84fd`。
需要 `datasets/data/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}`。
ResNet-50 权重由代码自动下载到 PyTorch 缓存，无需完整 ImageNet 数据集。
正式实验使用官方 `resnet50-11ad3fa6.pth`，可通过 `--backbone_weights v1` 重现旧初始化。
早期 V1、lr=0.01 试跑在完成 11 个 epoch 后最佳 mIoU 为 61.59%，未训练满 5000 次；
其完整日志保留在 `report/pilot_v1_miou.json`，其权重及代码快照保留在工作区
`experiments/pilot_imagenet_v1`。它不参与最终两种损失的比较。

## 检查与 Problem 1

```powershell
python check_implementation.py
python dataset_stats.py
python visualize.py samples
```

产出类别统计、类别权重、21 类样例（包含背景）和数值检查结果。

## Problem 3：按顺序运行两个实验

```powershell
python run_experiments.py
```

脚本依次训练两个模型，完成后调用 `finish_report.py` 重新评估最佳权重、生成图表、
填入报告并编译。重复运行时，完成的实验会跳过；未完成的实验从最近的 epoch 保存点继续。
完成状态会同时检查最佳模型文件。交付 ZIP 不包含大模型权重；在仅有随包日志的解压目录
重新运行时，脚本会把原日志备份到 `logs/previous_runs/`，然后从 ImageNet 初始化重新训练。
Windows 运行期间只阻止系统自动睡眠，不保持屏幕常亮。电脑关机或手动睡眠仍会中断。

也可单独运行：

```powershell
python main.py --amp --channels_last --loss_type cross_entropy
python main.py --amp --channels_last --loss_type focal_loss --class_weights class_weights.npy
```

最新进度见 `logs/pipeline_status.json` 和各自的 `*_console.log`。
每个 epoch 的完整指标见 `logs/*_miou.json`；实验配置见 `logs/*_config.json`。
`smoke_*` 是短测试，不用于最终结果或达标判断。

## 模型与最终结果

- `checkpoints/best_*_VOC_os16.pth`：按验证 mIoU 选出的模型。
- `checkpoints/latest_*_VOC_os16.pth`：带优化器、学习率、AMP 状态与历史的恢复点。
- `report/results.json`：最终数值、独立复评、模型 SHA256。
- `logs/*_evaluation.json`：完整验证集的混淆矩阵（行是真值、列是预测），以及背景误报和前景漏报比例。
- `report/per_class_iou.csv`：21 类的验证 IoU。
- `report/figs/`：样例、每个 epoch 的 mIoU 曲线、5 张图片的四列对比及包含 SAM 的五列对比。
- `report/main.tex` / `main.pdf`：以课程原始模板填写的报告。
- 代码目录 `output/`：报告 PDF 和不含大数据、模型权重的代码报告压缩包。
  可用 `python finish_report.py --output_dir <目录>` 指定交付位置。

交叉熵实验的最佳完整验证 mIoU 至少 0.65 才满足 Problem 3.3 的数值要求。
最终结果以完整验证集上的独立复评为准；短测试用于检查程序。Focal Loss 允许下降，报告按实际数据解释。
两次训练只做一个随机种子；对比图使用训练图片，定量指标使用独立验证集。

## 报告编译

本机的 Tectonic 位于项目根目录 `tools/tectonic/tectonic.exe`。

```powershell
python build_report.py
../../tools/tectonic/tectonic.exe --keep-logs report/main.tex
```

训练未完成时仅可使用 `python build_report.py --draft`，其报告明确标记为草稿。
报告姓名按要求留空。Problem 4 已补充原始 SAM ViT-B 的五图推理、第五列对比及分析。
报告包含课程代码、论文和 AI 辅助说明，便于按课程要求审核与署名。

## Problem 4：SAM（无需训练）

使用原始 SAM 的官方 ViT-B 权重。自动掩码生成采用 32×32 提示点网格、每批16点、
predicted-IoU阈值0.88、stability阈值0.95、crop层数0，FP32推理；不输入真值提示。

```powershell
python -m pip install -r requirements_sam.txt
curl.exe -L --fail -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python sam_predict.py
python visualize.py compare --ckpt1 checkpoints/best_deeplabv3plus_resnet50_cross_entropy_VOC_os16.pth --ckpt2 checkpoints/best_deeplabv3plus_resnet50_focal_loss_VOC_os16.pth --sam_dir sam_results
python build_report.py
../../tools/tectonic/tectonic.exe --keep-logs report/main.tex
```

SAM颜色表示不同掩码，不表示VOC类别；黑色表示没有保留掩码覆盖的像素。
`sam_results/*_masks.npz`保留所有二值掩码，JSON保留框、面积和评分，PNG为合成可视化。
`report/sam_results.json`记录图像SHA256、掩码数、权重SHA256、配置和显存峰值。
完整交付文件为 `HW3_Problems_1_4.pdf` 和 `HW3_Problems_1_4_code_report.zip`。
