# 首次备份范围

## 已包含

筛选并上传以下稳定文本类文件：

- 源码：`.py`、`.sh`、`.ps1`、`.bat`、C/C++/CUDA/Rust 源文件
- 配置：`.yaml`、`.yml`、`.json`、`.jsonl`、`.toml`、`.ini`、`.cfg`、`.conf`
- 记录：`.md`、`.txt`、`.log`、`.csv`、`.tsv`、`.xml`、`.ipynb`、`.tex`
- 工程文件：README、LICENSE、Makefile、Dockerfile、requirements、环境和代码质量配置
- 已完成或当前未写入实验中的结构化报告、状态、参数和日志

首次快照共有 2,333 个服务器文件，约 264.53 MiB；另新增本仓库说明与 SHA-256 清单。

## 本次明确排除

- 数据集目录：`dataset/`、`dataset_giis/`、`GC10-DET/`、`UWWT-Dataset-1500/`、`generalization_aluminum_pcb/`、`generalization_datasets/`、`generalization_wheat_stage/`、`gold_yolo_data/`
- 权重和部署二进制：`*.pt`、`*.pth`、`*.onnx`、`*.engine`
- 缓存和进程文件：`*.cache`、`*.pyc`、`*.pid`、`__pycache__/` 及框架缓存目录
- 嵌套仓库的 `.git/` 历史；其工作树源码作为普通目录保存
- 正在写入的 RT-DETR seed1/2/3 任务：`runs_sota_rtdetr_hgnetv2_l_seed123_e250/` 和 `sota_rtdetr_hgnetv2_l_seed123_e250.log`
- 普通图片、预测图、模型权重、数据集和其他大体积二进制训练工件
- Conda/Python 环境和服务器系统目录

## 一致性说明

制作快照时服务器上的 `train_sota_rtdetr_hgnetv2_l_seed123.py --train-worker neudet --seed 2` 仍在运行，因此该任务的运行目录和主日志被整体排除，避免提交不一致的半成品。任务完成后应进行增量同步。

## 原始归档校验

下载前在服务器生成的筛选归档：

```text
ZYH_stable_snapshot_20260804.tar.gz
SHA-256: bc0d3ebecb0679978570dfc29b06eef552f9e3e15d7e9499007e4d56c187effb
```