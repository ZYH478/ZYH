# ZYH YOLO26 服务器工作快照

这是服务器目录 `/root/autodl-tmp/neu-det-yolo26` 的私有 Git 快照，主要保存可复现研究工作所需的源码、训练/评估脚本、模型配置、日志和结构化实验报告。

## 快照信息

- 来源服务器：AutoDL SSH 别名 `autodl`
- 来源目录：`/root/autodl-tmp/neu-det-yolo26`
- 快照日期：2026-08-04（Asia/Shanghai）
- 首次上传范围：稳定源码、配置、日志和报告
- 文件完整性清单：`FILE_SHA256SUMS.txt`
- 详细范围：`BACKUP_SCOPE.md`

## 主要内容

- 根目录训练、评估、诊断、安装与队列脚本
- `scripts/` 中的实验编排脚本
- 各 `generated_models_*` 目录中的 YAML/配置
- 各 `runs_*` 目录中的 JSON、CSV、Markdown、YAML 和日志报告
- `SLF-YOLO/`、`RT-DETR-official/`、`Gold-YOLO-src/` 的源码与配置快照
- `.codestable/` 中可被筛选规则识别的 Markdown、JSON、YAML 等项目记录

## 恢复说明

该仓库不是服务器 48G 工作目录的完整二进制镜像。数据集、模型权重、缓存和大体积训练工件将在后续使用私有 Release 分卷备份；恢复训练环境时还需要重新准备 Python/Conda 环境和数据集路径。