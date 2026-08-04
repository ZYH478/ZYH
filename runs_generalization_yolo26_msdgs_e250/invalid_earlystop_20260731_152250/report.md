# YOLO26n 与 MSDGS 三数据集迁移性/鲁棒性对比

> 本报告是两个结构在各目标数据集上按同一协议重新训练后的迁移性与鲁棒性验证，不是 zero-shot 泛化。

## 统一口径

- 训练：250 epochs，imgsz=640，batch=32，seed=0，deterministic=True，cache=False，resume=False，均从 yolo26n.pt 初始化。
- 真值：每组训练后由独立新 Python 进程从磁盘重载 best.pt，在 test split 评测。
- FPS：同一 RTX 4090，模型 fuse 后，batch=32，固定 warmup/iterations，仅计 forward，排除预处理与后处理。

## gc10

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 待完成 | 待完成 | 待完成 | 待完成 |
| msdgs | 待完成 | 待完成 | 待完成 | 待完成 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| 待训练完成 | - | - | - |

## uwwt

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 待完成 | 待完成 | 待完成 | 待完成 |
| msdgs | 待完成 | 待完成 | 待完成 | 待完成 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| 待训练完成 | - | - | - |

## steel_weld

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 待完成 | 待完成 | 待完成 | 待完成 |
| msdgs | 待完成 | 待完成 | 待完成 | 待完成 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| 待训练完成 | - | - | - |

