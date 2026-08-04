# YOLO26n 与 MSDGS Aluminum/PCB 迁移性/鲁棒性对比

> 本报告是两个结构在各目标数据集上按同一协议重新训练后的迁移性与鲁棒性验证，不是 zero-shot 泛化。

## 统一口径

- 训练：250 epochs，imgsz=640，batch=32，seed=0，deterministic=True，cache=False，resume=False，patience=0（禁用 EarlyStopping），均从 yolo26n.pt 初始化。
- 真值：每组训练后由独立新 Python 进程从磁盘重载 best.pt，在 test split 评测。
- FPS：同一 RTX 4090，模型 fuse 后，batch=32，固定 warmup/iterations，仅计 forward，排除预处理与后处理。

## aluminum

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 0.965897 | 0.537342 | 2173.6 | 2,375,616 |
| msdgs | 0.982344 | 0.526203 | 2413.4 | 1,776,928 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| zhen_kong | 0.945880 | 0.985882 | +0.040003 |
| ca_shang | 0.930655 | 0.970169 | +0.039514 |
| zang_wu | 0.993265 | 0.992074 | -0.001191 |
| zhe_zhou | 0.993790 | 0.981251 | -0.012539 |

## pcb

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 0.734402 | 0.323580 | 2164.8 | 2,376,006 |
| msdgs | 0.823257 | 0.370845 | 2409.4 | 1,777,318 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| missing_hole | 0.585707 | 0.887251 | +0.301543 |
| mouse_bite | 0.651297 | 0.775812 | +0.124515 |
| open_circuit | 0.918994 | 0.940745 | +0.021751 |
| short | 0.945689 | 0.782465 | -0.163224 |
| spurious_copper | 0.791087 | 0.936705 | +0.145618 |
| spur | 0.513640 | 0.616563 | +0.102924 |

