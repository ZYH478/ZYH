# YOLOv8n Seed0 三数据集基线

- 训练：250 epochs，imgsz=640，batch=32，seed=0，deterministic=True，cache=False，patience=0。
- 初始化：官方 `yolov8n.pt`。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090，fused FP32，batch=32，warmup=50，iterations=200，仅计 forward。

## 总体性能

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.716441 | 0.382702 | 2237.98 | 3,006,818 |
| aluminum | 0.988181 | 0.568334 | 2243.03 | 3,006,428 |
| pcb | 0.798451 | 0.337164 | 2241.88 | 3,006,818 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.317986 | 0.121526 |
| inclusion | 0.758098 | 0.366472 |
| patches | 0.917330 | 0.568301 |
| pitted_surface | 0.778047 | 0.444376 |
| rolled-in_scale | 0.602838 | 0.288828 |
| scratches | 0.924346 | 0.506708 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.983974 | 0.381756 |
| ca_shang | 0.978751 | 0.566985 |
| zang_wu | 0.995000 | 0.663757 |
| zhe_zhou | 0.995000 | 0.660837 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.898857 | 0.441057 |
| mouse_bite | 0.640652 | 0.214473 |
| open_circuit | 0.873116 | 0.399141 |
| short | 0.852585 | 0.324110 |
| spurious_copper | 0.781287 | 0.348949 |
| spur | 0.744209 | 0.295252 |
