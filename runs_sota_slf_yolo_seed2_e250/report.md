# SLF-YOLO Seed2 三数据集训练结果

- 模型：`yolov8-SlimAsfNeck-CGLU.yaml`（Ultralytics 默认 `scale='n'`，本轮记作 `slf_yolo_slimasfneck_cglu_n`）。
- 训练：250 epochs、imgsz=640、batch=32、seed=2、deterministic=True、cache=False、patience=0、optimizer=SGD、amp=False、close_mosaic=10、from scratch。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090、fused FP32、batch=32、warmup=50、iterations=200，仅模型 forward。

## 总体指标

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.715876 | 0.394367 | 1059.50 | 2,665,278 |
| aluminum | 0.969903 | 0.575023 | 1063.84 | 2,664,888 |
| pcb | 0.764717 | 0.320560 | 1062.28 | 2,665,278 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.298862 | 0.119408 |
| inclusion | 0.732135 | 0.380198 |
| patches | 0.948891 | 0.609419 |
| pitted_surface | 0.791950 | 0.467138 |
| rolled-in_scale | 0.608337 | 0.284764 |
| scratches | 0.915082 | 0.505273 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.908794 | 0.376309 |
| ca_shang | 0.980816 | 0.575359 |
| zang_wu | 0.995000 | 0.659333 |
| zhe_zhou | 0.995000 | 0.689091 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.943399 | 0.503736 |
| mouse_bite | 0.554832 | 0.177832 |
| open_circuit | 0.812932 | 0.378817 |
| short | 0.832185 | 0.310214 |
| spurious_copper | 0.821015 | 0.355787 |
| spur | 0.623938 | 0.196973 |
