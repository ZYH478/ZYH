# SLF-YOLO Seed1 三数据集训练结果

- 模型：`yolov8-SlimAsfNeck-CGLU.yaml`（Ultralytics 默认 `scale='n'`，本轮记作 `slf_yolo_slimasfneck_cglu_n`）。
- 训练：250 epochs、imgsz=640、batch=32、seed=1、deterministic=True、cache=False、patience=0、optimizer=SGD、amp=False、close_mosaic=10、from scratch。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090、fused FP32、batch=32、warmup=50、iterations=200，仅模型 forward。

## 总体指标

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.739811 | 0.403105 | 1059.83 | 2,665,278 |
| aluminum | 0.980794 | 0.574127 | 1059.59 | 2,664,888 |
| pcb | 0.704978 | 0.290509 | 1057.35 | 2,665,278 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.356887 | 0.134186 |
| inclusion | 0.761939 | 0.392156 |
| patches | 0.934510 | 0.608090 |
| pitted_surface | 0.751972 | 0.448638 |
| rolled-in_scale | 0.705189 | 0.318520 |
| scratches | 0.928366 | 0.517041 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.956229 | 0.374781 |
| ca_shang | 0.976949 | 0.555983 |
| zang_wu | 0.995000 | 0.663005 |
| zhe_zhou | 0.995000 | 0.702741 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.977112 | 0.507442 |
| mouse_bite | 0.277768 | 0.082461 |
| open_circuit | 0.887479 | 0.362659 |
| short | 0.871154 | 0.333762 |
| spurious_copper | 0.567720 | 0.243236 |
| spur | 0.648637 | 0.213493 |
