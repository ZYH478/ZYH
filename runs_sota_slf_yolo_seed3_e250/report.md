# SLF-YOLO Seed3 三数据集训练结果

- 模型：`yolov8-SlimAsfNeck-CGLU.yaml`（Ultralytics 默认 `scale='n'`，本轮记作 `slf_yolo_slimasfneck_cglu_n`）。
- 训练：250 epochs、imgsz=640、batch=32、seed=3、deterministic=True、cache=False、patience=0、optimizer=SGD、amp=False、close_mosaic=10、from scratch。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090、fused FP32、batch=32、warmup=50、iterations=200，仅模型 forward。

## 总体指标

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.738293 | 0.399277 | 1060.33 | 2,665,278 |
| aluminum | 0.984453 | 0.571872 | 1059.59 | 2,664,888 |
| pcb | 0.706121 | 0.247856 | 1053.17 | 2,665,278 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.364407 | 0.125362 |
| inclusion | 0.757054 | 0.374206 |
| patches | 0.938610 | 0.612972 |
| pitted_surface | 0.815568 | 0.477596 |
| rolled-in_scale | 0.639133 | 0.289537 |
| scratches | 0.914987 | 0.515989 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.975706 | 0.389551 |
| ca_shang | 0.972633 | 0.557207 |
| zang_wu | 0.995000 | 0.657964 |
| zhe_zhou | 0.994474 | 0.682765 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.936935 | 0.336729 |
| mouse_bite | 0.474368 | 0.126299 |
| open_circuit | 0.877106 | 0.367679 |
| short | 0.755265 | 0.281416 |
| spurious_copper | 0.552445 | 0.170837 |
| spur | 0.640604 | 0.204176 |
