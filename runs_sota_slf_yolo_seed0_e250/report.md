# SLF-YOLO Seed0 三数据集训练结果

- 模型：`yolov8-SlimAsfNeck-CGLU.yaml`（Ultralytics 默认 `scale='n'`，本轮记作 `slf_yolo_slimasfneck_cglu_n`）。
- 训练：250 epochs、imgsz=640、batch=32、seed=0、deterministic=True、cache=False、patience=0、optimizer=SGD、amp=False、close_mosaic=10、from scratch。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090、fused FP32、batch=32、warmup=50、iterations=200，仅模型 forward。

## 总体指标

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.703639 | 0.400924 | 1056.60 | 2,665,278 |
| aluminum | 0.975666 | 0.575156 | 1059.65 | 2,664,888 |
| pcb | 0.746109 | 0.317125 | 1056.82 | 2,665,278 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.287428 | 0.107803 |
| inclusion | 0.731169 | 0.386497 |
| patches | 0.945407 | 0.616586 |
| pitted_surface | 0.726760 | 0.460164 |
| rolled-in_scale | 0.616899 | 0.310371 |
| scratches | 0.914172 | 0.524122 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.948415 | 0.374655 |
| ca_shang | 0.966704 | 0.579798 |
| zang_wu | 0.995000 | 0.664060 |
| zhe_zhou | 0.992544 | 0.682113 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.950117 | 0.496788 |
| mouse_bite | 0.581975 | 0.196440 |
| open_circuit | 0.901183 | 0.438553 |
| short | 0.689041 | 0.262426 |
| spurious_copper | 0.867126 | 0.341790 |
| spur | 0.487214 | 0.166751 |
