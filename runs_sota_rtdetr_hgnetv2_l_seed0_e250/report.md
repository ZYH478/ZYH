# RT-DETR-HGNetv2-L Seed0 三数据集基线

- 训练：250 epochs，imgsz=640，batch=8，seed=0，deterministic=False，cache=False，patience=0。
- 初始化：官方 Ultralytics `rtdetr-l.pt`（HGNetv2-L）。
- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。
- FPS：RTX 4090，fused FP32，batch=1，warmup=20，iterations=100，仅计 forward。

## 总体性能

| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| neudet | 0.692772 | 0.390422 | 40.60 | 31,996,070 |
| aluminum | 0.967322 | 0.559479 | 38.93 | 31,991,960 |
| pcb | 0.943891 | 0.405071 | 39.32 | 31,996,070 |

## neudet 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.329953 | 0.118224 |
| inclusion | 0.703807 | 0.348658 |
| patches | 0.925380 | 0.600711 |
| pitted_surface | 0.719708 | 0.455984 |
| rolled-in_scale | 0.560776 | 0.294278 |
| scratches | 0.917005 | 0.524675 |

## aluminum 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| zhen_kong | 0.946597 | 0.351064 |
| ca_shang | 0.961312 | 0.570492 |
| zang_wu | 0.995000 | 0.661970 |
| zhe_zhou | 0.966379 | 0.654391 |

## pcb 逐类 test AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| missing_hole | 0.968297 | 0.460728 |
| mouse_bite | 0.842185 | 0.285305 |
| open_circuit | 0.977337 | 0.545342 |
| short | 0.946245 | 0.305543 |
| spurious_copper | 0.980587 | 0.465031 |
| spur | 0.948692 | 0.368475 |
