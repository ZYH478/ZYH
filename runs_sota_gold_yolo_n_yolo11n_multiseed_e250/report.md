# Gold-YOLO-n 与 YOLOv11n 四 Seed SOTA 训练汇总

- 状态：partial
- 完成：0/24
- 顺序：Gold-YOLO-n seed0/1/2/3 → YOLOv11n seed0/1/2/3；每个 seed 内按 NEU-DET、Aluminum、PCB 串行。
- 真值：每组训练结束后独立新进程从磁盘重载 best checkpoint，在 test split 评测。

| Model | Seed | Dataset | mAP50 | mAP50-95 | FPS | Weights |
|---|---:|---|---:|---:|---:|---|

## Pending

| Model | Seed | Dataset | Project |
|---|---:|---|---|
| Gold-YOLO-n | 0 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed0_e250` |
| Gold-YOLO-n | 0 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed0_e250` |
| Gold-YOLO-n | 0 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed0_e250` |
| Gold-YOLO-n | 1 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed1_e250` |
| Gold-YOLO-n | 1 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed1_e250` |
| Gold-YOLO-n | 1 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed1_e250` |
| Gold-YOLO-n | 2 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed2_e250` |
| Gold-YOLO-n | 2 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed2_e250` |
| Gold-YOLO-n | 2 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed2_e250` |
| Gold-YOLO-n | 3 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed3_e250` |
| Gold-YOLO-n | 3 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed3_e250` |
| Gold-YOLO-n | 3 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_gold_yolo_n_seed3_e250` |
| yolo11n | 0 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed0_e250` |
| yolo11n | 0 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed0_e250` |
| yolo11n | 0 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed0_e250` |
| yolo11n | 1 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed1_e250` |
| yolo11n | 1 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed1_e250` |
| yolo11n | 1 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed1_e250` |
| yolo11n | 2 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed2_e250` |
| yolo11n | 2 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed2_e250` |
| yolo11n | 2 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed2_e250` |
| yolo11n | 3 | neudet | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed3_e250` |
| yolo11n | 3 | aluminum | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed3_e250` |
| yolo11n | 3 | pcb | `/root/autodl-tmp/neu-det-yolo26/runs_sota_yolo11n_seed3_e250` |
