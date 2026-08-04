# MSDGS Backbone P3 官方 C3k 主路径 seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / resume=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

## 总体结果

| split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.774231 | 0.634141 | 0.715198 | 0.390222 |
| test | 0.732660 | 0.689582 | 0.727753 | 0.391043 |

## Test 六类 AP

| 类别 | AP50 | AP50-95 |
|---|---:|---:|
| crazing | 0.402511 | 0.142931 |
| inclusion | 0.771233 | 0.401170 |
| patches | 0.939505 | 0.601655 |
| pitted_surface | 0.748256 | 0.404762 |
| rolled-in_scale | 0.570875 | 0.282533 |
| scratches | 0.934138 | 0.513206 |

## Gate

| test mAP50 | test mAP50-95 | crazing mAP50-95 | Owner gate | 理想 gate |
|---:|---:|---:|---|---|
| 0.727753 | 0.391043 | 0.142931 | FAIL | FAIL |

```json
{
  "owner_gate_pass": false,
  "owner_checks": {
    "test_map50_gt_msdgs": false,
    "test_map50_95_ge_msdgs": false,
    "test_crazing_map50_95_gt_msdgs": false
  },
  "ideal_gate_pass": false,
  "ideal_checks": {
    "test_map50_ge_0.738": false,
    "test_map50_95_ge_0.402": false,
    "test_crazing_map50_95_ge_0.187": false
  },
  "deltas_vs_msdgs": {
    "map50": -0.0046581562956286016,
    "map50_95": -0.007794436244087877,
    "crazing_map50_95": -0.034138942489095714
  }
}
```

## 效率

- fused 参数：`1779446`
- GFLOPs：`None`
- test infer ms/image：`1.9461938357572492`
- test FPS：`513.8234340419168`
