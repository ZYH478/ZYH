# MSDGS P5 分类纹理残差 seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate |
|---:|---:|---:|---:|---:|---|
| 0.733311 | 0.390902 | 0.145806 | 1779500 | None | FAIL |

## 相对 MSDGS

```json
{
  "owner_gate_pass": false,
  "owner_checks": {
    "test_map50_gt_msdgs": true,
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
    "map50": 0.0008993324827183402,
    "map50_95": -0.007935095533067316,
    "crazing_map50_95": -0.031263677500598136
  }
}
```
