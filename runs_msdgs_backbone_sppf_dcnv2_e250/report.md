# MSDGS Backbone pre-SPPF P5 DCNv2 单模块替换 seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate |
|---:|---:|---:|---:|---:|---|
| 0.731141 | 0.383264 | 0.123376 | 1808604 | None | FAIL |

## 相对 MSDGS

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
    "map50": -0.0012703140579227723,
    "map50_95": -0.015573532907592391,
    "crazing_map50_95": -0.05369399893287398
  }
}
```
