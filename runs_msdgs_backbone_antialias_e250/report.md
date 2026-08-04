# MSDGS Backbone P3→P4 抗混叠残差 seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate |
|---:|---:|---:|---:|---:|---|
| 0.714150 | 0.389900 | 0.118421 | 1793702 | None | FAIL |

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
    "map50": -0.01826157482654578,
    "map50_95": -0.008936862175637772,
    "crazing_map50_95": -0.05864809178543082
  }
}
```
