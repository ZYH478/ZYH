# MSDGS Backbone P4 Hybrid DCNv2 r50 seed0 报告

保护机制：50% 规则网格通道 + 50% 有界 DCNv2 通道；offset∈[-1,1]，mask∈[0.5,1.5]。
固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate | 理想三指标 |
|---:|---:|---:|---:|---:|---|---|
| 0.702100 | 0.387161 | 0.124018 | 1792988 | None | FAIL | FAIL |

## 相对 MSDGS 与 gate

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
    "map50": -0.030311719446464047,
    "map50_95": -0.011676280339328216,
    "crazing_map50_95": -0.0530514240750698
  }
}
```
