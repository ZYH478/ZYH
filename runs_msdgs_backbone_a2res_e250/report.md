# MSDGS Backbone P4 A2 残差 seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate |
|---:|---:|---:|---:|---:|---|
| 0.729772 | 0.404557 | 0.151306 | 1820166 | None | FAIL |

## 相对 MSDGS

```json
{
  "owner_gate_pass": false,
  "owner_checks": {
    "test_map50_gt_msdgs": false,
    "test_map50_95_ge_msdgs": true,
    "test_crazing_map50_95_gt_msdgs": false
  },
  "ideal_gate_pass": false,
  "ideal_checks": {
    "test_map50_ge_0.738": false,
    "test_map50_95_ge_0.402": true,
    "test_crazing_map50_95_ge_0.187": false
  },
  "deltas_vs_msdgs": {
    "map50": -0.002639420044424856,
    "map50_95": 0.005719358285646181,
    "crazing_map50_95": -0.025763660721535214
  }
}
```
