# MSDGS Backbone P4 DCNv2 多 seed 稳定性报告

固定协议：250 epochs / imgsz=640 / batch=32 / cache=False / resume=False / yolo26n.pt。
所有指标由独立新进程从磁盘重载各 seed 的 best.pt 后评测。

| seed | test mAP50 | test mAP50-95 | crazing mAP50-95 | 理想三项同时通过 |
|---:|---:|---:|---:|---|
| 0 | 0.747863 | 0.404772 | 0.144572 | FAIL |
| 1 | 0.725255 | 0.390626 | 0.142394 | FAIL |
| 2 | 0.733549 | 0.391468 | 0.144497 | FAIL |
| 3 | 0.700871 | 0.383203 | 0.127621 | FAIL |

## 汇总

```json
{
  "n": 4,
  "seeds": [
    0,
    1,
    2,
    3
  ],
  "test_map50": {
    "n": 4,
    "mean": 0.7268843764748913,
    "std": 0.019696936905340058,
    "min": 0.7008706200559977,
    "max": 0.7478634738742257,
    "range": 0.04699285381822804
  },
  "test_map50_95": {
    "n": 4,
    "mean": 0.39251696672415004,
    "std": 0.008974257023998471,
    "min": 0.3832028553620318,
    "max": 0.40477179495828175,
    "range": 0.021568939596249947
  },
  "test_crazing_map50_95": {
    "n": 4,
    "mean": 0.139771255988213,
    "std": 0.00816253632468406,
    "min": 0.12762143146133503,
    "max": 0.14457241431179094,
    "range": 0.016950982850455903
  },
  "ideal_pass_count": 0,
  "owner_pass_count": 0,
  "per_metric_ideal_pass_count": {
    "test_map50": 1,
    "test_map50_95": 1,
    "test_crazing_map50_95": 0
  },
  "all_four_ideal_pass": false,
  "all_seed123_ideal_pass": false
}
```
