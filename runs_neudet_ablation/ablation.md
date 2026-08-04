# NEU-DET seed0 消融（统一独立复评协议）

生成 2026-08-02 11:04:13；真值=独立进程 test split 复评 + forward-only FPS。全部 seed0。

| 配置 | 组 | dilations | split | fused params | Δparams vs YOLO26n | mAP50 | mAP50-95 | FPS |
|---|---|---|---|---:|---:|---:|---:|---:|
| YOLO26n baseline (C3k2 neck + Conv-down) | component | - | - | 2,376,006 | +0.00% | 0.704649 | 0.374058 | 2128.1 |
| GSConv-down + VoVGSCSP (Slim-neck) | component | - | - | 1,935,814 | -18.53% | 0.690810 | 0.363272 | 2290.6 |
| DualBranchGS dil {1,2} | dilation | {1,2} | equal | 1,777,318 | -25.20% | 0.699879 | 0.362483 | 2426.0 |
| DualBranchGS dil {1,3} | dilation | {1,3} | equal | 1,777,318 | -25.20% | 0.728326 | 0.377627 | 2422.3 |
| DualBranchGS dil {1,5} | dilation | {1,5} | equal | 1,777,318 | -25.20% | 0.695404 | 0.354715 | 2418.0 |
| MSDGS {1,3,5} equal (OURS) | design | {1,3,5} | equal | 1,777,318 | -25.20% | 0.711102 | 0.367406 | 2367.4 |
| MSDGS {1,3} local-split | design | {1,3} | local | 1,777,318 | -25.20% | 0.694280 | 0.364090 | 2418.8 |
| MSDGS {1,3,5} local-split | design | {1,3,5} | local | 1,777,318 | -25.20% | 0.699377 | 0.346814 | 2401.9 |
