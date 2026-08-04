# MSDGS P5 box residual seed0 报告

固定协议：250 epochs / imgsz=640 / batch=32 / cache=False / seed=0 / yolo26n.pt。
最终指标由独立新进程重载磁盘 `best.pt` 后分别评测 val/test。

| 候选 | 状态 | test mAP50 | test mAP50-95 | fused 参数 | GFLOPs | 晋级 |
|---|---|---:|---:|---:|---:|---|
| msdgs_p5box_res16 | done | 0.712320 | 0.380686 | 1816586 | 4.1138176 | FAIL |
| msdgs_p4detail_boxres | done | 0.724495 | 0.384811 | 1780730 | 4.085043199999999 | FAIL |

## 原始结果

```json
{
  "schema_version": 1,
  "started_at": "2026-07-30 07:20:29",
  "protocol": {
    "epochs": 250,
    "imgsz": 640,
    "batch": 32,
    "cache": false,
    "seed": 0,
    "init_weights": "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt",
    "data": "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml",
    "final_eval": "fresh subprocess reloads best.pt for val and test"
  },
  "baseline": {
    "map50": 0.7324113100352823,
    "map50_95": 0.3988372172746915,
    "per_class": {
      "crazing": 0.1770695825201802,
      "pitted_surface": 0.4686952771905249,
      "rolled-in_scale": 0.2584298478269593
    }
  },
  "experiments": {
    "msdgs_p5box_res16": {
      "status": "done",
      "candidate": "msdgs_p5box_res16",
      "seed": 0,
      "weights": "/root/autodl-tmp/neu-det-yolo26/runs_msdgs_boxres_e250/msdgs_p5box_res16_seed0/weights/best.pt",
      "params_fused": 1816586,
      "gflops_eval_graph": 4.712908799999999,
      "gflops_fused_inference": 4.1138176,
      "val": {
        "precision": 0.7040161433995807,
        "recall": 0.7061107983825098,
        "map50": 0.719305562040588,
        "map50_95": 0.3929224512343674,
        "infer_ms": 2.167254313826561,
        "fps_infer_only": 461.4133162039363,
        "per_class": {
          "crazing": {
            "precision": 0.5285219718239881,
            "recall": 0.42326320749570184,
            "map50": 0.4732229190065118,
            "map50_95": 0.18435552913836242
          },
          "inclusion": {
            "precision": 0.7820043712287652,
            "recall": 0.8238993710691824,
            "map50": 0.8136990197427033,
            "map50_95": 0.412645838121762
          },
          "patches": {
            "precision": 0.8243820975465086,
            "recall": 0.8823529411764706,
            "map50": 0.8957448395296329,
            "map50_95": 0.5832544383663383
          },
          "pitted_surface": {
            "precision": 0.7245750786193468,
            "recall": 0.676056338028169,
            "map50": 0.7122204752874776,
            "map50_95": 0.3911998903633456
          },
          "rolled-in_scale": {
            "precision": 0.5402231582750275,
            "recall": 0.5940977134987295,
            "map50": 0.5453351908217124,
            "map50_95": 0.2563895576063803
          },
          "scratches": {
            "precision": 0.824390182903848,
            "recall": 0.8369952190268055,
            "map50": 0.8756109278554898,
            "map50_95": 0.5296894538100156
          }
        }
      },
      "test": {
        "precision": 0.7157833208688501,
        "recall": 0.6714238049478983,
        "map50": 0.712320313218559,
        "map50_95": 0.38068573938845707,
        "infer_ms": 1.8888261405919309,
        "fps_infer_only": 529.4293521830518,
        "per_class": {
          "crazing": {
            "precision": 0.49732731916591977,
            "recall": 0.38095238095238093,
            "map50": 0.4111636759509341,
            "map50_95": 0.13044198556102493
          },
          "inclusion": {
            "precision": 0.6989819163528291,
            "recall": 0.74375,
            "map50": 0.7541101176007801,
            "map50_95": 0.3569781985296251
          },
          "patches": {
            "precision": 0.8389993625413759,
            "recall": 0.89937106918239,
            "map50": 0.9126874207948296,
            "map50_95": 0.586993212732488
          },
          "pitted_surface": {
            "precision": 0.7823444095232803,
            "recall": 0.6619718309859155,
            "map50": 0.7674238485344671,
            "map50_95": 0.4620583420206752
          },
          "rolled-in_scale": {
            "precision": 0.6217002024930128,
            "recall": 0.5813953488372093,
            "map50": 0.5658317118031773,
            "map50_95": 0.2612407322784801
          },
          "scratches": {
            "precision": 0.8553467151366824,
            "recall": 0.7611021997294941,
            "map50": 0.8627051046271659,
            "map50_95": 0.4864019652084498
          }
        }
      },
      "evaluated_at": "2026-07-30 07:42:35",
      "eval_pid": 3973,
      "cfg": "/root/autodl-tmp/neu-det-yolo26/generated_models_msdgs_boxres_e250/msdgs_p5box_res16.yaml",
      "gate": {
        "pass": false,
        "checks": {
          "map50_95_gt_0.401794": false,
          "map50_ge_0.722411": false,
          "target_class_gain_ge_0.01": false,
          "crazing_drop_le_0.02": false
        },
        "deltas_vs_msdgs": {
          "map50": -0.020090996816723217,
          "map50_95": -0.018151477886234446,
          "crazing": -0.046627596959155276,
          "pitted_surface": -0.0066369351698497,
          "rolled-in_scale": 0.0028108844515207787
        }
      }
    },
    "msdgs_p4detail_boxres": {
      "status": "done",
      "candidate": "msdgs_p4detail_boxres",
      "seed": 0,
      "weights": "/root/autodl-tmp/neu-det-yolo26/runs_msdgs_boxres_e250/msdgs_p4detail_boxres_seed0/weights/best.pt",
      "params_fused": 1780730,
      "gflops_eval_graph": 4.6560768,
      "gflops_fused_inference": 4.085043199999999,
      "val": {
        "precision": 0.7192789858286776,
        "recall": 0.688721038797746,
        "map50": 0.7232998879039547,
        "map50_95": 0.3937522332225903,
        "infer_ms": 2.2397511762877302,
        "fps_infer_only": 446.4781671227638,
        "per_class": {
          "crazing": {
            "precision": 0.4942708357383957,
            "recall": 0.37755102040816324,
            "map50": 0.43278589330881867,
            "map50_95": 0.17152305048478936
          },
          "inclusion": {
            "precision": 0.7418920077713759,
            "recall": 0.7169811320754716,
            "map50": 0.7543990946916521,
            "map50_95": 0.3841373891790352
          },
          "patches": {
            "precision": 0.8060394901015195,
            "recall": 0.8823529411764706,
            "map50": 0.8869886074949328,
            "map50_95": 0.5534734666967182
          },
          "pitted_surface": {
            "precision": 0.7700447856808557,
            "recall": 0.7323943661971831,
            "map50": 0.7570616607738763,
            "map50_95": 0.44301513987180224
          },
          "rolled-in_scale": {
            "precision": 0.6865092890687574,
            "recall": 0.5393258426966292,
            "map50": 0.6093990901166468,
            "map50_95": 0.28452384783108076
          },
          "scratches": {
            "precision": 0.816917506611161,
            "recall": 0.8837209302325582,
            "map50": 0.8991649810378016,
            "map50_95": 0.5258405052721156
          }
        }
      },
      "test": {
        "precision": 0.7097661148892613,
        "recall": 0.6945736831626942,
        "map50": 0.7244952256894371,
        "map50_95": 0.3848112615710157,
        "infer_ms": 1.853696042578356,
        "fps_infer_only": 539.46276899263,
        "per_class": {
          "crazing": {
            "precision": 0.4789922871619258,
            "recall": 0.4,
            "map50": 0.42538289211207986,
            "map50_95": 0.1500884523007276
          },
          "inclusion": {
            "precision": 0.7380323306694887,
            "recall": 0.70625,
            "map50": 0.7364531359727964,
            "map50_95": 0.3882908981621838
          },
          "patches": {
            "precision": 0.8084137894364065,
            "recall": 0.89937106918239,
            "map50": 0.895855610337903,
            "map50_95": 0.5625492065301236
          },
          "pitted_surface": {
            "precision": 0.7991173525100388,
            "recall": 0.7323943661971831,
            "map50": 0.7829996803939684,
            "map50_95": 0.4347658571717825
          },
          "rolled-in_scale": {
            "precision": 0.6029394128420754,
            "recall": 0.5581395348837209,
            "map50": 0.5783824773089554,
            "map50_95": 0.24761254353057244
          },
          "scratches": {
            "precision": 0.8311015167156331,
            "recall": 0.8712871287128713,
            "map50": 0.9278975580109199,
            "map50_95": 0.5255606117307042
          }
        }
      },
      "evaluated_at": "2026-07-30 08:05:02",
      "eval_pid": 7140,
      "cfg": "/root/autodl-tmp/neu-det-yolo26/generated_models_msdgs_boxres_e250/msdgs_p4detail_boxres.yaml",
      "gate": {
        "pass": false,
        "checks": {
          "map50_95_gt_0.401794": false,
          "map50_ge_0.722411": true,
          "target_class_gain_ge_0.01": false,
          "crazing_drop_le_0.02": false
        },
        "deltas_vs_msdgs": {
          "map50": -0.00791608434584512,
          "map50_95": -0.014025955703675819,
          "crazing": -0.026981130219452615,
          "pitted_surface": -0.033929420018742384,
          "rolled-in_scale": -0.010817304296386887
        }
      }
    }
  },
  "finished_at": "2026-07-30 08:05:03"
}
```
