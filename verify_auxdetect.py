#!/usr/bin/env python
"""AuxDetect 管线验证：build + 短跑 3 epoch + fuse 推理，铺开 6 模型前的 gate。

验证点：
1. build 基线+AuxDetect：可实例化、end2end=True、reg_max=1、nl=3、输出格式对。
2. 短跑 3 epoch：loss 不 NaN/不崩，训练态 forward 返回 dict 含 one2many/one2one/aux。
3. fuse 后：aux/one2many 分支被删，推理输出 [1,300,6]，参数与普通 Detect 一致。
"""
import sys
import copy

sys.path.insert(0, "/root/autodl-tmp/neu-det-yolo26")

import torch
import train_yolo26_module_sweep as sweep


def to_aux(doc):
    """把 head 最后一行的 Detect 换成 AuxDetect。"""
    d = copy.deepcopy(doc)
    for row in d["head"]:
        if row[2] == "Detect":
            row[2] = "AuxDetect"
    return d


def main():
    from ultralytics import YOLO
    from ultralytics.nn.modules.yolo26_auxdetect import AuxDetect

    sweep.GEN_DIR = sweep.ROOT / "generated_models_auxdetect_verify"
    doc = sweep.base_doc(sweep.BASE_BACKBONE, sweep.make_head())
    doc_aux = to_aux(doc)
    cfg = sweep.dump_yaml("verify_base_auxdetect", doc_aux)
    print("cfg", cfg)

    # 1. build
    model = YOLO(str(cfg))
    m = model.model.model[-1]
    print("HEAD_TYPE", type(m).__name__)
    print("IS_AUXDETECT", isinstance(m, AuxDetect))
    print("end2end", getattr(m, "end2end", None), "reg_max", m.reg_max, "nl", m.nl)
    print("has_aux_cv2", hasattr(m, "aux_cv2") and m.aux_cv2 is not None)
    n_params = sum(p.numel() for p in model.model.parameters())
    print("PARAMS_WITH_AUX", n_params)

    # 2. 训练态 forward：确认返回 dict 含 aux
    model.model.train()
    x = torch.randn(2, 3, 640, 640)
    out = model.model(x)
    if isinstance(out, dict):
        print("TRAIN_FORWARD_KEYS", sorted(out.keys()))
    else:
        # DetectionModel.forward 训练态需要 batch；直接调 head
        feats = model.model.model[:-1]
        print("TRAIN_FORWARD_TYPE", type(out).__name__)

    # 3. 短跑 3 epoch
    print("=== SHORT TRAIN 3 epoch ===")
    res = model.train(
        data=str(sweep.DATA), epochs=3, imgsz=640, batch=16, workers=4,
        seed=0, device=0, project=str(sweep.ROOT / "runs_auxdetect_verify"),
        name="verify_base_auxdetect", exist_ok=True, verbose=False, cache=False,
    )
    print("TRAIN_OK")

    # 4. fuse 后推理
    best = YOLO(str(sweep.ROOT / "runs_auxdetect_verify" / "verify_base_auxdetect" / "weights" / "best.pt"))
    mf = best.model.model[-1]
    best.model.fuse()
    print("AFTER_FUSE_cv2_none", mf.cv2 is None, "aux_cv2_none", getattr(mf, "aux_cv2", "MISSING") is None)
    best.model.eval()
    with torch.no_grad():
        y = best.model(torch.randn(1, 3, 640, 640))
    yt = y[0] if isinstance(y, tuple) else y
    print("INFER_OUT_SHAPE", list(yt.shape))
    n_params_fused = sum(p.numel() for p in best.model.parameters())
    print("PARAMS_FUSED", n_params_fused)
    print("VERIFY_DONE")


if __name__ == "__main__":
    sys.exit(main())
