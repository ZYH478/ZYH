import faulthandler
import os
from pathlib import Path

import torch

faulthandler.enable()

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
BASE_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
CFG = ROOT / "generated_models_msdgs_backbone_dcnv2_e250" / "msdgs_bb_p4_dcnv2.yaml"
OFFICIAL = ROOT / "yolo26n.pt"


def log(*a):
    print(*a, flush=True)


from ultralytics import YOLO
from ultralytics.nn.modules.yolo26_msdgs_backbone_dcnv2 import transfer_standard_to_dcnv2

log("STEP1 YOLO(cfg).load(official)")
model = YOLO(str(CFG), task="detect").load(str(OFFICIAL))
log("STEP1 ok")

log("STEP2 baseline load")
baseline_std = YOLO(str(BASE_YAML), task="detect").load(str(OFFICIAL))
log("STEP2 ok")

log("STEP3 transfer")
n = transfer_standard_to_dcnv2(model.model, baseline_std.model.state_dict())
log("STEP3 ok transferred", n)
del baseline_std

log("STEP4 model.info() [CPU flops path]")
try:
    model.model.info(verbose=False)
    log("STEP4 ok")
except Exception as e:
    log("STEP4 EXC", repr(e))

log("STEP5 get_flops CPU")
from ultralytics.utils.torch_utils import get_flops
try:
    g = get_flops(model.model, imgsz=640)
    log("STEP5 ok", g)
except Exception as e:
    log("STEP5 EXC", repr(e))

log("ALL DONE")
