import os
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
BASE_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
CFG = ROOT / "generated_models_msdgs_backbone_dcnv2_e250" / "msdgs_bb_p4_dcnv2.yaml"


def log(*a):
    print(*a, flush=True)


from ultralytics import YOLO
from ultralytics.nn.modules.yolo26_msdgs_backbone_dcnv2 import (
    DCNv2Unit,
    transfer_standard_to_dcnv2,
)

log("STEP build cfg model")
model = YOLO(str(CFG), task="detect").model
log("STEP build baseline")
baseline = YOLO(str(BASE_YAML), task="detect").model
log("STEP load_state_dict")
model.load_state_dict(baseline.state_dict(), strict=False)
log("STEP transfer")
n = transfer_standard_to_dcnv2(model, baseline.state_dict())
log("transferred", n)
model.eval()
baseline.eval()
x = torch.randn(1, 3, 640, 640)
log("STEP baseline forward")
with torch.no_grad():
    baseline(x.clone())
log("baseline fwd ok")
log("STEP candidate forward (deform on CPU)")
with torch.no_grad():
    model(x.clone())
log("candidate fwd ok")
log("STEP isolate: run one DCNv2Unit at layer6 real size on CPU")
unit = next(m for m in model.model[6].modules() if isinstance(m, DCNv2Unit))
c = unit.offset_mask.in_channels
log("unit in_channels", c)
with torch.no_grad():
    out = unit(torch.randn(1, c, 80, 80))
log("unit fwd ok", tuple(out.shape))

import copy

log("STEP deepcopy")
fused = copy.deepcopy(model).eval()
log("deepcopy ok")
log("STEP fuse")
fused.fuse()
log("fuse ok")
log("STEP fused forward")
with torch.no_grad():
    fused(torch.zeros(1, 3, 640, 640))
log("fused fwd ok")
log("STEP get_flops")
from ultralytics.utils.torch_utils import get_flops
g = get_flops(fused, imgsz=640)
log("get_flops ok", g)
