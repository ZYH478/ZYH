import copy
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
BASE_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
CFG = ROOT / "generated_models_msdgs_backbone_dcnv2_e250" / "msdgs_bb_p4_dcnv2.yaml"


def log(*a):
    print(*a, flush=True)


from ultralytics import YOLO
from ultralytics.nn.modules.yolo26_msdgs_backbone_dcnv2 import transfer_standard_to_dcnv2
from ultralytics.utils.torch_utils import get_flops


def build():
    model = YOLO(str(CFG), task="detect").model
    baseline = YOLO(str(BASE_YAML), task="detect").model
    model.load_state_dict(baseline.state_dict(), strict=False)
    transfer_standard_to_dcnv2(model, baseline.state_dict())
    return model.eval()


assert torch.cuda.is_available(), "need cuda"
log("=== GPU get_flops ===")
m = copy.deepcopy(build()).cuda()
m.fuse(verbose=False)
fused_params = int(sum(p.numel() for p in m.parameters()))
g = get_flops(m, imgsz=640)
log("gpu get_flops ok", g, "fused_params", fused_params)
log("DONE")
