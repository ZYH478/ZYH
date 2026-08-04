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


log("=== CPU get_flops ===")
m = copy.deepcopy(build())
m.fuse(verbose=False)
try:
    g = get_flops(m, imgsz=640)
    log("cpu get_flops ok", g)
except Exception as e:
    log("cpu get_flops EXC", repr(e))

if torch.cuda.is_available():
    log("=== GPU get_flops ===")
    m2 = copy.deepcopy(build()).cuda()
    m2.fuse(verbose=False)
    try:
        g2 = get_flops(m2, imgsz=640)
        log("gpu get_flops ok", g2)
    except Exception as e:
        log("gpu get_flops EXC", repr(e))

log("DONE")
