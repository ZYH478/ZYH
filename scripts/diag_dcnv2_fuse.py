import copy
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


def build():
    model = YOLO(str(CFG), task="detect").model
    baseline = YOLO(str(BASE_YAML), task="detect").model
    model.load_state_dict(baseline.state_dict(), strict=False)
    transfer_standard_to_dcnv2(model, baseline.state_dict())
    return model.eval()


# --- CPU fuse via ultralytics ---
log("=== CPU ultralytics fuse ===")
m_cpu = copy.deepcopy(build())
try:
    m_cpu.fuse(verbose=False)
    log("cpu fuse ok")
except Exception as e:
    log("cpu fuse EXC", repr(e))

# --- GPU fuse via ultralytics ---
if torch.cuda.is_available():
    log("=== GPU ultralytics fuse ===")
    m_gpu = copy.deepcopy(build()).cuda()
    try:
        m_gpu.fuse(verbose=False)
        log("gpu fuse ok")
        with torch.no_grad():
            out = m_gpu(torch.zeros(1, 3, 640, 640).cuda())
        log("gpu fused fwd ok", tuple(out[0].shape) if isinstance(out, (list, tuple)) else tuple(out.shape))
    except Exception as e:
        log("gpu fuse EXC", repr(e))
else:
    log("no cuda")

log("DONE")
