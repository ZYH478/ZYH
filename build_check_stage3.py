#!/usr/bin/env python
"""只 build 不训练，验证 stage3 GSConv 三个候选能实例化且 end2end/reg_max 正确。"""
import sys
import torch

import train_yolo26_stage3_gsconv as s3
import train_yolo26_module_sweep as sweep

sweep.ensure_modules()
import subprocess
subprocess.run([sys.executable, "install_gsconv_modules.py"], check=True)

from ultralytics import YOLO

specs = s3.stage3_specs()
s3.set_sweep_outputs()

ok = True
for name, spec in specs.items():
    print(f"\n===== BUILD_CHECK {name} =====")
    try:
        cfg = sweep.dump_yaml(name, spec["doc"])
        model = YOLO(str(cfg)).load(str(sweep.WEIGHTS))
        m = model.model
        end2end = getattr(m, "end2end", None)
        det = m.model[-1]
        reg_max = getattr(det, "reg_max", None)
        nl = getattr(det, "nl", None)
        params = int(sum(p.numel() for p in m.parameters()))
        x = torch.zeros(1, 3, 640, 640)
        m.eval()
        with torch.no_grad():
            out = m(x)
        shapes = out[0].shape if isinstance(out, (list, tuple)) else "dict"
        print(f"OK end2end={end2end} reg_max={reg_max} nl={nl} "
              f"n_layers={len(m.model)} params={params}")
        print(f"   forward_out={[shapes, type(out).__name__]}")
    except Exception as exc:
        ok = False
        print(f"BUILD_FAIL {name}: {exc!r}")

print("\nBUILD_CHECK_ALL_OK" if ok else "\nBUILD_CHECK_HAD_FAILURES")
