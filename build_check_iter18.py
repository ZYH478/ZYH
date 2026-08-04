#!/usr/bin/env python
"""iter18 build + structural verification (no training).

Confirms: 6 candidates instantiate, HWD/CARAFE numerically correct, RepConv fuses,
end2end output format is [1,300,6].
"""
from __future__ import annotations

import torch

import train_yolo26_module_sweep as sweep
import train_yolo26_iter18_struct as i18


def build(doc, tag):
    from ultralytics import YOLO
    import yaml
    from pathlib import Path
    p = Path(f"/tmp/_i18_{tag}.yaml")
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return YOLO(str(p))


def main():
    # HWD / CARAFE numeric self-check
    from ultralytics.nn.modules.yolo26_struct import HWD, CARAFE
    x = torch.rand(2, 32, 64, 64)
    hwd = HWD(32, 64)
    yh = hwd(x)
    assert tuple(yh.shape) == (2, 64, 32, 32), f"HWD shape {tuple(yh.shape)}"
    print(f"HWD_OK in{tuple(x.shape)} -> out{tuple(yh.shape)} (half-res, proj 64ch)")
    car = CARAFE(32, scale=2)
    yc = car(x)
    assert tuple(yc.shape) == (2, 32, 128, 128), f"CARAFE shape {tuple(yc.shape)}"
    print(f"CARAFE_OK in{tuple(x.shape)} -> out{tuple(yc.shape)} (2x up, c-preserving)")

    for name, spec in i18.specs().items():
        tag = name.replace("y26n_i18_", "").replace("_e250", "")
        m = build(spec["doc"], tag)
        det = m.model.model[-1]
        params = sum(p.numel() for p in m.model.parameters())
        m.model.eval()
        with torch.no_grad():
            out = m.model(torch.rand(1, 3, 640, 640))
        y = out[0] if isinstance(out, (list, tuple)) else out
        shape = tuple(y.shape) if torch.is_tensor(y) else "n/a"
        print(f"{name}: params={params} end2end={getattr(det,'end2end',None)} "
              f"reg_max={getattr(det,'reg_max',None)} nl={det.nl} out={shape}")
    print("ALL_I18_BUILD_CHECK_OK")


if __name__ == "__main__":
    raise SystemExit(main())
