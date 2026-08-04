#!/usr/bin/env python
"""iter19 build 验证：确认 4 个候选都能构建成功（end2end/reg_max/nl=3/输出[1,300,6]），
不训练。像 iter18 那样先 build 再训，避免训到一半才暴露模块 bug。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_backbone_modules.py
    python build_check_iter19.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

import train_yolo26_iter19_backbone as it19
import train_yolo26_module_sweep as sweep


def ensure_modules() -> None:
    here = Path(__file__).resolve().parent
    for installer in ("install_yolo26_exp_modules.py", "install_backbone_modules.py"):
        subprocess.run([sys.executable, str(here / installer)], check=True)


def check_one(name: str, spec: dict) -> dict:
    from ultralytics import YOLO

    cfg = it19.GEN_DIR / f"{name}.yaml"
    it19.GEN_DIR.mkdir(parents=True, exist_ok=True)
    import yaml

    cfg.write_text(yaml.safe_dump(spec["doc"], sort_keys=False, allow_unicode=True), encoding="utf-8")
    model = YOLO(str(cfg))
    m = model.model
    det = m.model[-1]
    end2end = getattr(det, "end2end", None)
    reg_max = getattr(det, "reg_max", None)
    nl = getattr(det, "nl", None)

    m.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 640, 640)
        y = m(x)
    if isinstance(y, (list, tuple)):
        shapes = [tuple(t.shape) for t in y if hasattr(t, "shape")]
    else:
        shapes = [tuple(y.shape)]

    n_params = sum(p.numel() for p in m.parameters())
    return {
        "name": name,
        "end2end": end2end,
        "reg_max": reg_max,
        "nl": nl,
        "out_shapes": shapes,
        "params": n_params,
    }


def main() -> int:
    it19.set_sweep_outputs()
    ensure_modules()
    all_specs = it19.specs()
    ok = True
    for name, spec in all_specs.items():
        try:
            info = check_one(name, spec)
            end2end_ok = info["end2end"] is True
            reg_ok = info["reg_max"] == 1
            nl_ok = info["nl"] == 3
            passed = end2end_ok and reg_ok and nl_ok
            ok = ok and passed
            print(
                f"[{'OK' if passed else 'FAIL'}] {name} "
                f"end2end={info['end2end']} reg_max={info['reg_max']} nl={info['nl']} "
                f"params={info['params']} out={info['out_shapes']}"
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {name} EXCEPTION: {exc!r}")
    print("BUILD_CHECK_ITER19_" + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
