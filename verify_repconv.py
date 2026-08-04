#!/usr/bin/env python
"""验证 RepConv 已进 base_modules，且 iter18 剩余 4 候选能 build 通过。"""
import re
import pathlib
import ultralytics

t = pathlib.Path(ultralytics.__file__).resolve().parent / "nn" / "tasks.py"
s = t.read_text(encoding="utf-8")
m = re.search(r"base_modules = frozenset\(\s*\{(.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", s, re.S)
body = m.group(1)
print("HWD in base_modules   :", "HWD" in body)
print("RepConv in base_modules:", bool(re.search(r"(?m)^\s*RepConv,\s*$", body)))

# build 剩余 4 候选
import sys
sys.path.insert(0, "/root/autodl-tmp/neu-det-yolo26")
import train_yolo26_iter18_struct as it18
from ultralytics import YOLO

it18.set_sweep_outputs()
it18.GEN_DIR.mkdir(parents=True, exist_ok=True)
specs = it18.specs()
targets = [
    "y26n_i18_gsdown_repconv_e250",
    "y26n_i18_winner_hwd_e250",
    "y26n_i18_winner_carafe_e250",
    "y26n_i18_winner_repconv_e250",
]
import yaml
for name in targets:
    doc = specs[name]["doc"]
    cfg = it18.GEN_DIR / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    try:
        model = YOLO(str(cfg))
        det = model.model.model[-1]
        info = model.model.info(verbose=False)
        params = info[1] if isinstance(info, tuple) else None
        e2e = getattr(det, "end2end", None)
        rmax = getattr(det, "reg_max", None)
        nl = getattr(det, "nl", None)
        print(f"OK  {name:32s} params={params} end2end={e2e} reg_max={rmax} nl={nl}")
    except Exception as exc:
        print(f"FAIL {name:32s} {type(exc).__name__}: {exc}")
