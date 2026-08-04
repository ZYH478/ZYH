#!/usr/bin/env python
"""打印 gsdown 候选 YAML 的 backbone + head 结构，确认 DWR 改动点。"""
import glob
import yaml

# gsdown 的 doc 由 stage3 gsconv head 生成，找一个 gsdown 的 yaml
cands = glob.glob("/root/autodl-tmp/neu-det-yolo26/generated_models_*/*gsdown*.yaml")
print("=== candidate yamls ===")
for c in cands:
    print(c)
if not cands:
    raise SystemExit("no gsdown yaml found")

path = cands[0]
print(f"\n=== USING {path} ===")
d = yaml.safe_load(open(path))
print("--- backbone ---")
for i, r in enumerate(d["backbone"]):
    print(i, r[2], r[3])
print("--- head ---")
base = len(d["backbone"])
for j, r in enumerate(d["head"]):
    print(base + j, r[0], r[2], r[3])
