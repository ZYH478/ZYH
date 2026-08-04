#!/usr/bin/env python
"""读取 GC10 训练日志尾部 + 进程/GPU 状态，避开引号地狱与会话抖动。"""
import os
import glob
import subprocess

ROOT = "/root/autodl-tmp/neu-det-yolo26"
RUNS = os.path.join(ROOT, "runs_gc10_e250")
LOG = os.path.join(RUNS, "train.log")

print("===RUNS_DIR_LIST===")
try:
    for f in sorted(glob.glob(RUNS + "/**", recursive=True)):
        try:
            sz = os.path.getsize(f)
        except Exception:
            sz = -1
        print(sz, f)
except Exception as e:
    print("RUNS_ERR", repr(e))

try:
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    print("===LOG_TAIL(4000)===")
    print(data[-4000:])
    print("===LOG_LEN===", len(data))
except Exception as e:
    print("LOG_ERR", repr(e))

print("===PROC===")
print(subprocess.run(["pgrep", "-af", "train_gc10"], capture_output=True, text=True).stdout or "(none)")
print("===GPU===")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout)
