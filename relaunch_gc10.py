#!/usr/bin/env python
"""诊断旧 GC10 训练死因 + 重启训练 + 确认进程存活。

一次调用完成，避开会话抖动与引号地狱：
1. 尝试读旧 train.log 尾部（若存在，输出死因线索）
2. 清理旧 run 目录（保留 report 若已 done）
3. setsid nohup 重启 train_gc10.py
4. 循环 25 秒确认进程存活 + 打印新日志尾部
"""
import os
import subprocess
import time
import glob

ROOT = "/root/autodl-tmp/neu-det-yolo26"
RUNS = os.path.join(ROOT, "runs_gc10_e250")
LOG = os.path.join(RUNS, "train.log")

os.chdir(ROOT)

# --- 1. 读旧日志死因 ---
print("=== OLD_LOG_TAIL ===")
try:
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    print(data[-3000:])
    print("=== OLD_LOG_LEN ===", len(data))
except Exception as e:
    print("OLD_LOG_ERR", repr(e))

print("=== RUNS_LIST ===")
try:
    for p in sorted(glob.glob(RUNS + "/*")):
        print(os.path.basename(p))
except Exception as e:
    print("LIST_ERR", repr(e))

# --- 2. 检查是否已有进程 ---
proc = subprocess.run(["pgrep", "-af", "train_gc10"], capture_output=True, text=True).stdout.strip()
if proc:
    print("=== ALREADY_RUNNING ===")
    print(proc)
    raise SystemExit(0)

# --- 3. 重启 ---
print("=== RELAUNCH ===")
launch = (
    "source /root/miniconda3/etc/profile.d/conda.sh && "
    "conda activate yolo26 && "
    "cd /root/autodl-tmp/neu-det-yolo26 && "
    "python install_yolo26_exp_modules.py >/dev/null 2>&1 ; "
    "python install_backbone_modules.py >/dev/null 2>&1 ; "
    "mkdir -p runs_gc10_e250 && "
    "setsid nohup python -u train_gc10.py > runs_gc10_e250/train.log 2>&1 < /dev/null & "
    "echo PID=$!"
)
r = subprocess.run(["bash", "-lc", launch], capture_output=True, text=True)
print(r.stdout.strip())
print(r.stderr.strip()[-500:] if r.stderr.strip() else "")

# --- 4. 存活确认 ---
for i in range(5):
    time.sleep(5)
    alive = subprocess.run(["pgrep", "-af", "train_gc10"], capture_output=True, text=True).stdout.strip()
    print(f"=== CHECK {i} (t+{(i+1)*5}s) ALIVE={'YES' if alive else 'NO'} ===")
    if alive:
        print(alive.splitlines()[0])

print("=== NEW_LOG_TAIL ===")
try:
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        print(f.read()[-1500:])
except Exception as e:
    print("NEW_LOG_ERR", repr(e))
