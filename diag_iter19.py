import subprocess
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("===GEN DIR===")
print(sh("ls -la /root/autodl-tmp/neu-det-yolo26/generated_models_iter19_backbone_e250/ 2>&1"))
print("===ITER19 LOGS===")
print(sh("ls -la /root/autodl-tmp/neu-det-yolo26/*iter19* /root/autodl-tmp/neu-det-yolo26/*i19* 2>&1"))
print("===ALL PYTHON PROC===")
print(sh("ps aux | grep -E 'python|train' | grep -v grep 2>&1"))
print("===TOP CPU===")
print(sh("ps aux --sort=-%cpu | head -8 2>&1"))
print("===RECENT LOG TAILS (find any iter19 log)===")
print(sh("find /root/autodl-tmp/neu-det-yolo26 -name '*.log' -newermt '2026-07-26 07:15' 2>&1"))
