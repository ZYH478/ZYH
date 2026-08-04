"""iter34 收尾:定位并还原被 O2M-QFL 打补丁的 loss.py,清除环境污染。
在远端执行:python scripts/restore_loss_o2mqfl.py
"""
import os
import shutil

MARKER = "# === iter34 O2M-QFL patch ==="
BAK_SUFFIX = ".o2mqfl_bak"


def find_files(root, name_endswith):
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(name_endswith):
                hits.append(os.path.join(dirpath, fn))
    return hits


def main():
    # 1) 找到 site-packages 下的 loss.py（只扫 conda 环境目录，加速）
    search_roots = [
        "/root/miniconda3/envs/yolo26/lib",
        "/root/autodl-tmp/neu-det-yolo26",
    ]
    loss_files = []
    bak_files = []
    for r in search_roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _dirs, files in os.walk(r):
            for fn in files:
                full = os.path.join(dirpath, fn)
                if fn == "loss.py" and dirpath.endswith(os.path.join("ultralytics", "utils")):
                    loss_files.append(full)
                if fn == ("loss.py" + BAK_SUFFIX.lstrip(".")) or fn.endswith(BAK_SUFFIX):
                    bak_files.append(full)

    print("=== loss.py found ===")
    for f in loss_files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        cnt = txt.count(MARKER)
        print(f"{f}  marker={cnt}")
    print("=== backups found ===")
    for f in bak_files:
        print(f)

    # 2) 还原
    for lf in loss_files:
        with open(lf, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        if MARKER not in txt:
            print(f"[skip] no marker: {lf}")
            continue
        bak = lf + BAK_SUFFIX
        if os.path.exists(bak):
            shutil.copy2(bak, lf)
            print(f"[restored] {lf}  <-  {bak}")
        else:
            # 无备份则截断补丁段（补丁是 append 到文件末尾的）
            idx = txt.find(MARKER)
            cleaned = txt[:idx].rstrip() + "\n"
            with open(lf, "w", encoding="utf-8") as fh:
                fh.write(cleaned)
            print(f"[truncated-patch] {lf}  (no bak, cut from marker)")

    # 3) 复核
    print("=== verify (marker should be 0) ===")
    for lf in loss_files:
        with open(lf, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        print(f"{lf}  marker={txt.count(MARKER)}")


if __name__ == "__main__":
    main()
