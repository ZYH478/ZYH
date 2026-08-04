#!/usr/bin/env python
"""Generate a lossless GIIS copy of NEU-DET for the MSDGS input ablation.

The network-facing RGB tensor is [mean_blur(gray), mean_blur(gray), gray]. Images
are written as PNG. Because OpenCV writes BGR and Ultralytics converts BGR to
RGB, the file is stored as BGR=[gray, blur, blur]. Network parameters and model
FLOPs are unchanged, while deployment still has a small preprocessing cost.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
SRC_DS = ROOT / "dataset"
DST_DS = ROOT / "dataset_giis"
SRC_YAML = SRC_DS / "neu-det.yaml"
DST_YAML = SRC_DS / "neu-det-giis.yaml"
AUDIT_JSON = DST_DS / "giis_audit.json"

BLUR_K = 3
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_DIRS = ("train", "valid", "test")


def giis_transform(img_bgr: np.ndarray) -> np.ndarray:
    """Return lossless-file BGR whose Ultralytics RGB view is [blur, blur, gray]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (BLUR_K, BLUR_K))
    # Ultralytics reverses BGR -> RGB, so persist BGR=[gray, blur, blur].
    return np.stack([gray, blur, blur], axis=-1)


def source_channel_audit(max_per_split: int = 64) -> dict:
    """Measure whether source images are genuinely grayscale-like before conversion."""
    result: dict[str, dict] = {}
    for split in SPLIT_DIRS:
        image_dir = SRC_DS / split / "images"
        paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS][:max_per_split]
        sample_stats = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
            delta = np.maximum.reduce((np.abs(b - g), np.abs(g - r), np.abs(b - r)))
            sample_stats.append({
                "file": path.name,
                "max_channel_delta": int(delta.max()),
                "mean_channel_delta": float(delta.mean()),
            })
        result[split] = {
            "samples": len(sample_stats),
            "max_channel_delta": max((x["max_channel_delta"] for x in sample_stats), default=None),
            "mean_channel_delta": float(np.mean([x["mean_channel_delta"] for x in sample_stats]))
            if sample_stats else None,
        }
        print("SOURCE_AUDIT", split, result[split])
    return result


def reset_destination() -> None:
    """Recreate only the generated GIIS tree to prevent stale mixed extensions."""
    resolved_root = ROOT.resolve()
    resolved_dst = DST_DS.resolve()
    if resolved_dst.parent != resolved_root:
        raise RuntimeError(f"unsafe GIIS destination: {resolved_dst}")
    if DST_DS.exists():
        shutil.rmtree(DST_DS)
    DST_DS.mkdir(parents=True, exist_ok=True)


def process_split(split: str) -> tuple[int, int]:
    src_img_dir = SRC_DS / split / "images"
    src_lbl_dir = SRC_DS / split / "labels"
    dst_img_dir = DST_DS / split / "images"
    dst_lbl_dir = DST_DS / split / "labels"
    if not src_img_dir.is_dir():
        raise FileNotFoundError(src_img_dir)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_img = 0
    for path in sorted(src_img_dir.iterdir()):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"unreadable image: {path}")
        out_path = dst_img_dir / f"{path.stem}.png"
        if not cv2.imwrite(str(out_path), giis_transform(img), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to write: {out_path}")
        n_img += 1

    n_lbl = 0
    for path in sorted(src_lbl_dir.glob("*.txt")):
        shutil.copy2(path, dst_lbl_dir / path.name)
        n_lbl += 1
    print(f"SPLIT {split}: images={n_img} labels={n_lbl}")
    if n_img != n_lbl:
        raise RuntimeError(f"split {split} image/label mismatch: {n_img}!={n_lbl}")
    return n_img, n_lbl


def write_yaml() -> None:
    src = yaml.safe_load(SRC_YAML.read_text(encoding="utf-8"))
    doc = {
        "path": str(DST_DS),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": int(src.get("nc", len(src["names"]))),
        "names": src["names"],
    }
    DST_YAML.write_text(
        "# GIIS-preprocessed NEU-DET; network RGB=[blur, blur, original_gray]\n"
        + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"WROTE_YAML {DST_YAML}")


def verify(max_per_split: int = 16) -> int:
    """Verify persisted files and the post-Ultralytics RGB channel semantics."""
    ok = True
    verification: dict[str, dict] = {}
    for split in SPLIT_DIRS:
        image_dir = DST_DS / split / "images"
        paths = sorted(image_dir.glob("*.png"))
        checked = paths[:max_per_split]
        split_ok = bool(checked)
        n_semantic_ok = 0
        for path in checked:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                split_ok = False
                continue
            rgb = bgr[..., ::-1]
            first_two_equal = np.array_equal(rgb[..., 0], rgb[..., 1])
            original_differs = not np.array_equal(rgb[..., 0], rgb[..., 2])
            semantic_ok = first_two_equal and original_differs
            n_semantic_ok += int(semantic_ok)
            split_ok = split_ok and semantic_ok
        verification[split] = {
            "images": len(paths),
            "checked": len(checked),
            "semantic_ok": n_semantic_ok,
            "ok": split_ok,
        }
        print("VERIFY", split, verification[split])
        ok = ok and split_ok
    audit = {}
    if AUDIT_JSON.exists():
        try:
            audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            audit = {}
    audit.update({"blur_kernel": BLUR_K, "network_rgb": ["blur", "blur", "original_gray"],
                  "verification": verification})
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("VERIFY_OK" if ok else "VERIFY_FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if args.audit_only:
        print(json.dumps(source_channel_audit(), indent=2, ensure_ascii=False))
        return 0
    if args.verify:
        return verify()

    print(f"SRC {SRC_DS}")
    print(f"DST {DST_DS}")
    print(f"BLUR_K {BLUR_K}")
    source_audit = source_channel_audit()
    reset_destination()
    total = 0
    for split in SPLIT_DIRS:
        n_img, _ = process_split(split)
        total += n_img
    write_yaml()
    AUDIT_JSON.write_text(json.dumps({"source": source_audit}, indent=2, ensure_ascii=False), encoding="utf-8")
    rc = verify()
    print(f"GIIS_DONE total_images={total}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
