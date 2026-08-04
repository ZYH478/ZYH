#!/usr/bin/env python
"""准备三数据集迁移性验证数据并执行严格数据 gate。

产物位于 runs_generalization_yolo26_msdgs_e250/datasets/，只生成绝对路径
image list 与 data YAML，不复制图像。压缩包解压到 generalization_datasets/raw/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import yaml

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STORAGE = Path(os.environ.get("YOLO26_DATA_STORAGE", "/root/autodl-fs"))
PROJECT = ROOT / "runs_generalization_yolo26_msdgs_e250"
RAW = ROOT / "generalization_datasets" / "raw"
OUT = PROJECT / "datasets"
PROFILE = PROJECT / "dataset_profile.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASETS = {
    "gc10": STORAGE / "GC10-DET.v2i.yolov8.zip",
    "uwwt": STORAGE / "UWWT-Dataset-1500.zip",
    "steel_weld": STORAGE / "钢管焊缝缺陷检测数据集.zip",
}
NAMES = {
    "gc10": ["10_yaozhe", "1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban", "6_siban", "7_yiwu", "8_yahen", "9_zhehen"],
    "uwwt": ["rubber char", "wire char", "wire broken", "side shedding", "wire stains"],
    "steel_weld": ["air-hole", "bite-edge", "broken-arc", "crack", "hollow-bead", "overlap", "slag-inclusion", "unfused"],
}


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> None:
    marker = dest / ".extract_complete.json"
    zip_stat = zip_path.stat()
    expected = {"zip": str(zip_path), "size": zip_stat.st_size, "mtime_ns": zip_stat.st_mtime_ns}
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in expected.items()):
            print(f"EXTRACT_SKIP {zip_path.name} -> {dest}")
            return
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    print(f"EXTRACT_START {zip_path} -> {dest}", flush=True)
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe zip member: {info.filename}")
        zf.extractall(dest)
    atomic_json(marker, expected)
    print(f"EXTRACT_DONE {zip_path.name}", flush=True)


def image_files(path: Path) -> list[Path]:
    return sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_for_image(img: Path) -> Path:
    parts = list(img.parts)
    indexes = [i for i, x in enumerate(parts) if x == "images"]
    if not indexes:
        raise RuntimeError(f"image path has no /images/ segment: {img}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_label(path: Path, nc: int) -> Counter:
    if not path.is_file():
        raise FileNotFoundError(f"missing label: {path}")
    counts = Counter()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(fields)}")
        cls_float, x, y, w, h = map(float, fields)
        cls = int(cls_float)
        if cls_float != cls or not 0 <= cls < nc:
            raise ValueError(f"{path}:{lineno}: invalid class {cls_float} for nc={nc}")
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"{path}:{lineno}: non-finite bbox")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{lineno}: invalid normalized bbox {(x, y, w, h)}")
        counts[cls] += 1
    return counts


def write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p.as_posix()}\n" for p in images), encoding="utf-8")


def write_yaml(dataset: str, root: Path, splits: dict[str, list[Path]]) -> Path:
    ds_out = OUT / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    lists = {}
    for split, images in splits.items():
        list_path = ds_out / f"{split}.txt"
        write_list(list_path, images)
        lists[split] = str(list_path.resolve())
    doc = {
        "path": str(root.resolve()),
        "train": lists["train"],
        "val": lists["val"],
        "test": lists["test"],
        "nc": len(NAMES[dataset]),
        "names": NAMES[dataset],
    }
    out = OUT / f"{dataset}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def validate(dataset: str, splits: dict[str, list[Path]]) -> dict:
    nc = len(NAMES[dataset])
    seen: dict[Path, str] = {}
    result = {}
    for split in ("train", "val", "test"):
        images = splits[split]
        inst = Counter()
        empty = 0
        for img in images:
            if not img.is_file():
                raise FileNotFoundError(f"missing image: {img}")
            if img in seen:
                raise RuntimeError(f"split overlap: {img} in {seen[img]} and {split}")
            seen[img] = split
            c = parse_label(label_for_image(img), nc)
            if not c:
                empty += 1
            inst.update(c)
        result[split] = {
            "images": len(images),
            "empty_labels": empty,
            "instances": {NAMES[dataset][i]: inst.get(i, 0) for i in range(nc)},
        }
    result["image_overlap_count"] = 0
    result["all_classes_in_test"] = all(result["test"]["instances"][n] > 0 for n in NAMES[dataset])
    if not result["all_classes_in_test"]:
        raise RuntimeError(f"{dataset}: test split does not contain all classes")
    return result


def prepare_gc10(root: Path) -> tuple[dict[str, list[Path]], dict]:
    splits = {
        "train": image_files(root / "train" / "images"),
        "val": image_files(root / "valid" / "images"),
        "test": image_files(root / "test" / "images"),
    }
    if {k: len(v) for k, v in splits.items()} != {"train": 1606, "val": 459, "test": 229}:
        raise RuntimeError(f"unexpected GC10 counts: { {k: len(v) for k, v in splits.items()} }")
    return splits, {"split_policy": "archive train/valid/test"}


def prepare_uwwt(root: Path) -> tuple[dict[str, list[Path]], dict]:
    manifest = root / "manifest.jsonl"
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    splits = {"train": [], "val": [], "test": []}
    sources = defaultdict(set)
    variants = defaultdict(Counter)
    for row in rows:
        split = row["split"]
        variant = row["variant"]
        rel = Path(str(row["image"]).replace("\\", "/"))
        img = (root / rel).resolve()
        variants[split][variant] += 1
        source_id = f"{row.get('source_split','')}::{row['source_image']}"
        sources[source_id].add(split)
        if split == "train" or variant == "orig":
            splits[split].append(img)
    cross = sorted(k for k, v in sources.items() if len(v) > 1)
    if cross:
        raise RuntimeError(f"UWWT source leakage across output splits: {cross[:10]}")
    expected = {"train": 1050, "val": 100, "test": 50}
    if {k: len(v) for k, v in splits.items()} != expected:
        raise RuntimeError(f"unexpected UWWT prepared counts: { {k: len(v) for k, v in splits.items()} }")
    return splits, {
        "split_policy": "train all variants; val/test variant=orig only",
        "manifest_rows": len(rows),
        "archive_variants": {k: dict(v) for k, v in variants.items()},
        "source_cross_split_count": 0,
    }


def steel_group(stem: str) -> str:
    m = re.match(r"^(.*)-([0-9]+)$", stem)
    return m.group(1) if m else re.sub(r"[0-9]+$", "", stem)


def frame_number(path: Path) -> int:
    m = re.search(r"([0-9]+)$", path.stem)
    return int(m.group(1)) if m else -1


def split_three_blocks(items: list[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    items = sorted(items, key=lambda p: (frame_number(p), p.name))
    if len(items) < 12:
        raise RuntimeError(f"too few items for guarded three-way block split: {len(items)}")
    usable = len(items) - 2
    n_train = int(usable * 0.70)
    n_val = int(usable * 0.20)
    n_test = usable - n_train - n_val
    g1 = n_train
    g2 = n_train + 1 + n_val
    blocks = {
        "train": items[:n_train],
        "val": items[n_train + 1:g2],
        "test": items[g2 + 1:g2 + 1 + n_test],
    }
    dropped = [items[g1], items[g2]]
    return blocks, dropped


def split_two_blocks(items: list[Path], left="val", right="test") -> tuple[dict[str, list[Path]], list[Path]]:
    items = sorted(items, key=lambda p: (frame_number(p), p.name))
    usable = len(items) - 1
    n_left = int(usable * 2 / 3)
    blocks = {left: items[:n_left], right: items[n_left + 1:]}
    return blocks, [items[n_left]]


def counts_for_paths(paths: list[Path], nc: int) -> Counter:
    out = Counter()
    for p in paths:
        out.update(parse_label(label_for_image(p), nc))
    return out


def optimize_regular_groups(groups: dict[str, list[Path]], pre: dict[str, list[Path]], retained_total: int) -> dict[str, str]:
    names = sorted(groups)
    stats = {g: (len(groups[g]), counts_for_paths(groups[g], 8)) for g in names}
    pre_stats = {s: counts_for_paths(pre[s], 8) for s in pre}
    total_cls = Counter()
    for s in pre:
        total_cls.update(pre_stats[s])
    for _, c in stats.values():
        total_cls.update(c)
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
    target_img = {s: retained_total * r for s, r in ratios.items()}
    target_cls = {s: {c: total_cls[c] * r for c in (0, 2, 7)} for s, r in ratios.items()}
    rng = random.Random(20260731)
    choices = ("train", "val", "test")
    weights = (0.70, 0.20, 0.10)
    best_score = float("inf")
    best = None
    for _ in range(120000):
        assign = {g: rng.choices(choices, weights=weights, k=1)[0] for g in names}
        imgs = {s: len(pre[s]) for s in choices}
        cls = {s: Counter(pre_stats[s]) for s in choices}
        group_counts = Counter(assign.values())
        for g, s in assign.items():
            imgs[s] += stats[g][0]
            cls[s].update(stats[g][1])
        if any(group_counts[s] == 0 for s in choices):
            continue
        if any(cls[s][c] == 0 for s in choices for c in (0, 2, 7)):
            continue
        score = 8.0 * sum(((imgs[s] - target_img[s]) / max(target_img[s], 1)) ** 2 for s in choices)
        score += sum(((cls[s][c] - target_cls[s][c]) / max(target_cls[s][c], 1)) ** 2 for s in choices for c in (0, 2, 7))
        score += 0.02 * sum((group_counts[s] - len(names) * ratios[s]) ** 2 for s in choices)
        if score < best_score:
            best_score, best = score, assign.copy()
    if best is None:
        raise RuntimeError("could not find deterministic steel group split")
    print(f"STEEL_GROUP_OPT score={best_score:.6f} assignments={best}")
    return best


def prepare_steel(root: Path) -> tuple[dict[str, list[Path]], dict]:
    all_images = image_files(root / "images")
    if len(all_images) != 3408:
        raise RuntimeError(f"unexpected steel image count: {len(all_images)}")
    groups = defaultdict(list)
    for img in all_images:
        groups[steel_group(img.stem)].append(img)
    if len(groups) != 25:
        raise RuntimeError(f"unexpected steel source group count: {len(groups)}")

    splits = {"train": [], "val": [], "test": []}
    dropped: list[Path] = []
    exceptions = {}

    # 单一来源类别：连续三段 + 两个边界隔离帧。
    for group in ("bite-edge2", "crack", "overlap"):
        blocks, gaps = split_three_blocks(groups.pop(group))
        for s in splits:
            splits[s].extend(blocks[s])
        dropped.extend(gaps)
        exceptions[group] = {"reason": "class has one source group", "counts": {s: len(v) for s, v in blocks.items()}, "gap_frames": [p.name for p in gaps]}

    # hollow-bead / slag-inclusion 各只有两个来源组：一组完整放 train，另一组只在 val/test 间连续分段。
    for train_group, eval_group, reason in (
        ("air-hole4(hollow-bead)", "air-hole12(hollow-bead)", "hollow-bead has two source groups"),
        ("air-hole9(slag-inclusion)", "slag-inclusion2", "slag-inclusion has two source groups"),
    ):
        splits["train"].extend(groups.pop(train_group))
        blocks, gaps = split_two_blocks(groups.pop(eval_group))
        splits["val"].extend(blocks["val"])
        splits["test"].extend(blocks["test"])
        dropped.extend(gaps)
        exceptions[eval_group] = {"reason": reason, "counts": {s: len(v) for s, v in blocks.items()}, "gap_frames": [p.name for p in gaps], "paired_train_group": train_group}

    retained_total = len(all_images) - len(dropped)
    assignment = optimize_regular_groups(dict(groups), splits, retained_total)
    for group, imgs in groups.items():
        splits[assignment[group]].extend(imgs)
    for s in splits:
        splits[s] = sorted(splits[s])

    # 常规来源组必须完全隔离；只有显式 exceptions 可跨 split。
    group_splits = defaultdict(set)
    for s, imgs in splits.items():
        for img in imgs:
            group_splits[steel_group(img.stem)].add(s)
    unexpected_cross = {g: sorted(v) for g, v in group_splits.items() if len(v) > 1 and g not in exceptions}
    if unexpected_cross:
        raise RuntimeError(f"unexpected steel source leakage: {unexpected_cross}")

    return splits, {
        "split_policy": "deterministic 7:2:1 target; source-group isolation except limited-source classes; guarded contiguous blocks",
        "archive_images": len(all_images),
        "retained_images": retained_total,
        "source_groups": 25,
        "regular_group_assignments": assignment,
        "limited_source_exceptions": exceptions,
        "dropped_gap_frames": [p.name for p in dropped],
        "unexpected_regular_group_cross_split_count": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    PROJECT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in DATASETS.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    roots = {
        "gc10": RAW / "gc10",
        "uwwt": RAW / "uwwt" / "UWWT-Dataset-1500",
        "steel_weld": RAW / "steel_weld" / "钢管焊缝缺陷检测数据集",
    }
    if not args.skip_extract:
        safe_extract(DATASETS["gc10"], RAW / "gc10")
        safe_extract(DATASETS["uwwt"], RAW / "uwwt")
        safe_extract(DATASETS["steel_weld"], RAW / "steel_weld")

    profile = {
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "zip_files": {k: {"path": str(v), "size": v.stat().st_size, "sha256": sha256(v)} for k, v in DATASETS.items()},
        "datasets": {},
        "gate_pass": False,
    }
    builders = {"gc10": prepare_gc10, "uwwt": prepare_uwwt, "steel_weld": prepare_steel}
    for dataset in ("gc10", "uwwt", "steel_weld"):
        root = roots[dataset]
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root missing: {root}")
        splits, notes = builders[dataset](root)
        yml = write_yaml(dataset, root, splits)
        gate = validate(dataset, splits)
        profile["datasets"][dataset] = {
            "root": str(root.resolve()),
            "data_yaml": str(yml.resolve()),
            "names": NAMES[dataset],
            **notes,
            **gate,
        }
        print(f"DATASET_GATE_OK {dataset} counts={ {s: len(v) for s, v in splits.items()} } yaml={yml}")
    profile["gate_pass"] = True
    atomic_json(PROFILE, profile)
    print(f"GENERALIZATION_DATA_GATE_PASS {PROFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
