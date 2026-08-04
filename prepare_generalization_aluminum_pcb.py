#!/usr/bin/env python
"""Prepare Aluminum and PCB datasets for fair YOLO26n/MSDGS retraining.

The script keeps uploaded archives immutable, creates absolute image lists, and
builds a leakage-safe PCB split by source board. PCB horizontal-flip variants
are admitted only to train; validation and test contain original images only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import yaml

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STORAGE = Path(os.environ.get("YOLO26_DATA_STORAGE", "/autodl-fs/data"))
PROJECT = ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250"
WORK = ROOT / "generalization_aluminum_pcb"
RAW = WORK / "raw"
PREPARED = WORK / "prepared"
OUT = PROJECT / "datasets"
PROFILE = PROJECT / "dataset_profile.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARCHIVES = {
    "aluminum": STORAGE / "aluminum_yolo26.zip",
    "pcb": STORAGE / "pcb_dataset.rar",
}
NAMES = {
    "aluminum": ["zhen_kong", "ca_shang", "zang_wu", "zhe_zhou"],
    "pcb": ["missing_hole", "mouse_bite", "open_circuit", "short", "spurious_copper", "spur"],
}
PCB_RE = re.compile(r"^(?P<board>\d+)_(?P<kind>[a-z_]+)_(?P<index>\d+?)(?P<hflip>_hflip)?$")


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


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rmtree(path: Path, allowed_parent: Path) -> None:
    path = path.resolve()
    allowed_parent = allowed_parent.resolve()
    if path == allowed_parent or allowed_parent not in path.parents:
        raise RuntimeError(f"unsafe cleanup target: {path}")
    if path.exists():
        shutil.rmtree(path)


def extract_archives() -> None:
    aluminum_root = RAW / "aluminum_yolo26"
    if not (aluminum_root / "data.yaml").is_file():
        safe_rmtree(aluminum_root, RAW)
        RAW.mkdir(parents=True, exist_ok=True)
        with ZipFile(ARCHIVES["aluminum"]) as zf:
            root = RAW.resolve()
            for info in zf.infolist():
                target = (RAW / info.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe zip member: {info.filename}")
            zf.extractall(RAW)

    pcb_root = RAW / "pcb_dataset"
    if not (pcb_root / "dataset.yaml").is_file():
        if shutil.which("7z") is None:
            raise RuntimeError("7z is required to extract the RAR5 PCB archive")
        safe_rmtree(pcb_root, RAW)
        RAW.mkdir(parents=True, exist_ok=True)
        listing = subprocess.run(["7z", "l", "-slt", str(ARCHIVES["pcb"])], check=True, text=True, stdout=subprocess.PIPE).stdout
        for raw in listing.splitlines():
            if not raw.startswith("Path = "):
                continue
            member = raw[7:].replace("\\", "/")
            if member == str(ARCHIVES["pcb"]).replace("\\", "/"):
                continue
            p = Path(member)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"unsafe RAR member: {member}")
        subprocess.run(["7z", "x", "-y", "-bd", f"-o{RAW}", str(ARCHIVES["pcb"])], check=True)


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def label_for_image(img: Path) -> Path:
    parts = list(img.parts)
    indexes = [i for i, value in enumerate(parts) if value == "images"]
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


def counts_for(paths: list[Path], nc: int) -> Counter:
    result = Counter()
    for img in paths:
        result.update(parse_label(label_for_image(img), nc))
    return result


def write_list(path: Path, images: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p.as_posix()}\n" for p in images), encoding="utf-8")


def write_yaml(dataset: str, root: Path, splits: dict[str, list[Path]]) -> Path:
    ds_out = OUT / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    list_paths = {}
    for split in ("train", "val", "test"):
        path = ds_out / f"{split}.txt"
        write_list(path, splits[split])
        list_paths[split] = str(path.resolve())
    doc = {
        "path": str(root.resolve()),
        "train": list_paths["train"],
        "val": list_paths["val"],
        "test": list_paths["test"],
        "nc": len(NAMES[dataset]),
        "names": NAMES[dataset],
    }
    output = OUT / f"{dataset}.yaml"
    output.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def validate(dataset: str, splits: dict[str, list[Path]], source_group=None) -> dict:
    nc = len(NAMES[dataset])
    seen_paths = {}
    seen_hashes = {}
    group_splits = defaultdict(set)
    result = {}
    for split in ("train", "val", "test"):
        images = splits[split]
        if not images:
            raise RuntimeError(f"{dataset}: empty {split} split")
        instances = Counter()
        empty_labels = 0
        for img in images:
            if not img.is_file():
                raise FileNotFoundError(f"missing image: {img}")
            resolved = img.resolve()
            if resolved in seen_paths:
                raise RuntimeError(f"path overlap: {resolved} in {seen_paths[resolved]} and {split}")
            seen_paths[resolved] = split
            digest = sha1(resolved)
            if digest in seen_hashes and seen_hashes[digest][0] != split:
                raise RuntimeError(f"content overlap: {img} and {seen_hashes[digest][1]}")
            seen_hashes[digest] = (split, str(img))
            c = parse_label(label_for_image(img), nc)
            if not c:
                empty_labels += 1
            instances.update(c)
            if source_group is not None:
                group_splits[source_group(img)].add(split)
        result[split] = {
            "images": len(images),
            "empty_labels": empty_labels,
            "instances": {NAMES[dataset][i]: instances.get(i, 0) for i in range(nc)},
        }
    result["path_overlap_count"] = 0
    result["content_hash_overlap_count"] = 0
    result["all_classes_in_test"] = all(result["test"]["instances"][name] > 0 for name in NAMES[dataset])
    if not result["all_classes_in_test"]:
        raise RuntimeError(f"{dataset}: test split does not contain all classes")
    if source_group is not None:
        cross = {g: sorted(v) for g, v in group_splits.items() if len(v) > 1}
        if cross:
            raise RuntimeError(f"{dataset}: source-group leakage: {cross}")
        result["source_group_overlap_count"] = 0
    return result


def prepare_aluminum(root: Path) -> tuple[dict[str, list[Path]], dict]:
    splits = {split: image_files(root / "images" / split) for split in ("train", "val", "test")}
    expected = {"train": 1000, "val": 200, "test": 200}
    actual = {key: len(value) for key, value in splits.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected Aluminum counts: {actual}")
    summary_path = root / "conversion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    return splits, {
        "split_policy": "archive deterministic 1000/200/200 split; no generated variants in val/test",
        "conversion_summary": summary,
    }


def pcb_meta(path: Path) -> dict:
    match = PCB_RE.match(path.stem)
    if match is None:
        raise RuntimeError(f"unexpected PCB filename: {path.name}")
    return {**match.groupdict(), "is_hflip": bool(match.group("hflip"))}


def pcb_source_group(path: Path) -> str:
    return pcb_meta(path)["board"]


def optimize_pcb_board_split(originals: list[Path]) -> tuple[dict[str, set[str]], dict]:
    boards = sorted({pcb_source_group(p) for p in originals})
    if len(boards) < 5:
        raise RuntimeError(f"too few PCB source boards: {boards}")
    board_images = Counter(pcb_source_group(p) for p in originals)
    board_instances = {board: Counter() for board in boards}
    for img in originals:
        board_instances[pcb_source_group(img)].update(parse_label(label_for_image(img), len(NAMES["pcb"])))
    total_images = len(originals)
    total_instances = Counter()
    for value in board_instances.values():
        total_instances.update(value)
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
    n_test = max(1, round(len(boards) * ratios["test"]))
    n_val = max(1, round(len(boards) * ratios["val"]))
    best = None
    best_score = float("inf")
    for test_tuple in itertools.combinations(boards, n_test):
        remain = [b for b in boards if b not in test_tuple]
        for val_tuple in itertools.combinations(remain, n_val):
            assignment = {
                "test": set(test_tuple),
                "val": set(val_tuple),
                "train": set(remain) - set(val_tuple),
            }
            image_counts = {s: sum(board_images[b] for b in bs) for s, bs in assignment.items()}
            instance_counts = {s: Counter() for s in ratios}
            for split, split_boards in assignment.items():
                for board in split_boards:
                    instance_counts[split].update(board_instances[board])
            if any(instance_counts[s][c] == 0 for s in ratios for c in range(len(NAMES["pcb"]))):
                continue
            score = 8.0 * sum(((image_counts[s] - total_images * ratios[s]) / max(total_images * ratios[s], 1)) ** 2 for s in ratios)
            score += sum(((instance_counts[s][c] - total_instances[c] * ratios[s]) / max(total_instances[c] * ratios[s], 1)) ** 2 for s in ratios for c in range(len(NAMES["pcb"])))
            key = (score, tuple(sorted(assignment["test"])), tuple(sorted(assignment["val"])))
            if best is None or key < best[0]:
                best = (key, assignment, image_counts, instance_counts)
                best_score = score
    if best is None:
        raise RuntimeError("could not find PCB board-disjoint split with all classes")
    _, assignment, image_counts, instance_counts = best
    notes = {
        "score": best_score,
        "boards": {s: sorted(v) for s, v in assignment.items()},
        "original_images": image_counts,
        "original_instances": {s: {NAMES["pcb"][c]: instance_counts[s][c] for c in range(len(NAMES["pcb"]))} for s in ratios},
    }
    return assignment, notes


def raw_pair_audit(root: Path) -> dict:
    audit = {}
    for split in ("train", "val"):
        images = {p.stem for p in image_files(root / "images" / split)}
        labels_dir = root / "labels" / split
        labels = {p.stem for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}
        audit[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": sorted(images - labels),
            "orphan_labels_excluded": sorted(labels - images),
        }
        if audit[split]["missing_labels"]:
            raise RuntimeError(f"PCB raw {split} has missing labels: {audit[split]['missing_labels'][:10]}")
    return audit


def materialize_pcb(prepared_root: Path, split_sources: dict[str, list[Path]]) -> dict[str, list[Path]]:
    safe_rmtree(prepared_root, PREPARED)
    for split in ("train", "val", "test"):
        (prepared_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (prepared_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    output = {"train": [], "val": [], "test": []}
    for split, sources in split_sources.items():
        seen_names = set()
        for src in sorted(sources):
            if src.name in seen_names:
                raise RuntimeError(f"duplicate PCB basename in {split}: {src.name}")
            seen_names.add(src.name)
            src_label = label_for_image(src)
            if not src_label.is_file():
                raise FileNotFoundError(src_label)
            dst_img = prepared_root / "images" / split / src.name
            dst_label = prepared_root / "labels" / split / src_label.name
            dst_img.symlink_to(src)
            dst_label.symlink_to(src_label)
            output[split].append(dst_img.absolute())
    return output


def prepare_pcb(root: Path) -> tuple[dict[str, list[Path]], dict]:
    raw_audit = raw_pair_audit(root)
    raw_train = image_files(root / "images" / "train")
    raw_val = image_files(root / "images" / "val")
    originals = [p for p in raw_train + raw_val if not pcb_meta(p)["is_hflip"]]
    hflips = [p for p in raw_train if pcb_meta(p)["is_hflip"]]
    original_keys = {(pcb_meta(p)["board"], pcb_meta(p)["kind"], pcb_meta(p)["index"]) for p in originals}
    orphan_hflip_sources = []
    for p in hflips:
        meta = pcb_meta(p)
        key = (meta["board"], meta["kind"], meta["index"])
        if key not in original_keys:
            orphan_hflip_sources.append(p.name)
    if orphan_hflip_sources:
        raise RuntimeError(f"PCB hflip without original source: {orphan_hflip_sources[:10]}")

    assignment, split_notes = optimize_pcb_board_split(originals)
    split_sources = {"train": [], "val": [], "test": []}
    for img in originals:
        board = pcb_source_group(img)
        split = next(s for s, boards in assignment.items() if board in boards)
        split_sources[split].append(img)
    for img in hflips:
        if pcb_source_group(img) in assignment["train"]:
            split_sources["train"].append(img)
    if any(pcb_meta(p)["is_hflip"] for p in split_sources["val"] + split_sources["test"]):
        raise RuntimeError("PCB val/test contains generated hflip variants")

    prepared_root = PREPARED / "pcb"
    splits = materialize_pcb(prepared_root, split_sources)
    return splits, {
        "split_policy": "source-board-disjoint exhaustive 7/2/1 board split optimized for image/class balance; hflip variants train-only",
        "raw_pair_audit": raw_audit,
        "raw_original_images": len(originals),
        "raw_hflip_images": len(hflips),
        "val_test_generated_variants": 0,
        "board_split_optimization": split_notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    PROJECT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for archive in ARCHIVES.values():
        if not archive.is_file():
            raise FileNotFoundError(archive)
    if not args.skip_extract:
        extract_archives()

    roots = {"aluminum": RAW / "aluminum_yolo26", "pcb": RAW / "pcb_dataset"}
    builders = {"aluminum": prepare_aluminum, "pcb": prepare_pcb}
    profile = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archives": {k: {"path": str(v), "size": v.stat().st_size, "sha256": sha256(v)} for k, v in ARCHIVES.items()},
        "datasets": {},
        "gate_pass": False,
    }
    for dataset in ("aluminum", "pcb"):
        root = roots[dataset]
        if not root.is_dir():
            raise FileNotFoundError(root)
        splits, notes = builders[dataset](root)
        yaml_path = write_yaml(dataset, root, splits)
        gate = validate(dataset, splits, pcb_source_group if dataset == "pcb" else None)
        profile["datasets"][dataset] = {
            "root": str(root.resolve()),
            "data_yaml": str(yaml_path.resolve()),
            "names": NAMES[dataset],
            **notes,
            **gate,
        }
        print(f"DATASET_GATE_OK {dataset} counts={ {s: len(v) for s, v in splits.items()} } yaml={yaml_path}", flush=True)
    profile["gate_pass"] = True
    atomic_json(PROFILE, profile)
    print(f"ALUMINUM_PCB_DATA_GATE_PASS {PROFILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
