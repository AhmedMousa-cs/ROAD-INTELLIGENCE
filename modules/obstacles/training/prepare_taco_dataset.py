"""
modules/obstacles/training/prepare_taco_dataset.py
====================================================
Dataset-prep script for Part A (road debris). NOT run as part of the
module's inference path -- this is an offline, one-time step you run
locally after downloading TACO, to produce a YOLO-format dataset with a
single class ("road_debris").

Run locally (requires TACO's images/ + annotations.json downloaded from
http://tacodataset.org/ -- not fetched by this script):

    python -m modules.obstacles.training.prepare_taco_dataset \\
        --taco-root /path/to/TACO \\
        --out-dir data/taco_road_debris \\
        --val-fraction 0.15 --test-fraction 0.15

What it does, matching the spec's data-prep rules (section "Dataset
Preparation" / Part A):
  1. Loads the real annotations.json and diffs its category list against
     class_mapping.TACO_TO_ROAD_DEBRIS via verify_taco_categories() --
     prints unmapped categories so you can extend the table deliberately
     instead of silently dropping them.
  2. Converts every mapped annotation's COCO bbox [x, y, w, h] to YOLO's
     normalized [x_center, y_center, w, h] format, single class id 0
     ("road_debris").
  3. Skips images with zero mapped annotations (no accidental empty-label
     "background" images from categories you excluded).
  4. Detects duplicate images by file hash before splitting, so the same
     photo can't land in both train and val/test.
  5. Writes a YOLO-detection folder layout (images/{train,val,test},
     labels/{train,val,test}) plus a data.yaml Ultralytics can train on
     directly.

This script does NOT invent dataset statistics -- it prints the actual
counts it computed from your local files, and labels the held-out split as
an "internal held-out test split" (TACO does not publish an official test
set), per spec section 15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from modules.obstacles.class_mapping import map_taco_category, verify_taco_categories

ROAD_DEBRIS_CLASS_ID = 0  # single-class YOLO dataset


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coco_to_yolo_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def prepare(taco_root: Path, out_dir: Path, val_fraction: float, test_fraction: float, seed: int = 42) -> None:
    ann_path = taco_root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"'{ann_path}' not found. Download TACO from http://tacodataset.org/ first -- "
            f"this script does not fetch datasets."
        )

    with open(ann_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    diff = verify_taco_categories(list(categories.values()))
    print(f"[prepare_taco_dataset] mapped categories:   {len(diff['mapped'])}")
    print(f"[prepare_taco_dataset] UNMAPPED categories:  {diff['unmapped']}  <- inspect these, see class_mapping.py")
    print(f"[prepare_taco_dataset] stale table entries:  {diff['stale']}  <- not present in this TACO release")

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_name = categories[ann["category_id"]]
        normalized_class, subclass = map_taco_category(cat_name)
        if normalized_class is None:
            continue  # excluded per class_mapping.py -- not force-mapped
        anns_by_image[ann["image_id"]].append(ann)

    # Only keep images that ended up with >= 1 mapped annotation.
    usable_image_ids = [img_id for img_id, anns in anns_by_image.items() if anns]
    print(f"[prepare_taco_dataset] images with >=1 mapped road_debris annotation: {len(usable_image_ids)}")

    # De-duplicate by file hash before splitting.
    hash_to_image_id: dict[str, int] = {}
    deduped_ids = []
    for img_id in usable_image_ids:
        img_path = taco_root / images_by_id[img_id]["file_name"]
        if not img_path.exists():
            print(f"[prepare_taco_dataset] WARNING: missing file, skipping: {img_path}")
            continue
        h = _file_hash(img_path)
        if h in hash_to_image_id:
            continue  # duplicate image content -- keep first occurrence only
        hash_to_image_id[h] = img_id
        deduped_ids.append(img_id)

    n_dupes = len(usable_image_ids) - len(deduped_ids)
    if n_dupes:
        print(f"[prepare_taco_dataset] removed {n_dupes} duplicate-content images")

    random.seed(seed)
    random.shuffle(deduped_ids)
    n = len(deduped_ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    split_ids = {
        "test": deduped_ids[:n_test],
        "val": deduped_ids[n_test : n_test + n_val],
        "train": deduped_ids[n_test + n_val :],
    }
    print(
        f"[prepare_taco_dataset] split -> train={len(split_ids['train'])} "
        f"val={len(split_ids['val'])} test(internal_held_out)={len(split_ids['test'])}"
    )

    for split, ids in split_ids.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_meta = images_by_id[img_id]
            src_path = taco_root / img_meta["file_name"]
            dst_name = f"{img_id}{src_path.suffix}"
            shutil.copy2(src_path, img_out / dst_name)

            lines = []
            for ann in anns_by_image[img_id]:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue  # invalid box, per spec's "invalid bounding boxes" check
                xc, yc, nw, nh = _coco_to_yolo_bbox(x, y, w, h, img_meta["width"], img_meta["height"])
                lines.append(f"{ROAD_DEBRIS_CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            (lbl_out / f"{img_id}.txt").write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "# Generated by prepare_taco_dataset.py -- internal_test is a held-out split,\n"
        "# NOT an official TACO test set (TACO does not publish one).\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: road_debris\n"
    )
    print(f"[prepare_taco_dataset] wrote {data_yaml}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TACO annotations into a YOLO road_debris dataset.")
    parser.add_argument("--taco-root", required=True, type=Path, help="Path to downloaded TACO/ folder")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output YOLO dataset folder")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    prepare(args.taco_root, args.out_dir, args.val_fraction, args.test_fraction, args.seed)
