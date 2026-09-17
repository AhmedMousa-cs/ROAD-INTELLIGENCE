"""
modules/obstacles/training/merge_roboflow_coco_taco.py
=========================================================
One-time adapter: merges a Roboflow COCO-JSON export of TACO (which ships
as separate train/valid/test folders, each with its own
_annotations.coco.json) into the single flat layout
prepare_taco_dataset.py expects (one taco_root/annotations.json +
images referenced by relative file_name).

This exists because prepare_taco_dataset.py was written against the
official TACO distribution format (one root/annotations.json), while
Roboflow always splits COCO exports by folder. Rather than force-fitting
Roboflow's already-split data through prepare_taco_dataset.py's own
splitting logic, this script:
  1. Copies every image into <out>/images/, preserving Roboflow's original
     filenames (already unique) as the file_name.
  2. Reassigns globally unique image_id / annotation_id across the three
     input splits (Roboflow's ids restart at 0 per split, which would
     collide if merged naively).
  3. Writes one combined <out>/annotations.json with the union of
     categories (by name) and every annotation, rewritten to point at the
     new ids.

Usage:
    python -m modules.obstacles.training.merge_roboflow_coco_taco \\
        --roboflow-export /path/to/downloaded/export \\
        --out data/raw/taco_merged

The output directory is then a valid --taco-root for
prepare_taco_dataset.py: it always contains flat images/ + annotations.json,
regardless of whether Roboflow's export used image subfolders per split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLIT_DIRS = ("train", "valid", "test")
ANNOTATION_FILENAME = "_annotations.coco.json"


def _find_split_dir(export_root: Path, split: str) -> Path | None:
    """Roboflow COCO exports usually put images at <root>/<split>/ with
    _annotations.coco.json alongside them, but some exports omit the
    'valid'/'test' folder name distinction -- check a couple of common
    layouts before giving up.
    """
    candidates = [export_root / split, export_root / split.replace("valid", "val")]
    for candidate in candidates:
        if (candidate / ANNOTATION_FILENAME).exists():
            return candidate
    return None


def merge(export_root: Path, out_dir: Path) -> None:
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    categories_by_name: dict[str, int] = {}
    merged_categories: list[dict] = []
    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    next_image_id = 0
    next_ann_id = 0

    found_any = False
    for split in SPLIT_DIRS:
        split_dir = _find_split_dir(export_root, split)
        if split_dir is None:
            print(f"[merge_roboflow_coco_taco] no {split} split found (looked for {ANNOTATION_FILENAME}), skipping")
            continue
        found_any = True

        with open(split_dir / ANNOTATION_FILENAME) as f:
            coco = json.load(f)

        # category id remap: keep one global id per category NAME (Roboflow
        # assigns fresh per-split ids that don't line up across splits).
        local_cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        for name in local_cat_id_to_name.values():
            if name not in categories_by_name:
                new_id = len(categories_by_name)
                categories_by_name[name] = new_id
                merged_categories.append({"id": new_id, "name": name, "supercategory": "litter"})

        local_image_id_to_new: dict[int, int] = {}
        for img in coco["images"]:
            src = split_dir / img["file_name"]
            if not src.exists():
                print(f"[merge_roboflow_coco_taco] WARNING missing image, skipping: {src}")
                continue
            new_id = next_image_id
            next_image_id += 1
            local_image_id_to_new[img["id"]] = new_id

            # Prefix with split name to guarantee filename uniqueness across
            # the three input folders even if Roboflow reused a name.
            dst_name = f"{split}_{src.name}"
            shutil.copy2(src, images_out / dst_name)

            merged_images.append(
                {
                    "id": new_id,
                    "file_name": f"images/{dst_name}",
                    "width": img["width"],
                    "height": img["height"],
                }
            )

        for ann in coco["annotations"]:
            if ann["image_id"] not in local_image_id_to_new:
                continue  # image was missing/skipped above
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = local_image_id_to_new[ann["image_id"]]
            new_ann["category_id"] = categories_by_name[local_cat_id_to_name[ann["category_id"]]]
            merged_annotations.append(new_ann)

        print(f"[merge_roboflow_coco_taco] {split}: {len(local_image_id_to_new)} images merged")

    if not found_any:
        raise FileNotFoundError(
            f"No train/valid/test split with {ANNOTATION_FILENAME} found under '{export_root}'. "
            f"Check you downloaded a COCO JSON export, not YOLOv8."
        )

    merged = {
        "info": {"description": "Merged Roboflow TACO COCO export (train+valid+test), for prepare_taco_dataset.py"},
        "categories": merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }
    (out_dir / "annotations.json").write_text(json.dumps(merged))
    print(f"[merge_roboflow_coco_taco] wrote {out_dir / 'annotations.json'}")
    print(f"[merge_roboflow_coco_taco] total: {len(merged_images)} images, {len(merged_annotations)} annotations, {len(merged_categories)} categories")
    print(f"[merge_roboflow_coco_taco] categories: {sorted(categories_by_name)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roboflow-export", required=True, type=Path, help="Root folder of the downloaded Roboflow COCO export")
    parser.add_argument("--out", required=True, type=Path, help="Output folder (becomes --taco-root for prepare_taco_dataset.py)")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    merge(args.roboflow_export, args.out)
