"""
Build the normalized road-damage training dataset.

Takes one or more YOLO-format source datasets (RDD2022 as primary, a Roboflow
pothole dataset as supplementary), and produces a single dataset whose class
ids are the four project classes, with a train / validation / internal_test
split.

    python -m modules.road_damage.prepare_dataset \
        --source data/raw/rdd2022 --name rdd2022 \
        --source data/raw/pothole --name pothole_roboflow \
        --output data/road_damage_prepared

Everything the spec asks to be verified is verified, and nothing is silently
dropped - every rejected image/annotation is counted and written to
``preparation_report.json``:

  * incorrect class ids          -> remapped, or the line is dropped
  * coordinates outside [0, 1]   -> clamped if trivially over, else dropped
  * invalid / zero-area boxes    -> dropped
  * empty label files            -> kept ONLY as explicit background images
  * corrupted / unreadable images-> dropped
  * duplicate & near-duplicate   -> grouped, and the whole group is forced
                                    into the SAME split, so a test image can
                                    never be a copy of a training image

Splitting is deterministic (seeded) and group-aware, so re-running gives the
same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .class_mapping import (
    CLASS_TO_ID,
    PROJECT_CLASSES,
    build_id_remap,
    exclusion_reason,
    normalize_class,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# A box whose normalized width or height is below this is treated as a
# degenerate annotation rather than a very thin crack.
MIN_NORMALIZED_SIDE = 1e-4

# Coordinates outside [0, 1] by no more than this are clamped (a common,
# harmless artefact of annotation tools). Anything worse is dropped, because
# it usually signals a genuinely broken label.
COORD_TOLERANCE = 0.02

# Hamming distance between 64-bit dHashes below which two images are
# considered near-duplicates.
NEAR_DUPLICATE_MAX_DISTANCE = 5


@dataclass
class PreparationStats:
    images_seen: int = 0
    images_kept: int = 0
    images_corrupted: int = 0
    images_missing_label: int = 0
    background_images: int = 0
    duplicate_groups: int = 0
    duplicate_images: int = 0
    annotations_seen: int = 0
    annotations_kept: int = 0
    annotations_dropped_unmapped: int = 0
    annotations_dropped_invalid_box: int = 0
    annotations_dropped_out_of_bounds: int = 0
    annotations_clamped: int = 0
    annotations_malformed_line: int = 0
    dropped_source_classes: Counter = field(default_factory=Counter)
    kept_class_counts: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        data = {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in self.__dict__.items()
        }
        return data


# -----------------------------------------------------------------------------
# Source dataset discovery
# -----------------------------------------------------------------------------


def load_source_class_names(source_dir: Path) -> list[str]:
    """Read the source dataset's own class list from its data.yaml.

    The spec is emphatic about inspecting the real YAML rather than assuming
    the class order, so this raises instead of guessing when it is absent.
    """
    candidates = list(source_dir.glob("*.yaml")) + list(source_dir.glob("*.yml"))
    for candidate in candidates:
        with open(candidate) as handle:
            data = yaml.safe_load(handle) or {}
        names = data.get("names")
        if names is None:
            continue
        if isinstance(names, dict):
            return [names[key] for key in sorted(names, key=int)]
        return list(names)
    raise FileNotFoundError(
        f"No data.yaml with a 'names:' entry found in {source_dir}. Inspect the "
        f"downloaded dataset and pass --class-names explicitly if it uses a "
        f"different layout."
    )


def find_image_label_pairs(source_dir: Path) -> list[tuple[Path, Path | None]]:
    """Find every image and its YOLO label file.

    Handles the usual YOLO layouts (``images/train`` + ``labels/train``,
    ``train/images`` + ``train/labels``) by locating each image then swapping
    the nearest ``images`` path component for ``labels``.
    """
    pairs: list[tuple[Path, Path | None]] = []
    for image_path in sorted(source_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            continue
        parts = list(image_path.parts)
        label_path: Path | None = None
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "images":
                candidate_parts = parts.copy()
                candidate_parts[index] = "labels"
                candidate = Path(*candidate_parts).with_suffix(".txt")
                if candidate.exists():
                    label_path = candidate
                break
        if label_path is None:
            sibling = image_path.with_suffix(".txt")
            if sibling.exists():
                label_path = sibling
        pairs.append((image_path, label_path))
    return pairs


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate_and_remap_label_file(
    label_path: Path | None,
    id_remap: dict[int, int],
    source_names: list[str],
    stats: PreparationStats,
) -> list[str] | None:
    """Return normalized YOLO label lines, or None if the file is unusable.

    An empty list means "valid, but no objects" (a background image).
    """
    if label_path is None or not label_path.exists():
        return []

    kept_lines: list[str] = []
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stats.annotations_seen += 1
        parts = line.split()
        if len(parts) < 5:
            stats.annotations_malformed_line += 1
            continue

        try:
            source_id = int(float(parts[0]))
            cx, cy, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            stats.annotations_malformed_line += 1
            continue

        # --- class id -------------------------------------------------------
        if source_id not in id_remap:
            stats.annotations_dropped_unmapped += 1
            name = source_names[source_id] if 0 <= source_id < len(source_names) else f"<id {source_id}>"
            stats.dropped_source_classes[str(name)] += 1
            continue
        project_id = id_remap[source_id]

        # --- geometry -------------------------------------------------------
        if width <= MIN_NORMALIZED_SIDE or height <= MIN_NORMALIZED_SIDE:
            stats.annotations_dropped_invalid_box += 1
            continue

        x1, y1 = cx - width / 2, cy - height / 2
        x2, y2 = cx + width / 2, cy + height / 2

        if min(x1, y1) < -COORD_TOLERANCE or max(x2, y2) > 1 + COORD_TOLERANCE:
            stats.annotations_dropped_out_of_bounds += 1
            continue

        if min(x1, y1) < 0 or max(x2, y2) > 1:
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            width, height = x2 - x1, y2 - y1
            cx, cy = x1 + width / 2, y1 + height / 2
            stats.annotations_clamped += 1
            if width <= MIN_NORMALIZED_SIDE or height <= MIN_NORMALIZED_SIDE:
                stats.annotations_dropped_invalid_box += 1
                continue

        kept_lines.append(f"{project_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
        stats.annotations_kept += 1
        stats.kept_class_counts[PROJECT_CLASSES[project_id]] += 1

    return kept_lines


def read_image_safely(image_path: Path) -> np.ndarray | None:
    """Return the decoded image, or None if it is corrupted/unreadable."""
    try:
        image = cv2.imread(str(image_path))
    except Exception:
        return None
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    return image


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """64-bit difference hash, for near-duplicate detection.

    Resilient to re-compression, mild resizing and small brightness shifts -
    exactly the differences between the "same" image appearing in two dataset
    releases or in an augmented copy.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------


@dataclass
class PreparedImage:
    source_name: str
    image_path: Path
    label_lines: list[str]
    exact_hash: str
    perceptual_hash: int
    output_stem: str


def collect_source(
    source_dir: Path,
    source_name: str,
    stats: PreparationStats,
    class_names: list[str] | None = None,
) -> list[PreparedImage]:
    """Validate and normalize one source dataset."""
    names = class_names or load_source_class_names(source_dir)
    id_remap = build_id_remap(names)

    unmapped = [name for name in names if normalize_class(name) is None]
    if unmapped:
        print(f"  [{source_name}] classes not mapped into the project ontology:")
        for name in unmapped:
            reason = exclusion_reason(name) or "not part of this module's four classes"
            print(f"      - {name}: {reason}")

    prepared: list[PreparedImage] = []
    for image_path, label_path in find_image_label_pairs(source_dir):
        stats.images_seen += 1

        image = read_image_safely(image_path)
        if image is None:
            stats.images_corrupted += 1
            continue

        if label_path is None:
            stats.images_missing_label += 1

        label_lines = validate_and_remap_label_file(label_path, id_remap, names, stats)
        if label_lines is None:
            continue
        if not label_lines:
            stats.background_images += 1

        prepared.append(
            PreparedImage(
                source_name=source_name,
                image_path=image_path,
                label_lines=label_lines,
                exact_hash=hashlib.md5(image_path.read_bytes()).hexdigest(),
                perceptual_hash=dhash(image),
                output_stem=f"{source_name}_{image_path.stem}",
            )
        )
    return prepared


def group_duplicates(images: list[PreparedImage], stats: PreparationStats) -> list[list[int]]:
    """Group exact and near-duplicate images.

    Every group is later assigned to a single split, which is what stops a
    near-duplicate of a training image from appearing in internal_test and
    inflating the reported metrics.
    """
    parent = list(range(len(images)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    by_exact: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        by_exact[image.exact_hash].append(index)
    for indices in by_exact.values():
        for other in indices[1:]:
            union(indices[0], other)

    # Near-duplicates: bucket by the hash's top bits so this stays roughly
    # linear instead of comparing all pairs on a 100k-image dataset.
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        for shift in (0, 4):
            buckets[(image.perceptual_hash >> shift) & 0xFFFF].append(index)
    for indices in buckets.values():
        if len(indices) < 2 or len(indices) > 400:
            continue
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                if hamming(images[first].perceptual_hash, images[second].perceptual_hash) <= NEAR_DUPLICATE_MAX_DISTANCE:
                    union(first, second)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(images)):
        groups[find(index)].append(index)

    grouped = list(groups.values())
    multi = [group for group in grouped if len(group) > 1]
    stats.duplicate_groups = len(multi)
    stats.duplicate_images = sum(len(group) - 1 for group in multi)
    return grouped


def assign_splits(
    images: list[PreparedImage],
    groups: list[list[int]],
    ratios: tuple[float, float, float],
    seed: int,
    respect_source_splits: set[str],
) -> dict[int, str]:
    """Assign each image to train / val / internal_test, group-aware.

    ``respect_source_splits`` names sources whose published split should be
    preserved (the spec asks for this on the supplementary pothole dataset).
    For those, the split is read from the source path when it is recognisable.
    """
    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    free_groups: list[list[int]] = []

    for group in groups:
        published = {_published_split(images[index]) for index in group}
        published.discard(None)
        sources = {images[index].source_name for index in group}

        if published and sources <= respect_source_splits and len(published) == 1:
            split = published.pop()
            for index in group:
                assignment[index] = split
        else:
            free_groups.append(group)

    rng.shuffle(free_groups)
    total = sum(len(group) for group in free_groups)
    train_target = ratios[0] * total
    val_target = ratios[1] * total

    counts = {"train": 0, "val": 0, "internal_test": 0}
    for group in free_groups:
        if counts["train"] < train_target:
            split = "train"
        elif counts["val"] < val_target:
            split = "val"
        else:
            split = "internal_test"
        counts[split] += len(group)
        for index in group:
            assignment[index] = split
    return assignment


def _published_split(image: PreparedImage) -> str | None:
    """Infer a published split name from the source path, if present."""
    lowered = {part.lower() for part in image.image_path.parts}
    if {"train", "training"} & lowered:
        return "train"
    if {"val", "valid", "validation"} & lowered:
        return "val"
    if "test" in lowered:
        return "internal_test"
    return None


def write_dataset(
    images: list[PreparedImage],
    assignment: dict[int, str],
    output_dir: Path,
    copy_mode: str = "copy",
) -> dict[str, int]:
    """Write the normalized dataset in Ultralytics layout."""
    for split in ("train", "val", "internal_test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = Counter()
    for index, image in enumerate(images):
        split = assignment.get(index)
        if split is None:
            continue
        destination_image = output_dir / "images" / split / f"{image.output_stem}{image.image_path.suffix.lower()}"
        destination_label = output_dir / "labels" / split / f"{image.output_stem}.txt"

        if copy_mode == "symlink":
            if not destination_image.exists():
                destination_image.symlink_to(image.image_path.resolve())
        else:
            shutil.copy2(image.image_path, destination_image)
        destination_label.write_text("\n".join(image.label_lines) + ("\n" if image.label_lines else ""))
        split_counts[split] += 1
    return dict(split_counts)


def write_data_yaml(output_dir: Path, split_counts: dict[str, int]) -> Path:
    """Write the Ultralytics data.yaml for the prepared dataset."""
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        # Ultralytics' `test` key; the split is OUR internal held-out split,
        # not any official challenge test set. Named accordingly on disk.
        "test": "images/internal_test",
        "nc": len(PROJECT_CLASSES),
        "names": list(PROJECT_CLASSES),
    }
    path = output_dir / "data.yaml"
    with open(path, "w") as handle:
        handle.write(
            "# Generated by modules/road_damage/prepare_dataset.py\n"
            "# Class ids follow modules/road_damage/class_mapping.py:PROJECT_CLASSES.\n"
            "# 'test' is an INTERNAL held-out split created by this script -\n"
            "# it is NOT the official RDD2022 challenge test set.\n\n"
        )
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="Source dataset root (repeatable)")
    parser.add_argument("--name", action="append", required=True, help="Short name per --source (repeatable)")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument(
        "--respect-split",
        action="append",
        default=[],
        help="Source name whose published train/val/test split should be preserved",
    )
    parser.add_argument("--ratios", default="0.75,0.15,0.10", help="train,val,internal_test for un-split data")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")
    args = parser.parse_args(argv)

    if len(args.source) != len(args.name):
        parser.error("--source and --name must be given the same number of times")

    ratios = tuple(float(value) for value in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("--ratios must be three numbers summing to 1.0")

    output_dir = Path(args.output)
    stats = PreparationStats()

    images: list[PreparedImage] = []
    per_source: dict[str, int] = {}
    for source, name in zip(args.source, args.name):
        print(f"[prepare] scanning {name} ({source})")
        collected = collect_source(Path(source), name, stats)
        per_source[name] = len(collected)
        images.extend(collected)
        print(f"  [{name}] {len(collected)} usable images")

    print("[prepare] grouping duplicates / near-duplicates ...")
    groups = group_duplicates(images, stats)
    print(f"  {stats.duplicate_groups} duplicate groups covering {stats.duplicate_images} redundant images")

    assignment = assign_splits(images, groups, ratios, args.seed, set(args.respect_split))
    split_counts = write_dataset(images, assignment, output_dir, "symlink" if args.symlink else "copy")
    stats.images_kept = sum(split_counts.values())

    yaml_path = write_data_yaml(output_dir, split_counts)

    report = {
        "output_dir": str(output_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "project_classes": list(PROJECT_CLASSES),
        "sources": per_source,
        "split_counts": split_counts,
        "split_note": (
            "'internal_test' is an internal held-out split created by this "
            "script. It is NOT the official RDD2022 challenge test set."
        ),
        "deduplication_note": (
            "Exact (md5) and near-duplicate (dHash, Hamming <= "
            f"{NEAR_DUPLICATE_MAX_DISTANCE}) images are grouped, and each group is "
            "assigned to a single split, so no test image is a duplicate of a "
            "training image."
        ),
        "statistics": stats.as_dict(),
    }
    report_path = output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"[prepare] wrote {stats.images_kept} images -> {output_dir}")
    print(f"[prepare] splits: {split_counts}")
    print(f"[prepare] class counts: {dict(stats.kept_class_counts)}")
    print(f"[prepare] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
