"""
modules/obstacles/training/train_speed_bump.py
=================================================
Trains the speed-bump YOLO detector via transfer learning. Unlike the
debris dataset, "Speed Bump Detection v10" ships its own train/val/test
split (5498/266/240 images per its Roboflow page) -- this script does NOT
re-split it. Download it in YOLOv8 format from Roboflow, point --data at
its data.yaml directly (after verifying its `names:` list only contains
what you expect -- see class_mapping.SPEED_BUMP_RAW_TO_NORMALIZED), and run:

    python -m modules.obstacles.training.train_speed_bump \\
        --data /path/to/speed-bump-detection-v10/data.yaml \\
        --epochs 100 --batch 16 --imgsz 640 \\
        --weights yolov8n.pt

Same run-recording behavior as train_debris.py -- see that file's
docstring.
"""

from __future__ import annotations

import argparse
import json
import time
import yaml
from pathlib import Path

from shared.config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD, resolve_device
from modules.obstacles.class_mapping import SPEED_BUMP_RAW_TO_NORMALIZED


def _check_dataset_names(data_yaml: Path) -> None:
    """Fail loudly (not force-map) if the downloaded dataset's names: list
    contains anything class_mapping.py doesn't already recognize -- per
    spec: 'Use its actual YAML class definitions. Do not assume additional
    classes exist.'
    """
    with open(data_yaml) as f:
        spec = yaml.safe_load(f)
    names = spec.get("names")
    if isinstance(names, dict):
        names = list(names.values())
    unrecognized = [n for n in (names or []) if n not in SPEED_BUMP_RAW_TO_NORMALIZED]
    if unrecognized:
        raise ValueError(
            f"data.yaml at '{data_yaml}' has class name(s) {unrecognized} not present in "
            f"class_mapping.SPEED_BUMP_RAW_TO_NORMALIZED. Inspect the dataset and either exclude "
            f"these classes or extend the mapping deliberately before training -- do not force it."
        )


def train(
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    weights: str,
    project: Path,
    name: str,
) -> Path:
    _check_dataset_names(data_yaml)

    from ultralytics import YOLO

    device = resolve_device()
    model = YOLO(weights)

    run_config = {
        "model": weights,
        "epochs": epochs,
        "batch_size": batch,
        "image_size": imgsz,
        "optimizer": "auto",
        "device": device,
        "data": str(data_yaml),
        "confidence_threshold_for_eval": CONFIDENCE_THRESHOLD,
        "iou_threshold_for_eval": IOU_THRESHOLD,
        "shared_image_size": IMAGE_SIZE,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Speed Bump Detection v10 ships its own train/val/test split; not re-split here.",
    }
    start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    run_config["training_time_seconds"] = round(time.time() - start, 1)

    # Ultralytics may nest the actual output under its own runs/<task>/...
    # prefix regardless of the `project` we passed, so ask it directly for
    # the real path rather than reconstructing project/name ourselves.
    run_dir = Path(getattr(results, "save_dir", project / name))
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"[train_speed_bump] training config recorded at {run_dir / 'run_config.json'}")
    print(f"[train_speed_bump] best weights at {run_dir / 'weights' / 'best.pt'}")
    print("[train_speed_bump] copy that file to models/speed_bump_best.pt once you're satisfied with it")
    return run_dir / "weights" / "best.pt"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the speed-bump YOLO detector.")
    parser.add_argument("--data", required=True, type=Path, help="Path to the downloaded dataset's data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="speed_bump")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(args.data, args.epochs, args.batch, args.imgsz, args.weights, args.project, args.name)
