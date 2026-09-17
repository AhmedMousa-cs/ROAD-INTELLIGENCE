"""
modules/obstacles/training/train_debris.py
=============================================
Trains the road-debris YOLO detector via transfer learning from a
pretrained checkpoint. Run this LOCALLY, after
prepare_taco_dataset.py has produced data/taco_road_debris/data.yaml.

    python -m modules.obstacles.training.train_debris \\
        --data data/taco_road_debris/data.yaml \\
        --epochs 100 --batch 16 --imgsz 640 \\
        --weights yolov8n.pt

Writes results (including the trained weights) to Ultralytics' own
runs/detect/<name>/ directory. Copy the resulting best.pt to
models/debris_best.pt when you're happy with it -- see
"11. MODEL NAMING" in the spec for why NOT to leave it as best.pt in models/.

This script only orchestrates training/records the run config; it does not
itself claim any metrics -- run evaluate_debris.py against the internal
held-out test split afterward and copy the real numbers into
evaluation_results.json / the module README.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from shared.config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD, resolve_device


def train(
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    weights: str,
    project: Path,
    name: str,
) -> Path:
    from ultralytics import YOLO

    device = resolve_device()
    model = YOLO(weights)

    run_config = {
        "model": weights,
        "epochs": epochs,
        "batch_size": batch,
        "image_size": imgsz,
        "optimizer": "auto",  # Ultralytics default (SGD/Adam auto-selection) -- record whichever it picks
        "device": device,
        "data": str(data_yaml),
        "confidence_threshold_for_eval": CONFIDENCE_THRESHOLD,
        "iou_threshold_for_eval": IOU_THRESHOLD,
        "shared_image_size": IMAGE_SIZE,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    start = time.time()
    model.train(
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

    run_dir = project / name
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"[train_debris] training config recorded at {run_dir / 'run_config.json'}")
    print(f"[train_debris] best weights at {run_dir / 'weights' / 'best.pt'}")
    print("[train_debris] copy that file to models/debris_best.pt once you're satisfied with it")
    return run_dir / "weights" / "best.pt"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the road-debris YOLO detector.")
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml from prepare_taco_dataset.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolov8n.pt", help="Pretrained checkpoint for transfer learning")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="road_debris")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train(args.data, args.epochs, args.batch, args.imgsz, args.weights, args.project, args.name)
