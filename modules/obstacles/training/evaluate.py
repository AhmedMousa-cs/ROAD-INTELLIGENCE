"""
modules/obstacles/training/evaluate.py
=========================================
Runs Ultralytics' built-in validation on a trained checkpoint against a
given data.yaml split, and writes real mAP50 / mAP50-95 / precision /
recall numbers to evaluation_results.json. Run this locally against your
own trained weights -- do not hand-copy numbers from anywhere else (spec
section 15: don't invent dataset statistics).

    python -m modules.obstacles.training.evaluate \\
        --weights models/debris_best.pt \\
        --data data/taco_road_debris/data.yaml \\
        --split test \\
        --out modules/obstacles/evaluation_results_debris.json

    python -m modules.obstacles.training.evaluate \\
        --weights models/speed_bump_best.pt \\
        --data /path/to/speed-bump-v10/data.yaml \\
        --split test \\
        --out modules/obstacles/evaluation_results_speed_bump.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.config import CONFIDENCE_THRESHOLD, IMAGE_SIZE, IOU_THRESHOLD, resolve_device


def evaluate(weights: Path, data_yaml: Path, split: str, out_path: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=resolve_device(),
    )

    results = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split_evaluated": split,
        "split_is_official_test_set": False,  # true only if you've verified the dataset itself calls it that
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_AP50": {
            name: float(ap) for name, ap in zip(metrics.names.values(), metrics.box.ap50)
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[evaluate] wrote {out_path}")
    print(json.dumps(results, indent=2))
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained obstacles-module detector.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", required=True, type=Path)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    evaluate(args.weights, args.data, args.split, args.out)
