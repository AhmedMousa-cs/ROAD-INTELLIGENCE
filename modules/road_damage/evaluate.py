"""
Evaluate the trained road-damage detector.

    python -m modules.road_damage.evaluate \
        --weights models/road_damage_best.pt \
        --data data/road_damage_prepared/data.yaml \
        --split val --split internal_test

Reports, per the spec: mAP50, mAP50-95, precision, recall, per-class AP, and
the confusion matrix - separately for each requested split, and separately
for each of the four classes.

Results are written to ``evaluation_results.json``. The split used is always
recorded alongside the numbers, and the internal test split is labelled as
such: it is a held-out split WE created, not the official RDD2022 challenge
test set.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from shared.config import resolve_device

from .class_mapping import PROJECT_CLASSES

SPLIT_NOTES = {
    "val": "Internal validation split created by prepare_dataset.py.",
    "internal_test": (
        "Internal held-out test split created by prepare_dataset.py before "
        "training. NOT the official RDD2022 challenge test set."
    ),
    "train": "Training split - reported only to check for overfitting.",
}


def _float(value) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def evaluate_split(weights: str, data_yaml: str, split: str, imgsz: int, device: str, conf: float, iou: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split="test" if split == "internal_test" else split,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    box = metrics.box
    per_class: dict[str, dict] = {}
    # ap_class_index maps row order in the per-class arrays to class ids.
    class_indices = list(getattr(box, "ap_class_index", range(len(PROJECT_CLASSES))))
    for row, class_id in enumerate(class_indices):
        name = PROJECT_CLASSES[int(class_id)] if int(class_id) < len(PROJECT_CLASSES) else str(class_id)
        try:
            per_class[name] = {
                "precision": _float(box.p[row]),
                "recall": _float(box.r[row]),
                "AP50": _float(box.ap50[row]),
                "AP50_95": _float(box.ap[row]),
            }
        except (IndexError, TypeError):
            per_class[name] = {"precision": None, "recall": None, "AP50": None, "AP50_95": None}

    result = {
        "split": split,
        "split_note": SPLIT_NOTES.get(split, ""),
        "overall": {
            "mAP50": _float(box.map50),
            "mAP50_95": _float(box.map),
            "precision": _float(box.mp),
            "recall": _float(box.mr),
        },
        "per_class": per_class,
        "thresholds": {"confidence": conf, "iou": iou, "image_size": imgsz},
    }

    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is not None and hasattr(confusion, "matrix"):
        matrix = confusion.matrix
        result["confusion_matrix"] = {
            "labels": list(PROJECT_CLASSES) + ["background"],
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
            "note": "Rows = predicted, columns = ground truth (Ultralytics convention).",
        }
    result["artifacts_dir"] = str(getattr(metrics, "save_dir", ""))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", action="append", default=None, help="Repeatable; default: val + internal_test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Low conf is standard for mAP computation")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="modules/road_damage/evaluation_results.json")
    args = parser.parse_args(argv)

    splits = args.split or ["val", "internal_test"]
    device = resolve_device(args.device)

    report = {
        "module": "road_damage",
        "weights": str(Path(args.weights).resolve()),
        "data_yaml": str(Path(args.data).resolve()),
        "classes": list(PROJECT_CLASSES),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": device},
        "splits": {},
    }

    for split in splits:
        print(f"[evaluate] {split} ...")
        report["splits"][split] = evaluate_split(
            args.weights, args.data, split, args.imgsz, device, args.conf, args.iou
        )
        overall = report["splits"][split]["overall"]
        print(f"  mAP50={overall['mAP50']} mAP50-95={overall['mAP50_95']} P={overall['precision']} R={overall['recall']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"[evaluate] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
