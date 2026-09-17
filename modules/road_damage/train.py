"""
Train the road-damage detector.

    python -m modules.road_damage.train \
        --data data/road_damage_prepared/data.yaml \
        --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640

Transfer learning from pretrained COCO weights is the default; training from
scratch is possible but not recommended (``--scratch``).

Every hyperparameter the spec asks to be recorded (model, epochs, batch size,
image size, learning rate, optimizer, augmentation, device, training time) is
written to ``training_run.json`` next to the exported weights, so the README's
training table can be filled in from a file rather than from memory.

On completion the best checkpoint is copied to ``models/road_damage_best.pt``
- the project naming convention, not Ultralytics' ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from shared.config import MODELS_DIR, resolve_device

from .class_mapping import PROJECT_CLASSES

EXPORTED_WEIGHTS_NAME = "road_damage_best.pt"

# Augmentation tuned for road imagery. Two deliberate choices:
#   * flipud=0.0 - road damage is photographed from a vehicle/phone looking
#     down at the surface; a vertically flipped road is not a real view, and
#     flipping it would swap the visual signature of longitudinal vs
#     transverse cracks relative to the camera.
#   * degrees kept small - a large rotation turns a longitudinal crack into a
#     transverse one, i.e. it would relabel the object.
ROAD_AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.4,   # road images vary a lot in exposure/shadow
    "degrees": 3.0,
    "translate": 0.1,
    "scale": 0.4,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 10,  # disable mosaic for the last epochs to stabilise boxes
}


def _verify_dataset_classes(data_yaml: Path) -> None:
    """Fail fast if the dataset is not the normalized one."""
    with open(data_yaml) as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if list(names or []) != list(PROJECT_CLASSES):
        raise SystemExit(
            f"{data_yaml} declares classes {names}, expected {list(PROJECT_CLASSES)}.\n"
            f"Train on the output of `python -m modules.road_damage.prepare_dataset`, "
            f"not on a raw dataset - otherwise the model's class ids will not match "
            f"class_mapping.PROJECT_CLASSES."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Prepared data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--optimizer", default="auto", choices=["auto", "SGD", "Adam", "AdamW"])
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/road_damage")
    parser.add_argument("--name", default="train")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (not recommended)")
    parser.add_argument("--export-to", default=str(MODELS_DIR / EXPORTED_WEIGHTS_NAME))
    args = parser.parse_args(argv)

    data_yaml = Path(args.data)
    _verify_dataset_classes(data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required to train: pip install ultralytics") from exc

    device = resolve_device(args.device)
    if device == "cpu":
        print("[train] WARNING: no CUDA device found. Training on CPU will be very slow.")

    weights = args.model.replace(".pt", ".yaml") if args.scratch else args.model
    model = YOLO(weights)

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=not args.scratch,
        **ROAD_AUGMENTATION,
    )
    elapsed = time.perf_counter() - started

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    export_path = Path(args.export_to)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, export_path)
        print(f"[train] exported {best} -> {export_path}")
    else:
        print(f"[train] WARNING: expected checkpoint not found at {best}")

    run_record = {
        "exported_weights": str(export_path),
        "ultralytics_run_dir": str(save_dir),
        "started_at_utc": started_at,
        "training_time_seconds": round(elapsed, 1),
        "training_time_human": f"{elapsed / 3600:.2f} h",
        "hyperparameters": {
            "model": args.model,
            "pretrained": not args.scratch,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "image_size": args.imgsz,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "device": device,
            "workers": args.workers,
        },
        "augmentation": ROAD_AUGMENTATION,
        "classes": list(PROJECT_CLASSES),
        "data_yaml": str(data_yaml.resolve()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    record_path = save_dir / "training_run.json"
    record_path.write_text(json.dumps(run_record, indent=2))
    print(f"[train] run record: {record_path}")
    print(f"[train] next: python -m modules.road_damage.evaluate --weights {export_path} --data {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
