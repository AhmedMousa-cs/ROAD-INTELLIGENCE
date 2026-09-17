"""
Road-damage inference entry point (images, folders, video) + CLI.

    # single image or a folder of images
    python -m modules.road_damage.inference \
        --source data/samples/road.jpg \
        --weights models/road_damage_best.pt \
        --json out/damage.json --image-out out/annotated.jpg

    # video
    python -m modules.road_damage.inference \
        --source road.mp4 --weights models/road_damage_best.pt \
        --json out/damage.jsonl --video-out out/annotated.mp4

    # no weights needed - synthetic scene with a scripted detector
    python -m modules.road_damage.inference --demo

Production inference lives in ``.py`` files only; no notebook is required to
run any of this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .road_damage_model import RoadDamageModel, ScriptedDamageDetector
from .visualization import draw_frame_result

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def predict_image(model: RoadDamageModel, image_path: str | Path) -> dict:
    """Run the model on one image file."""
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    result = model.predict(frame)
    result["image_path"] = str(image_path)
    return result


def predict_folder(model: RoadDamageModel, folder: str | Path) -> list[dict]:
    """Run the model on every image in a folder (non-recursive)."""
    paths = sorted(
        path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    results = []
    for index, path in enumerate(paths):
        result = predict_image(model, path)
        result["frame_id"] = index
        results.append(result)
    return results


def predict_video(
    model: RoadDamageModel,
    source: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Yield ``(frame_id, frame, result)`` for each processed video frame.

    A convenience for scripts - this module does not own a UI.
    """
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    try:
        index = -1
        processed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if stride > 1 and index % stride:
                continue
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            result = model.predict(frame, frame_id=index, timestamp=max(timestamp, 0.0))
            yield index, frame, result
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
    finally:
        capture.release()


# -----------------------------------------------------------------------------
# Demo (no weights required)
# -----------------------------------------------------------------------------


def _demo_frame(width: int = 960, height: int = 720) -> np.ndarray:
    """A synthetic road surface, so the demo is not drawn on a black square."""
    import cv2

    frame = np.full((height, width, 3), 70, dtype=np.uint8)
    frame[: height // 3, :] = (120, 110, 100)          # sky/horizon band
    cv2.rectangle(frame, (0, height // 3), (width, height), (78, 78, 80), -1)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 7, (height, width, 3))
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.line(frame, (width // 2, height), (width // 2 - 40, height // 3), (200, 200, 200), 6)
    return frame


def run_demo(json_out: str | None = None, image_out: str | None = None) -> dict:
    """Run the full pipeline with a scripted detector and no weights."""
    import cv2

    script = [
        [
            {"class": "D40", "confidence": 0.93, "bbox": [300, 430, 430, 530]},
            {"class": "D00", "confidence": 0.71, "bbox": [520, 400, 600, 690]},
            {"class": "D20", "confidence": 0.64, "bbox": [120, 520, 300, 640]},
            {"class": "D10", "confidence": 0.55, "bbox": [640, 470, 900, 520]},
            # Excluded label: must be dropped, never emitted.
            {"class": "D44", "confidence": 0.88, "bbox": [10, 600, 110, 640]},
        ]
    ]
    model = RoadDamageModel(detector=ScriptedDamageDetector(script))
    frame = _demo_frame()
    result = model.predict(frame, frame_id=0, timestamp=0.0)

    if image_out:
        Path(image_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_out), draw_frame_result(frame, result))
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(result, indent=2))

    print(json.dumps({
        "detections": len(result["detections"]),
        "classes": sorted({d["class"] for d in result["detections"]}),
        "dropped_excluded_labels": 1,
        "note": "D44 (white-line blur) was dropped by class_mapping, as designed.",
    }, indent=2))
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run road-damage detection.")
    parser.add_argument("--source", help="Image, folder of images, or video")
    parser.add_argument("--weights", default=None, help="Defaults to models/road_damage_best.pt")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", dest="json_out", default=None, help="Write results here")
    parser.add_argument("--image-out", default=None, help="Annotated image output")
    parser.add_argument("--video-out", default=None, help="Annotated video output")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--demo", action="store_true", help="Synthetic scene, no weights needed")
    args = parser.parse_args(argv)

    if args.demo:
        run_demo(args.json_out, args.image_out)
        return 0
    if not args.source:
        parser.error("--source is required (or use --demo)")

    overrides = {
        key: value
        for key, value in {
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "image_size": args.imgsz,
            "device": args.device,
        }.items()
        if value is not None
    }
    model = RoadDamageModel(args.weights, config=overrides or None)

    import cv2

    source = Path(args.source)
    if source.is_dir():
        results = predict_folder(model, source)
    elif source.suffix.lower() in VIDEO_SUFFIXES:
        results = []
        writer = None
        for frame_id, frame, result in predict_video(model, source, args.max_frames, args.stride):
            results.append(result)
            if args.video_out:
                annotated = draw_frame_result(frame, result)
                if writer is None:
                    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.video_out,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25.0,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                writer.write(annotated)
        if writer is not None:
            writer.release()
            print(f"[inference] wrote {args.video_out}")
    else:
        results = [predict_image(model, source)]
        if args.image_out:
            frame = cv2.imread(str(source))
            Path(args.image_out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.image_out), draw_frame_result(frame, results[0]))
            print(f"[inference] wrote {args.image_out}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1:
            Path(args.json_out).write_text(json.dumps(results[0], indent=2))
        else:
            with open(args.json_out, "w") as handle:
                for result in results:
                    handle.write(json.dumps(result) + "\n")
        print(f"[inference] wrote {args.json_out}")

    total = sum(len(result["detections"]) for result in results)
    print(f"[inference] {len(results)} frame(s), {total} detection(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
