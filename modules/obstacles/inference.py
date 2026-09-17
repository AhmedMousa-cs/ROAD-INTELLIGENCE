"""
modules/obstacles/inference.py
===============================
Standalone inference helpers + CLI for the obstacles module. This is what
"11. REQUIRED ARTIFACTS" / "13. PRODUCTION REQUIREMENT" mean by an
"inference Python module" that works without notebooks or a training
environment -- run it directly:

    python -m modules.obstacles.inference \\
        --debris-model models/debris_best.pt \\
        --speed-bump-model models/speed_bump_best.pt \\
        --image path/to/road.jpg \\
        --out annotated.jpg

    python -m modules.obstacles.inference \\
        --debris-model models/debris_best.pt \\
        --speed-bump-model models/speed_bump_best.pt \\
        --video path/to/road.mp4 \\
        --out annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .obstacle_model import RoadObstacleModel
from .visualization import draw_detections


def run_on_image(model: RoadObstacleModel, image_path: str | Path) -> tuple[np.ndarray, dict]:
    """Load one image file, run the combined obstacle model, return
    (frame, result) where result is shaped like shared.schemas.empty_frame_result().
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    result = model.predict(frame)
    return frame, result


def run_on_video(
    model: RoadObstacleModel,
    video_path: str | Path,
    out_path: str | Path | None = None,
    sample_every_n_frames: int = 1,
) -> list[dict]:
    """Run the combined obstacle model over every (or every Nth) frame of a
    video file. If `out_path` is given, writes an annotated video alongside
    it. Returns the list of per-frame result dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    results: list[dict] = []
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % sample_every_n_frames == 0:
                result = model.predict(frame)
                result["frame_id"] = frame_id
                result["timestamp"] = frame_id / fps
                results.append(result)

                if writer is not None:
                    writer.write(draw_detections(frame, result["detections"]))
            elif writer is not None:
                writer.write(frame)

            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the road-obstacles module on an image or video.")
    parser.add_argument("--debris-model", required=True, help="Path to debris_best.pt")
    parser.add_argument("--speed-bump-model", required=True, help="Path to speed_bump_best.pt")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--out", help="Optional path to write an annotated image/video to")
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=1,
        help="For --video: only run inference every Nth frame (default: every frame)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.image and not args.video:
        print("Error: pass --image or --video", file=sys.stderr)
        return 2
    if args.image and args.video:
        print("Error: pass only one of --image / --video", file=sys.stderr)
        return 2

    model = RoadObstacleModel(args.debris_model, args.speed_bump_model)

    if args.image:
        frame, result = run_on_image(model, args.image)
        print(json.dumps(result, default=str, indent=2))
        if args.out:
            annotated = draw_detections(frame, result["detections"])
            cv2.imwrite(args.out, annotated)
            print(f"Wrote annotated image to {args.out}", file=sys.stderr)
    else:
        results = run_on_video(
            model, args.video, out_path=args.out, sample_every_n_frames=args.sample_every_n_frames
        )
        print(json.dumps(results, default=str, indent=2))
        if args.out:
            print(f"Wrote annotated video to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
