"""
modules/road_segmentation/inference.py

Standalone CLI helpers for running the segmentation model outside of the
final integration pipeline — useful for sanity-checking a freshly trained
checkpoint on a single image/video before handing it to the team.

    python -m modules.road_segmentation.inference \
        --model models/road_segmentation_best.pt \
        --image data/samples/road1.jpg \
        --out outputs/road1_segmented.jpg

    python -m modules.road_segmentation.inference \
        --model models/road_segmentation_best.pt \
        --video data/samples/clip.mp4 \
        --out outputs/clip_segmented.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .road_segmentation_model import RoadSegmentationModel
from .visualization import draw_segmentations


def run_on_image(model: RoadSegmentationModel, image_path: str, out_path: str | None = None) -> dict:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = model.predict(frame)

    if out_path:
        vis = draw_segmentations(frame, result["segmentations"])
        cv2.imwrite(out_path, vis)

    return result


def run_on_video(model: RoadSegmentationModel, video_path: str, out_path: str | None = None, stride: int = 1) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    writer = None
    results = []
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride == 0:
                result = model.predict(frame)
                result["frame_id"] = frame_id
                result["timestamp"] = round(frame_id / fps, 3)
                results.append(result)

                if out_path:
                    if writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    vis = draw_segmentations(frame, result["segmentations"])
                    writer.write(vis)
            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return results


def _main():
    parser = argparse.ArgumentParser(description="Road segmentation standalone inference")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--out", help="Path to write visualization output")
    parser.add_argument("--json-out", help="Path to write raw results as JSON")
    parser.add_argument("--dataset", default="roboflow_road_defect", choices=["pothole_seg", "roboflow_road_defect"])
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("Provide --image or --video")

    model = RoadSegmentationModel(args.model, dataset=args.dataset)

    if args.image:
        result = run_on_image(model, args.image, args.out)
        print(json.dumps(result, indent=2, default=str))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2, default=str))
    else:
        results = run_on_video(model, args.video, args.out)
        print(f"Processed {len(results)} frames")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    _main()
