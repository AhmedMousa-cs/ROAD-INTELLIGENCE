"""
Evaluation for the tracking + analytics module.

What can and cannot be reported here
------------------------------------
This module trains nothing. The vehicle detector is pretrained COCO YOLO, so
its detection accuracy is whatever Ultralytics published for those weights -
we do not re-derive or restate those numbers here, and the README points at
the source instead of inventing figures.

What IS ours to measure, and what this script measures:

1. ``throughput``  - frames/second for the analytics stage and (optionally)
   the full pipeline including detection, on the machine you run it on.
2. ``tracking``    - identity quality against MOT-format ground truth, if you
   have any: ID switches, fragmentations, mostly-tracked ratio, and a
   simplified MOTA. Uses the ``motmetrics`` package when installed (proper
   MOTA/IDF1); falls back to a self-contained approximation otherwise, which
   is labelled as such in the output.
3. ``speed``       - estimated vs. known speed, on either a synthetic scene
   with simulated ground truth or a real clip where you know the true speeds.

Run:
    python -m modules.tracking_analytics.evaluation --synthetic \\
        --output evaluation_results.json

    python -m modules.tracking_analytics.evaluation --video clip.mp4 \\
        --gt gt.txt --weights models/vehicle_detection_yolov8n.pt \\
        --output evaluation_results.json

Ground-truth format (MOT16 ``gt.txt``, one detection per line):
    frame_id, track_id, x, y, width, height, conf, class, visibility
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .module_config import TrackingAnalyticsConfig
from .tracker import iou_matrix


# -----------------------------------------------------------------------------
# Ground truth
# -----------------------------------------------------------------------------


def load_mot_ground_truth(path: str | Path) -> dict[int, list[tuple[int, list[float]]]]:
    """Parse an MOT-format gt.txt into {frame_id: [(track_id, xyxy), ...]}."""
    by_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            frame_id, track_id = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
            by_frame[frame_id].append((track_id, [x, y, x + w, y + h]))
    return dict(by_frame)


# -----------------------------------------------------------------------------
# Tracking metrics
# -----------------------------------------------------------------------------


def evaluate_tracking(
    predictions: dict[int, list[tuple[int, list[float]]]],
    ground_truth: dict[int, list[tuple[int, list[float]]]],
    iou_threshold: float = 0.5,
) -> dict:
    """Identity-quality metrics. Uses motmetrics when available.

    The fallback is a simplified implementation: greedy IoU matching per
    frame, counting matches, misses, false positives and identity switches. It
    is close to MOTA but is NOT the official implementation, and the returned
    dict says so in ``method``.
    """
    try:
        import motmetrics as mm

        accumulator = mm.MOTAccumulator(auto_id=False)
        for frame_id in sorted(set(ground_truth) | set(predictions)):
            gt_items = ground_truth.get(frame_id, [])
            pred_items = predictions.get(frame_id, [])
            gt_ids = [i for i, _ in gt_items]
            pred_ids = [i for i, _ in pred_items]
            if gt_items and pred_items:
                gt_boxes = np.array([b for _, b in gt_items], dtype=np.float32)
                pred_boxes = np.array([b for _, b in pred_items], dtype=np.float32)
                distances = 1.0 - iou_matrix(gt_boxes, pred_boxes)
                distances[distances > 1.0 - iou_threshold] = np.nan
            else:
                distances = np.empty((len(gt_ids), len(pred_ids)))
                distances[:] = np.nan
            accumulator.update(gt_ids, pred_ids, distances, frameid=frame_id)

        metrics = mm.metrics.create()
        summary = metrics.compute(
            accumulator,
            metrics=["mota", "motp", "idf1", "num_switches", "num_fragmentations",
                     "mostly_tracked", "mostly_lost", "num_misses", "num_false_positives"],
            name="tracking",
        )
        row = summary.loc["tracking"].to_dict()
        row["method"] = "motmetrics (official implementation)"
        row["iou_threshold"] = iou_threshold
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}
    except ImportError:
        pass

    matches = misses = false_positives = switches = 0
    gt_to_pred: dict[int, int] = {}
    frames_tracked: dict[int, int] = defaultdict(int)
    frames_total: dict[int, int] = defaultdict(int)
    iou_sum = 0.0

    for frame_id in sorted(set(ground_truth) | set(predictions)):
        gt_items = ground_truth.get(frame_id, [])
        pred_items = predictions.get(frame_id, [])
        for gt_id, _ in gt_items:
            frames_total[gt_id] += 1

        if gt_items and pred_items:
            gt_boxes = np.array([b for _, b in gt_items], dtype=np.float32)
            pred_boxes = np.array([b for _, b in pred_items], dtype=np.float32)
            ious = iou_matrix(gt_boxes, pred_boxes)
            used_gt: set[int] = set()
            used_pred: set[int] = set()
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for r, c in order:
                r, c = int(r), int(c)
                if r in used_gt or c in used_pred or ious[r, c] < iou_threshold:
                    continue
                used_gt.add(r)
                used_pred.add(c)
                gt_id = gt_items[r][0]
                pred_id = pred_items[c][0]
                matches += 1
                iou_sum += float(ious[r, c])
                frames_tracked[gt_id] += 1
                if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                    switches += 1
                gt_to_pred[gt_id] = pred_id
            misses += len(gt_items) - len(used_gt)
            false_positives += len(pred_items) - len(used_pred)
        else:
            misses += len(gt_items)
            false_positives += len(pred_items)

    gt_total = sum(frames_total.values())
    mota = 1.0 - (misses + false_positives + switches) / gt_total if gt_total else None
    mostly_tracked = sum(
        1 for gt_id, total in frames_total.items() if frames_tracked[gt_id] / total >= 0.8
    )
    return {
        "method": "simplified fallback (install motmetrics for official MOTA/IDF1)",
        "iou_threshold": iou_threshold,
        "matches": matches,
        "misses": misses,
        "false_positives": false_positives,
        "num_switches": switches,
        "mota_approx": round(mota, 4) if mota is not None else None,
        "motp_iou_mean": round(iou_sum / matches, 4) if matches else None,
        "mostly_tracked": mostly_tracked,
        "gt_tracks": len(frames_total),
    }


# -----------------------------------------------------------------------------
# Synthetic benchmark (no weights, no video required)
# -----------------------------------------------------------------------------


def synthetic_benchmark(frames: int = 300, jitter_px: float = 0.0, seed: int = 0) -> dict:
    """Measure the analytics stage in isolation and validate speed estimation.

    The scene is generated in world metres and projected through a known
    homography, so the true speed of every vehicle is known exactly and the
    estimator can be scored against it. Detection is NOT included here: the
    scripted detector stands in for YOLO, so the FPS figure is the analytics
    stage only (tracking, speed, wrong-way, counting, scoring).
    """
    from .inference import TrackingAnalytics
    from .speed_estimation import SceneCalibration
    from .vehicle_detector import ScriptedVehicleDetector

    fps = 25.0
    calibration_data = {
        "image_points": [[120, 700], [840, 700], [620, 320], [340, 320]],
        "world_points": [[0, 0], [7.0, 0], [7.0, 40.0], [0, 40.0]],
    }
    calibration = SceneCalibration(**calibration_data)
    true_speeds = {"lane_a": 45.0, "lane_b": 30.0, "lane_c": 60.0}

    rng = np.random.default_rng(seed)
    script: list[list[dict]] = []
    for i in range(frames):
        detections = []
        for lane_index, (lane, speed_kmh) in enumerate(true_speeds.items()):
            lane_x = 1.0 + 2.5 * lane_index
            # Vehicles recycle every 60 frames so tracks are created and
            # retired repeatedly, which is what stresses the tracker.
            world_y = 2.0 + (speed_kmh / 3.6 / fps) * (i % 60)
            point = calibration.to_image([[lane_x, world_y]])[0]
            cx, cy = float(point[0]), float(point[1])
            if jitter_px:
                # Simulates detector box jitter, which is the dominant source
                # of speed error once a calibration is correct.
                cx += float(rng.normal(0.0, jitter_px))
                cy += float(rng.normal(0.0, jitter_px))
            detections.append(
                {
                    "class": "car",
                    "confidence": 0.9,
                    "bbox": [cx - 40, cy - 60, cx + 40, cy],
                }
            )
        script.append(detections)

    config = TrackingAnalyticsConfig.from_dict(
        {"fps": fps, "speed": {"calibration": calibration_data}}
    )
    pipeline = TrackingAnalytics(config=config, detector=ScriptedVehicleDetector(script))
    frame = np.zeros((720, 960, 3), dtype=np.uint8)

    # Warm-up: the first few calls pay one-off costs (lazy imports, numpy
    # buffer allocation, OpenCV init). Timing them would make whichever
    # scenario runs first look slower than the others, so they are run on a
    # throwaway pipeline before the clock starts.
    warmup_pipeline = TrackingAnalytics(
        config=TrackingAnalyticsConfig.from_dict(
            {"fps": fps, "speed": {"calibration": calibration_data}}
        ),
        detector=ScriptedVehicleDetector(script),
    )
    warmup_frame = np.zeros((720, 960, 3), dtype=np.uint8)
    for _ in range(min(20, frames)):
        warmup_pipeline.process(warmup_frame)

    errors: list[float] = []
    started = time.perf_counter()
    for _ in range(frames):
        result = pipeline.process(frame)
        for track, (lane, true_speed) in zip(result["tracks"], true_speeds.items()):
            estimated = track.get("estimated_speed_kmh")
            if estimated is not None:
                errors.append(abs(estimated - true_speed) / true_speed)
    elapsed = max(time.perf_counter() - started, 1e-9)

    return {
        "frames": frames,
        "box_jitter_px_sigma": jitter_px,
        "analytics_fps": round(frames / elapsed, 1),
        "analytics_ms_per_frame": round(elapsed / frames * 1000, 3),
        "note": (
            "Analytics stage only (tracking, speed, wrong-way, counting, "
            "scoring). Detection is replaced by a scripted stand-in, so this "
            "is NOT end-to-end pipeline FPS."
        ),
        "speed_validation": {
            "samples": len(errors),
            "mean_absolute_percentage_error": (
                round(float(np.mean(errors)) * 100, 2) if errors else None
            ),
            "max_absolute_percentage_error": (
                round(float(np.max(errors)) * 100, 2) if errors else None
            ),
            "note": (
                "Against simulated ground truth with a perfect homography. "
                "Real-world error will be larger still: it is dominated by "
                "calibration accuracy and detector box jitter, not by this "
                "arithmetic."
            ),
        },
    }


# -----------------------------------------------------------------------------
# Video benchmark
# -----------------------------------------------------------------------------


def video_benchmark(
    video: str, weights: str | None, config_path: str | None, max_frames: int | None
) -> tuple[dict, dict[int, list[tuple[int, list[float]]]]]:
    from .inference import TrackingAnalytics

    config = (
        TrackingAnalyticsConfig.from_yaml(config_path)
        if config_path
        else TrackingAnalyticsConfig()
    )
    pipeline = TrackingAnalytics(weights, config=config)

    predictions: dict[int, list[tuple[int, list[float]]]] = {}
    started = time.perf_counter()
    frames = 0
    for frame_id, _, result in pipeline.process_video(video, max_frames=max_frames):
        predictions[frame_id] = [
            (int(t["track_id"]), [float(v) for v in t["bbox"]]) for t in result["tracks"]
        ]
        frames += 1
    elapsed = max(time.perf_counter() - started, 1e-9)

    return (
        {
            "video": str(video),
            "frames": frames,
            "pipeline_fps": round(frames / elapsed, 2),
            "ms_per_frame": round(elapsed / frames * 1000, 2) if frames else None,
            "tracker_backend": pipeline.tracker.backend_in_use,
            "note": "End-to-end: detection + tracking + analytics.",
        },
        predictions,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def environment_info() -> dict:
    info = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the tracking/analytics module")
    parser.add_argument("--synthetic", action="store_true", help="Run the synthetic benchmark")
    parser.add_argument("--video", default=None, help="Video to benchmark end-to-end")
    parser.add_argument("--gt", default=None, help="MOT-format ground truth for --video")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default=None, help="Scene config YAML")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frames", type=int, default=300, help="Synthetic frame count")
    parser.add_argument("--output", default=None, help="Write results JSON here")
    args = parser.parse_args(argv)

    if not args.synthetic and not args.video:
        parser.error("pass --synthetic and/or --video")

    results: dict = {
        "module": "tracking_analytics",
        "environment": environment_info(),
        "scope_note": (
            "This module trains no model. Vehicle detection accuracy is that "
            "of the pretrained COCO weights in use; see the Ultralytics model "
            "card for those figures. Everything measured here is the tracking "
            "and analytics layer."
        ),
    }

    if args.synthetic:
        # Two runs: noise-free (validates the geometry) and with simulated
        # detector box jitter (shows how sensitive the estimate really is).
        results["synthetic_benchmark"] = synthetic_benchmark(args.frames, jitter_px=0.0)
        results["synthetic_benchmark_with_box_jitter"] = synthetic_benchmark(
            args.frames, jitter_px=3.0
        )

    if args.video:
        throughput, predictions = video_benchmark(
            args.video, args.weights, args.config, args.max_frames
        )
        results["video_benchmark"] = throughput
        if args.gt:
            results["tracking_metrics"] = evaluate_tracking(
                predictions, load_mot_ground_truth(args.gt)
            )
        else:
            results["tracking_metrics"] = (
                "Not computed: no --gt supplied. Tracking quality cannot be "
                "reported without ground-truth identities."
            )

    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
