"""
Vehicle tracking + traffic analytics: the module entry point.

    from modules.tracking_analytics import TrackingAnalytics

    analytics = TrackingAnalytics("models/vehicle_detection_yolov8n.pt")
    result = analytics.process(frame)          # or .predict(frame)

``result`` is a shared-schema FrameResult dict. This module fills ``tracks``,
``alerts`` and ``analytics``; ``detections`` and ``segmentations`` stay empty
so that ``shared.schemas.merge_frame_results`` can concatenate this with the
other three modules' outputs without duplicating vehicle boxes.

Unlike the other three modules this one is STATEFUL: tracking needs frame
history. Call ``reset()`` between videos.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from shared.schemas import empty_frame_result

from .module_config import TrackingAnalyticsConfig
from .road_health import RoadHealthScorer, summarize_conditions
from .speed_estimation import SpeedEstimator
from .tracker import VehicleTracker, tracks_to_dicts
from .traffic_analytics import TrafficAnalytics
from .vehicle_detector import VehicleDetector
from .wrong_way import WrongWayDetector

SOURCE = "tracking_analytics"


class TrackingAnalytics:
    """Vehicle detection -> tracking -> analytics -> alerts -> road health.

    Parameters
    ----------
    model_path:
        Pretrained COCO YOLO weights for vehicle detection. ``None`` uses
        ``models/vehicle_detection_yolov8n.pt`` if present, else the
        Ultralytics default ``yolov8n.pt``. Ignored when ``detector`` is
        given.
    config:
        ``TrackingAnalyticsConfig``, a shared ``InferenceConfig``, a plain
        dict, or ``None`` for the shared defaults.
    detector:
        Any object with ``detect(frame) -> list[detection dict]``. Injecting
        one skips loading YOLO entirely - used by the tests and by anyone
        who wants to swap in a custom-trained vehicle detector.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config=None,
        detector=None,
    ) -> None:
        self.config = TrackingAnalyticsConfig.coerce(config)

        self.detector = detector if detector is not None else VehicleDetector(
            model_path, self.config.inference
        )
        self.tracker = VehicleTracker(self.config.tracker)
        self.speed_estimator = SpeedEstimator(self.config.speed, fps=self.config.fps)
        self.wrong_way_detector = WrongWayDetector(self.config.wrong_way)
        self.traffic = TrafficAnalytics(self.config.traffic)
        self.road_health = RoadHealthScorer(self.config.road_health)

        self._frame_counter = -1

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cross-frame state. Call between videos/streams."""
        self.tracker.reset()
        self.traffic.reset()
        self.road_health.reset()
        self.wrong_way_detector.reset()
        self._frame_counter = -1
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    # -- main entry point -----------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        frame_id: int | None = None,
        timestamp: float | None = None,
        other_results=None,
    ) -> dict:
        """Process one BGR frame and return a shared-schema FrameResult.

        ``other_results`` is the other modules' output for the SAME frame -
        either one merged dict or an iterable of partial FrameResults. When
        supplied, the ``road_damage_count`` / ``obstacle_count`` /
        ``speed_bump_count`` analytics fields and the Road Health Score are
        computed from those real findings. When omitted, the counts stay 0 and
        the road-health fields are ``None`` rather than a fabricated 100.
        """
        if frame is None or not hasattr(frame, "shape") or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 numpy array in BGR order")

        self._frame_counter += 1
        frame_id = self._frame_counter if frame_id is None else int(frame_id)
        if timestamp is None:
            timestamp = frame_id / self.config.fps if self.config.fps > 0 else 0.0
        timestamp = float(timestamp)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame_id, timestamp, frame)

        for track in tracks:
            self.speed_estimator.update(track)

        alerts = list(self.wrong_way_detector.update(tracks, frame_id))

        # Attach per-track extras only after every component has written its
        # state, so the serialized track carries speed AND wrong-way status.
        for track in tracks:
            extra = self.speed_estimator.output_extra(track)
            extra["wrong_way"] = bool(track.state.get("wrong_way_active", False))
            track.state["output_extra"] = extra

        traffic_fields, line_alerts = self.traffic.update(tracks, frame_id)
        alerts.extend(line_alerts)

        result = empty_frame_result(frame_id=frame_id, timestamp=timestamp)
        result["tracks"] = tracks_to_dicts(
            tracks,
            include_trajectory=self.config.include_trajectory_in_output,
            trajectory_points=self.config.trajectory_output_points,
        )
        result["alerts"] = alerts
        result["analytics"].update(traffic_fields)
        result["analytics"].update(self._condition_fields(other_results))

        if self.config.traffic.counting_lines or self.wrong_way_detector.is_configured:
            result["analytics"]["scene_configured"] = True

        return result

    # The shared contract names the method predict(); this module's natural
    # name is process() (it is stateful, not a pure prediction). Both exist.
    def predict(self, frame: np.ndarray, **kwargs) -> dict:
        return self.process(frame, **kwargs)

    __call__ = process

    # -- helpers --------------------------------------------------------------

    def _condition_fields(self, other_results) -> dict:
        """Counts + road health derived from the OTHER modules' findings."""
        if other_results is None:
            return {
                "road_damage_count": 0,
                "obstacle_count": 0,
                "speed_bump_count": 0,
                "road_health_score": None,
                "road_health_grade": None,
                "road_health_note": (
                    "No road-damage/segmentation/obstacle results supplied for "
                    "this frame; condition counts and Road Health Score are not "
                    "computed from this module's own output."
                ),
            }

        if isinstance(other_results, dict):
            others: Iterable[dict] = [other_results]
        else:
            others = list(other_results)

        detections: list[dict] = []
        segmentations: list[dict] = []
        for partial in others:
            detections.extend(partial.get("detections", []))
            segmentations.extend(partial.get("segmentations", []))

        fields = summarize_conditions(detections, segmentations)
        fields.update(self.road_health.score_frame(detections, segmentations))
        return fields

    def session_summary(self) -> dict:
        """Aggregate statistics for everything processed since ``reset()``."""
        summary = {
            "frames_processed": self._frame_counter + 1,
            "unique_vehicles_seen": len(self.tracker.unique_ids_seen),
            "tracker_backend": self.tracker.backend_in_use,
            "line_counts": {
                name: dict(counts) for name, counts in self.traffic.line_counts.items()
            },
            "speed_calibrated": self.speed_estimator.is_calibrated,
            "wrong_way_configured": self.wrong_way_detector.is_configured,
        }
        summary["road_health"] = self.road_health.session_summary()
        return summary

    # -- video convenience ----------------------------------------------------

    def process_video(
        self,
        source: str | Path,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> Iterator[tuple[int, np.ndarray, dict]]:
        """Yield ``(frame_id, frame, result)`` for each processed frame.

        A convenience for scripts and the future dashboard - the module does
        NOT own a UI, this just avoids every caller rewriting the OpenCV loop.
        """
        import cv2

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.config.fps = float(fps)
            self.speed_estimator.fps = float(fps)

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
                result = self.process(
                    frame,
                    timestamp=timestamp if timestamp > 0 else None,
                )
                yield index, frame, result
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            capture.release()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vehicle tracking + traffic analytics over a video.",
    )
    parser.add_argument("--source", help="Video file or camera index")
    parser.add_argument("--weights", default=None, help="Vehicle detector weights")
    parser.add_argument("--config", default=None, help="Scene config YAML")
    parser.add_argument("--json", default=None, help="Write per-frame results here")
    parser.add_argument("--video-out", default=None, help="Write an annotated video")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a synthetic scene with a scripted detector (no weights needed)",
    )
    parser.add_argument(
        "--image-out", default=None, help="With --demo: save one annotated frame"
    )
    return parser


def _run_demo(output_json: str | None, image_out: str | None = None) -> dict:
    """Synthetic end-to-end run: no weights, no torch, no video file.

    Two cars driving up the frame and one driving down it (against the
    configured flow), so tracking, counting, line crossing, wrong-way and the
    Road Health Score all exercise. Used to produce example_output.json.
    """
    from .speed_estimation import SceneCalibration
    from .vehicle_detector import ScriptedVehicleDetector

    fps = 25.0
    frames = 40
    calibration_data = {
        # A 7 m wide x 40 m deep rectangle of road surface.
        "image_points": [[120, 700], [840, 700], [620, 320], [340, 320]],
        "world_points": [[0, 0], [7.0, 0], [7.0, 40.0], [0, 40.0]],
        "description": "Demo: 7 m wide, 40 m deep road rectangle.",
    }
    calibration = SceneCalibration(
        image_points=calibration_data["image_points"],
        world_points=calibration_data["world_points"],
    )

    # Motion is simulated in WORLD metres and projected into the image, so the
    # pixel step shrinks with distance the way real perspective does and the
    # estimated speeds are comparable with the true ones.
    vehicles = [
        # (class, lane x [m], start y [m], speed [km/h], direction, confidence)
        ("car", 2.0, 2.0, 45.0, +1, 0.93),
        ("truck", 5.0, 4.0, 30.0, +1, 0.88),
        ("car", 1.0, 20.0, 40.0, -1, 0.90),  # travelling against the flow
    ]

    script: list[list[dict]] = []
    for i in range(frames):
        frame_detections = []
        for label, lane_x, start_y, speed_kmh, direction, conf in vehicles:
            world_y = start_y + direction * (speed_kmh / 3.6 / fps) * i
            point = calibration.to_image([[lane_x, world_y]])[0]
            cx, cy = float(point[0]), float(point[1])
            half_width = 55.0 if label == "truck" else 40.0
            frame_detections.append(
                {
                    "class": label,
                    "confidence": conf,
                    "bbox": [cx - half_width, cy - 65, cx + half_width, cy],
                }
            )
        script.append(frame_detections)

    config = TrackingAnalyticsConfig.from_dict(
        {
            "fps": fps,
            "traffic": {
                "counting_lines": [
                    {"name": "stop_line", "p1": [0, 520], "p2": [960, 520]}
                ]
            },
            "wrong_way": {
                "zones": [
                    {
                        "name": "northbound_carriageway",
                        # Traffic should move AWAY from the camera, i.e. up
                        # the image (y decreasing).
                        "allowed_direction": [0, -1],
                        "tolerance_deg": 60,
                    }
                ],
                "min_frames": 6,
            },
            "speed": {"calibration": calibration_data},
        }
    )

    analytics = TrackingAnalytics(config=config, detector=ScriptedVehicleDetector(script))
    blank = np.zeros((720, 960, 3), dtype=np.uint8)

    # Stand-in findings from the other three modules, so the analytics block
    # and Road Health Score are populated the way they will be after merge.
    other = {
        "detections": [
            {"class": "pothole", "confidence": 0.91, "bbox": [400, 500, 460, 545], "source": "road_damage"},
            {"class": "alligator_crack", "confidence": 0.72, "bbox": [200, 560, 320, 620], "source": "road_damage"},
            {"class": "road_debris", "confidence": 0.66, "bbox": [640, 520, 690, 560], "source": "debris"},
            {"class": "speed_bump", "confidence": 0.94, "bbox": [100, 470, 860, 500], "source": "speed_bump"},
        ],
        "segmentations": [
            {
                "class": "surface_damage",
                "confidence": 0.83,
                "mask": "<rle-or-polygon omitted>",
                "area_px": 41_472,
                "area_ratio": 0.06,
                "source": "road_segmentation",
            }
        ],
    }

    results = []
    started = time.perf_counter()
    for _ in range(frames):
        results.append(analytics.process(blank.copy(), other_results=other))
    elapsed = time.perf_counter() - started

    summary = analytics.session_summary()
    summary["analytics_fps_excluding_detector"] = round(frames / elapsed, 1)

    if image_out:
        import cv2

        from .visualization import draw_frame_result

        canvas = blank.copy()
        canvas[360:, :] = 55  # a suggestion of road surface, for contrast
        annotated = draw_frame_result(canvas, results[-1], config=config)
        Path(image_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_out, annotated)

    if output_json:
        payload = {
            "note": (
                "Synthetic demo output produced by "
                "`python -m modules.tracking_analytics.inference --demo`. "
                "Detections come from ScriptedVehicleDetector, not a real "
                "model, and the damage/segmentation entries are stand-ins for "
                "the other three modules."
            ),
            "session_summary": summary,
            "frames": [results[0], results[len(results) // 2], results[-1]],
        }
        Path(output_json).write_text(json.dumps(payload, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.demo:
        summary = _run_demo(args.json, args.image_out)
        print(json.dumps(summary, indent=2))
        return 0

    if not args.source:
        print("--source is required (or use --demo)")
        return 2

    import cv2

    config = (
        TrackingAnalyticsConfig.from_yaml(args.config)
        if args.config
        else TrackingAnalyticsConfig()
    )
    analytics = TrackingAnalytics(args.weights, config=config)

    writer = None
    collected: list[dict] = []
    started = time.perf_counter()
    frames = 0

    for _, frame, result in analytics.process_video(
        args.source, max_frames=args.max_frames, stride=args.stride
    ):
        collected.append(result)
        frames += 1
        if args.video_out:
            from .visualization import draw_frame_result

            annotated = draw_frame_result(frame, result)
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    args.video_out,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    analytics.config.fps,
                    (width, height),
                )
            writer.write(annotated)

    if writer is not None:
        writer.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = analytics.session_summary()
    summary["pipeline_fps"] = round(frames / elapsed, 2)
    print(json.dumps(summary, indent=2))

    if args.json:
        Path(args.json).write_text(
            json.dumps({"session_summary": summary, "frames": collected}, indent=2)
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
