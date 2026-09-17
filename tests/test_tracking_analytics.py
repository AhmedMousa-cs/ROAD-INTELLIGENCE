"""
Tests for modules/tracking_analytics (Person 4).

Covers the eight checks required by the spec:
    1. Model loads successfully
    2. Inference runs on one image
    3. Inference runs on one video frame
    4. Output follows the shared schema
    5. Class names are normalized
    6. Confidence values are valid
    7. Bounding boxes are valid
    8. No unexpected exceptions

plus behavioural tests for the analytics this module owns (id stability,
counting, density, line crossing, wrong-way, speed, road health).

Most tests inject ``ScriptedVehicleDetector`` instead of loading YOLO. That
is deliberate: it makes the tracking/analytics logic deterministic and lets
the suite run in a clean environment without torch or weights. The tests that
genuinely need the real detector are skipped when ultralytics is missing.

Run with: pytest tests/test_tracking_analytics.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from modules.tracking_analytics import (
    CountingLine,
    RoadHealthScorer,
    SceneCalibration,
    ScriptedVehicleDetector,
    TrackingAnalytics,
    TrackingAnalyticsConfig,
    summarize_conditions,
)
from modules.tracking_analytics.class_mapping import normalize_class
from modules.tracking_analytics.traffic_analytics import TrafficAnalytics
from shared.config import MODELS_DIR
from shared.schemas import (
    load_allowed_classes,
    validate_bbox,
    validate_confidence,
    validate_frame_result,
)

FRAME_HEIGHT, FRAME_WIDTH = 720, 960
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


# -----------------------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------------------


def blank_frame(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """A BGR frame with a little texture (not a flat zero image)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[height // 2 :, :] = 60
    return frame


def upward_script(frames: int = 30, step: int = 14) -> list[list[dict]]:
    """One car and one truck moving UP the frame (decreasing y)."""
    return [
        [
            {"class": "car", "confidence": 0.93, "bbox": [300, 640 - step * i, 380, 700 - step * i]},
            {"class": "truck", "confidence": 0.85, "bbox": [430, 660 - 11 * i, 540, 730 - 11 * i]},
        ]
        for i in range(frames)
    ]


def downward_script(frames: int = 30, step: int = 15) -> list[list[dict]]:
    """One car moving DOWN the frame (increasing y)."""
    return [
        [{"class": "car", "confidence": 0.92, "bbox": [150, 60 + step * i, 230, 130 + step * i]}]
        for i in range(frames)
    ]


def build_analytics(script, config_dict: dict | None = None) -> TrackingAnalytics:
    config = TrackingAnalyticsConfig.from_dict(config_dict or {"fps": 25.0})
    return TrackingAnalytics(config=config, detector=ScriptedVehicleDetector(script))


@pytest.fixture
def analytics() -> TrackingAnalytics:
    return build_analytics(upward_script())


# -----------------------------------------------------------------------------
# 1. Model loads successfully
# -----------------------------------------------------------------------------


def test_module_constructs_with_injected_detector():
    """The pipeline is constructible without weights (clean-environment path)."""
    pipeline = build_analytics(upward_script())
    assert pipeline.tracker is not None
    assert pipeline.traffic is not None
    assert pipeline.road_health is not None


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_real_detector_loads():
    """Loads pretrained COCO weights and filters to our ontology's class ids."""
    from modules.tracking_analytics import VehicleDetector

    local = MODELS_DIR / "vehicle_detection_yolov8n.pt"
    detector = VehicleDetector(str(local) if local.exists() else "yolov8n.pt")
    assert detector.model is not None
    # person, bicycle, car, motorcycle, bus, truck -> 6 COCO ids
    assert detector._class_ids is not None and len(detector._class_ids) == 6


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="ultralytics not installed")
def test_real_detector_output_is_schema_valid():
    from modules.tracking_analytics import VehicleDetector

    local = MODELS_DIR / "vehicle_detection_yolov8n.pt"
    detector = VehicleDetector(str(local) if local.exists() else "yolov8n.pt")
    frame = blank_frame()
    allowed = load_allowed_classes()
    for detection in detector.detect(frame):
        assert detection["class"] in allowed
        validate_confidence(detection["confidence"])
        validate_bbox(detection["bbox"], FRAME_WIDTH, FRAME_HEIGHT)


def test_missing_weights_path_raises_clearly():
    from modules.tracking_analytics import VehicleDetector

    with pytest.raises((FileNotFoundError, ImportError)):
        VehicleDetector(str(Path("models") / "definitely_not_here" / "weights.pt"))


# -----------------------------------------------------------------------------
# 2 & 3. Inference on one image and on a video frame
# -----------------------------------------------------------------------------


def test_inference_runs_on_single_image(analytics):
    result = analytics.process(blank_frame())
    validate_frame_result(result)
    assert result["frame_id"] == 0


def test_inference_runs_on_video_frame(tmp_path, analytics):
    """Decode a real video file and run one frame through the pipeline."""
    cv2 = pytest.importorskip("cv2")

    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        pytest.skip("No video writer codec available in this environment")
    for _ in range(5):
        writer.write(blank_frame())
    writer.release()

    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    assert ok, "failed to read back the test video"

    result = analytics.process(frame)
    validate_frame_result(result)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_process_video_helper(tmp_path):
    cv2 = pytest.importorskip("cv2")

    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("No video writer codec available in this environment")
    for _ in range(8):
        writer.write(blank_frame())
    writer.release()

    pipeline = build_analytics(upward_script())
    results = [result for _, _, result in pipeline.process_video(video_path)]
    assert len(results) == 8
    for result in results:
        validate_frame_result(result)


# -----------------------------------------------------------------------------
# 4. Output follows the shared schema
# -----------------------------------------------------------------------------


def test_output_matches_shared_schema_over_a_sequence(analytics):
    for _ in range(30):
        result = analytics.process(blank_frame())
        validate_frame_result(result)
        for key in ("frame_id", "timestamp", "detections", "segmentations", "tracks", "alerts", "analytics"):
            assert key in result
        # This module owns tracks/alerts/analytics only.
        assert result["detections"] == []
        assert result["segmentations"] == []


def test_required_analytics_keys_always_present(analytics):
    result = analytics.process(blank_frame())
    for key in (
        "vehicle_count",
        "traffic_density",
        "road_damage_count",
        "obstacle_count",
        "speed_bump_count",
    ):
        assert key in result["analytics"]


def test_track_dicts_have_required_fields(analytics):
    for _ in range(10):
        result = analytics.process(blank_frame())
    assert result["tracks"], "expected confirmed tracks after 10 frames"
    for track in result["tracks"]:
        for key in ("track_id", "class", "bbox", "center", "velocity_px_s"):
            assert key in track
        assert isinstance(track["track_id"], int)
        assert len(track["center"]) == 2
        assert isinstance(track["velocity_px_s"], float)


def test_output_is_json_serializable(analytics):
    import json

    for _ in range(10):
        result = analytics.process(blank_frame())
    json.dumps(result)  # must not raise


def test_merges_with_other_modules_results(analytics):
    from shared.schemas import merge_frame_results

    for _ in range(10):
        tracking_result = analytics.process(blank_frame())

    damage_result = {
        "detections": [
            {"class": "pothole", "confidence": 0.9, "bbox": [10, 10, 60, 60], "source": "road_damage"}
        ]
    }
    merged = merge_frame_results(damage_result, tracking_result, frame_id=9, timestamp=0.36)
    validate_frame_result(merged)
    assert len(merged["detections"]) == 1
    assert merged["tracks"] == tracking_result["tracks"]


# -----------------------------------------------------------------------------
# 5. Class names are normalized
# -----------------------------------------------------------------------------


def test_track_classes_come_from_shared_ontology(analytics):
    allowed = load_allowed_classes()
    for _ in range(12):
        result = analytics.process(blank_frame())
    assert result["tracks"]
    for track in result["tracks"]:
        assert track["class"] in allowed


def test_non_ontology_detector_labels_are_dropped():
    """A COCO class outside our ontology must never reach the output."""
    assert normalize_class("train") is None
    assert normalize_class("traffic light") is None
    assert normalize_class("car") == "car"

    script = [
        [
            {"class": "car", "confidence": 0.9, "bbox": [100, 100, 180, 170]},
            {"class": "train", "confidence": 0.95, "bbox": [400, 100, 600, 300]},
        ]
    ] * 10
    pipeline = build_analytics(script)
    for _ in range(10):
        result = pipeline.process(blank_frame())
    classes = {track["class"] for track in result["tracks"]}
    assert classes == {"car"}
    assert "train" not in classes


def test_class_label_is_stabilised_by_majority_vote():
    """Per-frame car/truck flicker must not flip the reported track class."""
    script = []
    for i in range(20):
        label = "truck" if i == 7 else "car"  # one bad frame
        script.append(
            [{"class": label, "confidence": 0.9, "bbox": [300, 600 - 12 * i, 380, 660 - 12 * i]}]
        )
    pipeline = build_analytics(script)
    for _ in range(20):
        result = pipeline.process(blank_frame())
    assert result["tracks"][0]["class"] == "car"


# -----------------------------------------------------------------------------
# 6 & 7. Confidence and bounding boxes are valid
# -----------------------------------------------------------------------------


def test_confidences_and_boxes_are_valid(analytics):
    for _ in range(25):
        result = analytics.process(blank_frame())
        for track in result["tracks"]:
            validate_confidence(track["confidence"])
            validate_bbox(track["bbox"], FRAME_WIDTH, FRAME_HEIGHT)
        for alert in result["alerts"]:
            validate_confidence(alert["confidence"])


def test_boxes_are_clipped_to_the_frame():
    """Detections partly outside the frame are clipped, not passed through."""
    script = [
        [{"class": "car", "confidence": 0.9, "bbox": [-40, -30, 120, 140]}] for _ in range(10)
    ]
    pipeline = build_analytics(script)
    for _ in range(10):
        result = pipeline.process(blank_frame())
    for track in result["tracks"]:
        validate_bbox(track["bbox"], FRAME_WIDTH, FRAME_HEIGHT)
        assert track["bbox"][0] >= 0 and track["bbox"][1] >= 0


def test_varying_frame_resolution_is_supported():
    """The module must not assume a fixed input resolution."""
    script = [
        [{"class": "car", "confidence": 0.9, "bbox": [50, 50, 150, 150]}] for _ in range(6)
    ]
    pipeline = build_analytics(script)
    for height, width in ((480, 640), (720, 1280), (1080, 1920)):
        pipeline.reset()
        pipeline.detector.reset()
        for _ in range(6):
            result = pipeline.process(blank_frame(height, width))
        validate_frame_result(result)
        for track in result["tracks"]:
            validate_bbox(track["bbox"], width, height)


# -----------------------------------------------------------------------------
# 8. No unexpected exceptions
# -----------------------------------------------------------------------------


def test_empty_detections_are_handled():
    pipeline = build_analytics([[] for _ in range(10)])
    for _ in range(10):
        result = pipeline.process(blank_frame())
        validate_frame_result(result)
    assert result["tracks"] == []
    assert result["analytics"]["vehicle_count"] == 0
    assert result["analytics"]["traffic_density"] == "low"


def test_invalid_frame_raises_value_error(analytics):
    with pytest.raises(ValueError):
        analytics.process(None)
    with pytest.raises(ValueError):
        analytics.process(np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8))  # not BGR


def test_reset_clears_state(analytics):
    for _ in range(12):
        analytics.process(blank_frame())
    assert analytics.session_summary()["unique_vehicles_seen"] > 0

    analytics.reset()
    summary = analytics.session_summary()
    assert summary["frames_processed"] == 0
    assert summary["unique_vehicles_seen"] == 0


def test_long_run_is_stable():
    """200 frames with appearing/disappearing objects must not raise."""
    rng = np.random.default_rng(0)
    script = []
    for i in range(200):
        frame_dets = []
        if i % 7 != 0:
            frame_dets.append(
                {"class": "car", "confidence": 0.9, "bbox": [300, 600 - i, 380, 660 - i]}
            )
        if 50 < i < 150:
            x = 100 + int(rng.integers(0, 5))
            frame_dets.append(
                {"class": "bus", "confidence": 0.7, "bbox": [x, 200, x + 120, 320]}
            )
        script.append(frame_dets)

    pipeline = build_analytics(script)
    for _ in range(200):
        validate_frame_result(pipeline.process(blank_frame()))


# -----------------------------------------------------------------------------
# Tracking behaviour
# -----------------------------------------------------------------------------


def test_track_ids_are_stable_across_frames(analytics):
    seen: list[set[int]] = []
    for _ in range(25):
        result = analytics.process(blank_frame())
        if result["tracks"]:
            seen.append({t["track_id"] for t in result["tracks"]})
    assert seen, "expected confirmed tracks"
    # Same two ids for the whole run once confirmed.
    assert all(ids == seen[-1] for ids in seen[len(seen) // 2 :])
    assert len(seen[-1]) == 2


def test_track_survives_a_short_detection_gap():
    """A missed detection must not create a new id (occlusion robustness)."""
    script = []
    for i in range(30):
        if i in (12, 13):  # two dropped frames
            script.append([])
        else:
            script.append(
                [{"class": "car", "confidence": 0.9, "bbox": [300, 600 - 10 * i, 380, 660 - 10 * i]}]
            )
    pipeline = build_analytics(script)
    ids = set()
    for _ in range(30):
        result = pipeline.process(blank_frame())
        ids.update(t["track_id"] for t in result["tracks"])
    assert ids == {1}, f"expected a single stable id, got {ids}"


def test_unique_vehicle_count_accumulates(analytics):
    for _ in range(25):
        result = analytics.process(blank_frame())
    assert result["analytics"]["unique_vehicles_seen"] == 2
    assert result["analytics"]["unique_counts_by_type"] == {"car": 1, "truck": 1}


# -----------------------------------------------------------------------------
# Traffic analytics
# -----------------------------------------------------------------------------


def test_traffic_density_thresholds_are_configurable():
    analytics_layer = TrafficAnalytics()
    assert analytics_layer.density_label(0) == "low"
    assert analytics_layer.density_label(5) == "low"
    assert analytics_layer.density_label(6) == "medium"
    assert analytics_layer.density_label(15) == "medium"
    assert analytics_layer.density_label(16) == "high"

    from modules.tracking_analytics.traffic_analytics import TrafficAnalyticsConfig

    custom = TrafficAnalytics(
        TrafficAnalyticsConfig.from_dict(
            {"density_thresholds": {"low_max": 1, "medium_max": 3}}
        )
    )
    assert custom.density_label(2) == "medium"
    assert custom.density_label(4) == "high"


def test_person_is_tracked_but_not_counted_as_a_vehicle():
    script = [
        [
            {"class": "car", "confidence": 0.9, "bbox": [300, 600 - 10 * i, 380, 660 - 10 * i]},
            {"class": "person", "confidence": 0.8, "bbox": [700, 500, 730, 580]},
        ]
        for i in range(12)
    ]
    pipeline = build_analytics(script)
    for _ in range(12):
        result = pipeline.process(blank_frame())
    assert result["analytics"]["vehicle_count"] == 1
    assert result["analytics"]["person_count"] == 1
    assert {t["class"] for t in result["tracks"]} == {"car", "person"}


def test_line_crossing_counts_once_per_direction():
    config = {
        "fps": 25.0,
        "traffic": {"counting_lines": [{"name": "stop_line", "p1": [0, 400], "p2": [960, 400]}]},
    }
    pipeline = build_analytics(upward_script(frames=40), config)
    for _ in range(40):
        result = pipeline.process(blank_frame())

    counts = result["analytics"]["line_counts"]["stop_line"]
    assert sum(counts.values()) == 2, f"expected both vehicles to cross once: {counts}"


def test_counting_line_geometry():
    line = CountingLine(name="l", p1=(0, 100), p2=(100, 100))
    assert line.crossed((50, 90), (50, 110)) is not None
    assert line.crossed((50, 90), (50, 95)) is None  # never reaches the line
    assert line.crossed((500, 90), (500, 110)) is None  # outside the segment


# -----------------------------------------------------------------------------
# Wrong-way detection
# -----------------------------------------------------------------------------


WRONG_WAY_CONFIG = {
    "fps": 25.0,
    "wrong_way": {
        "zones": [
            {"name": "northbound", "allowed_direction": [0, -1], "tolerance_deg": 60}
        ],
        "min_frames": 6,
    },
}


def test_wrong_way_alert_is_raised_for_opposing_traffic():
    pipeline = build_analytics(downward_script(frames=30), WRONG_WAY_CONFIG)
    alerts: list[dict] = []
    for _ in range(30):
        alerts.extend(pipeline.process(blank_frame())["alerts"])

    wrong_way = [a for a in alerts if a["type"] == "wrong_way"]
    assert wrong_way, "a vehicle moving against the allowed direction should alert"
    alert = wrong_way[0]
    assert alert["message"] == "Possible wrong-way vehicle"
    assert alert["verified"] is False  # algorithmic, not an enforcement decision
    validate_confidence(alert["confidence"])
    assert alert["confidence"] >= 0.60


def test_compliant_traffic_raises_no_wrong_way_alert():
    pipeline = build_analytics(upward_script(frames=30), WRONG_WAY_CONFIG)
    for _ in range(30):
        result = pipeline.process(blank_frame())
        assert not [a for a in result["alerts"] if a["type"] == "wrong_way"]


def test_stationary_vehicle_raises_no_wrong_way_alert():
    script = [
        [{"class": "car", "confidence": 0.9, "bbox": [300, 500, 380, 570]}] for _ in range(30)
    ]
    pipeline = build_analytics(script, WRONG_WAY_CONFIG)
    for _ in range(30):
        result = pipeline.process(blank_frame())
        assert not result["alerts"]


def test_no_zones_configured_means_no_wrong_way_opinion():
    pipeline = build_analytics(downward_script(frames=30))
    for _ in range(30):
        result = pipeline.process(blank_frame())
        assert result["alerts"] == []


def test_wrong_way_zone_polygon_limits_the_area():
    """A vehicle outside every zone must not be judged."""
    config = {
        "fps": 25.0,
        "wrong_way": {
            "zones": [
                {
                    "name": "right_lane_only",
                    "allowed_direction": [0, -1],
                    "polygon": [[600, 0], [960, 0], [960, 720], [600, 720]],
                    "tolerance_deg": 60,
                }
            ],
            "min_frames": 6,
        },
    }
    # downward_script drives at x~150-230, i.e. outside the polygon.
    pipeline = build_analytics(downward_script(frames=30), config)
    for _ in range(30):
        assert pipeline.process(blank_frame())["alerts"] == []


# -----------------------------------------------------------------------------
# Speed estimation
# -----------------------------------------------------------------------------


def test_velocity_px_s_is_reported_and_speed_is_none_without_calibration(analytics):
    for _ in range(15):
        result = analytics.process(blank_frame())
    for track in result["tracks"]:
        assert track["velocity_px_s"] > 0
        assert track["estimated_speed_kmh"] is None, (
            "pixel velocity must never be presented as a real-world speed"
        )
        assert track["speed_method"] == "uncalibrated"
        assert track["speed_reliable"] is False


ROAD_CALIBRATION = {
    # A 7 m wide x 40 m deep rectangle of road surface.
    "image_points": [[120, 700], [840, 700], [620, 320], [340, 320]],
    "world_points": [[0, 0], [7.0, 0], [7.0, 40.0], [0, 40.0]],
}


def world_motion_script(
    speed_kmh: float,
    frames: int,
    fps: float = 25.0,
    lane_x_m: float = 2.0,
    start_y_m: float = 2.0,
) -> list[list[dict]]:
    """Synthesize detections for a vehicle moving at a KNOWN real speed.

    Motion is generated in world metres and projected into the image through
    the calibration, so the pixel step shrinks with distance exactly as real
    perspective does. That makes the estimated speed comparable against
    ground truth - a constant pixel step would be physically impossible and
    would tell us nothing.
    """
    calibration = SceneCalibration(**ROAD_CALIBRATION)
    metres_per_frame = speed_kmh / 3.6 / fps

    script = []
    for i in range(frames):
        world_y = start_y_m + metres_per_frame * i
        image = calibration.to_image([[lane_x_m, world_y]])[0]
        cx, cy = float(image[0]), float(image[1])
        script.append(
            [
                {
                    "class": "car",
                    "confidence": 0.92,
                    "bbox": [cx - 40, cy - 60, cx + 40, cy],
                }
            ]
        )
    return script


def test_estimated_speed_recovers_the_true_speed_with_calibration():
    """A vehicle simulated at 50 km/h must be estimated near 50 km/h."""
    true_speed = 50.0
    config = {"fps": 25.0, "speed": {"calibration": ROAD_CALIBRATION}}
    pipeline = build_analytics(world_motion_script(true_speed, frames=30), config)
    for _ in range(30):
        result = pipeline.process(blank_frame())

    assert result["tracks"], "expected a confirmed track"
    track = result["tracks"][0]
    assert track["speed_method"] == "homography_from_points"
    assert track["speed_reliable"] is True
    assert track["estimated_speed_kmh"] == pytest.approx(true_speed, rel=0.10), (
        f"estimated {track['estimated_speed_kmh']} km/h for a simulated {true_speed} km/h"
    )


def test_estimated_speed_tracks_a_different_true_speed():
    true_speed = 90.0
    config = {"fps": 30.0, "speed": {"calibration": ROAD_CALIBRATION}}
    pipeline = build_analytics(
        world_motion_script(true_speed, frames=25, fps=30.0), config
    )
    for _ in range(25):
        result = pipeline.process(blank_frame())
    assert result["tracks"][0]["estimated_speed_kmh"] == pytest.approx(true_speed, rel=0.10)


def test_implausible_speed_is_rejected_not_reported():
    """Near the horizon a few pixels span many metres; don't emit the number."""
    config = {
        "fps": 25.0,
        "speed": {"calibration": ROAD_CALIBRATION, "max_plausible_kmh": 250.0},
    }
    # Constant 14 px/frame all the way up the frame: physically impossible,
    # and enormous in world terms once the track approaches the horizon.
    pipeline = build_analytics(upward_script(frames=40), config)
    for _ in range(40):
        result = pipeline.process(blank_frame())

    rejected = [t for t in result["tracks"] if t["estimated_speed_kmh"] is None]
    assert rejected, "an implausible speed should be withheld, not reported"
    for track in rejected:
        assert track["speed_reliable"] is False
        assert track["speed_rejected_kmh"] > 250.0


def test_uniform_scale_calibration_is_flagged_unreliable():
    config = {"fps": 25.0, "speed": {"calibration": {"pixels_per_meter": 10.0}}}
    pipeline = build_analytics(upward_script(frames=20), config)
    for _ in range(20):
        result = pipeline.process(blank_frame())
    for track in result["tracks"]:
        assert track["estimated_speed_kmh"] is not None
        assert track["speed_reliable"] is False  # perspective ignored
        assert track["speed_method"] == "uniform_scale"


def test_scene_calibration_projects_points():
    calibration = SceneCalibration(pixels_per_meter=20.0)
    world = calibration.to_world([[0, 0], [20, 0]])
    assert world is not None
    assert np.isclose(np.linalg.norm(world[1] - world[0]), 1.0)  # 20 px == 1 m


# -----------------------------------------------------------------------------
# Road health score
# -----------------------------------------------------------------------------


def test_road_health_is_none_without_other_module_results(analytics):
    result = analytics.process(blank_frame())
    assert result["analytics"]["road_health_score"] is None
    assert result["analytics"]["road_damage_count"] == 0


def test_road_health_uses_other_modules_results(analytics):
    other = {
        "detections": [
            {"class": "pothole", "confidence": 0.9, "bbox": [10, 10, 60, 60], "source": "road_damage"},
            {"class": "road_debris", "confidence": 0.7, "bbox": [80, 80, 120, 120], "source": "debris"},
            {"class": "speed_bump", "confidence": 0.95, "bbox": [0, 300, 900, 330], "source": "speed_bump"},
        ],
        "segmentations": [
            {
                "class": "surface_damage",
                "confidence": 0.8,
                "mask": None,
                "area_px": 1000,
                "area_ratio": 0.10,
                "source": "road_segmentation",
            }
        ],
    }
    result = analytics.process(blank_frame(), other_results=other)
    stats = result["analytics"]

    assert stats["road_damage_count"] == 1  # pothole only
    assert stats["obstacle_count"] == 1
    assert stats["speed_bump_count"] == 1
    # 100 - (pothole 5.0 + debris 1.0) - (20.0 * 0.10 * 1.0) = 92.0
    assert stats["road_health_score"] == pytest.approx(92.0, abs=0.05)
    assert stats["road_health_grade"] == "A"


def test_speed_bump_is_not_penalised_as_damage():
    scorer = RoadHealthScorer()
    only_bump = scorer.score_frame(
        [{"class": "speed_bump", "confidence": 0.9, "bbox": [0, 0, 10, 10], "source": "speed_bump"}]
    )
    assert only_bump["road_health_score"] == 100.0


def test_road_health_score_is_clamped_and_graded():
    scorer = RoadHealthScorer()
    many_potholes = [
        {"class": "pothole", "confidence": 0.9, "bbox": [0, 0, 10, 10], "source": "road_damage"}
    ] * 50
    worst = scorer.score_frame(many_potholes)
    assert worst["road_health_score"] == 0.0
    assert worst["road_health_grade"] == "E"


def test_road_health_session_summary_uses_the_mean():
    scorer = RoadHealthScorer()
    scorer.score_frame([])  # 100
    scorer.score_frame(
        [{"class": "pothole", "confidence": 0.9, "bbox": [0, 0, 1, 1], "source": "road_damage"}] * 4
    )  # 80
    summary = scorer.session_summary()
    assert summary["frames_scored"] == 2
    assert summary["road_health_score"] == pytest.approx(90.0)
    assert summary["worst_frame_score"] == pytest.approx(80.0)


def test_summarize_conditions_ignores_vehicle_classes():
    stats = summarize_conditions(
        [
            {"class": "car", "confidence": 0.9, "bbox": [0, 0, 1, 1], "source": "tracking_analytics"},
            {"class": "pothole", "confidence": 0.9, "bbox": [0, 0, 1, 1], "source": "road_damage"},
        ]
    )
    assert stats["road_damage_count"] == 1
    assert stats["condition_counts_by_class"] == {"pothole": 1}


# -----------------------------------------------------------------------------
# Visualization (smoke test only - the dashboard owns the real theme)
# -----------------------------------------------------------------------------


def test_visualization_runs_without_mutating_the_input(analytics):
    pytest.importorskip("cv2")
    from modules.tracking_analytics import visualization as viz

    for _ in range(12):
        result = analytics.process(blank_frame())

    frame = blank_frame()
    original = frame.copy()
    annotated = viz.draw_frame_result(frame, result, config=analytics.config)
    assert annotated.shape == frame.shape
    assert np.array_equal(frame, original), "draw_frame_result must not mutate the input"
