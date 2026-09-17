"""
Tests for modules/road_damage (Person 1).

Covers the eight checks required by the spec:
    1. Model loads successfully
    2. Inference runs on one image
    3. Inference runs on one video frame
    4. Output follows the shared schema
    5. Class names are normalized
    6. Confidence values are valid
    7. Bounding boxes are valid
    8. No unexpected exceptions occur

plus the class mapping and the dataset-preparation validators, which are where
the RDD2022 normalization rule is actually enforced.

Most tests inject ``ScriptedDamageDetector`` instead of loading YOLO, so the
suite runs in a clean environment with no torch and no weights. The tests that
genuinely need real weights are skipped when they are absent.

Run with: pytest tests/test_road_damage.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from modules.road_damage import (
    PROJECT_CLASSES,
    RDD_TO_ONTOLOGY,
    RoadDamageModel,
    ScriptedDamageDetector,
    build_id_remap,
    clip_and_filter_boxes,
    normalize_class,
)
from modules.road_damage.class_mapping import (
    EXCLUDED_CLASSES,
    verify_mapping_against_ontology,
)
from modules.road_damage.prepare_dataset import (
    PreparationStats,
    dhash,
    hamming,
    validate_and_remap_label_file,
)
from shared.config import MODELS_DIR
from shared.schemas import (
    load_allowed_classes,
    validate_bbox,
    validate_confidence,
    validate_detection,
    validate_frame_result,
)

FRAME_HEIGHT, FRAME_WIDTH = 720, 960
HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
WEIGHTS_PATH = MODELS_DIR / "road_damage_best.pt"


# -----------------------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------------------


def road_frame(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """A BGR frame with some texture (not a flat zero image)."""
    frame = np.full((height, width, 3), 70, dtype=np.uint8)
    frame[: height // 3, :] = 110
    return frame


DEFAULT_SCRIPT = [
    [
        {"class": "D40", "confidence": 0.93, "bbox": [300, 430, 430, 530]},
        {"class": "D00", "confidence": 0.71, "bbox": [520, 400, 600, 690]},
        {"class": "D20", "confidence": 0.64, "bbox": [120, 520, 300, 640]},
        {"class": "D10", "confidence": 0.55, "bbox": [640, 470, 900, 520]},
    ]
]


@pytest.fixture
def model() -> RoadDamageModel:
    return RoadDamageModel(detector=ScriptedDamageDetector(DEFAULT_SCRIPT, loop=True))


# -----------------------------------------------------------------------------
# 1. Model loads
# -----------------------------------------------------------------------------


def test_model_loads_with_injected_detector(model):
    assert isinstance(model, RoadDamageModel)
    assert model.model_path is None  # no weights touched when a backend is injected


def test_missing_weights_raise_a_clear_error():
    with pytest.raises((FileNotFoundError, ImportError)):
        RoadDamageModel("models/definitely_not_here.pt")


@pytest.mark.skipif(
    not (HAS_ULTRALYTICS and WEIGHTS_PATH.exists()),
    reason="trained road_damage_best.pt not available",
)
def test_real_weights_load_and_predict():
    real = RoadDamageModel(str(WEIGHTS_PATH))
    result = real.predict(road_frame())
    validate_frame_result(result)
    assert set(real.model_class_names.values()) <= set(PROJECT_CLASSES)


# -----------------------------------------------------------------------------
# 2 & 3. Inference on an image and on a video frame
# -----------------------------------------------------------------------------


def test_inference_runs_on_one_image(model):
    result = model.predict(road_frame())
    validate_frame_result(result)
    assert len(result["detections"]) == 4


def test_inference_runs_on_one_video_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("No video writer codec available in this environment")
    for _ in range(5):
        writer.write(road_frame())
    writer.release()

    from modules.road_damage.inference import predict_video

    damage = RoadDamageModel(detector=ScriptedDamageDetector(DEFAULT_SCRIPT, loop=True))
    frames = list(predict_video(damage, video_path, max_frames=3))
    assert frames, "no frames were decoded"
    for frame_id, frame, result in frames:
        assert frame.ndim == 3
        validate_frame_result(result)
        assert result["frame_id"] == frame_id


def test_varying_frame_resolution_is_supported(model):
    """The module must not assume a fixed input resolution."""
    for height, width in ((480, 640), (720, 1280), (1080, 1920)):
        result = model.predict(road_frame(height, width))
        for detection in result["detections"]:
            validate_detection(detection, frame_width=width, frame_height=height)


# -----------------------------------------------------------------------------
# 4. Shared schema
# -----------------------------------------------------------------------------


def test_output_follows_shared_schema(model):
    result = model.predict(road_frame(), frame_id=12, timestamp=0.48)
    validate_frame_result(result)
    for key in ("frame_id", "timestamp", "detections", "segmentations", "tracks", "alerts", "analytics"):
        assert key in result
    assert result["frame_id"] == 12
    assert result["timestamp"] == pytest.approx(0.48)


def test_module_does_not_fill_other_modules_fields(model):
    result = model.predict(road_frame())
    assert result["segmentations"] == []
    assert result["tracks"] == []
    assert result["alerts"] == []
    assert result["analytics"]["vehicle_count"] == 0


def test_road_damage_count_matches_detections(model):
    result = model.predict(road_frame())
    assert result["analytics"]["road_damage_count"] == len(result["detections"])


def test_source_field_is_road_damage(model):
    result = model.predict(road_frame())
    assert {d["source"] for d in result["detections"]} == {"road_damage"}


def test_result_is_json_serializable(model):
    import json

    json.dumps(model.predict(road_frame()))


def test_merges_with_other_modules_results(model):
    from shared.schemas import empty_frame_result, merge_frame_results

    other = empty_frame_result()
    other["tracks"].append({"track_id": 1, "class": "car", "bbox": [0, 0, 10, 10]})

    merged = merge_frame_results(model.predict(road_frame()), other, frame_id=3, timestamp=1.0)
    validate_frame_result(merged)
    assert len(merged["detections"]) == 4
    assert len(merged["tracks"]) == 1


# -----------------------------------------------------------------------------
# 5. Class names are normalized
# -----------------------------------------------------------------------------


def test_rdd_codes_never_leak_into_output(model):
    result = model.predict(road_frame())
    emitted = {detection["class"] for detection in result["detections"]}
    assert emitted <= set(PROJECT_CLASSES)
    assert not emitted & set(RDD_TO_ONTOLOGY)  # no D00/D10/D20/D40


def test_emitted_classes_exist_in_shared_ontology(model):
    allowed = load_allowed_classes()
    for detection in model.predict(road_frame())["detections"]:
        assert detection["class"] in allowed


def test_mapping_targets_are_valid_ontology_names():
    verify_mapping_against_ontology()


def test_rdd_code_mapping_is_exactly_as_specified():
    assert normalize_class("D00") == "longitudinal_crack"
    assert normalize_class("D10") == "transverse_crack"
    assert normalize_class("D20") == "alligator_crack"
    assert normalize_class("D40") == "pothole"


def test_pothole_dataset_maps_to_the_single_pothole_class():
    for spelling in ("pothole", "potholes", "Potholes", "POTHOLE"):
        assert normalize_class(spelling) == "pothole"


def test_excluded_labels_are_dropped_not_passed_through():
    for label in EXCLUDED_CLASSES:
        assert normalize_class(label) is None

    damage = RoadDamageModel(
        detector=ScriptedDamageDetector(
            [[{"class": "D44", "confidence": 0.9, "bbox": [10, 10, 90, 90]}]]
        )
    )
    result = damage.predict(road_frame())
    assert result["detections"] == []


def test_unknown_labels_are_dropped():
    assert normalize_class("some_new_class") is None
    assert normalize_class("") is None
    assert normalize_class(None) is None


def test_project_class_order_matches_ids():
    from modules.road_damage import CLASS_TO_ID, ID_TO_CLASS

    assert PROJECT_CLASSES[0] == "pothole"
    assert len(PROJECT_CLASSES) == 4
    for name, class_id in CLASS_TO_ID.items():
        assert ID_TO_CLASS[class_id] == name


def test_build_id_remap_skips_unmappable_source_ids():
    # A realistic RDD export: four usable codes plus two marking classes.
    remap = build_id_remap(["D00", "D10", "D20", "D40", "D43", "D44"])
    assert remap == {0: 1, 1: 2, 2: 3, 3: 0}
    assert 4 not in remap and 5 not in remap


# -----------------------------------------------------------------------------
# 6 & 7. Confidence and bounding boxes
# -----------------------------------------------------------------------------


def test_confidence_values_are_valid(model):
    for detection in model.predict(road_frame())["detections"]:
        validate_confidence(detection["confidence"])


def test_bounding_boxes_are_valid(model):
    for detection in model.predict(road_frame())["detections"]:
        validate_bbox(detection["bbox"], frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)


def test_boxes_are_clipped_into_the_frame():
    """Boxes outside the image must be clipped, per the shared validator."""
    damage = RoadDamageModel(
        detector=ScriptedDamageDetector(
            [[{"class": "pothole", "confidence": 0.9, "bbox": [-50, -20, FRAME_WIDTH + 80, 300]}]]
        )
    )
    result = damage.predict(road_frame())
    assert len(result["detections"]) == 1
    x1, y1, x2, y2 = result["detections"][0]["bbox"]
    assert x1 == 0 and y1 == 0
    assert x2 <= FRAME_WIDTH and y2 <= FRAME_HEIGHT
    validate_detection(result["detections"][0], FRAME_WIDTH, FRAME_HEIGHT)


def test_degenerate_boxes_are_dropped():
    damage = RoadDamageModel(
        detector=ScriptedDamageDetector(
            [[{"class": "pothole", "confidence": 0.9, "bbox": [100, 100, 100, 100]}]]
        )
    )
    assert damage.predict(road_frame())["detections"] == []


def test_clip_and_filter_boxes_reports_original_indices():
    kept = clip_and_filter_boxes(
        [[0, 0, 0, 0], [10, 10, 50, 50]], FRAME_WIDTH, FRAME_HEIGHT
    )
    assert [index for index, _ in kept] == [1]


def test_bbox_is_xyxy_top_left_origin(model):
    """bbox must be [x1, y1, x2, y2] with x1<x2, y1<y2 and origin top-left."""
    detections = model.predict(road_frame())["detections"]
    pothole = next(d for d in detections if d["class"] == "pothole")
    x1, y1, x2, y2 = pothole["bbox"]
    assert (x1, y1, x2, y2) == (300.0, 430.0, 430.0, 530.0)


# -----------------------------------------------------------------------------
# 8. No unexpected exceptions
# -----------------------------------------------------------------------------


def test_empty_detections_are_handled():
    damage = RoadDamageModel(detector=ScriptedDamageDetector([[]]))
    result = damage.predict(road_frame())
    validate_frame_result(result)
    assert result["detections"] == []
    assert result["analytics"]["road_damage_count"] == 0


def test_invalid_frame_raises_value_error(model):
    for bad in (None, np.zeros((10, 10), dtype=np.uint8)):
        with pytest.raises((ValueError, AttributeError)):
            model.predict(bad)


def test_predict_batch_and_call_alias(model):
    results = model.predict_batch([road_frame(), road_frame()])
    assert len(results) == 2
    for index, result in enumerate(results):
        assert result["frame_id"] == index
        validate_frame_result(result)
    validate_frame_result(model(road_frame()))


def test_repeated_calls_are_stateless():
    """Unlike the tracking module, this one must not carry frame state."""
    damage = RoadDamageModel(detector=ScriptedDamageDetector(DEFAULT_SCRIPT, loop=True))
    first = damage.predict(road_frame())
    second = damage.predict(road_frame())
    assert first["detections"] == second["detections"]


def test_demo_runs_without_weights():
    from modules.road_damage.inference import run_demo

    result = run_demo()
    validate_frame_result(result)
    # The demo script includes an excluded D44 label; it must not be emitted.
    assert len(result["detections"]) == 4


# -----------------------------------------------------------------------------
# Dataset preparation validators
# -----------------------------------------------------------------------------


def test_label_validation_remaps_ids_and_drops_bad_lines(tmp_path):
    label = tmp_path / "sample.txt"
    label.write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.2 0.2",      # D00 -> longitudinal_crack (id 1)
                "3 0.4 0.4 0.1 0.1",      # D40 -> pothole (id 0)
                "5 0.4 0.4 0.1 0.1",      # D44 -> excluded
                "0 0.5 0.5 0.0 0.2",      # zero width -> dropped
                "0 1.9 0.5 0.2 0.2",      # far out of bounds -> dropped
                "0 0.5 0.5",              # malformed -> dropped
                "",                        # blank -> ignored
            ]
        )
    )
    stats = PreparationStats()
    remap = build_id_remap(["D00", "D10", "D20", "D40", "D43", "D44"])
    lines = validate_and_remap_label_file(label, remap, ["D00", "D10", "D20", "D40", "D43", "D44"], stats)

    assert [line.split()[0] for line in lines] == ["1", "0"]
    assert stats.annotations_kept == 2
    assert stats.annotations_dropped_unmapped == 1
    assert stats.annotations_dropped_invalid_box == 1
    assert stats.annotations_dropped_out_of_bounds == 1
    assert stats.annotations_malformed_line == 1
    assert stats.dropped_source_classes["D44"] == 1


def test_slightly_out_of_bounds_boxes_are_clamped(tmp_path):
    label = tmp_path / "edge.txt"
    label.write_text("3 0.02 0.5 0.06 0.2\n")  # x1 = -0.01, within tolerance
    stats = PreparationStats()
    lines = validate_and_remap_label_file(label, {3: 0}, ["D00", "D10", "D20", "D40"], stats)

    assert stats.annotations_clamped == 1
    assert len(lines) == 1
    _, cx, _, width, _ = lines[0].split()
    assert float(cx) - float(width) / 2 >= 0.0


def test_missing_label_file_is_treated_as_background(tmp_path):
    stats = PreparationStats()
    assert validate_and_remap_label_file(tmp_path / "nope.txt", {0: 0}, ["D00"], stats) == []


def test_dhash_detects_near_duplicates():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    # Re-compressed + mildly resized copy: should be a near-duplicate.
    resized = cv2.resize(cv2.resize(base, (160, 120)), (320, 240))
    assert hamming(dhash(base), dhash(resized)) <= 5

    # A genuinely different image should not be.
    other = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert hamming(dhash(base), dhash(other)) > 5


# -----------------------------------------------------------------------------
# Visualization (smoke test only - the dashboard owns the real theme)
# -----------------------------------------------------------------------------


def test_visualization_runs_without_mutating_the_input(model):
    pytest.importorskip("cv2")
    from modules.road_damage import visualization as viz

    result = model.predict(road_frame())
    frame = road_frame()
    original = frame.copy()
    annotated = viz.draw_frame_result(frame, result)
    assert annotated.shape == frame.shape
    assert np.array_equal(frame, original), "draw_frame_result must not mutate the input"


def test_visualization_label_format(model):
    from modules.road_damage import visualization as viz

    detection = model.predict(road_frame())["detections"][0]
    assert viz.format_label(detection) == "pothole 0.93"


def test_visualization_skips_other_modules_detections():
    """A merged result must not have Person 3's boxes drawn as road damage."""
    pytest.importorskip("cv2")
    from modules.road_damage import visualization as viz
    from shared.schemas import empty_frame_result

    result = empty_frame_result()
    result["detections"] = [
        {"class": "speed_bump", "confidence": 0.9, "bbox": [10, 10, 90, 90], "source": "speed_bump"}
    ]
    frame = road_frame()
    annotated = viz.draw_frame_result(frame, result)
    assert np.array_equal(annotated, frame), "no road_damage detections, so nothing should be drawn"
