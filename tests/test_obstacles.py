"""
tests/test_obstacles.py
========================
Tests for modules.obstacles, in two tiers (see modules/obstacles/README.md
"Tests" section):

1. Unit tests (always run) -- exercise class_mapping.py's mapping/exclusion
   logic and RoadObstacleModel's merge/schema behavior against a stub
   backend, so the interface contract is verified even before any weights
   are trained.
2. Integration tests -- load the real .pt files from models/ and run actual
   inference. Auto-skipped (not failed) if the weights aren't present,
   since this delivery ships code + training scripts but not trained
   weights (see module README "Status of this delivery").

Covers the "12. REQUIRED TEST" checklist:
    1. Model loads successfully               -> integration tier
    2. Inference runs on one image             -> integration tier
    3. Inference runs on one video frame       -> unit tier (synthetic frame)
       + integration tier (real frame)
    4. Output follows the shared schema        -> both tiers
    5. Class names are normalized               -> both tiers
    6. Confidence values are valid              -> both tiers
    7. Bounding boxes are valid                 -> both tiers
    8. No unexpected exceptions                -> both tiers
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shared.config import MODELS_DIR
from shared.schemas import validate_detection, validate_frame_result

from modules.obstacles.class_mapping import (
    map_speed_bump_category,
    map_taco_category,
    verify_taco_categories,
)
from modules.obstacles.debris_model import DebrisModel
from modules.obstacles.speed_bump_model import SpeedBumpModel
from modules.obstacles.obstacle_model import RoadObstacleModel

DEBRIS_WEIGHTS = MODELS_DIR / "debris_best.pt"
SPEED_BUMP_WEIGHTS = MODELS_DIR / "speed_bump_best.pt"


# -----------------------------------------------------------------------------
# Tier 1a: class_mapping.py unit tests
# -----------------------------------------------------------------------------


def test_map_taco_category_known_maps_to_road_debris():
    normalized, subclass = map_taco_category("Clear plastic bottle")
    assert normalized == "road_debris"
    assert subclass == "plastic_bottle"


def test_map_taco_category_unknown_is_excluded_not_forced():
    normalized, subclass = map_taco_category("Not A Real TACO Category")
    assert normalized is None
    assert subclass is None


def test_verify_taco_categories_reports_unmapped_and_stale():
    diff = verify_taco_categories(["Clear plastic bottle", "Some Brand New Category"])
    assert "Clear plastic bottle" in diff["mapped"]
    assert "Some Brand New Category" in diff["unmapped"]
    assert "Food waste" in diff["stale"]  # in the table, not in this fake "downloaded" list


def test_map_speed_bump_category_known_variants():
    assert map_speed_bump_category("speed_bump") == "speed_bump"
    assert map_speed_bump_category("speed bump") == "speed_bump"


def test_map_speed_bump_category_unknown_is_excluded():
    assert map_speed_bump_category("pothole") is None  # wrong dataset's label, must not leak through


# -----------------------------------------------------------------------------
# Tier 1b: interface/schema tests against a stub backend (no real weights
# required -- this is what lets CI validate the module before training).
# -----------------------------------------------------------------------------


class _StubBackend:
    """Stand-in for _yolo_backend.YoloDetectorBackend that returns canned
    raw (bbox, confidence, raw_class_name) tuples instead of running a real
    model, so DebrisModel/SpeedBumpModel's normalization + schema logic can
    be tested without trained weights.
    """

    def __init__(self, raw_detections):
        self._raw_detections = raw_detections

    def raw_predict(self, frame):
        return self._raw_detections


@pytest.fixture
def stub_debris_model():
    raw = [
        ([10.0, 10.0, 50.0, 50.0], 0.87, "Clear plastic bottle"),  # mapped -> road_debris
        ([60.0, 60.0, 90.0, 90.0], 0.55, "Battery"),  # unmapped -> must be dropped
    ]
    with patch("modules.obstacles.debris_model.YoloDetectorBackend", return_value=_StubBackend(raw)):
        yield DebrisModel(model_path="unused-for-stub.pt")


@pytest.fixture
def stub_speed_bump_model():
    raw = [
        ([20.0, 20.0, 80.0, 40.0], 0.94, "speed_bump"),  # mapped
        ([100.0, 100.0, 150.0, 120.0], 0.42, "pothole"),  # unmapped -> dropped
    ]
    with patch("modules.obstacles.speed_bump_model.YoloDetectorBackend", return_value=_StubBackend(raw)):
        yield SpeedBumpModel(model_path="unused-for-stub.pt")


def _synthetic_frame(width=640, height=480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_debris_model_normalizes_and_drops_unmapped(stub_debris_model):
    result = stub_debris_model.predict(_synthetic_frame())
    validate_frame_result(result)
    assert len(result["detections"]) == 1  # "Battery" dropped
    det = result["detections"][0]
    validate_detection(det, frame_width=640, frame_height=480)
    assert det["class"] == "road_debris"
    assert det["source"] == "debris"
    assert det["subclass"] == "plastic_bottle"


def test_speed_bump_model_normalizes_and_drops_unmapped(stub_speed_bump_model):
    result = stub_speed_bump_model.predict(_synthetic_frame())
    validate_frame_result(result)
    assert len(result["detections"]) == 1  # "pothole" dropped
    det = result["detections"][0]
    validate_detection(det, frame_width=640, frame_height=480)
    assert det["class"] == "speed_bump"
    assert det["source"] == "speed_bump"


def test_road_obstacle_model_merges_both_detectors(stub_debris_model, stub_speed_bump_model):
    model = RoadObstacleModel.__new__(RoadObstacleModel)  # bypass __init__'s real loading
    model.debris_model = stub_debris_model
    model.speed_bump_model = stub_speed_bump_model

    result = model.predict(_synthetic_frame())
    validate_frame_result(result)

    classes = sorted(d["class"] for d in result["detections"])
    assert classes == ["road_debris", "speed_bump"]
    assert result["analytics"]["obstacle_count"] == 1
    assert result["analytics"]["speed_bump_count"] == 1


def test_debris_model_rejects_non_ndarray_frame(stub_debris_model):
    with pytest.raises(TypeError):
        stub_debris_model.predict([[1, 2, 3]])  # not a numpy.ndarray


def test_debris_model_missing_weights_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        DebrisModel(model_path="models/this_file_does_not_exist.pt")


# -----------------------------------------------------------------------------
# Tier 2: integration tests against real trained weights. Skipped, not
# failed, when weights aren't present -- see README "Status of this
# delivery".
# -----------------------------------------------------------------------------

_weights_present = DEBRIS_WEIGHTS.exists() and SPEED_BUMP_WEIGHTS.exists()


@pytest.mark.skipif(
    not _weights_present,
    reason=(
        "models/debris_best.pt and models/speed_bump_best.pt not found -- "
        "train them via modules/obstacles/training/ first (see module README)"
    ),
)
class TestRoadObstacleModelIntegration:
    @pytest.fixture(scope="class")
    def model(self):
        return RoadObstacleModel(DEBRIS_WEIGHTS, SPEED_BUMP_WEIGHTS)

    def test_model_loads_successfully(self, model):
        assert model.debris_model is not None
        assert model.speed_bump_model is not None

    def test_inference_runs_on_synthetic_frame(self, model):
        result = model.predict(_synthetic_frame())
        validate_frame_result(result)

    def test_inference_runs_on_sample_image_if_available(self, model):
        sample = Path("tests/sample_images/road_sample.jpg")
        if not sample.exists():
            pytest.skip(f"no sample image at {sample} -- add one to exercise real-image inference")
        import cv2

        frame = cv2.imread(str(sample))
        result = model.predict(frame)
        validate_frame_result(result)
        for det in result["detections"]:
            validate_detection(det, frame_width=frame.shape[1], frame_height=frame.shape[0])
