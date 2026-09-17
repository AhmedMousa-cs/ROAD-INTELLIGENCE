"""
tests/test_road_segmentation.py

Covers spec section 12's required checks:
  1. Model loads successfully           (skipped if no real weights present)
  2. Inference runs on one image         (skipped if no real weights present)
  3. Inference runs on one video frame   (frames and images are both plain
                                           numpy arrays, so this is exercised
                                           via the postprocessing path below
                                           without needing a video file)
  4. Output follows the shared schema
  5. Class names are normalized
  6. Confidence values are valid
  7. Bounding boxes are valid            (N/A here — this module returns
                                           masks/polygons, not bboxes; we
                                           assert polygon/area validity instead)
  8. No unexpected exceptions occur

Tests 4-8 (and most of 1-3's *logic*) run against pure functions
(`build_segmentation_output`, `polygon_to_mask_payload`, `normalize_class`)
so they work in a clean environment with no trained weights and, for the
pure-mapping tests, no ultralytics/opencv install either. The two
weights-dependent tests are skipped (not failed) when
`models/road_segmentation_best.pt` doesn't exist yet, since no team member
has trained/committed it in this scaffold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.config import PROJECT_ROOT, SOURCE_ROAD_SEGMENTATION
from shared.schemas import validate_segmentation

from modules.road_segmentation.class_mapping import normalize_class
from modules.road_segmentation.road_segmentation_model import (
    RoadSegmentationModel,
    build_segmentation_output,
    polygon_to_mask_payload,
)

MODEL_PATH = PROJECT_ROOT / "models" / "road_segmentation_best.pt"
WEIGHTS_AVAILABLE = MODEL_PATH.exists()


# -----------------------------------------------------------------------------
# Synthetic fixtures — no real model/weights required
# -----------------------------------------------------------------------------
FRAME_SHAPE = (480, 640, 3)  # H, W, C

# A simple square polygon (100x100 px) well inside a 640x480 frame.
SQUARE_POLYGON = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

RAW_CLASS_NAMES = ["pothole", "alligator", "Major Surface Damage", "Minor Surface Damage", "Road Patch", "Ravelling"]


def _make_synthetic_frame() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


# -----------------------------------------------------------------------------
# 5. Class names are normalized
# -----------------------------------------------------------------------------
def test_normalize_class_maps_known_labels():
    assert normalize_class("pothole", "roboflow_road_defect") == "pothole"
    assert normalize_class("alligator", "roboflow_road_defect") == "alligator_crack"
    assert normalize_class("Major Surface Damage", "roboflow_road_defect") == "surface_damage"
    assert normalize_class("Minor Surface Damage", "roboflow_road_defect") == "surface_damage"
    assert normalize_class("Road Patch", "roboflow_road_defect") == "road_patch"
    assert normalize_class("Ravelling", "roboflow_road_defect") == "ravelling"
    assert normalize_class("pothole", "pothole_seg") == "pothole"


def test_normalize_class_returns_none_for_unmapped_label():
    # Unknown/unmapped raw class must never be force-mapped or leaked through.
    assert normalize_class("totally_unknown_dataset_label", "roboflow_road_defect") is None


def test_normalize_class_rejects_unknown_dataset_key():
    with pytest.raises(ValueError):
        normalize_class("pothole", "not_a_real_dataset")


# -----------------------------------------------------------------------------
# Mask geometry / area_ratio correctness (feeds "bounding boxes are valid"
# equivalent check for a segmentation module: polygon + area validity)
# -----------------------------------------------------------------------------
def test_polygon_to_mask_payload_area_and_ratio():
    geom = polygon_to_mask_payload(SQUARE_POLYGON, FRAME_SHAPE)
    # 100x100 square -> 10000 px area
    assert geom["area_px"] == pytest.approx(10000, rel=0.01)
    expected_ratio = 10000 / (480 * 640)
    assert geom["area_ratio"] == pytest.approx(expected_ratio, rel=0.01)
    assert 0.0 <= geom["area_ratio"] <= 1.0
    assert geom["mask"]["format"] == "polygon"
    assert len(geom["mask"]["points"]) == 4


def test_polygon_to_mask_payload_handles_degenerate_polygon():
    # Fewer than 3 points -> zero area, must not raise.
    degenerate = np.array([[10, 10], [20, 20]], dtype=np.float32)
    geom = polygon_to_mask_payload(degenerate, FRAME_SHAPE)
    assert geom["area_px"] == 0
    assert geom["area_ratio"] == 0.0


# -----------------------------------------------------------------------------
# 4, 5, 6. Output follows shared schema; classes normalized; confidence valid
# -----------------------------------------------------------------------------
def test_build_segmentation_output_schema_and_normalization():
    class_ids = [0, 1, 2]  # pothole, alligator, Major Surface Damage
    confidences = [0.91, 0.55, 0.72]
    polygons = [SQUARE_POLYGON, SQUARE_POLYGON, SQUARE_POLYGON]

    result = build_segmentation_output(
        raw_class_names=RAW_CLASS_NAMES,
        class_ids=class_ids,
        confidences=confidences,
        polygons_xy=polygons,
        frame_shape=FRAME_SHAPE,
        dataset="roboflow_road_defect",
        confidence_threshold=0.40,
    )

    assert "segmentations" in result
    assert len(result["segmentations"]) == 3

    seen_classes = {s["class"] for s in result["segmentations"]}
    assert seen_classes == {"pothole", "alligator_crack", "surface_damage"}

    for seg in result["segmentations"]:
        assert seg["source"] == SOURCE_ROAD_SEGMENTATION
        # Raises AssertionError if malformed / class not in shared/classes.yaml
        validate_segmentation(seg)


def test_build_segmentation_output_filters_low_confidence():
    result = build_segmentation_output(
        raw_class_names=RAW_CLASS_NAMES,
        class_ids=[0],
        confidences=[0.10],  # below default 0.40 threshold
        polygons_xy=[SQUARE_POLYGON],
        frame_shape=FRAME_SHAPE,
        dataset="roboflow_road_defect",
        confidence_threshold=0.40,
    )
    assert result["segmentations"] == []


def test_build_segmentation_output_drops_unmapped_raw_class():
    names_with_junk = RAW_CLASS_NAMES + ["some_unmapped_label"]
    result = build_segmentation_output(
        raw_class_names=names_with_junk,
        class_ids=[6],  # index of "some_unmapped_label"
        confidences=[0.99],
        polygons_xy=[SQUARE_POLYGON],
        frame_shape=FRAME_SHAPE,
        dataset="roboflow_road_defect",
        confidence_threshold=0.40,
    )
    # Must be dropped, not force-mapped or leaked through as-is.
    assert result["segmentations"] == []


# -----------------------------------------------------------------------------
# 8. No unexpected exceptions — input validation on predict()
# -----------------------------------------------------------------------------
def test_predict_rejects_non_ndarray_input():
    model = RoadSegmentationModel(str(MODEL_PATH))  # path need not exist for this check
    with pytest.raises(TypeError):
        model.predict("not an array")


def test_predict_rejects_wrong_shape_input():
    model = RoadSegmentationModel(str(MODEL_PATH))
    bad_frame = np.zeros((480, 640), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError):
        model.predict(bad_frame)


def test_predict_raises_clear_error_when_weights_missing():
    missing_path = Path("models/definitely_not_a_real_checkpoint.pt")
    model = RoadSegmentationModel(str(missing_path))
    frame = _make_synthetic_frame()
    with pytest.raises(FileNotFoundError):
        model.predict(frame)


# -----------------------------------------------------------------------------
# 1-3. Real model load + inference on an image and a "video frame"
# (a decoded video frame is just a numpy.ndarray, same as an image, so one
# real predict() call on a synthetic frame covers both once weights exist)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(
    not WEIGHTS_AVAILABLE,
    reason=f"No trained weights at {MODEL_PATH} yet — train the model per README.md",
)
def test_model_loads_and_predicts_on_real_weights():
    model = RoadSegmentationModel(str(MODEL_PATH))
    frame = _make_synthetic_frame()

    result = model.predict(frame)

    assert "segmentations" in result
    for seg in result["segmentations"]:
        validate_segmentation(seg)
