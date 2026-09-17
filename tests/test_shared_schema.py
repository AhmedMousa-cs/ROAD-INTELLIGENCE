"""
Sanity tests for shared/schemas.py and shared/config.py themselves.

Each member's tests/test_<module>.py should follow this pattern:
    1. Load the model from a plain model_path (clean-environment import).
    2. Run predict() on one real image and one synthetic frame.
    3. Call shared.schemas.validate_frame_result() (or validate_detection /
       validate_segmentation directly) on the output.
    4. Assert no unexpected exceptions.

Run with: pytest tests/test_shared_schema.py -v
"""

import numpy as np
import pytest

from shared.schemas import (
    empty_frame_result,
    merge_frame_results,
    validate_bbox,
    validate_confidence,
    validate_class_name,
    validate_detection,
    validate_segmentation,
    validate_frame_result,
)


def test_empty_frame_result_has_all_fields():
    result = empty_frame_result(frame_id=1, timestamp=0.5)
    validate_frame_result(result)
    assert result["frame_id"] == 1
    assert result["timestamp"] == 0.5


def test_merge_frame_results_concatenates_and_overwrites_analytics():
    a = empty_frame_result()
    a["detections"].append(
        {"class": "pothole", "confidence": 0.9, "bbox": [0, 0, 10, 10], "source": "road_damage"}
    )
    b = empty_frame_result()
    b["analytics"]["vehicle_count"] = 5

    merged = merge_frame_results(a, b, frame_id=3, timestamp=1.2)
    validate_frame_result(merged)
    assert len(merged["detections"]) == 1
    assert merged["analytics"]["vehicle_count"] == 5


def test_validate_bbox_rejects_malformed():
    validate_bbox([0, 0, 10, 10])  # should not raise
    with pytest.raises(AssertionError):
        validate_bbox([10, 10, 0, 0])  # x2 < x1
    with pytest.raises(AssertionError):
        validate_bbox([-1, 0, 10, 10])  # negative coordinate
    with pytest.raises(AssertionError):
        validate_bbox([0, 0, 700, 10], frame_width=640)  # exceeds frame


def test_validate_confidence_range():
    validate_confidence(0.0)
    validate_confidence(1.0)
    with pytest.raises(AssertionError):
        validate_confidence(1.1)
    with pytest.raises(AssertionError):
        validate_confidence(-0.1)


def test_validate_class_name_rejects_dataset_specific_labels():
    validate_class_name("pothole")  # normalized name: OK
    with pytest.raises(AssertionError):
        validate_class_name("D00")  # raw RDD2022 label: must be rejected


def test_validate_detection_end_to_end():
    det = {"class": "speed_bump", "confidence": 0.94, "bbox": [10, 10, 50, 50], "source": "speed_bump"}
    validate_detection(det, frame_width=640, frame_height=480)


def test_validate_segmentation_end_to_end():
    seg = {
        "class": "pothole",
        "confidence": 0.91,
        "mask": np.zeros((10, 10), dtype=np.uint8),
        "area_px": 12345,
        "area_ratio": 0.034,
        "source": "road_segmentation",
    }
    validate_segmentation(seg)
