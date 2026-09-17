"""
Road Intelligence — Shared Output Schema
==========================================
Defines the standardized result structure every module must conform to,
plus validation helpers that tests/test_<module>.py should call to verify
"class names are normalized", "bounding boxes are valid", etc.

Do not delete fields from FrameResult when a module doesn't use them —
leave them at their empty default (see `empty_frame_result()`).
"""

from __future__ import annotations

from typing import Any, TypedDict

import yaml

from shared.config import CLASSES_YAML_PATH

# -----------------------------------------------------------------------------
# TypedDicts describing the wire format. These match the JSON-compatible
# dict structure in the spec exactly — modules return plain dicts, these
# types exist for editor/type-checker support and for the validators below.
# -----------------------------------------------------------------------------


class Detection(TypedDict):
    class_: str  # NOTE: dict key is literally "class", see note below
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2], origin top-left, original frame dims
    source: str


class Segmentation(TypedDict):
    class_: str
    confidence: float
    mask: Any
    area_px: int
    area_ratio: float
    source: str


class Track(TypedDict, total=False):
    track_id: int
    class_: str
    bbox: list[float]
    center: list[float]
    velocity_px_s: float


class Alert(TypedDict, total=False):
    type: str
    track_id: int
    confidence: float
    message: str


class Analytics(TypedDict):
    vehicle_count: int
    traffic_density: str
    road_damage_count: int
    obstacle_count: int
    speed_bump_count: int


class FrameResult(TypedDict):
    frame_id: int
    timestamp: float
    detections: list[dict]
    segmentations: list[dict]
    tracks: list[dict]
    alerts: list[dict]
    analytics: Analytics


# -----------------------------------------------------------------------------
# NOTE ON "class" AS A DICT KEY
# -----------------------------------------------------------------------------
# The spec's wire format uses the literal key "class" (a Python builtin
# name), e.g. {"class": "pothole", ...}. Modules must emit plain dicts with
# a "class" key — do NOT construct a Detection(...) TypedDict positionally
# expecting class_ to become "class"; TypedDict does not rename keys.
# Build detections as plain dicts, e.g.:
#     {"class": "pothole", "confidence": 0.93, "bbox": [...], "source": "road_damage"}
# The TypedDicts above are for static-typing/documentation purposes only.


def empty_frame_result(frame_id: int = 0, timestamp: float = 0.0) -> dict:
    """Return a fresh, schema-complete, all-empty FrameResult dict.

    Every module should start from this (or merge into it) rather than
    hand-rolling the top-level dict, so no field is ever accidentally
    dropped.
    """
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "detections": [],
        "segmentations": [],
        "tracks": [],
        "alerts": [],
        "analytics": {
            "vehicle_count": 0,
            "traffic_density": "low",
            "road_damage_count": 0,
            "obstacle_count": 0,
            "speed_bump_count": 0,
        },
    }


def merge_frame_results(*results: dict, frame_id: int = 0, timestamp: float = 0.0) -> dict:
    """Merge several modules' partial FrameResult dicts into one final result.

    Mirrors the "FINAL SYSTEM API" merge step in shared/README.md. Later
    results win on scalar analytics fields; list fields are concatenated.
    """
    final = empty_frame_result(frame_id=frame_id, timestamp=timestamp)
    for result in results:
        final["detections"].extend(result.get("detections", []))
        final["segmentations"].extend(result.get("segmentations", []))
        final["tracks"].extend(result.get("tracks", []))
        final["alerts"].extend(result.get("alerts", []))
        final["analytics"].update(result.get("analytics", {}))
    return final


# -----------------------------------------------------------------------------
# Class ontology loading + validation
# -----------------------------------------------------------------------------

_ALLOWED_CLASSES_CACHE: set[str] | None = None


def load_allowed_classes() -> set[str]:
    """Load the flat set of every normalized class name from classes.yaml."""
    global _ALLOWED_CLASSES_CACHE
    if _ALLOWED_CLASSES_CACHE is not None:
        return _ALLOWED_CLASSES_CACHE

    with open(CLASSES_YAML_PATH) as f:
        raw = yaml.safe_load(f)

    allowed: set[str] = set()
    for key in ("road_damage", "road_surface", "road_obstacle", "road_feature", "vehicles"):
        allowed.update(raw.get(key, []))
    _ALLOWED_CLASSES_CACHE = allowed
    return allowed


# -----------------------------------------------------------------------------
# Validators — call these from tests/test_<module>.py
# -----------------------------------------------------------------------------


def validate_bbox(bbox: list[float], frame_width: int | None = None, frame_height: int | None = None) -> None:
    """Raise AssertionError if bbox is malformed.

    bbox must be [x1, y1, x2, y2], x1 < x2, y1 < y2, all finite and >= 0.
    If frame dimensions are given, also checks bbox lies within the frame.
    """
    assert isinstance(bbox, (list, tuple)) and len(bbox) == 4, f"bbox must have 4 values, got {bbox!r}"
    x1, y1, x2, y2 = bbox
    for v in (x1, y1, x2, y2):
        assert isinstance(v, (int, float)), f"bbox values must be numeric, got {bbox!r}"
        assert v >= 0, f"bbox coordinates must be >= 0, got {bbox!r}"
    assert x1 < x2, f"x1 must be < x2, got {bbox!r}"
    assert y1 < y2, f"y1 must be < y2, got {bbox!r}"
    if frame_width is not None:
        assert x2 <= frame_width, f"bbox x2 exceeds frame width {frame_width}: {bbox!r}"
    if frame_height is not None:
        assert y2 <= frame_height, f"bbox y2 exceeds frame height {frame_height}: {bbox!r}"


def validate_confidence(confidence: float) -> None:
    assert isinstance(confidence, (int, float)), f"confidence must be numeric, got {confidence!r}"
    assert 0.0 <= confidence <= 1.0, f"confidence must be in [0, 1], got {confidence!r}"


def validate_class_name(class_name: str) -> None:
    allowed = load_allowed_classes()
    assert class_name in allowed, (
        f"'{class_name}' is not a normalized class name from shared/classes.yaml. "
        f"Dataset-specific labels must be mapped before leaving the module. "
        f"Allowed: {sorted(allowed)}"
    )


def validate_detection(detection: dict, frame_width: int | None = None, frame_height: int | None = None) -> None:
    for key in ("class", "confidence", "bbox", "source"):
        assert key in detection, f"detection missing required key '{key}': {detection!r}"
    validate_class_name(detection["class"])
    validate_confidence(detection["confidence"])
    validate_bbox(detection["bbox"], frame_width, frame_height)
    assert isinstance(detection["source"], str) and detection["source"], "source must be a non-empty string"


def validate_segmentation(segmentation: dict) -> None:
    for key in ("class", "confidence", "mask", "area_px", "area_ratio", "source"):
        assert key in segmentation, f"segmentation missing required key '{key}': {segmentation!r}"
    validate_class_name(segmentation["class"])
    validate_confidence(segmentation["confidence"])
    assert isinstance(segmentation["area_px"], int) and segmentation["area_px"] >= 0
    assert 0.0 <= segmentation["area_ratio"] <= 1.0, (
        f"area_ratio must be segmented_area / image_area in [0, 1], got {segmentation['area_ratio']!r}"
    )
    assert isinstance(segmentation["source"], str) and segmentation["source"]


def validate_frame_result(result: dict) -> None:
    """Top-level check that a module's output dict has every required field."""
    for key in ("frame_id", "timestamp", "detections", "segmentations", "tracks", "alerts", "analytics"):
        assert key in result, f"FrameResult missing required key '{key}'"
    for det in result["detections"]:
        validate_detection(det)
    for seg in result["segmentations"]:
        validate_segmentation(seg)
    analytics = result["analytics"]
    for key in ("vehicle_count", "traffic_density", "road_damage_count", "obstacle_count", "speed_bump_count"):
        assert key in analytics, f"analytics missing required key '{key}'"
