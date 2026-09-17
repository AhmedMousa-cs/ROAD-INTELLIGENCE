"""
Road Intelligence — Dashboard logic (pure functions, no Streamlit UI code).

Kept separate from streamlit_app.py so this can be unit-tested and reused
(e.g. by a batch-processing script) without a Streamlit runtime.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from shared.config import MODELS_DIR
from shared.schemas import merge_frame_results, validate_frame_result

ROAD_DAMAGE_WEIGHTS = MODELS_DIR / "road_damage_best.pt"
ROAD_SEGMENTATION_WEIGHTS = MODELS_DIR / "road_segmentation_best.pt"
DEBRIS_WEIGHTS = MODELS_DIR / "debris_best.pt"
SPEED_BUMP_WEIGHTS = MODELS_DIR / "speed_bump_best.pt"
VEHICLE_WEIGHTS = None  # pretrained COCO default, per spec no custom training


def load_models() -> tuple[dict, dict[str, str]]:
    """Load whichever modules have weights on disk.

    Returns (models, status). status[name] is "loaded", "loaded (...)",
    "missing weights (...)", or "error: ...".
    """
    models: dict = {}
    status: dict[str, str] = {}

    if ROAD_DAMAGE_WEIGHTS.exists():
        try:
            from modules.road_damage import RoadDamageModel

            models["road_damage"] = RoadDamageModel(str(ROAD_DAMAGE_WEIGHTS))
            status["road_damage"] = "loaded"
        except Exception as exc:  # noqa: BLE001
            status["road_damage"] = f"error: {exc}"
    else:
        status["road_damage"] = f"missing weights ({ROAD_DAMAGE_WEIGHTS.name})"

    if ROAD_SEGMENTATION_WEIGHTS.exists():
        try:
            from modules.road_segmentation import RoadSegmentationModel

            models["road_segmentation"] = RoadSegmentationModel(str(ROAD_SEGMENTATION_WEIGHTS))
            status["road_segmentation"] = "loaded"
        except Exception as exc:  # noqa: BLE001
            status["road_segmentation"] = f"error: {exc}"
    else:
        status["road_segmentation"] = f"missing weights ({ROAD_SEGMENTATION_WEIGHTS.name})"

    if DEBRIS_WEIGHTS.exists() and SPEED_BUMP_WEIGHTS.exists():
        try:
            from modules.obstacles import RoadObstacleModel

            models["obstacles"] = RoadObstacleModel(str(DEBRIS_WEIGHTS), str(SPEED_BUMP_WEIGHTS))
            status["obstacles"] = "loaded"
        except Exception as exc:  # noqa: BLE001
            status["obstacles"] = f"error: {exc}"
    else:
        missing = [p.name for p in (DEBRIS_WEIGHTS, SPEED_BUMP_WEIGHTS) if not p.exists()]
        status["obstacles"] = f"missing weights ({', '.join(missing)})"

    try:
        from modules.tracking_analytics import TrackingAnalytics

        models["tracking_analytics"] = TrackingAnalytics(VEHICLE_WEIGHTS)
        status["tracking_analytics"] = "loaded (pretrained COCO — no training needed)"
    except Exception as exc:  # noqa: BLE001
        status["tracking_analytics"] = f"error: {exc}"

    return models, status


def load_image_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array."""
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode this file as an image.")
    return frame


def run_pipeline(models: dict, frame: np.ndarray) -> dict:
    """Run every loaded module on one frame and return a validated,
    merged shared.schemas FrameResult dict.
    """
    partials = []

    if "road_damage" in models:
        partials.append(models["road_damage"].predict(frame))
    if "road_segmentation" in models:
        partials.append(models["road_segmentation"].predict(frame))
    if "obstacles" in models:
        partials.append(models["obstacles"].predict(frame))

    if "tracking_analytics" in models:
        models["tracking_analytics"].reset()
        tracking_result = models["tracking_analytics"].process(frame, other_results=partials)
        partials.append(tracking_result)

    final_result = merge_frame_results(*partials, frame_id=0, timestamp=0.0)
    validate_frame_result(final_result)
    return final_result


def run_debris_only(models: dict, frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Run ONLY the road-debris detector on one frame, bypassing road_damage,
    road_segmentation, speed_bump and tracking entirely.

    Reuses the already-loaded `RoadObstacleModel.debris_model` sub-model
    (see modules/obstacles/obstacle_model.py) so no extra weights are loaded
    and thresholds stay consistent with the rest of the dashboard.

    Raises RuntimeError if the obstacles module isn't loaded (i.e.
    debris_best.pt and/or speed_bump_best.pt are missing from models/).

    Returns (annotated_frame, debris_count).
    """
    obstacles_model = models.get("obstacles")
    if obstacles_model is None:
        raise RuntimeError(
            "Debris model isn't loaded. Drop debris_best.pt "
            "(and speed_bump_best.pt) into models/ and reload the page."
        )

    result = obstacles_model.debris_model.predict(frame)
    detections = result["detections"]
    annotated = draw_all_detections(frame, detections)
    return annotated, len(detections)


def run_speed_bump_only(models: dict, frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Run ONLY the speed-bump detector on one frame, bypassing road_damage,
    road_segmentation, debris and tracking entirely.

    Reuses the already-loaded `RoadObstacleModel.speed_bump_model` sub-model
    (see modules/obstacles/obstacle_model.py) so no extra weights are loaded
    and thresholds stay consistent with the rest of the dashboard.

    Raises RuntimeError if the obstacles module isn't loaded (i.e.
    debris_best.pt and/or speed_bump_best.pt are missing from models/).

    Returns (annotated_frame, speed_bump_count).
    """
    obstacles_model = models.get("obstacles")
    if obstacles_model is None:
        raise RuntimeError(
            "Speed-bump model isn't loaded. Drop speed_bump_best.pt "
            "(and debris_best.pt) into models/ and reload the page."
        )

    result = obstacles_model.speed_bump_model.predict(frame)
    detections = result["detections"]
    annotated = draw_all_detections(frame, detections)
    return annotated, len(detections)


def draw_all_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw every bbox detection (road damage + obstacles + speed bumps +
    vehicles), color-grouped by class, using road_damage's generic drawer
    (any module's detection dicts share the same {class, confidence, bbox}
    shape, so it works for all of them). Extends road_damage's own
    DEFAULT_THEME.per_class with colors for the other modules' classes.
    """
    from modules.road_damage.visualization import DEFAULT_THEME, draw_detections

    theme = {
        **DEFAULT_THEME,
        "per_class": {
            **DEFAULT_THEME["per_class"],  # pothole/alligator/longitudinal/transverse
            "road_debris": (0, 200, 0),
            "speed_bump": (200, 200, 0),
            "car": (255, 0, 0),
            "truck": (255, 60, 0),
            "bus": (255, 120, 0),
            "motorcycle": (200, 0, 0),
            "bicycle": (150, 0, 0),
            "person": (255, 255, 0),
        },
    }
    return draw_detections(frame, detections, theme=theme)


def render_segmentation_overlay(frame: np.ndarray, segmentations: list[dict]) -> np.ndarray:
    """Best-effort translucent overlay for segmentation masks. Handles a
    boolean/uint8 numpy mask directly; skips anything else (e.g. an RLE
    string) since decoding that format belongs to road_segmentation, not
    the dashboard.
    """
    overlay = frame.copy()
    drawn = False
    for seg in segmentations:
        mask = seg.get("mask")
        if not isinstance(mask, np.ndarray):
            continue
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        color = np.array([0, 255, 255], dtype=np.uint8)
        region = mask_resized > 0
        overlay[region] = (0.5 * overlay[region] + 0.5 * color).astype(np.uint8)
        drawn = True
    return overlay if drawn else frame
