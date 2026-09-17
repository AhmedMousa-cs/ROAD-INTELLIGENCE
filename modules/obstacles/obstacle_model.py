"""
modules/obstacles/obstacle_model.py
====================================
Unified module interface for Person 3's two detectors (road debris, speed
bump), exposed as ONE class so the integration layer only has to load one
thing per the shared model interface (shared/README.md):

    model = RoadObstacleModel(debris_path, speed_bump_path)
    result = model.predict(frame)

Internally this simply runs both YOLO models on the same frame and
concatenates their already-normalized `detections` lists -- no merged
training, no shared weights, per the spec's "prefer two specialized
detectors behind one wrapper" instruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.config import InferenceConfig
from shared.schemas import empty_frame_result

from .debris_model import DebrisModel
from .speed_bump_model import SpeedBumpModel


class RoadObstacleModel:
    """Combined road-debris + speed-bump detector.

    Usage:
        model = RoadObstacleModel(
            "models/debris_best.pt",
            "models/speed_bump_best.pt",
        )
        result = model.predict(frame)   # frame: numpy.ndarray, BGR, HxWx3
        result["detections"]  # road_debris + speed_bump detections, merged
    """

    def __init__(
        self,
        debris_model_path: str | Path,
        speed_bump_model_path: str | Path,
        config: InferenceConfig | dict | None = None,
    ):
        # A single `config` is accepted (and forwarded to both sub-models)
        # for the common case of one confidence/IoU/image-size policy for
        # the whole obstacles module. Pass per-model configs by constructing
        # DebrisModel / SpeedBumpModel directly if the two detectors ever
        # need different thresholds.
        self.debris_model = DebrisModel(debris_model_path, config=config)
        self.speed_bump_model = SpeedBumpModel(speed_bump_model_path, config=config)

    def predict(self, frame: np.ndarray) -> dict:
        """Run both detectors on one frame and merge into one schema-shaped
        result (see shared.schemas.empty_frame_result). Only `detections`
        and `analytics.obstacle_count` / `analytics.speed_bump_count` are
        populated -- everything else is left at its empty default so this
        module's output can be passed straight into
        shared.schemas.merge_frame_results() alongside the other modules.
        """
        debris_result = self.debris_model.predict(frame)
        speed_bump_result = self.speed_bump_model.predict(frame)

        result = empty_frame_result()
        result["detections"] = debris_result["detections"] + speed_bump_result["detections"]
        result["analytics"]["obstacle_count"] = len(debris_result["detections"])
        result["analytics"]["speed_bump_count"] = len(speed_bump_result["detections"])
        return result
