"""
modules/obstacles/speed_bump_model.py
======================================
Speed-bump detector. Wraps a YOLO model trained on the Roboflow
"Speed Bump Detection v10" dataset, normalizing whatever raw label(s) the
dataset ships to the single project class `speed_bump`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shared.config import SOURCE_SPEED_BUMP, InferenceConfig
from shared.schemas import empty_frame_result

from ._yolo_backend import YoloDetectorBackend, validate_frame
from .class_mapping import map_speed_bump_category

logger = logging.getLogger(__name__)


class SpeedBumpModel:
    """Speed-bump detector (Roboflow Speed Bump Detection v10 -> speed_bump).

    Usage:
        model = SpeedBumpModel("models/speed_bump_best.pt")
        result = model.predict(frame)   # frame: numpy.ndarray, BGR, HxWx3
    """

    SOURCE = SOURCE_SPEED_BUMP

    def __init__(self, model_path: str | Path, config: InferenceConfig | dict | None = None):
        self._backend = YoloDetectorBackend(model_path, config)

    def predict(self, frame: np.ndarray) -> dict:
        """Run speed-bump detection on one frame.

        Returns a schema-shaped dict (see shared.schemas.empty_frame_result)
        with only `detections` populated.
        """
        validate_frame(frame)
        result = empty_frame_result()
        raw_dets = self._backend.raw_predict(frame)

        detections = []
        dropped = 0
        for bbox, confidence, raw_name in raw_dets:
            normalized_class = map_speed_bump_category(raw_name)
            if normalized_class is None:
                dropped += 1
                logger.debug("Dropping unrecognized speed-bump label %r (see class_mapping.py)", raw_name)
                continue
            detections.append(
                {
                    "class": normalized_class,
                    "confidence": confidence,
                    "bbox": bbox,
                    "source": self.SOURCE,
                }
            )

        if dropped:
            logger.info("SpeedBumpModel: dropped %d unrecognized-label detections this frame", dropped)

        result["detections"] = detections
        return result
