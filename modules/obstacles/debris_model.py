"""
modules/obstacles/debris_model.py
==================================
Road-debris detector. Wraps a YOLO model trained on TACO (Trash Annotations
in Context), normalizing every TACO category down to the single project
class `road_debris` via class_mapping.map_taco_category().

Per spec: TACO detections that don't have a safe mapping are DROPPED, not
force-mapped -- see class_mapping.py for the full rationale and how to
extend the mapping table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shared.config import SOURCE_DEBRIS, InferenceConfig
from shared.schemas import empty_frame_result

from ._yolo_backend import YoloDetectorBackend, validate_frame
from .class_mapping import map_taco_category

logger = logging.getLogger(__name__)


class DebrisModel:
    """Road-debris detector (TACO -> road_debris).

    Usage:
        model = DebrisModel("models/debris_best.pt")
        result = model.predict(frame)   # frame: numpy.ndarray, BGR, HxWx3
    """

    SOURCE = SOURCE_DEBRIS

    def __init__(self, model_path: str | Path, config: InferenceConfig | dict | None = None):
        self._backend = YoloDetectorBackend(model_path, config)

    def predict(self, frame: np.ndarray) -> dict:
        """Run debris detection on one frame.

        Returns a schema-shaped dict (see shared.schemas.empty_frame_result)
        with only `detections` populated. Each detection additionally
        carries an optional, non-schema `subclass` key (e.g. "plastic_bottle")
        for UI/analytics use -- downstream code should not rely on it being
        present, since it's metadata, not part of the shared contract.
        """
        validate_frame(frame)
        result = empty_frame_result()
        raw_dets = self._backend.raw_predict(frame)

        detections = []
        dropped = 0
        for bbox, confidence, raw_name in raw_dets:
            normalized_class, subclass = map_taco_category(raw_name)
            if normalized_class is None:
                dropped += 1
                logger.debug("Dropping unmapped TACO category %r (see class_mapping.py)", raw_name)
                continue
            detections.append(
                {
                    "class": normalized_class,
                    "confidence": confidence,
                    "bbox": bbox,
                    "source": self.SOURCE,
                    "subclass": subclass,
                }
            )

        if dropped:
            logger.info("DebrisModel: dropped %d unmapped-category detections this frame", dropped)

        result["detections"] = detections
        return result
