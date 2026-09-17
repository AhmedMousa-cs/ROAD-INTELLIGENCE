"""
modules/obstacles/_yolo_backend.py
===================================
Small internal helper shared by debris_model.py and speed_bump_model.py so
both detectors load and run Ultralytics YOLO the same way, honoring
shared/config.py thresholds. Not part of the public module interface --
external code should import DebrisModel / SpeedBumpModel / RoadObstacleModel
from `modules.obstacles`, not this file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from shared.config import InferenceConfig, resolve_device

logger = logging.getLogger(__name__)


def validate_frame(frame: np.ndarray) -> None:
    """Shared input check used by DebrisModel/SpeedBumpModel.predict() --
    lives here (not only inside YoloDetectorBackend.raw_predict) so it still
    applies when the backend is swapped for a test stub.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy.ndarray (BGR, HxWx3), got {type(frame)!r}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 BGR, got shape {frame.shape!r}")


class YoloDetectorBackend:
    """Thin wrapper around ultralytics.YOLO with a single, boring job:
    load a .pt file and return raw (xyxy box, confidence, raw class name)
    tuples for one BGR frame, honoring the shared confidence/IoU/image-size
    thresholds. All class-name normalization stays in the caller
    (DebrisModel / SpeedBumpModel), which is why this class never imports
    class_mapping.py.
    """

    def __init__(self, model_path: str | Path, config: InferenceConfig | dict | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. Train the detector first "
                f"(see modules/obstacles/training/) or point model_path at a valid .pt file."
            )

        self.config = self._normalize_config(config)
        self._device = resolve_device(self.config.device)

        # Imported lazily so `import modules.obstacles` never requires
        # ultralytics/torch to be installed just to inspect the interface
        # (e.g. from a lightweight test-collection environment).
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        self._names: dict[int, str] = self._model.names  # {class_id: raw_dataset_label}

    @staticmethod
    def _normalize_config(config: InferenceConfig | dict | None) -> InferenceConfig:
        if config is None:
            return InferenceConfig()
        if isinstance(config, InferenceConfig):
            return config
        if isinstance(config, dict):
            defaults = InferenceConfig()
            return InferenceConfig(
                confidence_threshold=config.get("confidence_threshold", defaults.confidence_threshold),
                iou_threshold=config.get("iou_threshold", defaults.iou_threshold),
                image_size=config.get("image_size", defaults.image_size),
                device=config.get("device", defaults.device),
            )
        raise TypeError(f"config must be InferenceConfig, dict, or None, got {type(config)!r}")

    def raw_predict(self, frame: np.ndarray) -> list[tuple[list[float], float, str]]:
        """Run inference on one BGR frame.

        Returns a list of (bbox_xyxy, confidence, raw_class_name) tuples in
        ORIGINAL frame pixel coordinates. Applies no class-name mapping --
        raw_class_name is whatever the .pt file's own names dict says.
        """
        validate_frame(frame)

        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self._device,
            verbose=False,
        )

        out: list[tuple[list[float], float, str]] = []
        if not results:
            return out

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            raw_name = self._names.get(int(cls_id), str(cls_id))
            out.append(([float(v) for v in box], float(conf), raw_name))
        return out
