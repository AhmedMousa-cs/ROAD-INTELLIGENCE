"""
Road damage detection model wrapper.

    from modules.road_damage import RoadDamageModel

    model = RoadDamageModel("models/road_damage_best.pt")
    result = model.predict(frame)

``result`` is a shared-schema FrameResult dict. This module fills
``detections`` only; ``segmentations`` (Person 2), ``tracks``/``alerts``/
``analytics`` (Person 4) are left at their empty defaults so
``shared.schemas.merge_frame_results()`` can concatenate all four modules'
output without duplication.

Unlike Person 4's tracking module this one is STATELESS: the same frame
always produces the same result, and no ``reset()`` is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from shared.config import MODELS_DIR, SOURCE_ROAD_DAMAGE, InferenceConfig, resolve_device
from shared.schemas import empty_frame_result

from .class_mapping import ID_TO_CLASS, PROJECT_CLASSES, normalize_class

SOURCE = SOURCE_ROAD_DAMAGE  # "road_damage"

DEFAULT_WEIGHTS = MODELS_DIR / "road_damage_best.pt"

# Road damage boxes are small and often thin (a hairline crack can be 3-4 px
# wide). The generic 2 px floor used for vehicles would be too aggressive at
# high resolution, so it is a parameter rather than a constant.
MIN_BOX_SIZE_PX = 1.0


def clip_and_filter_boxes(
    boxes: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
    min_size_px: float = MIN_BOX_SIZE_PX,
) -> list[tuple[int, list[float]]]:
    """Clip boxes into the frame and drop degenerate ones.

    The shared validator requires 0 <= x1 < x2 <= width (same for y). YOLO
    occasionally returns coordinates a pixel or two outside the image, so this
    must run before anything leaves the module.

    Returns (original_index, clipped_bbox) pairs.
    """
    kept: list[tuple[int, list[float]]] = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        x1 = max(0.0, min(x1, float(frame_width)))
        y1 = max(0.0, min(y1, float(frame_height)))
        x2 = max(0.0, min(x2, float(frame_width)))
        y2 = max(0.0, min(y2, float(frame_height)))
        if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
            continue
        kept.append((index, [x1, y1, x2, y2]))
    return kept


def _coerce_config(config: InferenceConfig | dict | None) -> InferenceConfig:
    """Accept an InferenceConfig, a plain dict, or None (shared defaults)."""
    if config is None:
        return InferenceConfig()
    if isinstance(config, InferenceConfig):
        return config
    if isinstance(config, dict):
        known = {
            key: config[key]
            for key in ("confidence_threshold", "iou_threshold", "image_size", "device")
            if key in config
        }
        return InferenceConfig(**known)
    raise TypeError(f"Unsupported config type: {type(config)!r}")


class RoadDamageModel:
    """YOLO road-damage detector, normalized to the shared ontology.

    Parameters
    ----------
    model_path:
        Trained weights. ``None`` falls back to ``models/road_damage_best.pt``.
    config:
        ``InferenceConfig``, a plain dict, or ``None`` for shared defaults.
    detector:
        Optional injected backend exposing ``detect(frame) -> list[dict]``.
        Used by the tests and by anyone wanting to swap the detector without
        touching the rest of the pipeline. When given, ``model_path`` is
        ignored and no weights are loaded.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: InferenceConfig | dict | None = None,
        detector=None,
    ) -> None:
        self.config = _coerce_config(config)
        self.device = resolve_device(self.config.device)
        self._injected = detector
        self._model = None
        self.model_class_names: dict[int, str] = {}

        if detector is None:
            self.model_path = str(model_path) if model_path is not None else str(DEFAULT_WEIGHTS)
            self._load()
        else:
            self.model_path = None

    # -- loading --------------------------------------------------------------

    def _load(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Road damage weights not found: {self.model_path}. Train them with "
                f"`python -m modules.road_damage.train` or place the exported "
                f"road_damage_best.pt in models/."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required to load road damage weights. Install it "
                "with `pip install ultralytics`, or inject a backend via "
                "RoadDamageModel(detector=...)."
            ) from exc

        self._model = YOLO(self.model_path)
        names = getattr(self._model, "names", {}) or {}
        self.model_class_names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
        self._warn_on_unexpected_class_names()

    def _warn_on_unexpected_class_names(self) -> None:
        """Flag weights whose class list is not the project ontology.

        A model trained on raw RDD codes still *works* here (the names are
        normalized on the way out), but the ordering assumption in
        ``class_mapping.PROJECT_CLASSES`` may not hold, which would silently
        mislabel everything. Better to say so at load time.
        """
        names = set(self.model_class_names.values())
        if not names:
            return
        if not names <= set(PROJECT_CLASSES):
            import warnings

            warnings.warn(
                f"Loaded weights expose class names {sorted(names)}, which are not "
                f"the project ontology {list(PROJECT_CLASSES)}. Names will be "
                f"normalized via class_mapping, but verify the model was trained "
                f"on the prepared dataset.",
                RuntimeWarning,
                stacklevel=3,
            )

    @property
    def model(self):
        return self._model

    # -- inference ------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return shared-schema detections for one BGR frame (no wrapper dict).

        Exposed separately from ``predict`` so the detections can be fed
        straight into a merge without unwrapping, and so this class satisfies
        the same ``detect(frame)`` backend protocol the other modules use.
        """
        if frame is None or not hasattr(frame, "shape") or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 numpy array in BGR order")

        if self._injected is not None:
            return self._injected.detect(frame)

        height, width = frame.shape[:2]

        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Ultralytics reverses its own letterbox internally, so xyxy is
        # already in ORIGINAL frame coordinates - which is what the shared
        # schema requires. The module must not resize the frame itself.
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[dict] = []
        for index, bbox in clip_and_filter_boxes(xyxy, width, height):
            class_id = int(class_ids[index])
            raw_name = self.model_class_names.get(class_id, ID_TO_CLASS.get(class_id))
            normalized = normalize_class(raw_name)
            if normalized is None:
                continue  # excluded/unknown label: drop, never pass through
            detections.append(
                {
                    "class": normalized,
                    "confidence": float(round(float(confidences[index]), 4)),
                    "bbox": [float(v) for v in bbox],
                    "source": SOURCE,
                    "class_id": class_id,
                }
            )
        return detections

    def predict(self, frame: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> dict:
        """Run detection and return a full shared-schema FrameResult dict."""
        detections = self.detect(frame)
        result = empty_frame_result(frame_id=int(frame_id), timestamp=float(timestamp))
        result["detections"] = detections
        # road_damage_count is this module's own finding, so it is safe to
        # fill. Person 4 recomputes it across all modules at merge time.
        result["analytics"]["road_damage_count"] = len(detections)
        return result

    __call__ = predict

    def predict_batch(self, frames: Sequence[np.ndarray]) -> list[dict]:
        """Convenience for scoring a list of frames/images."""
        return [self.predict(frame, frame_id=index) for index, frame in enumerate(frames)]


class ScriptedDamageDetector:
    """Replays fixed detections per frame. A test/demo double, not a model.

    Lets the team run and integrate the whole pipeline (and build the
    dashboard) before weights exist, and keeps this module's tests
    deterministic and torch-free.
    """

    def __init__(self, script: Sequence[Sequence[dict]], loop: bool = False) -> None:
        self.script = [list(frame_dets) for frame_dets in script]
        self.loop = loop
        self._frame_index = 0

    def reset(self) -> None:
        self._frame_index = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self._frame_index >= len(self.script):
            if not self.loop or not self.script:
                return []
            self._frame_index = 0

        raw = self.script[self._frame_index]
        self._frame_index += 1

        height, width = frame.shape[:2]
        detections: list[dict] = []
        for item in raw:
            # Same contract as the real detector: unmappable labels are
            # dropped rather than passed through.
            normalized = normalize_class(item["class"])
            if normalized is None:
                continue
            clipped = clip_and_filter_boxes([item["bbox"]], width, height)
            if not clipped:
                continue
            detections.append(
                {
                    "class": normalized,
                    "confidence": float(item.get("confidence", 0.9)),
                    "bbox": clipped[0][1],
                    "source": SOURCE,
                    "class_id": int(item.get("class_id", 0)),
                }
            )
        return detections
