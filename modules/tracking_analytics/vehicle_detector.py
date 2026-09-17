"""
Vehicle detection using pretrained COCO YOLO weights.

Per the spec, this module does NOT train a vehicle detector: COCO already
covers car / truck / bus / motorcycle / bicycle / person well, and training a
new one would spend GPU time for no gain. We load pretrained Ultralytics
weights, let the model filter to the class ids we care about, and map the
COCO names onto the shared ontology (see ``class_mapping.py``).

The detector is pluggable: anything implementing ``detect(frame) ->
list[detection dict]`` can be injected into ``TrackingAnalytics``. That keeps
the tracking/analytics layer testable without torch, and lets the team swap in
a custom-trained detector later without touching the analytics code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from shared.config import MODELS_DIR, InferenceConfig, resolve_device

from .class_mapping import allowed_model_class_ids, normalize_class

SOURCE = "tracking_analytics"

# Default weights. A bare filename like "yolov8n.pt" is resolved (and
# downloaded once) by Ultralytics itself; the repo-local path is preferred so
# a clean checkout with the weights committed to models/ works offline.
DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_LOCAL_WEIGHTS = MODELS_DIR / "vehicle_detection_yolov8n.pt"


@runtime_checkable
class DetectionBackend(Protocol):
    """Minimal interface the tracking layer needs from a detector."""

    def detect(self, frame: np.ndarray) -> list[dict]:
        ...


def clip_and_filter_boxes(
    boxes: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
    min_size_px: float = 2.0,
) -> list[tuple[int, list[float]]]:
    """Clip boxes into the frame; drop degenerate ones.

    Returns (original_index, clipped_bbox) pairs. The shared validator
    requires 0 <= x1 < x2 <= width (same for y), so this must run before
    anything leaves the module - YOLO occasionally returns boxes a pixel or
    two outside the image.
    """
    kept: list[tuple[int, list[float]]] = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in box)
        x1 = max(0.0, min(x1, frame_width))
        y1 = max(0.0, min(y1, frame_height))
        x2 = max(0.0, min(x2, frame_width))
        y2 = max(0.0, min(y2, frame_height))
        if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
            continue
        kept.append((index, [x1, y1, x2, y2]))
    return kept


class VehicleDetector:
    """Pretrained-COCO YOLO detector, normalized to the shared ontology."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: InferenceConfig | dict | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self.model_path = self._resolve_weights(model_path)
        self._model = None
        self._class_ids: list[int] | None = None
        self._load()

    # -- loading --------------------------------------------------------------

    @staticmethod
    def _resolve_weights(model_path: str | Path | None) -> str:
        if model_path is not None:
            return str(model_path)
        if DEFAULT_LOCAL_WEIGHTS.exists():
            return str(DEFAULT_LOCAL_WEIGHTS)
        return DEFAULT_WEIGHTS

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for VehicleDetector. Install it with "
                "`pip install ultralytics`, or inject your own detector via "
                "TrackingAnalytics(detector=...)."
            ) from exc

        path = Path(self.model_path)
        if path.suffix and not path.exists() and path.parent != Path("."):
            raise FileNotFoundError(
                f"Vehicle detector weights not found: {self.model_path}. "
                f"Either place the weights there or pass a bare Ultralytics "
                f"model name (e.g. 'yolov8n.pt') to download them."
            )

        self._model = YOLO(self.model_path)
        self.device = resolve_device(self.config.device)
        names = getattr(self._model, "names", {}) or {}
        self._class_ids = allowed_model_class_ids(names) or None
        self.model_class_names = names

    @property
    def model(self):
        return self._model

    # -- inference ------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on one BGR frame; return shared-schema detections."""
        if frame is None or not hasattr(frame, "shape") or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 numpy array (BGR)")
        height, width = frame.shape[:2]

        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.device,
            classes=self._class_ids,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Ultralytics returns xyxy in ORIGINAL frame coordinates already (it
        # reverses the letterbox internally), which is what the shared schema
        # requires.
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[dict] = []
        for index, bbox in clip_and_filter_boxes(xyxy, width, height):
            raw_name = self.model_class_names.get(int(class_ids[index]))
            normalized = normalize_class(raw_name)
            if normalized is None:
                continue  # never emit a detector-native label
            detections.append(
                {
                    "class": normalized,
                    "confidence": float(round(float(confidences[index]), 4)),
                    "bbox": [float(v) for v in bbox],
                    "source": SOURCE,
                    "class_id": int(class_ids[index]),
                }
            )
        return detections


class ScriptedVehicleDetector:
    """A detector that replays a fixed list of detections per frame.

    Not a model - a test/demo double. It lets the team run the whole tracking,
    analytics, wrong-way, speed and road-health pipeline (and wire up the
    Streamlit dashboard) before any weights exist, and it makes the module's
    tests deterministic and torch-free.

    ``script`` is a list (one entry per frame) of lists of partial detection
    dicts; missing ``source``/``confidence`` are filled in.
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

        height, width = (frame.shape[:2] if frame is not None else (10_000, 10_000))
        detections: list[dict] = []
        for item in raw:
            # Same contract as the real detector: an unmappable label is
            # dropped, never passed through.
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


def _coerce_config(config: InferenceConfig | dict | None) -> InferenceConfig:
    """Accept an InferenceConfig, a plain dict, or None (shared defaults)."""
    if config is None:
        return InferenceConfig()
    if isinstance(config, InferenceConfig):
        return config
    if isinstance(config, dict):
        known = {
            key: config[key]
            for key in (
                "confidence_threshold",
                "iou_threshold",
                "image_size",
                "device",
            )
            if key in config
        }
        return InferenceConfig(**known)
    raise TypeError(f"Unsupported config type: {type(config)!r}")
