"""
modules/road_segmentation/road_segmentation_model.py

Person 2 — Road Segmentation.

    from modules.road_segmentation import RoadSegmentationModel

    model = RoadSegmentationModel("models/road_segmentation_best.pt")
    result = model.predict(frame)   # -> {"segmentations": [...]}

`predict()` only returns the keys this module owns (`segmentations`);
callers that need a full shared.schemas FrameResult should start from
`shared.schemas.empty_frame_result()` and merge this in, exactly as
described in shared/README.md's "FINAL SYSTEM API".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from shared.config import InferenceConfig, SOURCE_ROAD_SEGMENTATION, resolve_device

from .class_mapping import normalize_class

# Which raw dataset each class in a mixed-training run came from. If you
# train road_segmentation_best.pt on both datasets merged (see README for
# the recommended two-stage approach), pass the model's own class-name list
# through here at postprocess time.
_DEFAULT_DATASET_FOR_UNKNOWN_LABEL = "roboflow_road_defect"


# -----------------------------------------------------------------------------
# Pure, unit-testable postprocessing — no ultralytics/torch dependency here,
# so tests/test_road_segmentation.py can exercise this without real weights.
# -----------------------------------------------------------------------------
def polygon_to_mask_payload(
    polygon_xy: np.ndarray, frame_shape: Sequence[int]
) -> Dict[str, Any]:
    """Convert one YOLO-seg polygon (already in original-frame pixel coords,
    as returned by ultralytics `result.masks.xy`) into the standardized
    JSON-compatible mask payload plus area_px / area_ratio.

    We store the mask as a polygon rather than a dense bitmap: it's
    lightweight, JSON-serializable, and sufficient for the dashboard to
    redraw an overlay with cv2.fillPoly. Modules needing a dense mask can
    rasterize this polygon themselves.
    """
    import cv2  # local import: keep module importable without opencv for pure unit tests

    h, w = frame_shape[0], frame_shape[1]
    pts = np.asarray(polygon_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        area_px = 0.0
    else:
        area_px = float(abs(cv2.contourArea(pts)))
    image_area = float(h * w) if h and w else 0.0
    area_ratio = (area_px / image_area) if image_area > 0 else 0.0

    mask_payload = {
        "format": "polygon",
        "points": np.round(pts, 2).tolist(),
    }
    return {
        "mask": mask_payload,
        "area_px": int(round(area_px)),
        "area_ratio": min(max(area_ratio, 0.0), 1.0),
    }


def build_segmentation_output(
    raw_class_names: List[str],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    polygons_xy: Sequence[np.ndarray],
    frame_shape: Sequence[int],
    dataset: str,
    confidence_threshold: float,
) -> Dict[str, Any]:
    """Turn raw per-instance YOLO-seg outputs into the standardized
    `{"segmentations": [...]}` dict. Skips any instance whose raw class
    doesn't have a safe mapping (see class_mapping.normalize_class),
    exactly per spec section 2's "do not force the mapping" rule.
    """
    segmentations = []
    for cls_id, conf, poly in zip(class_ids, confidences, polygons_xy):
        if conf < confidence_threshold:
            continue
        raw_name = raw_class_names[int(cls_id)]
        normalized = normalize_class(raw_name, dataset)
        if normalized is None:
            # Unmapped / intentionally excluded raw class — drop, don't guess.
            continue

        geom = polygon_to_mask_payload(poly, frame_shape)
        segmentations.append(
            {
                "class": normalized,
                "confidence": float(conf),
                "mask": geom["mask"],
                "area_px": geom["area_px"],
                "area_ratio": geom["area_ratio"],
                "source": SOURCE_ROAD_SEGMENTATION,
            }
        )
    return {"segmentations": segmentations}


class RoadSegmentationModel:
    """YOLOv8-seg-backed road surface / defect segmentation model.

    Parameters
    ----------
    model_path : str
        Path to trained weights, e.g. "models/road_segmentation_best.pt".
    config : InferenceConfig | dict | None
        Overrides for confidence_threshold / image_size / device. Falls
        back to shared/config.py defaults when omitted.
    dataset : str
        Which class_mapping table to use to normalize this model's own
        class names ("pothole_seg" or "roboflow_road_defect"). Set this to
        match whichever dataset(s) `model_path` was trained on. If you
        trained on a merged/relabeled dataset, extend class_mapping.py with
        a new key rather than hacking this default.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[InferenceConfig | dict] = None,
        dataset: str = _DEFAULT_DATASET_FOR_UNKNOWN_LABEL,
    ):
        self.model_path = Path(model_path)
        self.config = self._coerce_config(config)
        self.dataset = dataset
        self._model = None  # lazy load

    @staticmethod
    def _coerce_config(config) -> InferenceConfig:
        if config is None:
            return InferenceConfig()
        if isinstance(config, InferenceConfig):
            return config
        if isinstance(config, dict):
            fields = {k: v for k, v in config.items() if k in InferenceConfig.__dataclass_fields__}
            return InferenceConfig(**fields)
        raise TypeError(f"config must be None, dict, or InferenceConfig, got {type(config)}")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Road segmentation weights not found at '{self.model_path}'. "
                f"Train the model per README.md and place the .pt file there, "
                f"or pass the correct path."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for inference (`pip install ultralytics`)."
            ) from e
        self._model = YOLO(str(self.model_path))

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Input: BGR numpy.ndarray (H, W, 3), any resolution.
        Output: {"segmentations": [ {class, confidence, mask, area_px,
        area_ratio, source}, ... ]} — only the field this module owns.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray (BGR image)")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be H x W x 3, got shape {frame.shape}")

        self._ensure_loaded()
        device = resolve_device(self.config.device)
        results = self._model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            device=device,
            verbose=False,
        )
        result = results[0]

        if result.masks is None or len(result.masks.xy) == 0:
            return {"segmentations": []}

        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        polygons = result.masks.xy  # list of (N,2) arrays, already in original-frame coords
        raw_class_names = [result.names[i] for i in range(len(result.names))]

        return build_segmentation_output(
            raw_class_names=raw_class_names,
            class_ids=class_ids,
            confidences=confidences,
            polygons_xy=polygons,
            frame_shape=frame.shape,
            dataset=self.dataset,
            confidence_threshold=self.config.confidence_threshold,
        )
