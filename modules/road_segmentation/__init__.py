"""
modules/road_segmentation

Person 2 — Road Segmentation module.

    from modules.road_segmentation import RoadSegmentationModel

    model = RoadSegmentationModel("models/road_segmentation_best.pt")
    result = model.predict(frame)  # {"segmentations": [...]}
"""

from .road_segmentation_model import RoadSegmentationModel, build_segmentation_output, polygon_to_mask_payload
from .class_mapping import normalize_class, verify_mapping_against_dataset

__all__ = [
    "RoadSegmentationModel",
    "build_segmentation_output",
    "polygon_to_mask_payload",
    "normalize_class",
    "verify_mapping_against_dataset",
]
