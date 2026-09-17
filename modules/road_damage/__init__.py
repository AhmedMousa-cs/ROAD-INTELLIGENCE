"""
Road Intelligence — Person 1: Road Damage Detection
====================================================

Bounding-box detection of four road-damage types:
``pothole``, ``longitudinal_crack``, ``transverse_crack``, ``alligator_crack``.

Follows the shared module interface:

    from modules.road_damage import RoadDamageModel

    model = RoadDamageModel("models/road_damage_best.pt")
    result = model.predict(frame)

``result`` is a dict shaped like ``shared.schemas.empty_frame_result()``.
This module owns the ``detections`` field only; ``segmentations`` (Person 2)
and ``tracks``/``alerts``/``analytics`` (Person 4) are left at their empty
defaults and are never deleted.

What lives where
----------------
``road_damage_model.py``  ``RoadDamageModel`` inference class
``class_mapping.py``      RDD2022 / pothole-dataset labels -> shared ontology
``prepare_dataset.py``    annotation validation, dedup, normalized splits
``train.py``              transfer-learning training + run record
``evaluate.py``           mAP50 / mAP50-95 / per-class AP / confusion matrix
``inference.py``          image / folder / video runner + CLI
``visualization.py``      optional overlays (theme is injectable)
"""

from .class_mapping import (
    CLASS_TO_ID,
    EXCLUDED_CLASSES,
    ID_TO_CLASS,
    PROJECT_CLASSES,
    RDD_TO_ONTOLOGY,
    SUPPORTED_CLASSES,
    build_id_remap,
    exclusion_reason,
    is_excluded,
    normalize_class,
)
from .inference import predict_folder, predict_image, predict_video, run_demo
from .road_damage_model import (
    RoadDamageModel,
    ScriptedDamageDetector,
    clip_and_filter_boxes,
)

__all__ = [
    # Primary interface
    "RoadDamageModel",
    "ScriptedDamageDetector",
    # Runners
    "predict_image",
    "predict_folder",
    "predict_video",
    "run_demo",
    # Class mapping
    "normalize_class",
    "is_excluded",
    "exclusion_reason",
    "build_id_remap",
    "PROJECT_CLASSES",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "RDD_TO_ONTOLOGY",
    "EXCLUDED_CLASSES",
    "SUPPORTED_CLASSES",
    # Helpers
    "clip_and_filter_boxes",
]
