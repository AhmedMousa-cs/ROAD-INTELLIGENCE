"""
Vehicle class mapping: COCO (detector-native) -> shared ontology.

Person 4 does NOT train a vehicle detector. We use pretrained COCO weights
(YOLOv8/YOLO11 from Ultralytics) and map the COCO label set to the
normalized names in ``shared/classes.yaml``.

Rule from the shared contract: a dataset/detector-specific label must never
leave this module. Anything not in COCO_TO_ONTOLOGY below is dropped, not
force-mapped.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# COCO name -> normalized ontology name (shared/classes.yaml: `vehicles`)
# -----------------------------------------------------------------------------
# COCO already uses five of our six names verbatim, so most entries are
# identity mappings. They are listed explicitly anyway so that the mapping is
# auditable and so that a custom detector with different names can be
# supported by editing one dict.
COCO_TO_ONTOLOGY: dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}

# -----------------------------------------------------------------------------
# Deliberately EXCLUDED COCO classes (documented, not force-mapped)
# -----------------------------------------------------------------------------
# Per the shared contract: "If a dataset class cannot be mapped safely, do not
# force the mapping. Document it and exclude it."
EXCLUDED_COCO_CLASSES: dict[str, str] = {
    "train": "Rail vehicle; not a road vehicle in our ontology. Excluded.",
    "boat": "Not a road vehicle. Excluded.",
    "airplane": "Not a road vehicle. Excluded.",
    "traffic light": (
        "Not a vehicle. Would only be useful for red-light violations, which "
        "this module explicitly does NOT claim to detect (no traffic-light "
        "STATE detector is available - COCO gives the box, not red/green)."
    ),
    "stop sign": "Not a vehicle; no downstream consumer in the current spec.",
    "fire hydrant": "Not a vehicle; road-side furniture, not an obstacle class.",
    "parking meter": "Not a vehicle.",
    "bench": "Not a vehicle.",
}

# -----------------------------------------------------------------------------
# Known limitations of the COCO ontology for this project (see README)
# -----------------------------------------------------------------------------
MAPPING_CAVEATS: dict[str, str] = {
    "car": (
        "COCO 'car' covers sedans, hatchbacks, SUVs, vans and most pickups. "
        "There is no separate van/pickup class in the shared ontology either, "
        "so this is a lossless mapping for our purposes."
    ),
    "truck": (
        "COCO 'truck' ranges from small delivery vans to articulated lorries. "
        "car/truck confusion on large SUVs and vans is a known COCO weakness; "
        "the tracker's majority-vote class stabilisation mitigates (but does "
        "not eliminate) per-frame label flicker."
    ),
    "person": (
        "'person' is in the shared `vehicles` group but is NOT counted as a "
        "vehicle by the analytics layer - see TrafficAnalytics.VEHICLE_CLASSES. "
        "Pedestrians are reported separately as `person_count`."
    ),
    "motorcycle": (
        "Rider + motorcycle are two separate COCO detections (one 'person', "
        "one 'motorcycle'). We do not merge them; counts are therefore "
        "per-object, not per-road-user."
    ),
}

# Classes that the analytics layer treats as *vehicles* for counting and
# traffic-density purposes. 'person' is intentionally not in this set.
VEHICLE_CLASSES: frozenset[str] = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle"}
)

# Every normalized class this module is allowed to emit.
SUPPORTED_CLASSES: frozenset[str] = frozenset(COCO_TO_ONTOLOGY.values())


def normalize_class(detector_class_name: str) -> str | None:
    """Map a detector-native class name to the shared ontology.

    Returns None when the class is not part of this module's responsibility,
    in which case the caller must DROP the detection (never pass the raw
    label through).
    """
    if detector_class_name is None:
        return None
    return COCO_TO_ONTOLOGY.get(str(detector_class_name).strip().lower())


def allowed_model_class_ids(model_names: dict[int, str] | list[str]) -> list[int]:
    """Return the detector's own class ids that map into our ontology.

    Passed to YOLO's ``classes=`` argument so the detector filters internally
    (faster than post-filtering, and keeps NMS from being polluted by
    irrelevant categories).

    Works with either the ``{id: name}`` dict Ultralytics exposes as
    ``model.names`` or a plain list of names.
    """
    if isinstance(model_names, dict):
        items = model_names.items()
    else:
        items = enumerate(model_names)
    return sorted(
        class_id for class_id, name in items if normalize_class(name) is not None
    )
