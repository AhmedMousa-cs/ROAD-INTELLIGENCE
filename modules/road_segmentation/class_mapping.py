"""
modules/road_segmentation/class_mapping.py

Maps dataset-specific segmentation class names to the shared ontology in
shared/classes.yaml. Only these normalized names may leave the module:

    pothole, surface_damage, road_patch, ravelling
    (alligator_crack only if Dataset 2 is included — see note below)

IMPORTANT — per the spec, these mappings must NOT be trusted blindly.
Before training, run `python -m modules.road_segmentation.class_mapping
--verify <path/to/data.yaml> --dataset roboflow_road_defect` (or
`pothole_seg`) to diff the dataset's actual class list against what's
mapped here, and update this file if the real dataset differs from what's
assumed below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import yaml

# -----------------------------------------------------------------------------
# Dataset 1 — "Pothole Image Segmentation Dataset" (YOLOv8-seg format)
# Published as a single-class dataset. Verify the exact label string in the
# downloaded data.yaml (`names:`) before trusting this dict — some Roboflow
# exports title-case single classes ("Pothole") instead of lowercase.
# -----------------------------------------------------------------------------
POTHOLE_SEG_MAPPING: Dict[str, str] = {
    "pothole": "pothole",
    "Pothole": "pothole",  # tolerate title-case export variants
}

# -----------------------------------------------------------------------------
# Dataset 2 — Roboflow "Road Defect Segmentation"
#
# PROVISIONAL mapping from the spec. Section 2 of the spec explicitly warns:
# "Do NOT assume these mappings blindly. Inspect data.yaml, class names,
# annotation files before finalizing." Treat every key below as a hypothesis
# to confirm, not a fact.
#
# "alligator" -> "alligator_crack" is a road_damage class, not a
# road_surface class. It's included here only because this module's mask
# output CAN represent it (a segmentation mask), but if a dataset field
# actually distinguishes alligator cracking as a *detection* concern owned
# by Person 1, exclude it here rather than double-owning a class.
# -----------------------------------------------------------------------------
ROBOFLOW_ROAD_DEFECT_MAPPING: Dict[str, str] = {
    "pothole": "pothole",
    "alligator": "alligator_crack",
    "Major Surface Damage": "surface_damage",
    "Minor Surface Damage": "surface_damage",
    "Road Patch": "road_patch",
    "Ravelling": "ravelling",
}

# Classes we have explicitly decided NOT to map (ambiguous, or not part of
# the project's road-surface ontology). Document every excluded raw class
# here instead of silently dropping it, per spec section 2 ("If a dataset
# class cannot be mapped safely, do not force the mapping. Document it and
# exclude it.").
EXCLUDED_CLASSES: Set[str] = set()

_ALL_MAPPINGS = {
    "pothole_seg": POTHOLE_SEG_MAPPING,
    "roboflow_road_defect": ROBOFLOW_ROAD_DEFECT_MAPPING,
}


def normalize_class(raw_class: str, dataset: str) -> Optional[str]:
    """Map one dataset-specific class name to a shared ontology name.

    Returns None (and the caller should skip/exclude the annotation) if the
    raw class isn't in the mapping for that dataset — this is intentional:
    an unmapped class must never silently leak through as-is.
    """
    if dataset not in _ALL_MAPPINGS:
        raise ValueError(f"Unknown dataset key '{dataset}'. Known: {list(_ALL_MAPPINGS)}")
    if raw_class in EXCLUDED_CLASSES:
        return None
    return _ALL_MAPPINGS[dataset].get(raw_class)


def verify_mapping_against_dataset(data_yaml_path: str, dataset: str) -> None:
    """Load a real data.yaml's `names` list and report any class that is
    NOT covered by the mapping above (or vice versa). Run this before
    trusting the mapping for training — do not skip this step.
    """
    path = Path(data_yaml_path)
    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    names = spec.get("names")
    if isinstance(names, dict):
        raw_classes = list(names.values())
    elif isinstance(names, list):
        raw_classes = names
    else:
        raise ValueError(f"Could not find a `names` list/dict in {data_yaml_path}")

    mapping = _ALL_MAPPINGS[dataset]
    unmapped = [c for c in raw_classes if c not in mapping and c not in EXCLUDED_CLASSES]
    unused = [c for c in mapping if c not in raw_classes]

    print(f"Dataset '{dataset}' at {data_yaml_path}")
    print(f"  Raw classes found:      {raw_classes}")
    if unmapped:
        print(f"  ⚠ UNMAPPED (not in class_mapping.py, not excluded): {unmapped}")
        print("    -> Add to the mapping dict or EXCLUDED_CLASSES before training.")
    else:
        print("  ✓ Every raw class is either mapped or explicitly excluded.")
    if unused:
        print(f"  ⚠ Mapping entries not found in this dataset's names: {unused}")
        print("    -> Possibly stale; confirm this dataset version still has them.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, help="Path to the dataset's data.yaml")
    parser.add_argument("--dataset", required=True, choices=list(_ALL_MAPPINGS))
    args = parser.parse_args()
    verify_mapping_against_dataset(args.verify, args.dataset)
