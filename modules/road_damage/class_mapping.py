"""
Road damage class mapping: dataset-native labels -> shared ontology.

Two source datasets feed this module:

  * RDD2022 (primary) uses damage codes: D00, D10, D20, D40, ...
  * A Roboflow pothole dataset (supplementary) uses "potholes"/"pothole".

The shared contract is explicit: a dataset-specific label must NEVER leave
this module. ``D00`` becomes ``longitudinal_crack`` or it is dropped. Codes
that cannot be mapped safely are listed in ``EXCLUDED_CLASSES`` with a reason
rather than being force-fitted into the four project classes.

This mapping is applied twice, at two different stages:

  1. ``prepare_dataset.py`` — offline, when building the normalized training
     dataset (dataset class id -> project class id).
  2. ``road_damage_model.py`` — at inference, as a second line of defence, so
     that even a model accidentally trained on raw RDD ids cannot leak one.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# The four project classes, in the fixed order used by data.yaml and by the
# trained model's class ids. Changing this order invalidates trained weights.
# -----------------------------------------------------------------------------
PROJECT_CLASSES: tuple[str, ...] = (
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
)

CLASS_TO_ID: dict[str, int] = {name: index for index, name in enumerate(PROJECT_CLASSES)}
ID_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_TO_ID.items()}

# -----------------------------------------------------------------------------
# RDD2022 damage codes -> project classes
# -----------------------------------------------------------------------------
# The four codes required by the spec. RDD2022's own documentation describes
# D00/D10 as longitudinal/transverse *linear* cracks, D20 as alligator
# (fatigue) cracking, and D40 as pothole/rutting damage.
RDD_TO_ONTOLOGY: dict[str, str] = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}

# Roboflow / supplementary pothole datasets. Case and pluralisation vary
# between exports, so several spellings map to the single pothole class.
# Per the spec: do NOT create a second pothole class.
POTHOLE_DATASET_TO_ONTOLOGY: dict[str, str] = {
    "pothole": "pothole",
    "potholes": "pothole",
    "pot-hole": "pothole",
    "pot_hole": "pothole",
}
# Some RDD2022 re-uploads (e.g. the Kaggle "RDD2022 YOLO" republish) ship
# human-readable names instead of the raw D00/D10/D20/D40 codes. These are
# the SAME four classes, just spelled out — verified against that dataset's
# own data.yaml comment block, which states:
#   0: Longitudinal (D00)  1: Transverse (D10)
#   2: Alligator    (D20)  3: Pothole    (D40)
RDD_READABLE_NAME_TO_ONTOLOGY: dict[str, str] = {
    "longitudinal": "longitudinal_crack",
    "transverse": "transverse_crack",
    "alligator": "alligator_crack",
    # "pothole" already covered by POTHOLE_DATASET_TO_ONTOLOGY above.
}
# -----------------------------------------------------------------------------
# Deliberately EXCLUDED source labels (documented, never force-mapped)
# -----------------------------------------------------------------------------
EXCLUDED_CLASSES: dict[str, str] = {
    "D01": (
        "Construction-joint / lane-edge linear crack in some RDD releases. "
        "Overlaps D00 but is not identical; merging it into "
        "longitudinal_crack would blur a distinction the source data makes. "
        "Excluded pending team agreement."
    ),
    "D11": "Transverse construction-joint variant; same reasoning as D01.",
    "D43": (
        "Crosswalk blur — a road-MARKING defect, not structural damage. "
        "There is no marking class in shared/classes.yaml."
    ),
    "D44": "White-line blur — road marking wear, not structural damage.",
    "D50": (
        "Utility/manhole-related in several RDD releases. Not one of the four "
        "project damage types; would pollute the pothole class."
    ),
    "D0w0": "Repair/patch marker in some RDD variants. 'road_patch' belongs to "
    "the SEGMENTATION ontology (Person 2), not to this detector.",
    "repair": "Same as D0w0 — road_patch is Person 2's class, emitted as a mask.",
    "block_crack": (
        "Present in some crack datasets. Visually distinct from alligator "
        "cracking; no block_crack class exists in the shared ontology."
    ),
    "hcrack": (
        "Ambiguous abbreviation ('horizontal crack'). Whether that means "
        "transverse or longitudinal depends on the capture orientation of the "
        "source dataset, which is not documented. Excluded rather than guessed."
    ),
    "vcrack": "Ambiguous abbreviation ('vertical crack'); same reasoning as hcrack.",
}

# Every source spelling this module understands.
SOURCE_TO_ONTOLOGY: dict[str, str] = {
    **RDD_TO_ONTOLOGY,
    **POTHOLE_DATASET_TO_ONTOLOGY,
    **RDD_READABLE_NAME_TO_ONTOLOGY,   # <- add this line
    **{name: name for name in PROJECT_CLASSES},
}

SUPPORTED_CLASSES: frozenset[str] = frozenset(PROJECT_CLASSES)


def normalize_class(source_name: str) -> str | None:
    """Map a dataset-native label to a shared-ontology class name.

    Returns None when the label is excluded or unknown, in which case the
    caller must DROP the annotation/detection. Never pass a raw label through.

    Lookup is case-insensitive, but RDD codes are matched in upper case so
    that "d00" and "D00" behave identically.
    """
    if source_name is None:
        return None
    raw = str(source_name).strip()
    if not raw:
        return None
    if raw.upper() in RDD_TO_ONTOLOGY:
        return RDD_TO_ONTOLOGY[raw.upper()]
    return SOURCE_TO_ONTOLOGY.get(raw.lower().replace(" ", "_"))


def is_excluded(source_name: str) -> bool:
    """True if the label is knowingly excluded (vs. simply unrecognised)."""
    if source_name is None:
        return False
    raw = str(source_name).strip()
    return raw.upper() in EXCLUDED_CLASSES or raw.lower() in EXCLUDED_CLASSES


def exclusion_reason(source_name: str) -> str | None:
    raw = str(source_name).strip()
    return EXCLUDED_CLASSES.get(raw.upper()) or EXCLUDED_CLASSES.get(raw.lower())


def build_id_remap(source_names: dict[int, str] | list[str]) -> dict[int, int]:
    """Map a source dataset's class ids onto project class ids.

    Used by ``prepare_dataset.py`` to rewrite YOLO label files. Source ids
    whose name is excluded/unknown are absent from the returned dict, and
    their annotation lines must be dropped.

    Accepts either Ultralytics' ``{id: name}`` dict or a plain ordered list.
    """
    items = source_names.items() if isinstance(source_names, dict) else enumerate(source_names)
    remap: dict[int, int] = {}
    for source_id, name in items:
        normalized = normalize_class(name)
        if normalized is not None:
            remap[int(source_id)] = CLASS_TO_ID[normalized]
    return remap


def verify_mapping_against_ontology() -> None:
    """Assert every target name exists in shared/classes.yaml.

    Called by the tests so a typo fails loudly here rather than producing an
    off-ontology class at inference time.
    """
    from shared.schemas import load_allowed_classes

    allowed = load_allowed_classes()
    unknown = sorted(set(SOURCE_TO_ONTOLOGY.values()) - allowed)
    if unknown:
        raise AssertionError(
            f"class_mapping.py targets names absent from shared/classes.yaml: {unknown}"
        )
