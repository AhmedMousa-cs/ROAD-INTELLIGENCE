"""
modules/obstacles/class_mapping.py
===================================
Dataset-specific -> normalized (shared/classes.yaml) class mapping for
Person 3's two detectors (road debris, speed bump).

Design principle (per the project spec, section 2 + Person 3 Part A):
    "If a dataset class cannot be mapped safely, do not force the mapping.
     Document it and exclude it."

So this module is deliberately **fail-closed**, not fail-open:
  - `map_taco_category()` / `map_speed_bump_category()` return `None` for
    anything not explicitly listed, and the caller (debris_model.py /
    speed_bump_model.py) drops the detection and logs it rather than
    guessing.
  - The TACO table below is a *starter* mapping only. TACO's exact category
    list/IDs depend on which `annotations.json` release you download — the
    spec explicitly says "inspect the actual downloaded dataset ... before
    training". Call `verify_taco_categories()` against your real
    annotations.json during dataset prep; it tells you exactly which
    categories are unmapped so you can extend the table deliberately
    instead of silently dropping them.

Normalized target classes used here (see shared/classes.yaml):
    road_obstacle: road_debris
    road_feature:  speed_bump
"""

from __future__ import annotations

from shared.config import SOURCE_DEBRIS, SOURCE_SPEED_BUMP  # noqa: F401  (re-exported for convenience)

ROAD_DEBRIS = "road_debris"
SPEED_BUMP = "speed_bump"

# -----------------------------------------------------------------------------
# TACO (Trash Annotations in Context) -> road_debris
# -----------------------------------------------------------------------------
# TACO ships ~60 leaf categories under a handful of supercategories. The
# project only needs ONE final class (road_debris) plus an *optional*,
# non-authoritative `subclass` metadata field for downstream UI/analytics.
#
# This table is a best-effort starting point based on TACO's published
# category list. TREAT IT AS A DRAFT: before training, run
# `verify_taco_categories(your_loaded_categories)` against the categories
# actually present in the annotations.json you downloaded, and extend/trim
# this table to match. Do not assume it is complete or exact.
#
# Categories deliberately left OUT of this table (and therefore excluded,
# not force-mapped) are ones that are ambiguous for a *road-obstacle* use
# case rather than general litter classification (e.g. "Battery", which is
# small/rare debris but also hazardous waste with different handling
# implications than "an object a vehicle might need to avoid"). Extend this
# table only after inspecting real annotated examples.
TACO_TO_ROAD_DEBRIS: dict[str, str] = {
    # -- plastic bottles / containers -----------------------------------
    "Clear plastic bottle": "plastic_bottle",
    "Other plastic bottle": "plastic_bottle",
    "Plastic bottle cap": "plastic_bottle",
    "Disposable plastic cup": "plastic_bottle",
    "Other plastic cup": "plastic_bottle",
    "Foam cup": "plastic_bottle",
    "Tupperware": "plastic_bottle",
    "Disposable food container": "plastic_bottle",
    "Foam food container": "plastic_bottle",
    "Other plastic container": "plastic_bottle",
    "Spread tub": "plastic_bottle",
    # -- cans -------------------------------------------------------------
    "Drink can": "can",
    "Food Can": "can",
    "Metal bottle cap": "can",
    "Aerosol": "can",
    "Scrap metal": "can",
    # -- glass --------------------------------------------------------------
    "Glass bottle": "glass",
    "Glass jar": "glass",
    "Glass cup": "glass",
    "Broken glass": "glass",
    # -- paper / cardboard -------------------------------------------------
    "Normal paper": "paper",
    "Paper cup": "paper",
    "Paper bag": "paper",
    "Plastified paper bag": "paper",
    "Corrugated carton": "paper",
    "Other carton": "paper",
    "Egg carton": "paper",
    "Drink carton": "paper",
    "Meal carton": "paper",
    "Pizza box": "paper",
    "Magazine paper": "paper",
    "Tissues": "paper",
    "Wrapping paper": "paper",
    "Toilet tube": "paper",
    # -- food waste -----------------------------------------------------
    "Food waste": "food_waste",
    # -- generic plastic film / bags / wrappers -----------------------
    "Plastic film": "plastic_wrapper",
    "Garbage bag": "plastic_wrapper",
    "Other plastic wrapper": "plastic_wrapper",
    "Single-use carrier bag": "plastic_wrapper",
    "Polypropylene bag": "plastic_wrapper",
    "Crisp packet": "plastic_wrapper",
    "Six pack rings": "plastic_wrapper",
    "Plastic lid": "plastic_wrapper",
    "Metal lid": "plastic_wrapper",
    "Other plastic": "plastic_wrapper",
    "Styrofoam piece": "plastic_wrapper",
    # -- misc road-relevant solid objects --------------------------------
    "Shoe": "misc_debris",
    "Rope & strings": "misc_debris",
    "Plastic straw": "misc_debris",
    "Paper straw": "misc_debris",
    "Squeezable tube": "misc_debris",
    "Pop tab": "misc_debris",
    "Unlabeled litter": "misc_debris",
    # "Cigarette", "Battery", "Plastic glooves", "Plastic utensils",
    # "Aluminium foil", "Aluminium blister pack", "Carded blister pack" are
    # intentionally NOT mapped here: too small / ambiguous as a driving
    # hazard to include by default. Verify against real annotations and
    # add explicitly if your team decides they belong in road_debris.
}


def map_taco_category(taco_category_name: str) -> tuple[str | None, str | None]:
    """Map one TACO category name to (normalized_class, subclass).

    Returns (None, None) if the category is not in TACO_TO_ROAD_DEBRIS —
    callers must exclude the detection rather than force a mapping.
    """
    subclass = TACO_TO_ROAD_DEBRIS.get(taco_category_name)
    if subclass is None:
        return None, None
    return ROAD_DEBRIS, subclass


def verify_taco_categories(actual_categories: list[str]) -> dict:
    """Diff the categories in a real, downloaded annotations.json against
    this table. Call this during dataset prep, NOT at inference time.

    Returns a dict with:
        mapped:    categories present in both the file and this table
        unmapped:  categories present in the file but NOT in this table
                   (these will be silently excluded from training unless
                   you add them above -- inspect them before deciding)
        stale:     categories in this table but absent from the file
                   (likely a different TACO release/category set)
    """
    actual = set(actual_categories)
    known = set(TACO_TO_ROAD_DEBRIS)
    return {
        "mapped": sorted(actual & known),
        "unmapped": sorted(actual - known),
        "stale": sorted(known - actual),
    }


# -----------------------------------------------------------------------------
# Speed Bump Detection v10 (Roboflow) -> speed_bump
# -----------------------------------------------------------------------------
# Per spec: "Use its actual YAML class definitions. Do not assume additional
# classes exist." This map is intentionally a passthrough for the single
# expected label, plus a couple of common raw-label variants seen across
# Roboflow speed-bump dataset versions. Always cross-check against the
# `names:` list in the data.yaml you actually download (see
# training/speed_bump_data.yaml.template) before training — if it contains
# names beyond a speed-bump concept, do NOT map them here; exclude and
# document instead.
SPEED_BUMP_RAW_TO_NORMALIZED: dict[str, str] = {
    "speed_bump": SPEED_BUMP,
    "speed bump": SPEED_BUMP,
    "speedbump": SPEED_BUMP,
    "Speed Bump": SPEED_BUMP,
    "speed-bump": SPEED_BUMP,
}


def map_speed_bump_category(raw_name: str) -> str | None:
    """Map one raw Speed Bump Detection v10 label to 'speed_bump'.

    Returns None (exclude) for anything not recognized -- do not assume the
    dataset only ever contains the expected label.
    """
    return SPEED_BUMP_RAW_TO_NORMALIZED.get(raw_name)
