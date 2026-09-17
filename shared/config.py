"""
Road Intelligence — Shared Configuration
=========================================
Central location for every threshold / setting used by more than one module.

Rule: no module should hard-code a confidence threshold, IoU threshold,
image size, device string, or path convention. Import from here instead.

Values can be tuned after testing, but they must stay centralized — if a
module needs a value not listed here, add it to this file (with a comment
on why) rather than hard-coding it locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Path anchors (never hard-code absolute/project-specific paths in modules —
# derive everything from these).
# -----------------------------------------------------------------------------
SHARED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SHARED_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CLASSES_YAML_PATH = SHARED_DIR / "classes.yaml"

# -----------------------------------------------------------------------------
# Inference defaults (module-level code should accept overrides via a
# `config` argument rather than importing these directly wherever possible —
# see shared/schemas.py:InferenceConfig).
# -----------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = 0.40
IOU_THRESHOLD: float = 0.50
IMAGE_SIZE: int = 640

# "cuda" if available at runtime, else "cpu" — resolved lazily by
# `resolve_device()` below so importing this module never requires torch.
DEVICE: str = os.environ.get("ROAD_INTEL_DEVICE", "auto")

# -----------------------------------------------------------------------------
# Traffic / analytics thresholds (Person 4's module reads these — must stay
# configurable, not hard-coded inside the tracking algorithm).
# -----------------------------------------------------------------------------
TRAFFIC_DENSITY_THRESHOLDS = {
    # active tracked vehicle count -> density label
    "low_max": 5,      # 0-5 active vehicles      => "low"
    "medium_max": 15,  # 6-15 active vehicles     => "medium"
    # anything above medium_max                   => "high"
}

WRONG_WAY_CONFIDENCE_MIN: float = 0.60  # min confidence to raise a wrong_way alert

# -----------------------------------------------------------------------------
# Road Health Score weights (project-defined, not a civil-engineering
# standard — documented in shared/README.md and each module's README).
# -----------------------------------------------------------------------------
ROAD_HEALTH_WEIGHTS = {
    "pothole": 5.0,
    "longitudinal_crack": 1.5,
    "transverse_crack": 1.5,
    "alligator_crack": 3.0,
    "surface_damage_area_ratio": 20.0,  # multiplier on cumulative area_ratio
    "road_patch": 0.5,   # patches indicate prior repair, minor penalty
    "ravelling": 2.0,
    "road_debris": 1.0,
}
ROAD_HEALTH_MAX_SCORE: float = 100.0

# -----------------------------------------------------------------------------
# Standard dict keys / field names (import these instead of retyping string
# literals, to avoid silent schema drift between modules).
# -----------------------------------------------------------------------------
SOURCE_ROAD_DAMAGE = "road_damage"
SOURCE_ROAD_SEGMENTATION = "road_segmentation"
SOURCE_DEBRIS = "debris"
SOURCE_SPEED_BUMP = "speed_bump"
SOURCE_TRACKING = "tracking_analytics"


def resolve_device(preferred: str | None = None) -> str:
    """Resolve 'auto' / None to an actual torch device string.

    Falls back to 'cpu' if torch isn't importable or CUDA isn't available,
    so this function is safe to call even in a minimal environment.
    """
    choice = preferred or DEVICE
    if choice != "auto":
        return choice
    try:
        import torch  # local import: config.py must not require torch to import

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class InferenceConfig:
    """Per-model override of the shared defaults.

    Every ModuleModel.__init__(model_path, config=None) should accept an
    instance of this (or a plain dict with the same keys) and fall back to
    the module-level defaults above when config is None.
    """

    confidence_threshold: float = CONFIDENCE_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    image_size: int = IMAGE_SIZE
    device: str = field(default_factory=lambda: resolve_device())
