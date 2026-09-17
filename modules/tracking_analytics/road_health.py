"""
Road Health Score - a PROJECT-DEFINED road condition indicator.

NOT a civil-engineering certification. It is not PCI, not IRI, not an
official road safety rating, and it has not been validated against any
standard. It is a weighted, configurable summary of what the detection and
segmentation modules found, designed to give the dashboard one number that
moves in the right direction. Any UI that shows it must say so.

Formula
-------
    penalty = count_penalty + area_penalty

    count_penalty = SUM over detections of W[class]
        (bbox-based findings from Person 1's road_damage module and Person 3's
         obstacle module: potholes, cracks, debris. `speed_bump` is a road
         FEATURE, not damage, and carries no penalty.)

    area_penalty  = AREA_WEIGHT * SUM over segmentations of
                        area_ratio * severity[class]
        (mask-based findings from Person 2's segmentation module. Using
         area_ratio rather than a count is what makes a road with one large
         damaged patch score worse than one with a small one.)

    score = clamp(MAX_SCORE - penalty, 0, MAX_SCORE)

A pothole that is both *detected* (Person 1) and *segmented* (Person 2)
contributes to both terms. That is deliberate: count captures "how many
discrete defects", area captures "how much of the surface is affected", and
a defect that scores on both dimensions genuinely is worse.

Weights come from ``shared/config.py:ROAD_HEALTH_WEIGHTS`` so they stay
centralized and tunable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.config import ROAD_HEALTH_MAX_SCORE, ROAD_HEALTH_WEIGHTS

# Class groupings, aligned with shared/classes.yaml.
DAMAGE_CLASSES = frozenset(
    {"pothole", "longitudinal_crack", "transverse_crack", "alligator_crack"}
)
SURFACE_CLASSES = frozenset({"surface_damage", "road_patch", "ravelling"})
OBSTACLE_CLASSES = frozenset({"road_debris"})
FEATURE_CLASSES = frozenset({"speed_bump"})

# Grade bands (project-defined).
GRADE_BANDS: tuple[tuple[float, str, str], ...] = (
    (85.0, "A", "Good"),
    (70.0, "B", "Fair"),
    (55.0, "C", "Poor"),
    (35.0, "D", "Very poor"),
    (0.0, "E", "Critical"),
)


@dataclass
class RoadHealthConfig:
    count_weights: dict = field(default_factory=lambda: dict(ROAD_HEALTH_WEIGHTS))
    max_score: float = ROAD_HEALTH_MAX_SCORE
    # Multiplier applied to summed area_ratio; keyed in the shared weights as
    # "surface_damage_area_ratio".
    area_weight: float = float(
        ROAD_HEALTH_WEIGHTS.get("surface_damage_area_ratio", 20.0)
    )
    # Relative severity per segmented class (a patched road is repaired, so it
    # is penalised far less than raw surface damage).
    area_severity: dict = field(
        default_factory=lambda: {
            "pothole": 1.0,
            "surface_damage": 1.0,
            "ravelling": 0.6,
            "road_patch": 0.2,
        }
    )

    @classmethod
    def from_dict(cls, data: dict | None) -> "RoadHealthConfig":
        if not data:
            return cls()
        weights = dict(ROAD_HEALTH_WEIGHTS)
        weights.update(data.get("count_weights", {}))
        config = cls(
            count_weights=weights,
            max_score=float(data.get("max_score", ROAD_HEALTH_MAX_SCORE)),
            area_weight=float(
                data.get("area_weight", weights.get("surface_damage_area_ratio", 20.0))
            ),
        )
        config.area_severity.update(data.get("area_severity", {}))
        return config


def grade_for(score: float) -> tuple[str, str]:
    """Map a score to (grade_letter, label)."""
    for threshold, letter, label in GRADE_BANDS:
        if score >= threshold:
            return letter, label
    return "E", "Critical"


def summarize_conditions(
    detections: list[dict] | None, segmentations: list[dict] | None = None
) -> dict:
    """Count findings by ontology group.

    Source of the ``road_damage_count`` / ``obstacle_count`` /
    ``speed_bump_count`` fields of the shared analytics block. These are
    computed from the OTHER modules' outputs at merge time - this module never
    guesses them.
    """
    detections = detections or []
    segmentations = segmentations or []

    damage = sum(1 for d in detections if d.get("class") in DAMAGE_CLASSES)
    obstacles = sum(1 for d in detections if d.get("class") in OBSTACLE_CLASSES)
    bumps = sum(1 for d in detections if d.get("class") in FEATURE_CLASSES)
    surface_ratio = sum(
        float(s.get("area_ratio", 0.0))
        for s in segmentations
        if s.get("class") in SURFACE_CLASSES | {"pothole"}
    )

    per_class: dict[str, int] = {}
    for det in detections:
        name = det.get("class")
        if name in DAMAGE_CLASSES | OBSTACLE_CLASSES | FEATURE_CLASSES:
            per_class[name] = per_class.get(name, 0) + 1

    return {
        "road_damage_count": int(damage),
        "obstacle_count": int(obstacles),
        "speed_bump_count": int(bumps),
        "damaged_surface_ratio": round(float(min(surface_ratio, 1.0)), 4),
        "condition_counts_by_class": per_class,
    }


class RoadHealthScorer:
    """Computes the per-frame score and a session (video) aggregate."""

    def __init__(self, config: RoadHealthConfig | None = None) -> None:
        self.config = config or RoadHealthConfig()
        self._frame_scores: list[float] = []

    def reset(self) -> None:
        self._frame_scores.clear()

    # -- scoring --------------------------------------------------------------

    def score_frame(
        self,
        detections: list[dict] | None,
        segmentations: list[dict] | None = None,
        record: bool = True,
    ) -> dict:
        detections = detections or []
        segmentations = segmentations or []
        cfg = self.config

        count_penalty = 0.0
        for det in detections:
            name = det.get("class")
            if name in FEATURE_CLASSES:
                continue  # a speed bump is infrastructure, not damage
            count_penalty += float(cfg.count_weights.get(name, 0.0))

        area_penalty = 0.0
        for seg in segmentations:
            name = seg.get("class")
            severity = cfg.area_severity.get(name)
            if severity is None:
                continue
            ratio = float(seg.get("area_ratio", 0.0))
            area_penalty += cfg.area_weight * ratio * severity

        penalty = count_penalty + area_penalty
        score = max(0.0, min(cfg.max_score, cfg.max_score - penalty))
        letter, label = grade_for(score)

        if record:
            self._frame_scores.append(score)

        return {
            "road_health_score": round(score, 1),
            "road_health_grade": letter,
            "road_health_label": label,
            "road_health_penalty": round(penalty, 2),
            "road_health_breakdown": {
                "count_penalty": round(count_penalty, 2),
                "area_penalty": round(area_penalty, 2),
            },
            "road_health_basis": (
                "Project-defined road condition indicator - not an official "
                "civil-engineering road safety certification."
            ),
        }

    # -- session aggregate ----------------------------------------------------

    def session_summary(self) -> dict:
        """Aggregate over all scored frames of the current video/session.

        The session score is the MEAN of per-frame scores, not a sum: the same
        pothole appears in many consecutive frames, so summing would punish a
        defect for being visible longer. ``worst_frame_score`` is kept
        alongside because a single very bad stretch matters to a road
        inspector even if the average looks fine.
        """
        if not self._frame_scores:
            return {
                "frames_scored": 0,
                "road_health_score": None,
                "road_health_grade": None,
                "worst_frame_score": None,
            }
        mean_score = sum(self._frame_scores) / len(self._frame_scores)
        worst = min(self._frame_scores)
        letter, label = grade_for(mean_score)
        return {
            "frames_scored": len(self._frame_scores),
            "road_health_score": round(mean_score, 1),
            "road_health_grade": letter,
            "road_health_label": label,
            "worst_frame_score": round(worst, 1),
            "worst_frame_grade": grade_for(worst)[0],
            "road_health_basis": (
                "Project-defined road condition indicator - not an official "
                "civil-engineering road safety certification."
            ),
        }
