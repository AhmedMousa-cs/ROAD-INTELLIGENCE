"""
Traffic analytics: vehicle counting, traffic density, and line/zone crossing.

Counting semantics (documented so the dashboard shows the right number):

* ``vehicle_count``          - vehicles *currently visible and tracked* in
                               this frame. This is the number the density
                               rule uses.
* ``unique_vehicles_seen``   - cumulative distinct track ids since the last
                               ``reset()``. Grows over a video; never
                               decreases. Equal to "how many vehicles passed"
                               only if the tracker never fragments an id, so
                               treat it as an estimate.
* ``counts_by_type``         - per-class breakdown of ``vehicle_count``.
* ``unique_counts_by_type``  - per-class breakdown of ``unique_vehicles_seen``,
                               using each track's final majority-vote class.
* ``line_counts``            - directional crossing tallies per counting line.
                               This is the most reliable "throughput" number
                               when a line is configured.

``person`` is tracked and reported, but is NOT a vehicle: it is excluded from
``vehicle_count`` and from the density rule, and surfaced as ``person_count``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from shared.config import TRAFFIC_DENSITY_THRESHOLDS

from .class_mapping import VEHICLE_CLASSES

LINE_CROSSING_ALERT_TYPE = "line_crossing"


# -----------------------------------------------------------------------------
# Counting lines
# -----------------------------------------------------------------------------


@dataclass
class CountingLine:
    """A virtual line; crossings are counted per direction.

    ``p1``/``p2`` are image coordinates. "Positive" is the side the line's
    normal points to (cross-product sign), so which label ends up on which
    side depends on point order - check once against a sample video and swap
    the labels if needed.
    """

    name: str = "line_1"
    p1: tuple[float, float] = (0.0, 0.0)
    p2: tuple[float, float] = (0.0, 0.0)
    positive_label: str = "in"
    negative_label: str = "out"
    emit_alert: bool = False  # crossings are usually counts, not alerts

    @classmethod
    def from_dict(cls, data: dict) -> "CountingLine":
        return cls(
            name=data.get("name", "line_1"),
            p1=tuple(data["p1"]),
            p2=tuple(data["p2"]),
            positive_label=data.get("positive_label", "in"),
            negative_label=data.get("negative_label", "out"),
            emit_alert=bool(data.get("emit_alert", False)),
        )

    def side(self, point: Sequence[float]) -> float:
        """Signed side of the (infinite) line: >0, <0, or 0 when on it."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def crossed(
        self, previous: Sequence[float], current: Sequence[float]
    ) -> str | None:
        """Return the direction label if the segment crosses this line."""
        s_prev, s_cur = self.side(previous), self.side(current)
        if s_prev == 0.0 or s_cur == 0.0 or (s_prev > 0) == (s_cur > 0):
            return None
        if not self._within_segment(previous, current):
            return None
        return self.positive_label if s_cur > 0 else self.negative_label

    def _within_segment(
        self, previous: Sequence[float], current: Sequence[float]
    ) -> bool:
        """True if the crossing point lies on the finite line segment."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        x3, y3 = previous
        x4, y4 = current
        denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        if abs(denom) < 1e-9:
            return False
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
        return 0.0 <= t <= 1.0


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrafficAnalyticsConfig:
    # Copied (not referenced) from shared/config.py so a caller can override
    # per-scene without mutating the shared defaults for other modules.
    density_thresholds: dict = field(
        default_factory=lambda: dict(TRAFFIC_DENSITY_THRESHOLDS)
    )
    counting_lines: list[CountingLine] = field(default_factory=list)
    count_person_as_vehicle: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrafficAnalyticsConfig":
        if not data:
            return cls()
        thresholds = dict(TRAFFIC_DENSITY_THRESHOLDS)
        thresholds.update(data.get("density_thresholds", {}))
        return cls(
            density_thresholds=thresholds,
            counting_lines=[CountingLine.from_dict(l) for l in data.get("counting_lines", [])],
            count_person_as_vehicle=bool(data.get("count_person_as_vehicle", False)),
        )


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------


class TrafficAnalytics:
    """Per-frame traffic statistics derived from confirmed tracks."""

    def __init__(self, config: TrafficAnalyticsConfig | None = None) -> None:
        self.config = config or TrafficAnalyticsConfig()
        self.line_counts: dict[str, dict[str, int]] = {
            line.name: {line.positive_label: 0, line.negative_label: 0}
            for line in self.config.counting_lines
        }
        self._unique_ids: set[int] = set()
        self._unique_class_by_id: dict[int, str] = {}

    def reset(self) -> None:
        for counts in self.line_counts.values():
            for key in counts:
                counts[key] = 0
        self._unique_ids.clear()
        self._unique_class_by_id.clear()

    # -- density --------------------------------------------------------------

    def density_label(self, active_vehicles: int) -> str:
        """Project-defined density rule; thresholds come from config."""
        low_max = self.config.density_thresholds.get("low_max", 5)
        medium_max = self.config.density_thresholds.get("medium_max", 15)
        if active_vehicles <= low_max:
            return "low"
        if active_vehicles <= medium_max:
            return "medium"
        return "high"

    def _is_vehicle(self, class_name: str) -> bool:
        if class_name in VEHICLE_CLASSES:
            return True
        return self.config.count_person_as_vehicle and class_name == "person"

    # -- main entry point -----------------------------------------------------

    def update(self, tracks, frame_id: int) -> tuple[dict, list[dict]]:
        """Return (analytics_fields, alerts) for this frame."""
        alerts: list[dict] = []

        active_by_type: Counter[str] = Counter()
        for track in tracks:
            active_by_type[track.class_name] += 1
            self._unique_ids.add(track.track_id)
            self._unique_class_by_id[track.track_id] = track.class_name
            alerts.extend(self._check_lines(track, frame_id))

        vehicle_count = sum(
            count for name, count in active_by_type.items() if self._is_vehicle(name)
        )
        person_count = active_by_type.get("person", 0)

        unique_by_type: Counter[str] = Counter(self._unique_class_by_id.values())
        unique_vehicles = sum(
            count for name, count in unique_by_type.items() if self._is_vehicle(name)
        )

        analytics = {
            "vehicle_count": int(vehicle_count),
            "traffic_density": self.density_label(vehicle_count),
            "person_count": int(person_count),
            "counts_by_type": {k: int(v) for k, v in sorted(active_by_type.items())},
            "unique_vehicles_seen": int(unique_vehicles),
            "unique_counts_by_type": {
                k: int(v) for k, v in sorted(unique_by_type.items())
            },
            "line_counts": {
                name: dict(counts) for name, counts in self.line_counts.items()
            },
        }
        return analytics, alerts

    # -- line crossing --------------------------------------------------------

    def _check_lines(self, track, frame_id: int) -> list[dict]:
        if not self.config.counting_lines:
            return []
        points = list(track.trajectory)
        if len(points) < 2:
            return []

        previous = (points[-2].cx, points[-2].cy)
        current = (points[-1].cx, points[-1].cy)
        crossed_state: set[str] = track.state.setdefault("lines_crossed", set())

        alerts: list[dict] = []
        for line in self.config.counting_lines:
            direction = line.crossed(previous, current)
            if direction is None:
                continue
            # One count per (track, line, direction) so a vehicle hovering on
            # the line cannot inflate the tally.
            key = f"{line.name}:{direction}"
            if key in crossed_state:
                continue
            crossed_state.add(key)
            self.line_counts[line.name][direction] += 1

            if line.emit_alert:
                alerts.append(
                    {
                        "type": LINE_CROSSING_ALERT_TYPE,
                        "track_id": track.track_id,
                        "confidence": float(track.confidence),
                        "message": f"{track.class_name} crossed {line.name} ({direction})",
                        "class": track.class_name,
                        "line": line.name,
                        "direction": direction,
                        "frame_id": int(frame_id),
                        "verified": False,
                    }
                )
        return alerts
