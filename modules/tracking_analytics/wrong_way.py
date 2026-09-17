"""
Wrong-way driving detection.

Approach
--------
The scene is annotated with one or more ``DirectionZone``s: an optional
polygon (image coordinates) plus the direction traffic is *allowed* to move
in that polygon. For every confirmed track we compare its smoothed motion
vector against the allowed direction of the zone its ground point falls in:

    angle <= tolerance_deg            -> compliant   (decays the counter)
    angle >= 180 - tolerance_deg      -> violating   (increments the counter)
    otherwise                         -> ambiguous   (ignored: a turning or
                                          lane-changing vehicle is crossing,
                                          not opposing, the flow)

An alert is raised only after the violating evidence is *consistent* over
``min_frames`` and the vehicle has actually moved ``min_displacement_px``, so
parked cars and single-frame tracker jitter cannot trigger it.

IMPORTANT
---------
This is an algorithmic alert, not a legally verified traffic violation. The
emitted alert carries ``"verified": False`` and the message is phrased as a
possibility. Do not present it to a user as an enforcement decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from shared.config import WRONG_WAY_CONFIDENCE_MIN

ALERT_TYPE = "wrong_way"


@dataclass
class DirectionZone:
    """A region of the frame plus the direction traffic may travel in it.

    ``allowed_direction`` is an (dx, dy) vector in IMAGE coordinates, where y
    grows downward. Examples for a camera facing oncoming traffic:
        (0, 1)   traffic moves toward the camera (down the frame)
        (0, -1)  traffic moves away from the camera (up the frame)
        (1, 0)   traffic moves left-to-right

    ``polygon`` is a list of (x, y) image points. ``None`` means the whole
    frame - only sensible for a one-way road filling the view.
    """

    name: str = "default"
    allowed_direction: tuple[float, float] = (0.0, -1.0)
    polygon: Sequence[Sequence[float]] | None = None
    tolerance_deg: float = 60.0

    _polygon_np: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        dx, dy = self.allowed_direction
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            raise ValueError(
                f"DirectionZone '{self.name}': allowed_direction must be a non-zero vector"
            )
        self.allowed_direction = (dx / norm, dy / norm)
        if self.polygon is not None:
            poly = np.asarray(self.polygon, dtype=np.float32).reshape(-1, 2)
            if len(poly) < 3:
                raise ValueError(
                    f"DirectionZone '{self.name}': polygon needs at least 3 points"
                )
            self._polygon_np = poly

    @property
    def allowed_direction_deg(self) -> float:
        dx, dy = self.allowed_direction
        return math.degrees(math.atan2(dy, dx)) % 360.0

    def contains(self, x: float, y: float) -> bool:
        if self._polygon_np is None:
            return True
        import cv2

        return cv2.pointPolygonTest(self._polygon_np, (float(x), float(y)), False) >= 0

    @classmethod
    def from_dict(cls, data: dict) -> "DirectionZone":
        return cls(
            name=data.get("name", "default"),
            allowed_direction=tuple(data.get("allowed_direction", (0.0, -1.0))),
            polygon=data.get("polygon"),
            tolerance_deg=float(data.get("tolerance_deg", 60.0)),
        )


@dataclass
class WrongWayConfig:
    zones: list[DirectionZone] = field(default_factory=list)
    min_frames: int = 8  # consecutive-ish violating frames before alerting
    min_displacement_px: float = 25.0  # ignore stationary/near-stationary tracks
    min_confidence: float = WRONG_WAY_CONFIDENCE_MIN
    decay_on_compliant: int = 2  # counter decrement per compliant frame
    emit_while_active: bool = True  # re-emit each frame while still violating
    ignore_classes: tuple[str, ...] = ("person",)  # pedestrians have no lane

    @classmethod
    def from_dict(cls, data: dict | None) -> "WrongWayConfig":
        if not data:
            return cls()
        return cls(
            zones=[DirectionZone.from_dict(z) for z in data.get("zones", [])],
            min_frames=int(data.get("min_frames", 8)),
            min_displacement_px=float(data.get("min_displacement_px", 25.0)),
            min_confidence=float(data.get("min_confidence", WRONG_WAY_CONFIDENCE_MIN)),
            decay_on_compliant=int(data.get("decay_on_compliant", 2)),
            emit_while_active=bool(data.get("emit_while_active", True)),
            ignore_classes=tuple(data.get("ignore_classes", ("person",))),
        )


def angle_between_deg(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Unsigned angle between two 2-D vectors, in degrees [0, 180]."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


class WrongWayDetector:
    """Raises `wrong_way` alerts for tracks opposing the configured flow."""

    def __init__(self, config: WrongWayConfig | None = None) -> None:
        self.config = config or WrongWayConfig()

    @property
    def is_configured(self) -> bool:
        """False when no zone is defined - the detector then stays silent."""
        return bool(self.config.zones)

    def reset(self) -> None:
        """No global state; per-track state lives on the tracks themselves."""

    def zone_for(self, x: float, y: float) -> DirectionZone | None:
        for zone in self.config.zones:
            if zone.contains(x, y):
                return zone
        return None

    def update(self, tracks, frame_id: int) -> list[dict]:
        """Evaluate all active tracks; return alerts for this frame."""
        if not self.is_configured:
            return []

        alerts: list[dict] = []
        cfg = self.config

        for track in tracks:
            if track.class_name in cfg.ignore_classes:
                continue

            velocity = track.state.get("velocity_vector_px_s")
            if not velocity:
                continue

            # Use the same ground anchor as the speed estimator when present.
            x1, y1, x2, y2 = track.bbox
            anchor_x, anchor_y = (x1 + x2) / 2.0, y2
            zone = self.zone_for(anchor_x, anchor_y)
            if zone is None:
                continue

            displacement = self._displacement_px(track)
            if displacement < cfg.min_displacement_px:
                continue

            angle = angle_between_deg(velocity, zone.allowed_direction)
            hits = track.state.get("wrong_way_hits", 0)
            considered = track.state.get("wrong_way_considered", 0)

            if angle >= 180.0 - zone.tolerance_deg:
                hits += 1
                considered += 1
            elif angle <= zone.tolerance_deg:
                hits = max(0, hits - cfg.decay_on_compliant)
                considered += 1
            # else: ambiguous (crossing/turning) - no evidence either way.

            track.state["wrong_way_hits"] = hits
            track.state["wrong_way_considered"] = considered

            if hits < cfg.min_frames:
                track.state["wrong_way_active"] = False
                continue

            coverage = min(1.0, hits / max(1, cfg.min_frames))
            ratio = hits / max(1, considered)
            confidence = round(0.5 * coverage + 0.5 * ratio, 2)
            if confidence < cfg.min_confidence:
                continue

            already_alerted = track.state.get("wrong_way_alerted", False)
            track.state["wrong_way_active"] = True
            if already_alerted and not cfg.emit_while_active:
                continue
            track.state["wrong_way_alerted"] = True

            alerts.append(
                {
                    "type": ALERT_TYPE,
                    "track_id": track.track_id,
                    "confidence": float(min(confidence, 1.0)),
                    "message": "Possible wrong-way vehicle",
                    # --- context (additive; the dashboard can ignore it) ---
                    "class": track.class_name,
                    "zone": zone.name,
                    "frame_id": int(frame_id),
                    "observed_direction_deg": track.state.get("direction_deg"),
                    "allowed_direction_deg": round(zone.allowed_direction_deg, 1),
                    "angle_to_allowed_deg": round(angle, 1),
                    "evidence_frames": int(hits),
                    "verified": False,  # algorithmic alert, not an enforcement decision
                }
            )

        return alerts

    @staticmethod
    def _displacement_px(track) -> float:
        points = list(track.trajectory)
        if len(points) < 2:
            return 0.0
        first, last = points[0], points[-1]
        return math.hypot(last.cx - first.cx, last.cy - first.cy)
