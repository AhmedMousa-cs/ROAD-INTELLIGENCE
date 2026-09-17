"""
Approximate speed estimation for tracked vehicles.

HONESTY CONTRACT (from the spec)
--------------------------------
Pixel velocity is NOT speed. A vehicle far up the road moves few pixels per
second while travelling fast; the same vehicle near the camera moves many.
Converting pixels to km/h without scene geometry produces a number that looks
authoritative and is meaningless.

So this module reports two separate things:

* ``velocity_px_s``      - always available. Pure image-space motion. Useful
                           for tracking sanity checks and for "is it moving",
                           never labelled as a real-world speed.
* ``estimated_speed_kmh`` - ONLY populated when a ``SceneCalibration`` is
                           supplied. Always exposed under a name containing
                           "estimated", and accompanied by
                           ``speed_method`` / ``speed_reliable`` so the
                           dashboard can caveat it.

With no calibration, ``estimated_speed_kmh`` is ``None``. It is never
back-filled with a scaled pixel velocity.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# -----------------------------------------------------------------------------
# Scene calibration
# -----------------------------------------------------------------------------


@dataclass
class SceneCalibration:
    """Maps image pixels to road-plane metres.

    Three ways to specify it, in descending order of accuracy:

    1. ``homography``: a 3x3 matrix mapping image -> world (metres) on the
       road plane. Use this if you already computed one.
    2. ``image_points`` + ``world_points``: four corresponding points, e.g.
       the corners of a road rectangle whose real dimensions you know (lane
       width ~3.5 m, dashed-line pitch, etc.). The homography is derived from
       them. This is the practical option for a fixed camera.
    3. ``pixels_per_meter``: a single uniform scale. This IGNORES perspective
       and is only sane for a near-nadir/overhead camera. Flagged as
       ``approximate`` and reported with ``speed_reliable=False``.

    All coordinates are in ORIGINAL frame pixels, top-left origin.
    """

    homography: Sequence[Sequence[float]] | None = None
    image_points: Sequence[Sequence[float]] | None = None
    world_points: Sequence[Sequence[float]] | None = None
    pixels_per_meter: float | None = None
    description: str = ""

    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.homography is not None:
            self._matrix = np.asarray(self.homography, dtype=np.float64).reshape(3, 3)
        elif self.image_points is not None and self.world_points is not None:
            src = np.asarray(self.image_points, dtype=np.float32)
            dst = np.asarray(self.world_points, dtype=np.float32)
            if src.shape != (4, 2) or dst.shape != (4, 2):
                raise ValueError(
                    "image_points and world_points must each be exactly 4 (x, y) pairs"
                )
            import cv2

            self._matrix = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        elif self.pixels_per_meter is not None:
            if self.pixels_per_meter <= 0:
                raise ValueError("pixels_per_meter must be > 0")
            scale = 1.0 / float(self.pixels_per_meter)
            self._matrix = np.array(
                [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

    @property
    def is_configured(self) -> bool:
        return self._matrix is not None

    @property
    def is_approximate(self) -> bool:
        """True when perspective is ignored (uniform-scale calibration)."""
        return self.homography is None and self.image_points is None

    @property
    def method(self) -> str:
        if not self.is_configured:
            return "uncalibrated"
        if self.homography is not None:
            return "homography"
        if self.image_points is not None:
            return "homography_from_points"
        return "uniform_scale"

    def to_world(self, points: Sequence[Sequence[float]]) -> np.ndarray | None:
        """Project image points onto the road plane, in metres."""
        if self._matrix is None:
            return None
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
        projected = homogeneous @ self._matrix.T
        w = projected[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            world = np.where(np.abs(w) > 1e-9, projected[:, :2] / w, np.nan)
        return world

    def to_image(self, points: Sequence[Sequence[float]]) -> np.ndarray | None:
        """Inverse of :meth:`to_world`: road-plane metres -> image pixels.

        Useful for drawing a world-space grid over the video to sanity-check
        a calibration, and for generating synthetic test scenes with a known
        ground-truth speed.
        """
        if self._matrix is None:
            return None
        inverse = np.linalg.inv(self._matrix)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
        projected = homogeneous @ inverse.T
        w = projected[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(w) > 1e-9, projected[:, :2] / w, np.nan)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SceneCalibration | None":
        if not data:
            return None
        return cls(
            homography=data.get("homography"),
            image_points=data.get("image_points"),
            world_points=data.get("world_points"),
            pixels_per_meter=data.get("pixels_per_meter"),
            description=data.get("description", ""),
        )


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class SpeedConfig:
    calibration: SceneCalibration | None = None
    # Displacement is measured over this many trajectory points. Longer =
    # less sensitive to detector box jitter (measured on the synthetic
    # benchmark with 3 px box noise: 16% mean speed error at window=5, 8.5%
    # at 10, 5.6% at 15) but slower to react to real acceleration.
    smoothing_window: int = 15
    # No speed is reported until a track has this many points, so the first
    # (noisiest) estimates of a new track are withheld rather than shown.
    min_points: int = 8
    min_dt_s: float = 1e-3
    max_plausible_kmh: float = 250.0  # above this -> reported but flagged
    # Which point on the box touches the road plane. "bottom_center" is
    # correct for a ground-plane homography; "center" only for overhead views.
    ground_anchor: str = "bottom_center"


# -----------------------------------------------------------------------------
# Estimator
# -----------------------------------------------------------------------------


class SpeedEstimator:
    """Computes image-space velocity and (if calibrated) estimated speed.

    Writes onto ``track.state`` so that serialization stays in tracker.py:
        velocity_px_s, velocity_vector_px_s, direction_deg,
        estimated_speed_kmh, speed_method, speed_reliable
    """

    def __init__(self, config: SpeedConfig | None = None, fps: float = 30.0) -> None:
        self.config = config or SpeedConfig()
        self.fps = float(fps) if fps and fps > 0 else 30.0
        calibration = self.config.calibration
        if calibration is not None and calibration.is_approximate and calibration.is_configured:
            warnings.warn(
                "SpeedEstimator is using a uniform pixels_per_meter calibration; "
                "perspective is ignored, so speeds are rough and are flagged "
                "speed_reliable=False.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def is_calibrated(self) -> bool:
        return self.config.calibration is not None and self.config.calibration.is_configured

    def _ground_point(self, bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        if self.config.ground_anchor == "center":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (x1 + x2) / 2.0, y2  # bottom-center: wheel contact line

    def update(self, track) -> None:
        """Populate speed fields on one track (called once per frame)."""
        cfg = self.config

        # Ground-anchor history lives on the track so it dies with the track.
        history: list[tuple[float, float, float]] = track.state.setdefault("ground_points", [])
        gx, gy = self._ground_point(track.bbox)
        history.append((track.last_timestamp, gx, gy))
        if len(history) > max(cfg.smoothing_window * 2, 10):
            del history[: len(history) - max(cfg.smoothing_window * 2, 10)]

        track.state.setdefault("velocity_px_s", 0.0)
        track.state.setdefault("estimated_speed_kmh", None)
        track.state["speed_method"] = (
            cfg.calibration.method if self.is_calibrated else "uncalibrated"
        )

        points = list(track.trajectory)
        if len(points) < cfg.min_points:
            track.state["speed_reliable"] = False
            return

        window = min(cfg.smoothing_window, len(points) - 1)
        first, last = points[-1 - window], points[-1]
        dt = last.timestamp - first.timestamp
        if dt <= cfg.min_dt_s:
            # Timestamps unavailable or identical: fall back to frame count.
            frame_gap = max(1, last.frame_id - first.frame_id)
            dt = frame_gap / self.fps

        dx = last.cx - first.cx
        dy = last.cy - first.cy
        track.state["velocity_vector_px_s"] = [dx / dt, dy / dt]
        track.state["velocity_px_s"] = float(math.hypot(dx, dy) / dt)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # Image coordinates: y grows downward, so this angle is measured
            # clockwise from the +x axis. 0=right, 90=down, 180=left, 270=up.
            track.state["direction_deg"] = float(math.degrees(math.atan2(dy, dx)) % 360.0)

        if not self.is_calibrated:
            track.state["estimated_speed_kmh"] = None
            track.state["speed_reliable"] = False
            return

        # Ground-plane displacement over the same window.
        if len(history) < 2:
            track.state["speed_reliable"] = False
            return
        window_h = min(cfg.smoothing_window, len(history) - 1)
        t0, x0, y0 = history[-1 - window_h]
        t1, x1_, y1_ = history[-1]
        dt_world = t1 - t0
        if dt_world <= cfg.min_dt_s:
            dt_world = max(1, window_h) / self.fps

        world = cfg.calibration.to_world([[x0, y0], [x1_, y1_]])
        if world is None or not np.all(np.isfinite(world)):
            track.state["estimated_speed_kmh"] = None
            track.state["speed_reliable"] = False
            return

        metres = float(np.linalg.norm(world[1] - world[0]))
        kmh = metres / dt_world * 3.6

        if not (0.0 <= kmh <= cfg.max_plausible_kmh):
            # A value we already know is wrong must not reach the dashboard,
            # where it would be read as a real measurement. Usually means the
            # track is near the horizon (where a few pixels span many metres)
            # or the calibration is off. Kept under a separate key for
            # debugging the calibration.
            track.state["estimated_speed_kmh"] = None
            track.state["speed_rejected_kmh"] = round(kmh, 1)
            track.state["speed_reliable"] = False
            return

        track.state.pop("speed_rejected_kmh", None)
        track.state["estimated_speed_kmh"] = round(kmh, 1)
        track.state["speed_reliable"] = not cfg.calibration.is_approximate

    def output_extra(self, track) -> dict:
        """Speed fields to attach to the serialized track dict."""
        extra = {
            "estimated_speed_kmh": track.state.get("estimated_speed_kmh"),
            "speed_method": track.state.get("speed_method", "uncalibrated"),
            "speed_reliable": bool(track.state.get("speed_reliable", False)),
            "direction_deg": track.state.get("direction_deg"),
        }
        if "speed_rejected_kmh" in track.state:
            extra["speed_rejected_kmh"] = track.state["speed_rejected_kmh"]
        return extra
