"""
Multi-object tracking for the vehicle layer.

Design
------
Track *state* (id, history, class voting, motion) is owned by this module's
``Track`` / ``VehicleTracker`` classes. Only the *association* step (which
detection belongs to which existing track) is pluggable:

    NativeByteTrackAssociator  - dependency-free ByteTrack-style association
                                 (high/low confidence two-stage matching +
                                 lost-track recovery). Default.
    UltralyticsAssociator      - delegates id assignment to Ultralytics'
                                 BYTETracker or BOTSORT when ultralytics is
                                 installed (BoT-SORT adds camera-motion
                                 compensation, which helps on dashcam video).

Keeping history in our own ``Track`` objects means trajectories, velocity and
stable class labels behave identically whichever backend is used, and the
module still works in a clean environment with only numpy installed.
"""

from __future__ import annotations

import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrackerConfig:
    """Tracking parameters.

    Thresholds live here (and are surfaced through
    ``TrackingAnalyticsConfig``) rather than being hard-coded inside the
    association algorithm, per the shared contract.
    """

    # "native" | "bytetrack" | "botsort" | "auto"
    # "auto" -> bytetrack via ultralytics if importable, else native.
    backend: str = "native"

    # Two-stage (ByteTrack) association thresholds.
    track_high_thresh: float = 0.50  # detections used in the 1st association
    track_low_thresh: float = 0.10  # detections used in the 2nd association
    new_track_thresh: float = 0.60  # min confidence to spawn a new track
    match_thresh: float = 0.20  # min IoU, 1st association
    second_match_thresh: float = 0.15  # min IoU, 2nd association (low-conf dets)
    recover_match_thresh: float = 0.25  # min IoU, lost-track recovery

    max_age: int = 30  # frames a lost track is kept before removal
    min_hits: int = 3  # updates before a track is reported (confirmed)

    class_aware: bool = False  # if True, never match across different classes
    stable_class_vote_window: int = 15  # majority vote window for the label
    trajectory_max_points: int = 60  # center points retained per track

    frame_rate: float = 30.0  # used by the Ultralytics backends' buffers


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two [N,4] / [M,4] arrays of [x1,y1,x2,y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Uses scipy when available, else greedy.

    Greedy is a slightly worse (but perfectly usable) approximation; it only
    kicks in if scipy is missing, which keeps the module installable with
    numpy alone.
    """
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
        for r, c in flat:
            r, c = int(r), int(c)
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        return pairs


# -----------------------------------------------------------------------------
# Track state
# -----------------------------------------------------------------------------


@dataclass
class TrajectoryPoint:
    frame_id: int
    timestamp: float
    cx: float
    cy: float


class Track:
    """State for one tracked road user.

    Holds everything the downstream components need: bbox, center history
    (with timestamps, so speed estimation can use real dt), a stabilised
    class label, and simple linear motion for prediction.
    """

    def __init__(
        self,
        track_id: int,
        detection: dict,
        frame_id: int,
        timestamp: float,
        config: TrackerConfig,
    ) -> None:
        self.track_id = int(track_id)
        self.config = config

        self.bbox: list[float] = [float(v) for v in detection["bbox"]]
        self.confidence: float = float(detection["confidence"])
        self._class_votes: deque[str] = deque(maxlen=config.stable_class_vote_window)
        self._class_votes.append(detection["class"])
        self.last_class: str = detection["class"]

        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0
        self.start_frame: int = int(frame_id)
        self.last_frame: int = int(frame_id)
        self.last_timestamp: float = float(timestamp)

        self.trajectory: deque[TrajectoryPoint] = deque(
            maxlen=max(2, config.trajectory_max_points)
        )
        cx, cy = self._center_of(self.bbox)
        self.trajectory.append(TrajectoryPoint(int(frame_id), float(timestamp), cx, cy))

        # Linear motion model (pixels/frame), EMA-smoothed.
        self._velocity_px_frame = np.zeros(2, dtype=np.float32)

        # Scratch space for the analytics components (speed, wrong-way, lines).
        # Kept on the track so per-track state travels with the track and is
        # garbage-collected with it.
        self.state: dict = {}

    # -- properties -----------------------------------------------------------

    @staticmethod
    def _center_of(bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def center(self) -> list[float]:
        cx, cy = self._center_of(self.bbox)
        return [cx, cy]

    @property
    def class_name(self) -> str:
        """Majority-vote label over the recent history.

        COCO detectors flicker between car/truck on large vehicles; voting
        gives the dashboard a stable label per id.
        """
        if not self._class_votes:
            return self.last_class
        return Counter(self._class_votes).most_common(1)[0][0]

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.config.min_hits

    @property
    def is_active(self) -> bool:
        return self.time_since_update == 0

    # -- lifecycle ------------------------------------------------------------

    def predict(self) -> list[float]:
        """Predicted bbox for the next frame (association only).

        Never reported as output: only observed boxes leave the tracker, so
        emitted bboxes are always real detections inside the frame.
        """
        dx, dy = float(self._velocity_px_frame[0]), float(self._velocity_px_frame[1])
        x1, y1, x2, y2 = self.bbox
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def update(self, detection: dict, frame_id: int, timestamp: float) -> None:
        prev_cx, prev_cy = self._center_of(self.bbox)
        self.bbox = [float(v) for v in detection["bbox"]]
        self.confidence = float(detection["confidence"])
        self.last_class = detection["class"]
        self._class_votes.append(detection["class"])

        cx, cy = self._center_of(self.bbox)
        frame_gap = max(1, int(frame_id) - self.last_frame)
        instantaneous = np.array(
            [(cx - prev_cx) / frame_gap, (cy - prev_cy) / frame_gap], dtype=np.float32
        )
        # EMA keeps the motion model from chasing detector jitter.
        self._velocity_px_frame = 0.5 * self._velocity_px_frame + 0.5 * instantaneous

        self.trajectory.append(TrajectoryPoint(int(frame_id), float(timestamp), cx, cy))
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.last_frame = int(frame_id)
        self.last_timestamp = float(timestamp)

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1

    @property
    def is_expired(self) -> bool:
        return self.time_since_update > self.config.max_age

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Track(id={self.track_id}, class={self.class_name}, "
            f"hits={self.hits}, missed={self.time_since_update})"
        )


# -----------------------------------------------------------------------------
# Association backends
# -----------------------------------------------------------------------------


class NativeByteTrackAssociator:
    """Dependency-free ByteTrack-style association.

    Follows the ByteTrack idea: match high-confidence detections first, then
    use the *low*-confidence detections (usually occluded/blurred objects
    that a plain tracker would throw away) to keep existing tracks alive,
    then try to recover recently lost ids.

    Motion prediction is a smoothed constant-velocity model rather than a
    Kalman filter - simpler, and adequate because association here is
    IoU-based at road-camera frame rates. Documented as a deviation from the
    reference implementation in the module README.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    def associate(
        self,
        tracks: list[Track],
        detections: list[dict],
        frame: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Return (matches, unmatched_track_indices, unmatched_detection_indices).

        ``matches`` are (track_index, detection_index) pairs.
        ``unmatched_detection_indices`` only contains detections eligible to
        start a new track (confidence >= new_track_thresh).
        """
        cfg = self.config
        confidences = np.array([d["confidence"] for d in detections], dtype=np.float32)
        high_idx = [i for i, c in enumerate(confidences) if c >= cfg.track_high_thresh]
        low_idx = [
            i
            for i, c in enumerate(confidences)
            if cfg.track_low_thresh <= c < cfg.track_high_thresh
        ]

        active_idx = [i for i, t in enumerate(tracks) if t.time_since_update == 0]
        lost_idx = [i for i, t in enumerate(tracks) if t.time_since_update > 0]

        matches: list[tuple[int, int]] = []

        # Stage 1: active tracks vs high-confidence detections.
        m1, rem_tracks, rem_high = self._match(
            tracks, active_idx, detections, high_idx, cfg.match_thresh
        )
        matches += m1

        # Stage 2: still-unmatched active tracks vs low-confidence detections.
        m2, rem_tracks, _ = self._match(
            tracks, rem_tracks, detections, low_idx, cfg.second_match_thresh
        )
        matches += m2

        # Stage 3: recover recently lost tracks from leftover high-conf dets.
        m3, rem_lost, rem_high = self._match(
            tracks, lost_idx, detections, rem_high, cfg.recover_match_thresh
        )
        matches += m3

        unmatched_tracks = rem_tracks + rem_lost
        new_detections = [
            i for i in rem_high if confidences[i] >= cfg.new_track_thresh
        ]
        return matches, unmatched_tracks, new_detections

    def _match(
        self,
        tracks: list[Track],
        track_indices: list[int],
        detections: list[dict],
        detection_indices: list[int],
        min_iou: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detection_indices:
            return [], list(track_indices), list(detection_indices)

        track_boxes = np.array([tracks[i].predict() for i in track_indices], dtype=np.float32)
        det_boxes = np.array(
            [detections[i]["bbox"] for i in detection_indices], dtype=np.float32
        )
        ious = iou_matrix(track_boxes, det_boxes)

        if self.config.class_aware:
            for r, ti in enumerate(track_indices):
                for c, di in enumerate(detection_indices):
                    if tracks[ti].class_name != detections[di]["class"]:
                        ious[r, c] = 0.0

        pairs = _assign(1.0 - ious)

        matches: list[tuple[int, int]] = []
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()
        for r, c in pairs:
            if ious[r, c] < min_iou:
                continue
            matches.append((track_indices[r], detection_indices[c]))
            matched_rows.add(r)
            matched_cols.add(c)

        unmatched_tracks = [
            ti for r, ti in enumerate(track_indices) if r not in matched_rows
        ]
        unmatched_dets = [
            di for c, di in enumerate(detection_indices) if c not in matched_cols
        ]
        return matches, unmatched_tracks, unmatched_dets


class UltralyticsAssociator:
    """Delegate id assignment to Ultralytics' BYTETracker / BOTSORT.

    Only the *ids* are taken from Ultralytics; history, class voting and
    velocity stay in our ``Track`` objects. If ultralytics is missing or its
    tracker API has drifted, construction raises and the caller falls back to
    the native associator.
    """

    def __init__(self, config: TrackerConfig, kind: str = "bytetrack") -> None:
        from types import SimpleNamespace

        self.config = config
        self.kind = kind
        args = SimpleNamespace(
            tracker_type=kind,
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.max_age,
            match_thresh=1.0 - config.match_thresh,
            fuse_score=True,
            # BoT-SORT specific
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
            model="auto",
        )
        if kind == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT

            self._tracker = BOTSORT(args, frame_rate=int(config.frame_rate))
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker

            self._tracker = BYTETracker(args, frame_rate=int(config.frame_rate))

        self._external_to_internal: dict[int, int] = {}

    def update(self, detections: list[dict], frame: np.ndarray) -> list[tuple[int, int]]:
        """Return (external_track_id, detection_index) pairs."""
        from types import SimpleNamespace

        if not detections:
            boxes = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
            conf = np.array([d["confidence"] for d in detections], dtype=np.float32)
            cls = np.array([d.get("class_id", 0) for d in detections], dtype=np.float32)

        xywh = np.zeros_like(boxes)
        if len(boxes):
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
            xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        results = SimpleNamespace(conf=conf, cls=cls, xywh=xywh, xyxy=boxes)
        tracked = self._tracker.update(results, frame)

        pairs: list[tuple[int, int]] = []
        for row in np.atleast_2d(tracked):
            if row.size < 8:
                # Older ultralytics builds omit the original detection index;
                # without it we cannot map ids back to our detections.
                raise RuntimeError(
                    "Ultralytics tracker output lacks the detection-index column"
                )
            track_id = int(row[4])
            det_index = int(row[7])
            if 0 <= det_index < len(detections):
                pairs.append((track_id, det_index))
        return pairs


# -----------------------------------------------------------------------------
# Tracker
# -----------------------------------------------------------------------------


class VehicleTracker:
    """Frame-to-frame multi-object tracker over normalized detections.

    Input detections are shared-schema detection dicts
    (``{"class", "confidence", "bbox", "source"}``); output is a list of
    confirmed, currently-visible ``Track`` objects.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self._seen_ids: set[int] = set()
        self._id_class: dict[int, str] = {}

        self._ultralytics = None
        backend = self.config.backend.lower()
        if backend in ("bytetrack", "botsort", "auto"):
            kind = "botsort" if backend == "botsort" else "bytetrack"
            try:
                self._ultralytics = UltralyticsAssociator(self.config, kind=kind)
            except Exception as exc:  # ImportError or API drift
                if backend != "auto":
                    warnings.warn(
                        f"Could not initialise the Ultralytics '{kind}' tracker "
                        f"({exc}); falling back to the native ByteTrack-style "
                        f"associator.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self._ultralytics = None
        self._native = NativeByteTrackAssociator(self.config)

    # -- public API -----------------------------------------------------------

    @property
    def backend_in_use(self) -> str:
        return self._ultralytics.kind if self._ultralytics is not None else "native"

    def reset(self) -> None:
        """Clear all state. Call between videos."""
        self.tracks.clear()
        self._next_id = 1
        self._seen_ids.clear()
        self._id_class.clear()
        if self._ultralytics is not None:
            self._ultralytics = UltralyticsAssociator(
                self.config, kind=self._ultralytics.kind
            )

    def update(
        self,
        detections: list[dict],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        detections = [d for d in detections if d.get("confidence", 0.0) >= self.config.track_low_thresh]

        if self._ultralytics is not None and frame is not None:
            try:
                self._update_with_ultralytics(detections, frame_id, timestamp, frame)
            except Exception as exc:
                warnings.warn(
                    f"Ultralytics tracker failed at frame {frame_id} ({exc}); "
                    f"switching to the native associator for the rest of the run.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._ultralytics = None
                self._update_native(detections, frame_id, timestamp, frame)
        else:
            self._update_native(detections, frame_id, timestamp, frame)

        self.tracks = [t for t in self.tracks if not t.is_expired]
        for track in self.tracks:
            if track.is_confirmed:
                self._seen_ids.add(track.track_id)
                self._id_class[track.track_id] = track.class_name

        return [t for t in self.tracks if t.is_confirmed and t.is_active]

    # -- cumulative stats (used by TrafficAnalytics) --------------------------

    @property
    def unique_ids_seen(self) -> set[int]:
        return set(self._seen_ids)

    @property
    def unique_class_by_id(self) -> dict[int, str]:
        return dict(self._id_class)

    # -- internals ------------------------------------------------------------

    def _update_native(
        self,
        detections: list[dict],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray | None,
    ) -> None:
        matches, unmatched_tracks, new_detections = self._native.associate(
            self.tracks, detections, frame
        )
        for track_index, det_index in matches:
            self.tracks[track_index].update(detections[det_index], frame_id, timestamp)
        for track_index in unmatched_tracks:
            self.tracks[track_index].mark_missed()
        for det_index in new_detections:
            self._spawn(detections[det_index], frame_id, timestamp)

    def _update_with_ultralytics(
        self,
        detections: list[dict],
        frame_id: int,
        timestamp: float,
        frame: np.ndarray,
    ) -> None:
        pairs = self._ultralytics.update(detections, frame)
        by_id = {t.track_id: t for t in self.tracks}
        updated: set[int] = set()

        for external_id, det_index in pairs:
            detection = detections[det_index]
            track = by_id.get(external_id)
            if track is None:
                track = self._spawn(detection, frame_id, timestamp, forced_id=external_id)
            else:
                track.update(detection, frame_id, timestamp)
            updated.add(track.track_id)
            self._next_id = max(self._next_id, external_id + 1)

        for track in self.tracks:
            if track.track_id not in updated:
                track.mark_missed()

    def _spawn(
        self,
        detection: dict,
        frame_id: int,
        timestamp: float,
        forced_id: int | None = None,
    ) -> Track:
        track_id = forced_id if forced_id is not None else self._next_id
        if forced_id is None:
            self._next_id += 1
        track = Track(track_id, detection, frame_id, timestamp, self.config)
        self.tracks.append(track)
        return track


# -----------------------------------------------------------------------------
# Serialization to the shared schema
# -----------------------------------------------------------------------------


def track_to_dict(
    track: Track,
    include_trajectory: bool = True,
    trajectory_points: int = 30,
    extra: dict | None = None,
) -> dict:
    """Serialize a Track into the shared-schema track dict.

    Required keys (from the spec):
        track_id, class, bbox, center, velocity_px_s

    Additional keys are additive (the shared validator tolerates extras) and
    exist so the dashboard does not have to recompute them: confidence,
    source, age_frames, direction_deg, estimated_speed_kmh, trajectory.
    """
    payload = {
        "track_id": track.track_id,
        "class": track.class_name,
        "bbox": [float(v) for v in track.bbox],
        "center": [float(v) for v in track.center],
        "velocity_px_s": float(track.state.get("velocity_px_s", 0.0)),
        "confidence": float(track.confidence),
        "source": "tracking_analytics",
        "age_frames": int(track.age),
    }
    if extra:
        payload.update(extra)
    if include_trajectory and trajectory_points > 0:
        points = list(track.trajectory)[-trajectory_points:]
        payload["trajectory"] = [[round(p.cx, 2), round(p.cy, 2)] for p in points]
    return payload


def tracks_to_dicts(
    tracks: Iterable[Track],
    include_trajectory: bool = True,
    trajectory_points: int = 30,
) -> list[dict]:
    return [
        track_to_dict(t, include_trajectory, trajectory_points, extra=t.state.get("output_extra"))
        for t in tracks
    ]
