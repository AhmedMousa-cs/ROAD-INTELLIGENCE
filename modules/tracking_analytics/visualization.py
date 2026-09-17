"""
Optional visualization helpers for tracks, trajectories and alerts.

These are a debugging/demo aid. The final Streamlit dashboard may define its
own theme, so nothing here bakes in a colour: every drawing function takes a
``theme`` dict and falls back to ``DEFAULT_THEME``, which the dashboard can
replace wholesale:

    from modules.tracking_analytics import visualization as viz
    viz.draw_frame_result(frame, result, theme={**viz.DEFAULT_THEME,
                                                "track": (255, 0, 0)})

All colours are BGR tuples, matching OpenCV.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_THEME: dict = {
    "track": (0, 200, 0),
    "alert": (0, 0, 255),
    "trajectory": (200, 200, 0),
    "line": (255, 180, 0),
    "zone": (120, 120, 255),
    "text": (255, 255, 255),
    "text_bg": (0, 0, 0),
    "panel_bg": (32, 32, 32),
    "font_scale": 0.5,
    "thickness": 2,
    # Per-id palette: track colour is palette[track_id % len(palette)] unless
    # the track is flagged wrong-way, which always uses "alert".
    "palette": [
        (0, 200, 0),
        (255, 160, 0),
        (0, 190, 255),
        (200, 0, 200),
        (0, 140, 255),
        (180, 220, 0),
    ],
    "use_palette": True,
}


def _theme(theme: dict | None) -> dict:
    return {**DEFAULT_THEME, **(theme or {})}


def color_for_track(track: dict, theme: dict | None = None) -> tuple[int, int, int]:
    resolved = _theme(theme)
    if track.get("wrong_way"):
        return tuple(resolved["alert"])
    if resolved.get("use_palette") and resolved.get("palette"):
        palette = resolved["palette"]
        return tuple(palette[int(track.get("track_id", 0)) % len(palette)])
    return tuple(resolved["track"])


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Sequence[int],
    color: Sequence[int],
    theme: dict | None = None,
) -> np.ndarray:
    """Draw text with a filled background box for legibility."""
    import cv2

    resolved = _theme(theme)
    scale = float(resolved["font_scale"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = int(origin[0]), int(origin[1])
    y = max(y, height + baseline + 2)
    cv2.rectangle(
        image,
        (x, y - height - baseline - 2),
        (x + width + 4, y),
        tuple(int(c) for c in color),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline),
        font,
        scale,
        tuple(int(c) for c in resolved["text"]),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_tracks(
    image: np.ndarray,
    tracks: list[dict],
    theme: dict | None = None,
    show_speed: bool = True,
) -> np.ndarray:
    import cv2

    resolved = _theme(theme)
    for track in tracks:
        color = color_for_track(track, resolved)
        x1, y1, x2, y2 = (int(v) for v in track["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, int(resolved["thickness"]))

        parts = [f"#{track['track_id']}", track["class"]]
        if show_speed:
            speed = track.get("estimated_speed_kmh")
            if speed is not None:
                # "~" and "est" make it visually obvious this is an estimate.
                parts.append(f"~{speed:.0f} km/h est")
            else:
                parts.append(f"{track.get('velocity_px_s', 0.0):.0f} px/s")
        if track.get("wrong_way"):
            parts.append("WRONG WAY?")
        draw_label(image, " ".join(parts), (x1, y1), color, resolved)
    return image


def draw_trajectories(
    image: np.ndarray,
    tracks: list[dict],
    theme: dict | None = None,
    max_points: int = 30,
) -> np.ndarray:
    import cv2

    resolved = _theme(theme)
    for track in tracks:
        points = track.get("trajectory") or []
        if len(points) < 2:
            continue
        color = color_for_track(track, resolved)
        pts = np.asarray(points[-max_points:], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], False, color, max(1, int(resolved["thickness"]) - 1))
        cv2.circle(image, tuple(pts[-1][0]), 3, color, -1)
    return image


def draw_counting_lines(
    image: np.ndarray, lines, theme: dict | None = None
) -> np.ndarray:
    """``lines`` is an iterable of CountingLine objects."""
    import cv2

    resolved = _theme(theme)
    for line in lines:
        p1 = tuple(int(v) for v in line.p1)
        p2 = tuple(int(v) for v in line.p2)
        cv2.line(image, p1, p2, tuple(resolved["line"]), int(resolved["thickness"]))
        draw_label(image, line.name, p1, tuple(resolved["line"]), resolved)
    return image


def draw_direction_zones(
    image: np.ndarray, zones, theme: dict | None = None
) -> np.ndarray:
    """``zones`` is an iterable of DirectionZone objects."""
    import cv2

    resolved = _theme(theme)
    for zone in zones:
        if zone.polygon is None:
            continue
        pts = np.asarray(zone.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, tuple(resolved["zone"]), 1)
        centroid = pts.reshape(-1, 2).mean(axis=0).astype(int)
        dx, dy = zone.allowed_direction
        tip = (int(centroid[0] + dx * 60), int(centroid[1] + dy * 60))
        cv2.arrowedLine(
            image, tuple(centroid), tip, tuple(resolved["zone"]), 2, tipLength=0.3
        )
        draw_label(image, zone.name, tuple(centroid), tuple(resolved["zone"]), resolved)
    return image


def draw_analytics_panel(
    image: np.ndarray, result: dict, theme: dict | None = None
) -> np.ndarray:
    import cv2

    resolved = _theme(theme)
    analytics = result.get("analytics", {})
    lines = [
        f"Vehicles (now): {analytics.get('vehicle_count', 0)}",
        f"Density: {str(analytics.get('traffic_density', 'low')).upper()}",
        f"Unique seen: {analytics.get('unique_vehicles_seen', 0)}",
    ]
    by_type = analytics.get("counts_by_type") or {}
    if by_type:
        lines.append(", ".join(f"{k}:{v}" for k, v in by_type.items()))
    score = analytics.get("road_health_score")
    if score is not None:
        lines.append(
            f"Road health: {score} ({analytics.get('road_health_grade', '-')}) *"
        )
        lines.append("* project-defined indicator")

    pad = 8
    scale = float(resolved["font_scale"])
    row = int(22 * max(scale / 0.5, 1.0))
    width = 260
    height = pad * 2 + row * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), tuple(resolved["panel_bg"]), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (pad, pad + row * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            tuple(resolved["text"]),
            1,
            cv2.LINE_AA,
        )
    return image


def draw_alerts(
    image: np.ndarray, alerts: list[dict], theme: dict | None = None
) -> np.ndarray:
    import cv2

    resolved = _theme(theme)
    if not alerts:
        return image
    height = image.shape[0]
    for index, alert in enumerate(alerts[:5]):
        text = (
            f"[{alert.get('type')}] #{alert.get('track_id')} "
            f"{alert.get('message', '')} ({alert.get('confidence', 0):.2f})"
        )
        cv2.putText(
            image,
            text,
            (8, height - 10 - index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(resolved["font_scale"]),
            tuple(resolved["alert"]),
            1,
            cv2.LINE_AA,
        )
    return image


def draw_frame_result(
    frame: np.ndarray,
    result: dict,
    theme: dict | None = None,
    config=None,
    copy: bool = True,
) -> np.ndarray:
    """Draw everything this module produces onto a copy of the frame."""
    image = frame.copy() if copy else frame
    tracks = result.get("tracks", [])
    if config is not None:
        draw_direction_zones(image, config.wrong_way.zones, theme)
        draw_counting_lines(image, config.traffic.counting_lines, theme)
    draw_trajectories(image, tracks, theme)
    draw_tracks(image, tracks, theme)
    draw_analytics_panel(image, result, theme)
    draw_alerts(image, result.get("alerts", []), theme)
    return image
