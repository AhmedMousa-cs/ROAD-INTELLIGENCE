"""
modules/obstacles/visualization.py
===================================
Optional drawing helper for road_debris / speed_bump detections. Colors are
NOT hard-coded as a fixed brand palette -- callers (including the eventual
Streamlit dashboard) can pass their own `color_map` to override per-class
colors, per the spec ("the final dashboard may define its own visualization
theme").
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only without opencv installed
    raise ImportError(
        "modules.obstacles.visualization requires opencv-python (see requirements.txt)"
    ) from exc

# Sensible default colors (BGR, since frames are BGR) -- override via
# `color_map` if the dashboard wants a different theme.
DEFAULT_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "road_debris": (0, 165, 255),  # orange
    "speed_bump": (255, 200, 0),  # light blue
}
FALLBACK_COLOR: tuple[int, int, int] = (0, 255, 0)  # green, for any unlisted class


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    color_map: dict[str, tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes + "class conf" labels for a list of detection
    dicts shaped like shared.schemas.Detection.

    Does not mutate `frame` in place -- returns a copy, so callers can keep
    the original frame for further processing (e.g. feeding it to another
    module).
    """
    colors = {**DEFAULT_COLOR_MAP, **(color_map or {})}
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
        color = colors.get(det["class"], FALLBACK_COLOR)
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1 - text_h - baseline, 0)
        cv2.rectangle(out, (x1, label_y1), (x1 + text_w, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1, y1 - baseline // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out
