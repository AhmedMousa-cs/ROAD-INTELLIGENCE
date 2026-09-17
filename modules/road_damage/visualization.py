"""
Optional visualization helpers for road-damage detections.

A debugging/demo aid. The final Streamlit dashboard may define its own theme,
so nothing here bakes in a colour permanently: every drawing function takes a
``theme`` dict and falls back to ``DEFAULT_THEME``, which callers can replace
wholesale or per key:

    from modules.road_damage import visualization as viz

    annotated = viz.draw_detections(frame, result["detections"])
    annotated = viz.draw_detections(
        frame, result["detections"],
        theme={**viz.DEFAULT_THEME, "per_class": {"pothole": (0, 0, 255)}},
    )

Labels render as ``pothole 0.93``. All colours are BGR tuples (OpenCV order),
matching the convention in modules/tracking_analytics/visualization.py.

None of these functions mutate the input frame.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .class_mapping import PROJECT_CLASSES

DEFAULT_THEME: dict = {
    "box": (0, 165, 255),
    "text": (255, 255, 255),
    "text_bg": (0, 0, 0),
    "panel_bg": (32, 32, 32),
    "font_scale": 0.5,
    "thickness": 2,
    "show_confidence": True,
    # Per-class colours. The dashboard can override any subset; anything not
    # listed falls back to "box".
    "per_class": {
        "pothole": (0, 0, 220),
        "alligator_crack": (0, 120, 255),
        "longitudinal_crack": (0, 200, 200),
        "transverse_crack": (200, 200, 0),
    },
    "use_per_class": True,
}


def _theme(theme: dict | None) -> dict:
    resolved = {**DEFAULT_THEME, **(theme or {})}
    resolved["per_class"] = {**DEFAULT_THEME["per_class"], **(theme or {}).get("per_class", {})}
    return resolved


def color_for_class(class_name: str, theme: dict | None = None) -> tuple[int, int, int]:
    resolved = _theme(theme)
    if resolved.get("use_per_class"):
        return tuple(resolved["per_class"].get(class_name, resolved["box"]))
    return tuple(resolved["box"])


def format_label(detection: dict, theme: dict | None = None) -> str:
    """``pothole 0.93`` (or just ``pothole`` when confidence is hidden)."""
    resolved = _theme(theme)
    name = detection.get("class", "?")
    if not resolved.get("show_confidence"):
        return str(name)
    return f"{name} {float(detection.get('confidence', 0.0)):.2f}"


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Sequence[float],
    color: Sequence[int],
    theme: dict | None = None,
) -> None:
    """Draw text with a filled background. Mutates `image` in place."""
    resolved = _theme(theme)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(resolved["font_scale"])
    thickness = max(1, int(resolved["thickness"]) - 1)

    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(origin[0]), int(origin[1])

    # Keep the label inside the frame when the box touches the top edge.
    top = y - height - baseline - 2
    if top < 0:
        top = y + 2
    bottom = top + height + baseline + 2
    right = min(x + width + 4, image.shape[1])

    cv2.rectangle(image, (x, top), (right, bottom), tuple(int(c) for c in color), -1)
    cv2.putText(
        image,
        text,
        (x + 2, bottom - baseline),
        font,
        scale,
        tuple(int(c) for c in resolved["text"]),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[dict],
    theme: dict | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw bounding box + class + confidence for each detection."""
    resolved = _theme(theme)
    image = frame.copy() if copy else frame

    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        color = color_for_class(detection.get("class", ""), resolved)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, int(resolved["thickness"]))
        draw_label(image, format_label(detection, resolved), (x1, y1), color, resolved)

    return image


def draw_summary_panel(
    frame: np.ndarray,
    detections: Sequence[dict],
    theme: dict | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Small per-class count panel in the top-left corner."""
    resolved = _theme(theme)
    image = frame.copy() if copy else frame

    counts = {name: 0 for name in PROJECT_CLASSES}
    for detection in detections:
        name = detection.get("class")
        if name in counts:
            counts[name] += 1

    lines = [f"Road damage: {sum(counts.values())}"]
    lines += [f"  {name}: {count}" for name, count in counts.items() if count]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(resolved["font_scale"])
    line_height = int(22 * max(scale / 0.5, 1.0))
    width = 10 + max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines)
    height = 10 + line_height * len(lines)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), tuple(int(c) for c in resolved["panel_bg"]), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (6, 8 + line_height * (index + 1) - 6),
            font,
            scale,
            tuple(int(c) for c in resolved["text"]),
            1,
            cv2.LINE_AA,
        )
    return image


def draw_frame_result(
    frame: np.ndarray,
    result: dict,
    theme: dict | None = None,
    show_panel: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """Draw a whole shared-schema FrameResult (this module's detections only).

    Detections from other modules are skipped, so this stays safe to call on a
    merged result without drawing Person 3's obstacles in road-damage colours.
    """
    detections = [
        detection
        for detection in result.get("detections", [])
        if detection.get("source") == "road_damage"
    ]
    image = draw_detections(frame, detections, theme, copy=copy)
    if show_panel and detections:
        image = draw_summary_panel(image, detections, theme, copy=False)
    return image
