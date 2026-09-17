"""
modules/road_segmentation/visualization.py

Optional drawing utility. Colors here are DEFAULTS ONLY — the final
Streamlit dashboard may define its own theme and call
`draw_segmentations(frame, segs, colors={...})` with an override, per
spec section "Do not permanently hard-code colors".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    # BGR
    "pothole": (0, 0, 255),
    "surface_damage": (0, 165, 255),
    "road_patch": (255, 191, 0),
    "ravelling": (0, 255, 255),
    "alligator_crack": (255, 0, 255),
}
_FALLBACK_COLOR = (0, 255, 0)


def draw_segmentations(
    frame: np.ndarray,
    segmentations: List[Dict[str, Any]],
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    alpha: float = 0.45,
    draw_labels: bool = True,
) -> np.ndarray:
    """Draw filled, semi-transparent polygon overlays + "class conf" labels.

    Only understands the {"format": "polygon", "points": [[x,y], ...]}
    mask payload produced by road_segmentation_model.py.
    """
    palette = {**DEFAULT_COLORS, **(colors or {})}
    vis = frame.copy()
    overlay = frame.copy()

    for seg in segmentations:
        cls = seg.get("class", "unknown")
        conf = seg.get("confidence", 0.0)
        mask = seg.get("mask") or {}
        color = palette.get(cls, _FALLBACK_COLOR)

        if mask.get("format") == "polygon" and mask.get("points"):
            pts = np.array(mask["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

            if draw_labels:
                x, y = pts[0][0]
                label = f"{cls} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (int(x), int(y) - th - 6), (int(x) + tw + 4, int(y)), color, -1)
                cv2.putText(
                    vis, label, (int(x) + 2, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
