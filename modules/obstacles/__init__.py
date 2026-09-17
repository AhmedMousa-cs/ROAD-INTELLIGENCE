"""
modules/obstacles
==================
Person 3's module: road-debris (TACO) + speed-bump (Roboflow Speed Bump
Detection v10) detection, behind one unified interface.

Public interface (per shared/README.md's model interface contract):

    from modules.obstacles import RoadObstacleModel

    model = RoadObstacleModel(
        "models/debris_best.pt",
        "models/speed_bump_best.pt",
    )
    result = model.predict(frame)   # frame: numpy.ndarray, BGR, HxWx3

`result` is shaped like shared.schemas.empty_frame_result(), with
`detections` populated (each a plain dict with "class", "confidence",
"bbox", "source") and `analytics.obstacle_count` / `analytics.speed_bump_count`
filled in.

DebrisModel and SpeedBumpModel are also exported individually for anyone
who wants to run just one detector without loading both.
"""

from .debris_model import DebrisModel
from .obstacle_model import RoadObstacleModel
from .speed_bump_model import SpeedBumpModel

__all__ = ["RoadObstacleModel", "DebrisModel", "SpeedBumpModel"]
