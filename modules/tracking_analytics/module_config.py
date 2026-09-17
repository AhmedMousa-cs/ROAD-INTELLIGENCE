"""
Runtime configuration for the tracking_analytics module.

``shared/config.py`` holds values used by MORE THAN ONE module (confidence
thresholds, density thresholds, road-health weights, device). This file holds
values that are specific to *this* module and to *this scene* - direction
zones, counting lines, camera calibration - which cannot live in shared
config because they differ per camera.

Everything defaults to the shared values, so ``TrackingAnalytics(model_path)``
with no config behaves exactly as the shared contract specifies. Nothing here
is hard-coded at a call site: a scene is described by a JSON/YAML file and
loaded with ``TrackingAnalyticsConfig.from_yaml(path)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shared.config import InferenceConfig

from .road_health import RoadHealthConfig
from .speed_estimation import SceneCalibration, SpeedConfig
from .tracker import TrackerConfig
from .traffic_analytics import CountingLine, TrafficAnalyticsConfig
from .wrong_way import DirectionZone, WrongWayConfig


@dataclass
class TrackingAnalyticsConfig:
    """Everything ``TrackingAnalytics`` needs, in one object."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    traffic: TrafficAnalyticsConfig = field(default_factory=TrafficAnalyticsConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    wrong_way: WrongWayConfig = field(default_factory=WrongWayConfig)
    road_health: RoadHealthConfig = field(default_factory=RoadHealthConfig)

    # Video/stream properties.
    fps: float = 30.0

    # Output shaping.
    include_trajectory_in_output: bool = True
    trajectory_output_points: int = 30

    def __post_init__(self) -> None:
        # Keep the frame rate consistent everywhere it is used.
        self.tracker.frame_rate = self.fps

    # -- loading --------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrackingAnalyticsConfig":
        if not data:
            return cls()

        inference_data = data.get("inference", {})
        inference = InferenceConfig(
            **{
                key: inference_data[key]
                for key in ("confidence_threshold", "iou_threshold", "image_size", "device")
                if key in inference_data
            }
        )

        tracker_data = data.get("tracker", {})
        tracker = TrackerConfig(
            **{
                key: value
                for key, value in tracker_data.items()
                if key in TrackerConfig.__dataclass_fields__
            }
        )

        speed_data = dict(data.get("speed", {}))
        calibration = SceneCalibration.from_dict(speed_data.pop("calibration", None))
        speed = SpeedConfig(
            calibration=calibration,
            **{
                key: value
                for key, value in speed_data.items()
                if key in SpeedConfig.__dataclass_fields__ and key != "calibration"
            },
        )

        config = cls(
            inference=inference,
            tracker=tracker,
            traffic=TrafficAnalyticsConfig.from_dict(data.get("traffic")),
            speed=speed,
            wrong_way=WrongWayConfig.from_dict(data.get("wrong_way")),
            road_health=RoadHealthConfig.from_dict(data.get("road_health")),
            fps=float(data.get("fps", 30.0)),
            include_trajectory_in_output=bool(
                data.get("include_trajectory_in_output", True)
            ),
            trajectory_output_points=int(data.get("trajectory_output_points", 30)),
        )
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrackingAnalyticsConfig":
        import yaml

        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def coerce(
        cls, config: "TrackingAnalyticsConfig | InferenceConfig | dict | None"
    ) -> "TrackingAnalyticsConfig":
        """Accept any of the config forms the shared interface allows.

        The shared contract says ``__init__(model_path, config=None)`` where
        config may be an ``InferenceConfig`` or a plain dict; this module also
        accepts its own richer config object.
        """
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, InferenceConfig):
            return cls(inference=config)
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError(f"Unsupported config type: {type(config)!r}")


__all__ = [
    "TrackingAnalyticsConfig",
    "TrackerConfig",
    "TrafficAnalyticsConfig",
    "SpeedConfig",
    "SceneCalibration",
    "WrongWayConfig",
    "DirectionZone",
    "CountingLine",
    "RoadHealthConfig",
]
