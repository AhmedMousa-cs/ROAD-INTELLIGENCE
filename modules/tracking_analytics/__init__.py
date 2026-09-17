"""
Road Intelligence — Person 4: Vehicles, Tracking & Traffic Analytics
=====================================================================

Public entry point for the vehicle-intelligence layer of the Road
Intelligence system. Follows the shared module interface:

    from modules.tracking_analytics import TrackingAnalytics

    analytics = TrackingAnalytics("models/yolov8n.pt")
    result = analytics.process(frame)      # or .predict(frame)

``result`` is a dict shaped like ``shared.schemas.empty_frame_result()``.
This module owns the ``tracks``, ``alerts`` and ``analytics`` fields; it
leaves ``detections`` and ``segmentations`` for the other three modules and
never deletes them.

What lives where
----------------
``vehicle_detector.py``   pretrained COCO YOLO, filtered + normalized to the
                          six ontology road-user classes (no new training)
``tracker.py``            ByteTrack-style multi-object tracking, trajectory
                          history, pixel velocity
``traffic_analytics.py``  counts (current / unique / by type), traffic
                          density, line crossings
``wrong_way.py``          configurable direction zones -> wrong_way alerts
``speed_estimation.py``   pixel velocity -> estimated_speed_kmh, only when
                          the scene is calibrated
``road_health.py``        project-defined road condition indicator
``module_config.py``      per-scene configuration (zones, lines, calibration)
``visualization.py``      optional overlay helpers (dashboard may override
                          the theme)
``inference.py``          ``TrackingAnalytics`` orchestrator + CLI
"""

from .class_mapping import (
    COCO_TO_ONTOLOGY,
    EXCLUDED_COCO_CLASSES,
    SUPPORTED_CLASSES,
    VEHICLE_CLASSES,
    normalize_class,
)
from .inference import TrackingAnalytics
from .module_config import (
    CountingLine,
    DirectionZone,
    RoadHealthConfig,
    SceneCalibration,
    SpeedConfig,
    TrackerConfig,
    TrackingAnalyticsConfig,
    TrafficAnalyticsConfig,
    WrongWayConfig,
)
from .road_health import RoadHealthScorer, grade_for, summarize_conditions
from .speed_estimation import SpeedEstimator
from .tracker import Track, VehicleTracker, tracks_to_dicts
from .traffic_analytics import TrafficAnalytics
from .vehicle_detector import ScriptedVehicleDetector, VehicleDetector
from .wrong_way import WrongWayDetector

__all__ = [
    # Primary interface
    "TrackingAnalytics",
    # Configuration
    "TrackingAnalyticsConfig",
    "TrackerConfig",
    "TrafficAnalyticsConfig",
    "SpeedConfig",
    "SceneCalibration",
    "WrongWayConfig",
    "DirectionZone",
    "CountingLine",
    "RoadHealthConfig",
    # Components (usable standalone by the integration layer)
    "VehicleDetector",
    "ScriptedVehicleDetector",
    "VehicleTracker",
    "Track",
    "tracks_to_dicts",
    "TrafficAnalytics",
    "WrongWayDetector",
    "SpeedEstimator",
    "RoadHealthScorer",
    "summarize_conditions",
    "grade_for",
    # Class mapping
    "normalize_class",
    "COCO_TO_ONTOLOGY",
    "EXCLUDED_COCO_CLASSES",
    "VEHICLE_CLASSES",
    "SUPPORTED_CLASSES",
]
