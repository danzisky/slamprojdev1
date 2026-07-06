"""Chair-based particle filter localization package for slam_rover."""

from .association import AssociationResult, JointCompatibilityAssociator
from .detector import ChairObservationDetector
from .map_data import ChairLocalizationMap
from .particle_filter import ParticleFilterLocalizer
from .tof import ToFConfig, ToFIntegrator
from .types import (
    ChairLandmark,
    ChairObservation,
    LocalizationEstimate,
    LocalizationUpdate,
    MotionCommand,
    Pose2D,
    PredictedLandmarkMeasurement,
    ToFFrame,
)
from .visualization import LocalizationVisualizer, VisualizationConfig

__all__ = [
    "AssociationResult",
    "ChairLandmark",
    "ChairLocalizationMap",
    "ChairObservation",
    "ChairObservationDetector",
    "JointCompatibilityAssociator",
    "LocalizationEstimate",
    "LocalizationUpdate",
    "LocalizationVisualizer",
    "MotionCommand",
    "ParticleFilterLocalizer",
    "Pose2D",
    "PredictedLandmarkMeasurement",
    "ToFConfig",
    "ToFFrame",
    "ToFIntegrator",
    "VisualizationConfig",
]