"""Typed data containers for the localization package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Pose2D:
    """Robot pose in map coordinates, expressed in meters and radians."""

    x_m: float
    y_m: float
    heading_rad: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.heading_rad)


@dataclass(frozen=True)
class MotionCommand:
    """Odometry increment applied before the measurement update."""

    forward_m: float = 0.0
    turn_rad: float = 0.0


@dataclass(frozen=True)
class ChairLandmark:
    """Known chair landmark in the map frame."""

    landmark_id: int
    x_m: float
    y_m: float
    label: str = "chair"


@dataclass(frozen=True)
class ChairObservation:
    """Range-bearing chair observation extracted from the current image."""

    range_m: float
    bearing_rad: float
    confidence: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "detector"
    range_std_m: Optional[float] = None
    bearing_std_rad: Optional[float] = None


@dataclass(frozen=True)
class PredictedLandmarkMeasurement:
    """Expected measurement to a known landmark from a candidate particle pose."""

    landmark: ChairLandmark
    range_m: float
    bearing_rad: float


@dataclass
class LocalizationEstimate:
    """Summary of the current particle cloud after an update."""

    pose: Pose2D
    position_std_m: Tuple[float, float]
    heading_std_rad: float
    effective_sample_size: float
    matched_landmarks: int
    observation_count: int


@dataclass
class LocalizationUpdate:
    """Full update result including the extracted observations."""

    estimate: LocalizationEstimate
    observations: Tuple[ChairObservation, ...]
    matches: Tuple[Tuple[int, int], ...]
    best_particle_index: int