"""Map loading and visibility queries for chair-based localization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .math_utils import wrap_to_pi
from .types import ChairLandmark, Pose2D, PredictedLandmarkMeasurement


class ChairLocalizationMap:
    """
    Occupancy map wrapper used by the particle filter.

    The map frame follows the image axes of the occupancy grid:
    x grows to the right, y grows downward, and heading grows toward +y.
    """

    def __init__(
        self,
        occupancy_grid: np.ndarray,
        resolution_m_per_px: float,
        landmarks: Optional[Sequence[ChairLandmark]] = None,
        map_path: Optional[str] = None,
    ):
        self.occupancy_grid = occupancy_grid.astype(np.uint8, copy=True)
        self.resolution_m_per_px = float(resolution_m_per_px)
        self.map_path = map_path

        self.height_px, self.width_px = self.occupancy_grid.shape
        self.size_m = (
            self.width_px * self.resolution_m_per_px,
            self.height_px * self.resolution_m_per_px,
        )
        self.landmarks: List[ChairLandmark] = list(landmarks or [])

        self._free_pixels_rc = np.argwhere(self.occupancy_grid == 0)
        if self._free_pixels_rc.size == 0:
            raise ValueError("The supplied map contains no free cells.")

    @classmethod
    def from_files(
        cls,
        map_path: str,
        landmarks_path: Optional[str] = None,
        resolution_m_per_px: Optional[float] = None,
        free_threshold: int = 20,
        occupied_threshold: int = 150,
    ) -> "ChairLocalizationMap":
        """Create a map instance from an occupancy image and an optional landmarks file."""
        map_image = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
        if map_image is None:
            raise FileNotFoundError(f"Could not load map image: {map_path}")

        occupancy_grid = cls._process_map(
            map_image,
            free_threshold=free_threshold,
            occupied_threshold=occupied_threshold,
        )

        landmarks: List[ChairLandmark] = []
        resolution_from_landmarks = None
        if landmarks_path is not None:
            landmarks, resolution_from_landmarks = cls._load_landmarks(
                landmarks_path,
                resolution_m_per_px,
            )

        if resolution_m_per_px is None:
            resolution_m_per_px = resolution_from_landmarks or 0.0197

        return cls(
            occupancy_grid=occupancy_grid,
            resolution_m_per_px=resolution_m_per_px,
            landmarks=landmarks,
            map_path=map_path,
        )

    @staticmethod
    def _process_map(
        grayscale_map: np.ndarray,
        free_threshold: int,
        occupied_threshold: int,
    ) -> np.ndarray:
        """Convert a grayscale occupancy image into free/unknown/occupied classes."""
        grid = np.full(grayscale_map.shape, 127, dtype=np.uint8)
        grid[grayscale_map < free_threshold] = 0
        grid[grayscale_map > occupied_threshold] = 255
        return grid

    @classmethod
    def _load_landmarks(
        cls,
        landmarks_path: str,
        resolution_m_per_px: Optional[float],
    ) -> Tuple[List[ChairLandmark], Optional[float]]:
        with open(landmarks_path, "r", encoding="utf-8") as file_handle:
            raw_data = json.load(file_handle)

        if isinstance(raw_data, dict):
            raw_landmarks = raw_data.get("landmarks", [])
            file_resolution = raw_data.get("resolution")
        else:
            raw_landmarks = raw_data
            file_resolution = None

        chosen_resolution = resolution_m_per_px
        if chosen_resolution is None:
            chosen_resolution = file_resolution or 0.0197

        landmarks: List[ChairLandmark] = []
        for default_id, entry in enumerate(raw_landmarks):
            if isinstance(entry, dict):
                x_px = float(entry["x"])
                y_px = float(entry["y"])
                landmark_id = int(entry.get("id", default_id))
                label = str(entry.get("label", "chair"))
            else:
                x_px = float(entry[0])
                y_px = float(entry[1])
                landmark_id = default_id
                label = "chair"

            landmarks.append(
                ChairLandmark(
                    landmark_id=landmark_id,
                    x_m=x_px * chosen_resolution,
                    y_m=y_px * chosen_resolution,
                    label=label,
                )
            )

        return landmarks, file_resolution

    def pixel_to_world(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Convert map pixel coordinates to meters."""
        return (
            float(x_px) * self.resolution_m_per_px,
            float(y_px) * self.resolution_m_per_px,
        )

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert map-frame meters to nearest pixel indices."""
        x_px = int(round(x_m / self.resolution_m_per_px))
        y_px = int(round(y_m / self.resolution_m_per_px))
        return x_px, y_px

    def is_in_bounds(self, x_m: float, y_m: float) -> bool:
        """Return True if a world coordinate lies within the occupancy image."""
        x_px, y_px = self.world_to_pixel(x_m, y_m)
        return 0 <= x_px < self.width_px and 0 <= y_px < self.height_px

    def is_free(self, x_m: float, y_m: float) -> bool:
        """Return True if the location is a free cell in the map."""
        x_px, y_px = self.world_to_pixel(x_m, y_m)
        if not (0 <= x_px < self.width_px and 0 <= y_px < self.height_px):
            return False
        return bool(self.occupancy_grid[y_px, x_px] == 0)

    def are_free(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        """Vectorized free-space query for particle arrays."""
        x_px = np.rint(np.asarray(x_m) / self.resolution_m_per_px).astype(np.int32)
        y_px = np.rint(np.asarray(y_m) / self.resolution_m_per_px).astype(np.int32)

        in_bounds = (
            (x_px >= 0)
            & (x_px < self.width_px)
            & (y_px >= 0)
            & (y_px < self.height_px)
        )
        free = np.zeros_like(in_bounds, dtype=bool)
        if np.any(in_bounds):
            free[in_bounds] = self.occupancy_grid[y_px[in_bounds], x_px[in_bounds]] == 0
        return free

    def sample_free_positions(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample free-space positions uniformly from the occupancy image."""
        indices = rng.integers(0, len(self._free_pixels_rc), size=int(count))
        samples_rc = self._free_pixels_rc[indices]
        x_m = samples_rc[:, 1].astype(np.float64) * self.resolution_m_per_px
        y_m = samples_rc[:, 0].astype(np.float64) * self.resolution_m_per_px
        return np.column_stack((x_m, y_m))

    def line_of_sight(self, start_x_m: float, start_y_m: float, end_x_m: float, end_y_m: float) -> bool:
        """
        Check line of sight between two world points using Bresenham stepping.

        The endpoint is not treated as an obstacle because chair landmarks live on
        occupied cells by construction.
        """
        x0, y0 = self.world_to_pixel(start_x_m, start_y_m)
        x1, y1 = self.world_to_pixel(end_x_m, end_y_m)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx - dy

        current_x, current_y = x0, y0
        while True:
            if not (0 <= current_x < self.width_px and 0 <= current_y < self.height_px):
                return False

            if current_x == x1 and current_y == y1:
                return True

            if self.occupancy_grid[current_y, current_x] == 255:
                return False

            doubled_error = 2 * error
            if doubled_error > -dy:
                error -= dy
                current_x += step_x
            if doubled_error < dx:
                error += dx
                current_y += step_y

    def raycast_distance(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float,
        max_range_m: float,
        step_m: Optional[float] = None,
    ) -> float:
        """Raycast from a world pose until an obstacle or map boundary is reached."""
        step_m = float(step_m or max(self.resolution_m_per_px, 0.02))
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)

        distance_m = 0.0
        while distance_m <= max_range_m:
            probe_x = x_m + distance_m * cos_heading
            probe_y = y_m + distance_m * sin_heading
            if not self.is_in_bounds(probe_x, probe_y):
                return distance_m
            probe_px, probe_py = self.world_to_pixel(probe_x, probe_y)
            if self.occupancy_grid[probe_py, probe_px] == 255:
                return distance_m
            distance_m += step_m

        return float(max_range_m)

    def visible_landmarks(
        self,
        pose: Pose2D,
        max_range_m: Optional[float] = None,
        half_fov_rad: Optional[float] = None,
    ) -> List[PredictedLandmarkMeasurement]:
        """Return the landmarks that are visible from the supplied pose."""
        visible: List[PredictedLandmarkMeasurement] = []
        for landmark in self.landmarks:
            dx = landmark.x_m - pose.x_m
            dy = landmark.y_m - pose.y_m
            range_m = math.hypot(dx, dy)
            if max_range_m is not None and range_m > max_range_m:
                continue

            bearing_rad = wrap_to_pi(math.atan2(dy, dx) - pose.heading_rad)
            if half_fov_rad is not None and abs(bearing_rad) > half_fov_rad:
                continue

            if not self.line_of_sight(pose.x_m, pose.y_m, landmark.x_m, landmark.y_m):
                continue

            visible.append(
                PredictedLandmarkMeasurement(
                    landmark=landmark,
                    range_m=range_m,
                    bearing_rad=bearing_rad,
                )
            )

        visible.sort(key=lambda prediction: prediction.range_m)
        return visible