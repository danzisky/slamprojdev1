"""Map loading and preprocessing utilities for the interactive mapper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class MapManager:
    """Owns map image loading and occupancy grid preprocessing."""

    def __init__(self, map_file: str | Path) -> None:
        self.map_file = Path(map_file)
        self.original_map_image: np.ndarray
        self.occupancy_grid: np.ndarray
        self.grid_height = 0
        self.grid_width = 0
        self.grid_size = 0
        self.load(map_file)

    @staticmethod
    def load_map_image(map_file: Path) -> np.ndarray:
        """Load a map image from disk or raise an explicit error."""
        image = cv2.imread(str(map_file))
        if image is None:
            raise FileNotFoundError(f"Could not load map file: {map_file}")
        return image

    @staticmethod
    def process_map_image(image: np.ndarray) -> np.ndarray:
        """Convert BGR map image to occupancy values used by the planner.

        Output convention:
        - 0: free
        - 128: unknown
        - 255: occupied
        """
        gray_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray_map = cv2.bitwise_not(gray_map)  # Invert so that occupied areas are white (255) and free areas are black (0).
        grid = np.full(gray_map.shape, 128, dtype=np.uint8)
        grid[gray_map < 50] = 0
        grid[gray_map > 130] = 255
        return grid

    def load(self, map_file: str | Path) -> np.ndarray:
        """Reload map file and refresh occupancy/grid dimensions."""
        self.map_file = Path(map_file)
        self.original_map_image = self.load_map_image(self.map_file)
        self.occupancy_grid = self.process_map_image(self.original_map_image)
        self.grid_height, self.grid_width = self.occupancy_grid.shape[:2]

        # Planner utilities in frontier_exploration use one scalar grid_size.
        # We keep the existing behavior and use the map height as that scalar.
        self.grid_size = int(self.grid_height)
        return self.occupancy_grid
