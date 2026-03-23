"""Small regression tests for the local interactive mapper planner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    SLAM_ROVER_DIR = PACKAGE_DIR.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from interactive_mapper.map_manager import MapManager  # type: ignore
    from interactive_mapper.planner import GridPathPlanner  # type: ignore
else:
    from .map_manager import MapManager
    from .planner import GridPathPlanner


class MapManagerTests(unittest.TestCase):
    def test_load_map_converts_image_to_occupancy_grid(self) -> None:
        image = np.full((6, 6, 3), 255, dtype=np.uint8)
        image[2, 2] = (0, 0, 0)
        image[1, 1] = (160, 160, 160)

        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "map.png"
            saved = cv2.imwrite(str(map_path), image)
            self.assertTrue(saved)

            manager = MapManager(map_path)

        self.assertEqual(manager.occupancy_grid.shape, (6, 6))
        self.assertEqual(manager.grid_height, 6)
        self.assertEqual(manager.grid_width, 6)
        self.assertEqual(manager.occupancy_grid[0, 0], 0)
        self.assertEqual(manager.occupancy_grid[2, 2], 255)
        self.assertEqual(manager.occupancy_grid[1, 1], 128)


class GridPathPlannerTests(unittest.TestCase):
    def test_plan_path_returns_direct_path_when_clear(self) -> None:
        occupancy = np.zeros((10, 10), dtype=np.uint8)
        planner = GridPathPlanner(grid_size=10, resolution=1.0, min_wall_distance=0.0, occupancy_grid=occupancy)
        planner.robot_pos = np.array([1, 1, 0.0], dtype=np.float32)

        path = planner.plan_path((8, 1))

        self.assertEqual(path, [(1, 1), (8, 1)])

    def test_plan_path_returns_none_when_wall_fully_blocks_route(self) -> None:
        occupancy = np.zeros((10, 10), dtype=np.uint8)
        occupancy[:, 5] = 255
        planner = GridPathPlanner(grid_size=10, resolution=1.0, min_wall_distance=0.0, occupancy_grid=occupancy)
        planner.robot_pos = np.array([1, 1, 0.0], dtype=np.float32)

        path = planner.plan_path((8, 1))

        self.assertIsNone(path)

    def test_plan_path_adjusts_goal_when_target_is_blocked(self) -> None:
        occupancy = np.zeros((10, 10), dtype=np.uint8)
        occupancy[1, 8] = 255
        planner = GridPathPlanner(grid_size=10, resolution=1.0, min_wall_distance=0.0, occupancy_grid=occupancy)
        planner.robot_pos = np.array([1, 1, 0.0], dtype=np.float32)

        path = planner.plan_path((8, 1))

        self.assertIsNotNone(path)
        assert path is not None
        self.assertNotEqual(path[-1], (8, 1))
        self.assertEqual(path[0], (1, 1))
        self.assertIn(path[-1], {(7, 0), (7, 1), (7, 2), (8, 0), (8, 2), (9, 0), (9, 1), (9, 2)})


if __name__ == "__main__":
    unittest.main()