"""
Simple frontier exploration helper:
1) Scan N times over a specified angle.
2) Combine scans into a single occupancy grid.
3) Find frontiers and pick the best goal.
4) Navigate to that frontier.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mapper import (
    get_model,
    image_to_3d_pointcloud,
    pointcloud_to_occupancy_grid,
    _process_occupancy_grid,
    _trim_3d_pointcloud,
    visualize_occupancy_grid,
)
from frontier_utils import find_frontiers, extract_frontier_goals


class FrontierExplorer:
    def __init__(
        self,
        robot,
        camera,
        intrinsics: Dict[str, float],
        grid_size: Tuple[float, float] = (10.0, 15.0),
        grid_resolution: float = 0.05,
        height_range: Tuple[float, float] = (-0.5, 2.0),
        obstacle_threshold: float = 0.8,
        encoder: str = "vitl",
        dataset: str = "hypersim",
        max_depth: float = 20.0,
        model=None,
    ):
        self.robot = robot
        self.camera = camera
        self.intrinsics = intrinsics
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.height_range = height_range
        self.obstacle_threshold = obstacle_threshold
        self.model = model or get_model(encoder=encoder, dataset=dataset, max_depth=max_depth)

        self._combined_grid: Optional[np.ndarray] = None
        self._last_frontier_mask: Optional[np.ndarray] = None

    def scan_n_times(self, num_scans: int, total_angle_deg: float, settle_time: float = 0.2) -> List[Dict]:
        """
        Rotate the robot to capture multiple frames across a sweep.

        Returns a list of dicts: {"angle_deg": float, "image": np.ndarray}
        where angle_deg is the relative yaw from the starting heading.
        """
        if num_scans <= 0:
            return []

        if num_scans == 1:
            step_deg = 0.0
        else:
            step_deg = total_angle_deg / float(num_scans - 1)

        half_sweep = total_angle_deg / 2.0

        # Move to the starting edge of the sweep.
        if half_sweep != 0:
            self._turn_degrees(-half_sweep)
            time.sleep(settle_time)

        scans: List[Dict] = []
        current_angle = -half_sweep

        for i in range(num_scans):
            frame = self.camera.get_frame()
            if frame is not None:
                scans.append({"angle_deg": current_angle, "image": frame})
            else:
                print(f"Warning: No frame for scan {i + 1}/{num_scans}")

            if i < num_scans - 1:
                self._turn_degrees(step_deg)
                current_angle += step_deg
                time.sleep(settle_time)

        return scans

    def build_combined_map(self, scans: List[Dict], debug: bool = False) -> Optional[np.ndarray]:
        """
        Build a combined occupancy grid from scan images and angles.
        """
        if not scans:
            return None

        grid_w = int(np.ceil(self.grid_size[0] / self.grid_resolution))
        grid_h = int(np.ceil(self.grid_size[1] / self.grid_resolution))
        padding_factor = 2
        combined_grid = np.full((grid_h * padding_factor, grid_w * padding_factor), -1, dtype=np.int8)

        all_grids_in_series = []
        all_combined_grids_in_series = []

        for scan in scans:
            rgb = scan["image"]
            angle_deg = scan["angle_deg"]

            points = image_to_3d_pointcloud(rgb, self.intrinsics, model=self.model)
            pointcloud_trimmed = _trim_3d_pointcloud(points, self.grid_size[1])

            # Align with mapper convention: forward-positive Z, right-positive X.
            pointcloud_trimmed[:, 2] = -pointcloud_trimmed[:, 2]
            pointcloud_trimmed[:, 0] = -pointcloud_trimmed[:, 0]

            grid = pointcloud_to_occupancy_grid(
                pointcloud=pointcloud_trimmed,
                grid_size=self.grid_size,
                grid_resolution=self.grid_resolution,
                height_range=self.height_range,
                obstacle_threshold=self.obstacle_threshold,
                use_data_bounds=False,
            )

            processed_grid = _process_occupancy_grid(grid)
            warped_grid = self._warp_to_combined(processed_grid, angle_deg, combined_grid.shape)

            occupied_mask = (warped_grid == 1)
            free_mask = (warped_grid == 0)
            # combined_grid[occupied_mask] = 1

            # combine the current scan's warped grid into the overall combined grid, ensuring that free cells only overwrite unknown cells, while occupied cells overwrite everything.
            combined_grid[occupied_mask] = 1
            combined_grid[free_mask & (combined_grid != 1)] = 0

            if debug:
                all_grids_in_series.append(visualize_occupancy_grid(grid))
                all_combined_grids_in_series.append(visualize_occupancy_grid(combined_grid))

        if debug:
                xvis = cv2.hconcat(all_grids_in_series)
                yvis = cv2.hconcat(all_combined_grids_in_series)
                cv2.imshow("All Scans", xvis)
                cv2.imshow("All Combined Grids", yvis)
                cv2.waitKey(500)    
        self._combined_grid = combined_grid
        return combined_grid

    def select_best_frontier(
        self,
        combined_grid: np.ndarray,
        min_region_size: int = 20,
        criteria: str = "closest",
    ) -> Optional[Tuple[int, int]]:
        """
        Return the best frontier goal as (row, col) in grid coordinates.

        criteria options: "closest", "farthest", "largest", "smallest".
        """
        frontier_mask = find_frontiers(combined_grid)
        self._last_frontier_mask = frontier_mask

        regions = self._extract_frontier_regions(frontier_mask, min_region_size=min_region_size)
        if not regions:
            return None

        center = (combined_grid.shape[0] // 2, combined_grid.shape[1] // 2)
        criteria = criteria.lower().strip()

        if criteria == "closest":
            best = min(regions, key=lambda r: self._grid_distance(center, r["goal"]))
        elif criteria == "farthest":
            best = max(regions, key=lambda r: self._grid_distance(center, r["goal"]))
        elif criteria == "largest":
            best = max(regions, key=lambda r: (r["area"], -self._grid_distance(center, r["goal"])))
        elif criteria == "smallest":
            best = min(regions, key=lambda r: (r["area"], self._grid_distance(center, r["goal"])))
        else:
            raise ValueError("criteria must be one of: closest, farthest, largest, smallest")

        return best["goal"]

    def navigate_to_frontier(
        self,
        goal: Tuple[int, int],
        current_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> Optional[np.ndarray]:
        """
        Convert a frontier goal to meters and call the robot's move_to.
        """
        if goal is None:
            return None

        target_pos = self._grid_to_robot_xy(goal, self._combined_grid)
        if target_pos is None:
            return None

        return self.robot.move_to(target_pos, current_pos)

    def _turn_degrees(self, degrees: float):
        """
        Turn the robot by a specified number of degrees (positive = counter-clockwise).
        """
        self.robot.turn_degrees_PRECISE(degrees=degrees, min_speed=0.33, max_speed=0.4)

    def _warp_to_combined(
        self,
        occupancy_grid: np.ndarray,
        angle_deg: float,
        combined_shape: Tuple[int, int],
        anchor: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Rotate an occupancy grid around the robot anchor, then place it into the combined grid.
        """
        height, width = occupancy_grid.shape
        if anchor is None:
            anchor = (width / 2.0, 0.0)

        combined_h, combined_w = combined_shape
        combined_center = (combined_w / 2.0, combined_h / 2.0)

        rot_mat = cv2.getRotationMatrix2D(anchor, angle_deg, 1.0)
        rot_mat[0, 2] += combined_center[0] - anchor[0]
        rot_mat[1, 2] += combined_center[1] - anchor[1]

        return cv2.warpAffine(
            occupancy_grid,
            rot_mat,
            (combined_w, combined_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )

    @staticmethod
    def _extract_frontier_regions(
        frontier_mask: np.ndarray,
        min_region_size: int = 20,
    ) -> List[Dict[str, object]]:
        """
        Extract frontier regions as goals with areas.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mask.astype(np.uint8),
            connectivity=8,
        )

        regions: List[Dict[str, object]] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_region_size:
                continue
            cy, cx = int(round(centroids[i][1])), int(round(centroids[i][0]))
            regions.append({"goal": (cy, cx), "area": area})

        return regions

    def _grid_to_robot_xy(
        self,
        goal: Tuple[int, int],
        combined_grid: Optional[np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        """
        Convert a grid cell to robot-centric meters (x right, y forward).
        """
        if combined_grid is None:
            return None

        row, col = goal
        center_row = combined_grid.shape[0] // 2
        center_col = combined_grid.shape[1] // 2

        dx = (col - center_col) * self.grid_resolution
        dy = (center_row - row) * self.grid_resolution

        return (dx, dy)

    @staticmethod
    def _grid_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))
