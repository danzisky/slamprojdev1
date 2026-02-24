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
from waveshare_robot_controller import WaveRoverController


class FrontierExplorer:
    def __init__(
        self,
        robot: WaveRoverController,
        camera,
        intrinsics: Dict[str, float],
        grid_size: Tuple[float, float] = (10.0, 15.0),
        grid_resolution: float = 0.05,
        height_range: Tuple[float, float] = (-0.5, 2.0),
        obstacle_threshold: float = 0.5,
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

        Returns a list of dicts: {"angle_deg": float, "imu_heading_deg": float, "image": np.ndarray}
        where angle_deg is the relative yaw offset and imu_heading_deg is the actual IMU heading.
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
            # Get actual IMU heading
            imu_heading = self.robot.get_current_heading()
            
            if frame is not None:
                scans.append({
                    "angle_deg": current_angle,
                    "imu_heading_deg": imu_heading,
                    "image": frame
                })
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
        If debug=True, saves images, pointclouds, and occupancy grids to output directory.
        """
        if not scans:
            return None

        import os
        
        # Setup debug output directory
        if debug:
            output_dir = "output/build_combined_map_debug"
            os.makedirs(output_dir, exist_ok=True)

        grid_w = int(np.ceil(self.grid_size[0] / self.grid_resolution))
        grid_h = int(np.ceil(self.grid_size[1] / self.grid_resolution))
        padding_factor = 2
        combined_shape = (grid_h * padding_factor, grid_w * padding_factor)
        log_odds = np.zeros(combined_shape, dtype=np.float32)
        combined_grid = np.full(combined_shape, 0.5, dtype=np.float32)

        # p_occ = 0.7
        # p_free = 0.3
        p_occ = 0.6
        p_free = 0.2
        l_occ = float(np.log(p_occ / (1.0 - p_occ)))
        l_free = float(np.log(p_free / (1.0 - p_free)))
        l_min, l_max = -4.0, 4.0

        all_grids_in_series = []
        all_combined_grids_in_series = []

        for scan_idx, scan in enumerate(scans):
            rgb = scan["image"]
            angle_deg = scan["angle_deg"]
            real_imu_heading = scan["imu_heading_deg"]

            points = image_to_3d_pointcloud(rgb, self.intrinsics, model=self.model)

            pointcloud_trimmed = _trim_3d_pointcloud(points, self.grid_size[1])
            # print size and bounds of point cloud for debugging
            print(f"Scan {scan_idx + 1}/{len(scans)} | Angle: {angle_deg:.1f}° | IMU Heading: {real_imu_heading:.1f}°")
            print(f"  X bounds: [{pointcloud_trimmed[:, 0].min():.2f}, {pointcloud_trimmed[:, 0].max():.2f}]")
            print(f"  Y bounds: [{pointcloud_trimmed[:, 1].min():.2f}, {pointcloud_trimmed[:, 1].max():.2f}]")
            print(f"  Z bounds: [{pointcloud_trimmed[:, 2].min():.2f}, {pointcloud_trimmed[:, 2].max():.2f}]")

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
            warped_grid = self._warp_to_combined(processed_grid, real_imu_heading, combined_grid.shape)

            occupied_mask = (warped_grid == 1)
            free_mask = (warped_grid == 0)

            log_odds[occupied_mask] += l_occ
            log_odds[free_mask] += l_free
            np.clip(log_odds, l_min, l_max, out=log_odds)
            combined_grid = 1.0 / (1.0 + np.exp(-log_odds))

            if debug:
                # Save image
                cv2.imwrite(f"{output_dir}/scan_{scan_idx:03d}_angle_{real_imu_heading:06.1f}_image.jpg", rgb)
                
                # Save pointcloud as PLY
                pc_path = f"{output_dir}/scan_{scan_idx:03d}_angle_{real_imu_heading:06.1f}_pointcloud.ply"
                with open(pc_path, 'w') as f:
                    f.write('ply\n')
                    f.write('format ascii 1.0\n')
                    f.write(f'element vertex {len(pointcloud_trimmed)}\n')
                    f.write('property float x\n')
                    f.write('property float y\n')
                    f.write('property float z\n')
                    f.write('end_header\n')
                    for p in pointcloud_trimmed:
                        f.write(f'{p[0]} {p[1]} {p[2]}\n')
                
                # Save individual occupancy grid
                grid_vis = visualize_occupancy_grid(processed_grid)
                cv2.imwrite(f"{output_dir}/scan_{scan_idx:03d}_angle_{real_imu_heading:06.1f}_grid.jpg", grid_vis)
                
                all_grids_in_series.append(grid_vis)
                all_combined_grids_in_series.append(visualize_occupancy_grid(combined_grid))

        if debug:
            xvis = cv2.hconcat(all_grids_in_series)
            yvis = cv2.hconcat(all_combined_grids_in_series)
            cv2.imshow("All Scans", xvis)
            cv2.imshow("All Combined Grids", yvis)
            cv2.waitKey(500)

        self._combined_grid = combined_grid
        
        # Add visualization with robot center and start/stop from actual IMU headings
        if debug and len(scans) > 0:
            combined_vis = visualize_occupancy_grid(combined_grid)
            
            center_h, center_w = combined_grid.shape[0] // 2, combined_grid.shape[1] // 2
            
            # Draw robot center
            cv2.circle(combined_vis, (center_w, center_h), 8, (0, 0, 255), -1)
            cv2.putText(combined_vis, "Robot Center", (center_w + 10, center_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Use actual IMU headings from first and last scans
            first_scan_heading = scans[0].get("imu_heading_deg", 0.0)
            last_scan_heading = scans[-1].get("imu_heading_deg", 0.0)
            
            # Draw start heading ray
            ray_length = 100
            rotation_rad = np.radians(first_scan_heading)
            start_x = int(center_w + ray_length * np.sin(rotation_rad))
            start_y = int(center_h - ray_length * np.cos(rotation_rad))
            cv2.line(combined_vis, (center_w, center_h), (start_x, start_y), (255, 0, 0), 2)
            cv2.putText(combined_vis, f"Start: {first_scan_heading:.1f}°", (start_x + 10, start_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw stop heading ray
            rotation_rad = np.radians(last_scan_heading)
            stop_x = int(center_w + ray_length * np.sin(rotation_rad))
            stop_y = int(center_h - ray_length * np.cos(rotation_rad))
            cv2.line(combined_vis, (center_w, center_h), (stop_x, stop_y), (0, 255, 0), 2)
            cv2.putText(combined_vis, f"Stop: {last_scan_heading:.1f}°", (stop_x + 10, stop_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated visualization
            cv2.imwrite(f"output/build_combined_map_debug/final_combined_map_annotated.jpg", combined_vis)
            cv2.imshow("Final Map with Annotations", combined_vis)
            cv2.waitKey(500)
        
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
        self.robot.turn_degrees_PRECISE(degrees=degrees, min_speed=0.26, max_speed=0.3)

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
