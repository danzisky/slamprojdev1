"""Main interactive mapper controller class."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .map_manager import MapManager
from .planner import GridPathPlanner, grid_to_robot_frame, robot_to_grid_frame
from .runtime_imports import WaveRoverController
from .sensor_hub import SensorHub
from .types import SensorCallback, SensorSnapshot


class InteractiveMapper:
    """Load a saved map, plan paths, and optionally execute them on the robot."""

    def __init__(
        self,
        map_file: str | Path,
        start_pos: Tuple[int, int, float],
        robot_ip: str = "192.168.1.24",
        resolution: float = 0.0197,
        min_wall_distance: float = 0.3,
        speed_multiplier: float = 1.0,
        use_fused_internal_yaw: bool = False,
        auto_execute_path: bool = True,
        window_name: str = "Interactive Mapper",
    ) -> None:
        self.resolution = float(resolution)
        self.min_wall_distance = float(min_wall_distance)
        self.auto_execute_path = bool(auto_execute_path)
        self.window_name = window_name

        # Map manager owns image loading and occupancy conversion.
        self.map_manager = MapManager(map_file)

        # Robot controller is optional at runtime until connect_robot() is called.
        self.robot = WaveRoverController(
            robot_ip=robot_ip,
            speed_multiplier=speed_multiplier,
            external_imu=None,
            use_fused_internal_yaw=use_fused_internal_yaw,
        )

        # Sensor hub handles camera/IMU attachments and the latest sensor snapshot.
        self.sensor_hub = SensorHub(self.robot)

        # Local path planner owned by this package.
        self.planner = GridPathPlanner(
            grid_size=self.map_manager.grid_size,
            resolution=self.resolution,
            min_wall_distance=self.min_wall_distance,
            occupancy_grid=self.map_manager.occupancy_grid,
        )

        # Runtime state used by planning, drawing, and execution.
        self.robot_pos_meters = np.zeros(2, dtype=np.float32)
        self.robot_heading_rad = 0.0
        self.navigation_history: list[Tuple[int, int]] = []
        self.current_path: Optional[list[Tuple[int, int]]] = None
        self.target_grid_pos: Optional[Tuple[int, int]] = None
        self.current_waypoint_idx = 0
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.is_moving = False
        self._window_initialized = False
        self.start_pos = start_pos

        self.set_robot_pose(start_pos[0], start_pos[1], start_pos[2], record_history=True)

    @property
    def map_file(self) -> Path:
        return self.map_manager.map_file

    @property
    def occupancy_grid(self) -> np.ndarray:
        return self.map_manager.occupancy_grid

    @staticmethod
    def _normalize_angle_rad(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    @staticmethod
    def _coerce_motion_vector(motion_raw: Any) -> np.ndarray:
        """Normalize controller output into [dx, dy, dtheta]."""
        motion = np.asarray(motion_raw if motion_raw is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
        if motion.size < 3:
            motion = np.pad(motion, (0, 3 - motion.size))
        return motion[:3]

    def load_map(self, map_file: str | Path) -> np.ndarray:
        """Reload map file and update planner occupancy."""
        occupancy = self.map_manager.load(map_file)
        self.planner.grid_size = self.map_manager.grid_size
        self.planner.occupancy_grid = occupancy.copy()
        return occupancy

    def set_robot_pose(self, grid_x: int, grid_y: int, heading_deg: float, record_history: bool = False) -> None:
        """Set robot pose used by planner and visualization."""
        heading_rad = math.radians(float(heading_deg))
        grid_pos = np.array([float(grid_x), float(grid_y)], dtype=np.float32)

        # Convert map pixel position into robot-centric metric coordinates.
        self.robot_pos_meters = grid_to_robot_frame(
            grid_pos,
            self.map_manager.grid_size,
            self.resolution,
        ).astype(np.float32)

        self.robot_heading_rad = heading_rad
        self.planner.robot_pos = np.array([grid_pos[0], grid_pos[1], heading_rad], dtype=np.float32)

        if record_history:
            self.navigation_history.append((int(grid_x), int(grid_y)))

    # ------------------------------------------------------------------
    # Sensor integration API
    # ------------------------------------------------------------------
    def register_sensor_callback(self, callback: Optional[SensorCallback]) -> None:
        self.sensor_hub.register_callback(callback)

    def attach_camera(self, sensor_type: str = "usb", start: bool = False, **kwargs: Any):
        return self.sensor_hub.attach_camera(sensor_type=sensor_type, start=start, **kwargs)

    def attach_imu(self, sensor_type: str = "android", start: bool = False, **kwargs: Any):
        return self.sensor_hub.attach_imu(sensor_type=sensor_type, start=start, **kwargs)

    def start_sensor_receivers(self) -> None:
        self.sensor_hub.start_receivers()

    def stop_sensor_receivers(self) -> None:
        self.sensor_hub.stop_receivers()

    def receive_sensor_data(
        self,
        frame: Optional[np.ndarray] = None,
        imu_data: Optional[object] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SensorSnapshot:
        return self.sensor_hub.receive_sensor_data(frame=frame, imu_data=imu_data, metadata=metadata)

    def poll_sensor_receivers(self, include_robot_imu: bool = False) -> SensorSnapshot:
        return self.sensor_hub.poll_receivers(include_robot_imu=include_robot_imu)

    def get_sensor_snapshot(self) -> SensorSnapshot:
        return self.sensor_hub.get_snapshot()

    # ------------------------------------------------------------------
    # Robot control API
    # ------------------------------------------------------------------
    def connect_robot(self) -> bool:
        return self.robot.connect()

    def disconnect_robot(self) -> None:
        self.robot.stop()
        self.robot.disconnect()

    def send_robot_command(self, command: Dict[str, Any]) -> bool:
        return self.robot.send_command(command)

    def sync_heading_from_robot(self) -> Optional[float]:
        """Pull heading from robot IMU (the single source of orientation truth).

        After calibrate_heading() has been called, robot.get_current_heading()
        returns an offset-corrected yaw in the map frame.  We treat that as
        authoritative and overwrite the local heading + planner state.
        """
        heading_deg = self.robot.get_current_heading()
        if heading_deg is None:
            return None

        self.robot_heading_rad = math.radians(float(heading_deg))
        self.planner.robot_pos[2] = self.robot_heading_rad
        return heading_deg

    def calibrate_heading(self, map_heading_deg: float) -> None:
        self.robot.calibrate_heading(float(map_heading_deg))
        self.robot_heading_rad = math.radians(float(map_heading_deg))
        self.planner.robot_pos[2] = self.robot_heading_rad

    # ------------------------------------------------------------------
    # Planning and execution API
    # ------------------------------------------------------------------
    def plan_path(self, goal_grid_pos: Sequence[int]) -> Optional[list[Tuple[int, int]]]:
        """Plan from current robot pixel pose to target pixel pose."""
        goal = (int(goal_grid_pos[0]), int(goal_grid_pos[1]))
        path = self.planner.plan_path(goal=goal)
        plan_details = self.planner.last_plan_details

        if path is None:
            self.current_path = None
            self.target_grid_pos = goal
            error_reason = plan_details.get("reason", "Unknown planning error.")
            print(f"Path planning failed: start={plan_details.get('start')} requested_goal={goal} error={error_reason}")
            return None

        self.current_path = [tuple(map(int, point)) for point in path]
        self.target_grid_pos = goal
        self.current_waypoint_idx = 0
        actual_goal = plan_details.get("actual_goal", goal)
        path_type = plan_details.get("path_type", "unknown")
        waypoint_count = plan_details.get("waypoint_count", len(self.current_path))
        path_length_cells = float(plan_details.get("path_length_cells", 0.0))
        path_length_m = path_length_cells * self.resolution
        print(
            "Planned path: "
            f"start={plan_details.get('start')} "
            f"requested_goal={goal} "
            f"actual_goal={actual_goal} "
            f"type={path_type} "
            f"waypoints={waypoint_count} "
            f"length={path_length_cells:.2f} cells ({path_length_m:.2f} m)"
        )
        return self.current_path

    def execute_path(
        self,
        path: Sequence[Sequence[int]],
        move_speed: float = 0.22,
        waypoint_pause_s: float = 0.3,
    ) -> list[Tuple[int, int]]:
        """Execute a path by sending waypoint-by-waypoint commands to the robot."""
        if len(path) < 2:
            return []

        if not self.robot.connected and not self.connect_robot():
            print("Robot connection failed; path will not be executed.")
            return []

        visited: list[Tuple[int, int]] = []
        self.is_moving = True

        try:
            for index, waypoint_grid in enumerate(path[1:], start=1):
                self.current_waypoint_idx = index

                target_pos_meters = grid_to_robot_frame(
                    np.array(waypoint_grid, dtype=np.float32),
                    self.map_manager.grid_size,
                    self.resolution,
                )

                executed_motion = self._coerce_motion_vector(
                    self.robot.move_to(target_pos_meters, self.robot_pos_meters, move_speed)
                )

                # Update position from executed motion vector.
                self.robot_pos_meters += executed_motion[:2]

                # Heading: prefer live IMU reading over dead-reckoned delta.
                heading_deg = self.robot.get_current_heading()
                if heading_deg is not None:
                    self.robot_heading_rad = math.radians(float(heading_deg))
                else:
                    self.robot_heading_rad = self._normalize_angle_rad(
                        self.robot_heading_rad + float(executed_motion[2])
                    )

                current_grid_pos = robot_to_grid_frame(
                    self.robot_pos_meters,
                    self.map_manager.grid_size,
                    self.resolution,
                )
                current_grid_tuple = (int(current_grid_pos[0]), int(current_grid_pos[1]))

                self.planner.robot_pos = np.array(
                    [current_grid_tuple[0], current_grid_tuple[1], self.robot_heading_rad],
                    dtype=np.float32,
                )

                self.navigation_history.append(current_grid_tuple)
                visited.append(current_grid_tuple)

                if self._window_initialized:
                    self.draw_state()

                time.sleep(max(0.0, waypoint_pause_s))
        finally:
            self.is_moving = False
            self.current_waypoint_idx = 0
            self.current_path = None

        return visited

    def move_to_target(
        self,
        target_grid_pos: Sequence[int],
        execute: Optional[bool] = None,
        move_speed: float = 0.22,
        waypoint_pause_s: float = 0.3,
    ) -> Optional[list[Tuple[int, int]]]:
        """Plan to target and optionally execute immediately."""
        path = self.plan_path(target_grid_pos)
        if path is None:
            return None

        should_execute = self.auto_execute_path if execute is None else bool(execute)
        if should_execute:
            self.execute_path(path, move_speed=move_speed, waypoint_pause_s=waypoint_pause_s)
        return path

    # ------------------------------------------------------------------
    # UI and visualization
    # ------------------------------------------------------------------
    def _ensure_window(self) -> None:
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self._window_initialized = True

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Click-to-plan callback for the map window."""
        self.mouse_pos = (int(x), int(y))

        if event != cv2.EVENT_LBUTTONDOWN or self.is_moving:
            return

        if not (0 <= x < self.map_manager.grid_width and 0 <= y < self.map_manager.grid_height):
            return

        self.move_to_target((x, y))

    def draw_state(self) -> np.ndarray:
        """Render planner, robot, path, and sensor summary to a debug image."""
        viz = cv2.cvtColor(self.map_manager.occupancy_grid, cv2.COLOR_GRAY2BGR)

        if len(self.navigation_history) > 1:
            for start_point, end_point in zip(self.navigation_history[:-1], self.navigation_history[1:]):
                cv2.line(viz, start_point, end_point, (255, 255, 0), 2)
            for point in self.navigation_history:
                cv2.circle(viz, point, 3, (255, 200, 0), -1)

        if self.current_path:
            for start_point, end_point in zip(self.current_path[:-1], self.current_path[1:]):
                cv2.line(viz, start_point, end_point, (255, 0, 255), 2)
            for index, point in enumerate(self.current_path):
                if index < self.current_waypoint_idx:
                    color = (0, 0, 255)
                elif index == self.current_waypoint_idx:
                    color = (0, 255, 255)
                else:
                    color = (255, 0, 0)
                cv2.circle(viz, point, 4, color, -1)

        if self.target_grid_pos is not None:
            cv2.drawMarker(viz, self.target_grid_pos, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        robot_px_pos = tuple(self.planner.robot_pos[:2].astype(int))
        cv2.circle(viz, robot_px_pos, 8, (0, 255, 0), -1)

        # Heading arrow is drawn in image coordinates (y axis points down).
        arrow_length = 20
        arrow_end_x = int(robot_px_pos[0] + arrow_length * math.cos(self.robot_heading_rad))
        arrow_end_y = int(robot_px_pos[1] - arrow_length * math.sin(self.robot_heading_rad))
        cv2.arrowedLine(viz, robot_px_pos, (arrow_end_x, arrow_end_y), (0, 255, 0), 2, tipLength=0.35)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 255, 0)

        heading_deg = math.degrees(self.robot_heading_rad)

        cv2.putText(viz, f"Map: {self.map_file.name}", (10, 24), font, 0.55, text_color, 2)
        cv2.putText(viz, f"Robot: ({robot_px_pos[0]}, {robot_px_pos[1]})", (10, 48), font, 0.55, text_color, 2)
        cv2.putText(viz, f"Heading: {heading_deg:.1f} deg", (10, 72), font, 0.55, text_color, 2)

        imu_data = self.robot.last_imu_data
        if imu_data is not None:
            accel_x = getattr(imu_data, "accel_x", 0.0)
            accel_y = getattr(imu_data, "accel_y", 0.0)
            cv2.putText(viz, f"Accel: ({float(accel_x):.2f}, {float(accel_y):.2f})", (10, 96), font, 0.55, text_color, 2)

        if self.mouse_pos is not None:
            delta_x = (self.mouse_pos[0] - robot_px_pos[0]) * self.resolution
            delta_y = (self.mouse_pos[1] - robot_px_pos[1]) * self.resolution
            distance = math.hypot(delta_x, delta_y)
            angle_deg = math.degrees(math.atan2(-delta_y, delta_x) - self.robot_heading_rad)
            y_offset = 120 if imu_data is not None else 96
            cv2.putText(viz, f"Cursor dist: {distance:.2f} m", (10, y_offset), font, 0.55, text_color, 2)
            cv2.putText(viz, f"Cursor angle: {angle_deg:.1f} deg", (10, y_offset + 24), font, 0.55, text_color, 2)

        if self._window_initialized:
            cv2.imshow(self.window_name, viz)

        return viz

    def run(self, connect_robot: bool = False, wait_ms: int = 100, poll_sensors: bool = True) -> None:
        """Open the interactive map window and process keyboard/mouse commands."""
        self._ensure_window()

        if connect_robot:
            self.connect_robot()

        print("Interactive mapper ready.")
        print("Left click: plan or plan+execute to a target")
        print("c: calibrate heading to current map heading")
        print("p: poll attached sensors")
        print("s: stop robot")
        print("q: quit")

        while True:
            if poll_sensors and (self.sensor_hub.camera is not None or self.sensor_hub.external_imu is not None):
                self.poll_sensor_receivers(include_robot_imu=False)

            if not self.is_moving:
                self.sync_heading_from_robot()

            self.draw_state()
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                self.calibrate_heading(self.start_pos[2])
            if key == ord("r"):
                start_x = input("Enter start X coordinate (pixels): ")
                start_y = input("Enter start Y coordinate (pixels): ")
                self.set_robot_pose(int(start_x), int(start_y), self.start_pos[2], record_history=True)
            if key == ord("p"):
                self.poll_sensor_receivers(include_robot_imu=True)
            if key == ord("s"):
                self.robot.stop()

        self.close()

    def close(self) -> None:
        """Release sensor streams, robot connection, and UI resources."""
        self.stop_sensor_receivers()

        if self.robot.connected:
            self.disconnect_robot()

        if self._window_initialized:
            cv2.destroyWindow(self.window_name)
            self._window_initialized = False
