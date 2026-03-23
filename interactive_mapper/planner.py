"""Local path planning utilities for the interactive mapper package."""

from __future__ import annotations

import heapq
import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


GridPoint = Tuple[int, int]


def _grid_center_components(grid_size: int | Sequence[int]) -> tuple[float, float]:
    """Return grid center as (x_center, y_center)."""
    if isinstance(grid_size, Sequence) and not isinstance(grid_size, (str, bytes)):
        if len(grid_size) >= 2:
            width = float(grid_size[0])
            height = float(grid_size[1])
            return width / 2.0, height / 2.0
    center = float(grid_size) / 2.0
    return center, center


def grid_to_robot_frame(grid_pos: Sequence[float], grid_size: int | Sequence[int], resolution: float) -> np.ndarray:
    """Convert grid coordinates (image frame) to robot-centric metric coordinates."""
    center_x, center_y = _grid_center_components(grid_size)
    robot_x = (float(grid_pos[0]) - center_x) * float(resolution)
    robot_y = -(float(grid_pos[1]) - center_y) * float(resolution)
    return np.array([robot_x, robot_y], dtype=np.float32)


def robot_to_grid_frame(robot_pos_meters: Sequence[float], grid_size: int | Sequence[int], resolution: float) -> np.ndarray:
    """Convert robot-centric metric coordinates to grid/image coordinates."""
    center_x, center_y = _grid_center_components(grid_size)
    grid_x = (float(robot_pos_meters[0]) / float(resolution)) + center_x
    grid_y = (-float(robot_pos_meters[1]) / float(resolution)) + center_y
    return np.array([grid_x, grid_y], dtype=np.int32)


class GridPathPlanner:
    """Minimal occupancy-grid planner owned by the interactive mapper package."""

    def __init__(
        self,
        grid_size: int,
        resolution: float,
        min_wall_distance: float = 0.1,
        occupancy_grid: Optional[np.ndarray] = None,
    ) -> None:
        self.grid_size = int(grid_size)
        self.resolution = float(resolution)
        self.min_wall_distance = float(min_wall_distance)
        self.occupancy_grid = (
            occupancy_grid.copy()
            if occupancy_grid is not None
            else np.full((self.grid_size, self.grid_size), 128, dtype=np.uint8)
        )
        self.robot_pos = np.array([self.grid_size // 2, self.grid_size // 2, 0.0], dtype=np.float32)
        self.last_plan_details: dict[str, object] = {}

    def plan_path(self, goal: Sequence[int], max_path_length: Optional[float] = None) -> Optional[list[GridPoint]]:
        """Plan a safe path from the current robot pose to the requested grid point."""
        start = tuple(self.robot_pos[:2].astype(int))
        target = (int(goal[0]), int(goal[1]))
        min_wall_cells = max(0, int(math.ceil(self.min_wall_distance / self.resolution)))
        self.last_plan_details = {
            "start": start,
            "requested_goal": target,
            "actual_goal": target,
            "min_wall_cells": min_wall_cells,
            "status": "planning",
            "reason": None,
        }

        actual_goal = target
        if not self._is_cell_safe(actual_goal, min_wall_cells):
            actual_goal = self._find_nearest_safe_point(actual_goal, min_wall_cells, search_radius=50)
            if actual_goal is None:
                self.last_plan_details.update(
                    {
                        "status": "failed",
                        "reason": "No safe goal cell found near the requested target.",
                    }
                )
                return None
            self.last_plan_details["actual_goal"] = actual_goal

        if not self._is_cell_safe(start, min_wall_cells):
            safe_start = self._find_nearest_safe_point(start, min_wall_cells, search_radius=20)
            if safe_start is None:
                self.last_plan_details.update(
                    {
                        "status": "failed",
                        "reason": "Robot start position is not in reachable free space.",
                    }
                )
                return None
            start = safe_start
            self.last_plan_details["start"] = start

        if self._is_line_clear(start, actual_goal, min_wall_cells):
            path = [start, actual_goal]
            self.last_plan_details.update(
                {
                    "status": "ok",
                    "path_type": "direct",
                    "waypoint_count": len(path),
                    "path_length_cells": self._path_length_cells(path),
                }
            )
            return path

        path = self._astar_path(start, actual_goal, max_path_length=max_path_length, min_wall_cells=min_wall_cells)
        if not path:
            self.last_plan_details.update(
                {
                    "status": "failed",
                    "reason": "No collision-free path found from start to goal.",
                }
            )
            return None
        simplified_path = self._simplify_path(path, min_wall_cells)
        self.last_plan_details.update(
            {
                "status": "ok",
                "path_type": "astar",
                "raw_waypoint_count": len(path),
                "waypoint_count": len(simplified_path),
                "path_length_cells": self._path_length_cells(simplified_path),
            }
        )
        return simplified_path

    @staticmethod
    def _path_length_cells(path: Sequence[GridPoint]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for start, end in zip(path[:-1], path[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        return total

    def _find_nearest_safe_point(self, point: GridPoint, min_wall_cells: int, search_radius: int) -> Optional[GridPoint]:
        x, y = point
        for radius in range(search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if radius > 0 and abs(dx) != radius and abs(dy) != radius:
                        continue
                    candidate = (x + dx, y + dy)
                    if self._is_cell_safe(candidate, min_wall_cells):
                        return candidate
        return None

    def _is_cell_safe(self, cell: GridPoint, min_wall_cells: int) -> bool:
        x, y = int(cell[0]), int(cell[1])
        grid_height, grid_width = self.occupancy_grid.shape[:2]
        if not (0 <= x < grid_width and 0 <= y < grid_height):
            return False

        cell_value = int(self.occupancy_grid[y, x])
        if cell_value != 0:
            return False

        if min_wall_cells <= 0:
            return True

        x_min = max(0, x - min_wall_cells)
        x_max = min(grid_width, x + min_wall_cells + 1)
        y_min = max(0, y - min_wall_cells)
        y_max = min(grid_height, y + min_wall_cells + 1)
        region = self.occupancy_grid[y_min:y_max, x_min:x_max]
        return not np.any(region == 255)

    def _is_line_clear(self, start: GridPoint, goal: GridPoint, min_wall_cells: int) -> bool:
        x0, y0 = start
        x1, y1 = goal
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if not self._is_cell_safe((x, y), min_wall_cells):
                return False
            if x == x1 and y == y1:
                return True

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _compute_safe_map(self, min_wall_cells: int) -> np.ndarray:
        safe_map = self.occupancy_grid == 0
        if min_wall_cells <= 0:
            return safe_map

        obstacle_mask = (self.occupancy_grid == 255).astype(np.uint8)
        kernel_size = 2 * min_wall_cells + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated_obstacles = cv2.dilate(obstacle_mask, kernel, iterations=1).astype(bool)
        return safe_map & ~dilated_obstacles

    def _astar_path(
        self,
        start: GridPoint,
        goal: GridPoint,
        max_path_length: Optional[float],
        min_wall_cells: int,
    ) -> Optional[list[GridPoint]]:
        safe_map = self._compute_safe_map(min_wall_cells)
        grid_height, grid_width = safe_map.shape

        if not (0 <= start[0] < grid_width and 0 <= start[1] < grid_height and safe_map[start[1], start[0]]):
            return None
        if not (0 <= goal[0] < grid_width and 0 <= goal[1] < grid_height and safe_map[goal[1], goal[0]]):
            return None

        def heuristic(a: GridPoint, b: GridPoint) -> float:
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

        directions: Iterable[tuple[GridPoint, float]] = (
            ((-1, 0), 1.0),
            ((1, 0), 1.0),
            ((0, -1), 1.0),
            ((0, 1), 1.0),
            ((-1, -1), math.sqrt(2.0)),
            ((-1, 1), math.sqrt(2.0)),
            ((1, -1), math.sqrt(2.0)),
            ((1, 1), math.sqrt(2.0)),
        )

        open_heap: list[tuple[float, int, GridPoint]] = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
        came_from: dict[GridPoint, Optional[GridPoint]] = {start: None}
        g_score: dict[GridPoint, float] = {start: 0.0}
        closed: set[GridPoint] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path: list[GridPoint] = []
                node: Optional[GridPoint] = current
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return list(reversed(path))

            closed.add(current)
            current_cost = g_score[current]
            if max_path_length is not None and current_cost > max_path_length:
                continue

            cx, cy = current
            for (dx, dy), step_cost in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                    continue
                if not safe_map[ny, nx]:
                    continue

                neighbor = (nx, ny)
                tentative_cost = current_cost + step_cost
                if tentative_cost >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                counter += 1
                heapq.heappush(open_heap, (tentative_cost + heuristic(neighbor, goal), counter, neighbor))

        return None

    def _simplify_path(self, path: Sequence[GridPoint], min_wall_cells: int) -> list[GridPoint]:
        if len(path) <= 2:
            return list(path)

        simplified = [path[0]]
        index = 0
        while index < len(path) - 1:
            furthest = index + 1
            for candidate in range(index + 2, len(path)):
                if self._is_line_clear(path[index], path[candidate], min_wall_cells):
                    furthest = candidate
                else:
                    break
            simplified.append(path[furthest])
            index = furthest

        if simplified[-1] != path[-1]:
            simplified.append(path[-1])
        return simplified


__all__ = ["GridPathPlanner", "grid_to_robot_frame", "robot_to_grid_frame"]