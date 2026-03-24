"""Map partitioning and section-constrained patrol planning for multi-robot setups.

This module provides three building blocks:
1) map partitioning into one free-space mask per robot,
2) colored overlay visualization of robot regions,
3) patrol route generation constrained to a robot's assigned section.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    SLAM_ROVER_DIR = PACKAGE_DIR.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from interactive_mapper.map_manager import MapManager  # type: ignore
else:
    from .map_manager import MapManager

GridPoint = tuple[int, int]


@dataclass(frozen=True)
class RobotSection:
    """Container for a single robot's assigned operating region."""

    robot_index: int
    start: GridPoint
    mask: np.ndarray


@dataclass(frozen=True)
class PatrolQualityMetrics:
    """Patrol quality indicators for one robot section."""

    robot_index: int
    section_area_px: int
    route_points: int
    route_length_px: float
    route_length_m: float
    estimated_cycle_time_s: float
    coverage_percent: float
    overlap_percent: float


class MultiRobotMapPartitioner:
    """Partition free space into obstacle-aware Voronoi-style robot sections."""

    def __init__(self, occupancy_grid: np.ndarray, free_value: int = 0, obstacle_clearance_px: int = 0) -> None:
        self.occupancy_grid = occupancy_grid
        self.height, self.width = occupancy_grid.shape[:2]
        self.free_value = int(free_value)
        self.obstacle_clearance_px = max(0, int(obstacle_clearance_px))

        raw_free_mask = occupancy_grid == self.free_value
        if self.obstacle_clearance_px <= 0:
            self.free_mask = raw_free_mask
        else:
            free_u8 = (raw_free_mask.astype(np.uint8)) * 255
            distance = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)
            self.free_mask = raw_free_mask & (distance >= float(self.obstacle_clearance_px))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _nearest_free(self, point: GridPoint, max_radius: int = 80) -> Optional[GridPoint]:
        px, py = int(point[0]), int(point[1])
        if self._in_bounds(px, py) and self.free_mask[py, px]:
            return (px, py)

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    x, y = px + dx, py + dy
                    if self._in_bounds(x, y) and self.free_mask[y, x]:
                        return (x, y)
        return None

    def partition(self, robot_starts: Sequence[GridPoint]) -> tuple[np.ndarray, list[RobotSection], list[GridPoint]]:
        """Assign each free cell to the closest robot in traversable distance.

        Returns:
        - labels: int32 grid of robot index per free cell, -1 elsewhere
        - sections: list of RobotSection with uint8 masks (255=in section, 0=out)
        - adjusted_starts: start points nudged to nearest free cell when needed
        """
        if not robot_starts:
            raise ValueError("At least one robot start point is required.")

        adjusted_starts: list[GridPoint] = []
        for idx, start in enumerate(robot_starts):
            free_start = self._nearest_free(start)
            if free_start is None:
                raise ValueError(f"Robot {idx} start {start} is not near reachable free space.")
            adjusted_starts.append(free_start)

        labels = np.full((self.height, self.width), -1, dtype=np.int32)
        distances = np.full((self.height, self.width), np.inf, dtype=np.float32)

        # Multi-source Dijkstra gives obstacle-aware nearest-section assignment.
        pq: list[tuple[float, int, int, int]] = []
        for robot_idx, (sx, sy) in enumerate(adjusted_starts):
            labels[sy, sx] = robot_idx
            distances[sy, sx] = 0.0
            heapq.heappush(pq, (0.0, robot_idx, sx, sy))

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        ]

        while pq:
            dist, robot_idx, x, y = heapq.heappop(pq)
            if dist > distances[y, x] + 1e-6:
                continue

            for dx, dy, step_cost in neighbors:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if not self.free_mask[ny, nx]:
                    continue
                nd = dist + step_cost
                if nd + 1e-6 < distances[ny, nx]:
                    distances[ny, nx] = nd
                    labels[ny, nx] = robot_idx
                    heapq.heappush(pq, (nd, robot_idx, nx, ny))
                elif abs(nd - distances[ny, nx]) <= 1e-6 and robot_idx < labels[ny, nx]:
                    labels[ny, nx] = robot_idx

        sections: list[RobotSection] = []
        for robot_idx, start in enumerate(adjusted_starts):
            section_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            section_mask[labels == robot_idx] = 255
            sections.append(RobotSection(robot_index=robot_idx, start=start, mask=section_mask))

        return labels, sections, adjusted_starts


class SectionVisualizer:
    """Utilities to render color-highlighted section overlays."""

    DEFAULT_COLORS = [
        (255, 80, 80),
        (80, 200, 120),
        (70, 140, 255),
        (255, 190, 70),
        (230, 120, 255),
        (70, 220, 220),
    ]

    @staticmethod
    def render(
        occupancy_grid: np.ndarray,
        sections: Sequence[RobotSection],
        alpha: float = 0.45,
        show_starts: bool = True,
        colors: Optional[Sequence[tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Return a BGR image with each section tinted with a distinct color."""
        base = np.zeros((*occupancy_grid.shape[:2], 3), dtype=np.uint8)
        base[occupancy_grid == 0] = (40, 40, 40)
        base[occupancy_grid == 128] = (120, 120, 120)
        base[occupancy_grid == 255] = (245, 245, 245)

        palette = list(colors) if colors else SectionVisualizer.DEFAULT_COLORS
        overlay = base.copy()

        for section in sections:
            color = palette[section.robot_index % len(palette)]
            region = section.mask > 0
            overlay[region] = color

        blended = cv2.addWeighted(base, 1.0 - float(alpha), overlay, float(alpha), 0.0)

        if show_starts:
            for section in sections:
                color = palette[section.robot_index % len(palette)]
                x, y = section.start
                cv2.circle(blended, (x, y), 6, color, -1)
                cv2.circle(blended, (x, y), 8, (0, 0, 0), 1)
                cv2.putText(
                    blended,
                    f"R{section.robot_index}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return blended


class SectionPatrolPlanner:
    """Build patrol paths that stay strictly within one robot section."""

    def __init__(self, section_mask: np.ndarray) -> None:
        self.section_mask = section_mask > 0
        self.height, self.width = section_mask.shape[:2]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_free(self, x: int, y: int) -> bool:
        return self._in_bounds(x, y) and bool(self.section_mask[y, x])

    def _astar(self, start: GridPoint, goal: GridPoint) -> Optional[list[GridPoint]]:
        if not self._is_free(start[0], start[1]) or not self._is_free(goal[0], goal[1]):
            return None

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        ]

        def h(a: GridPoint, b: GridPoint) -> float:
            return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))

        open_heap: list[tuple[float, int, GridPoint]] = []
        heapq.heappush(open_heap, (h(start, goal), 0, start))

        came_from: dict[GridPoint, Optional[GridPoint]] = {start: None}
        g: dict[GridPoint, float] = {start: 0.0}
        closed: set[GridPoint] = set()
        counter = 0

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
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current
            for dx, dy, step_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if not self._is_free(nx, ny):
                    continue
                tentative = g[current] + step_cost
                if tentative + 1e-6 < g.get(nxt, np.inf):
                    came_from[nxt] = current
                    g[nxt] = tentative
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h(nxt, goal), counter, nxt))

        return None

    def build_patrol_route(
        self,
        start: GridPoint,
        sweep_step_px: int = 16,
        min_segment_points: int = 3,
        min_path_closeness_px: int = 0,
        close_loop: bool = False,
    ) -> list[GridPoint]:
        """Create a boustrophedon-style patrol route inside the section mask.

        The route alternates left-to-right and right-to-left sweep lines and uses
        A* to connect between sweep waypoints while staying inside the section.
        """
        if not self._is_free(start[0], start[1]):
            return []

        step = max(2, int(sweep_step_px), int(min_path_closeness_px))
        waypoints: list[GridPoint] = [start]
        flip = False

        for y in range(0, self.height, step):
            xs = np.flatnonzero(self.section_mask[y])
            if xs.size < min_segment_points:
                continue

            x_min, x_max = int(xs[0]), int(xs[-1])
            left = (x_min, y)
            right = (x_max, y)
            if flip:
                waypoints.extend([right, left])
            else:
                waypoints.extend([left, right])
            flip = not flip

        # Deduplicate sequential repeats while preserving order.
        cleaned_waypoints: list[GridPoint] = []
        for wp in waypoints:
            if not cleaned_waypoints or cleaned_waypoints[-1] != wp:
                cleaned_waypoints.append(wp)

        if len(cleaned_waypoints) <= 1:
            return cleaned_waypoints

        route: list[GridPoint] = [cleaned_waypoints[0]]
        for target in cleaned_waypoints[1:]:
            segment = self._astar(route[-1], target)
            if not segment:
                continue
            route.extend(segment[1:])

        if close_loop and len(route) > 2:
            tail = self._astar(route[-1], route[0])
            if tail:
                route.extend(tail[1:])

        return route


def _route_length_px(route: Sequence[GridPoint]) -> float:
    if len(route) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(route[:-1], route[1:]):
        total += float(np.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])))
    return total


def _route_coverage_mask(shape_hw: tuple[int, int], route: Sequence[GridPoint], path_radius_px: int) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if len(route) >= 2:
        pts = np.array(route, dtype=np.int32).reshape((-1, 1, 2))
        thickness = max(1, 2 * int(path_radius_px) + 1)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
    elif len(route) == 1:
        cv2.circle(mask, route[0], max(1, int(path_radius_px)), 255, -1)
    return mask


def compute_patrol_quality_metrics(
    sections: Sequence[RobotSection],
    patrols: dict[int, list[GridPoint]],
    resolution_m_per_px: float,
    estimated_speed_mps: float,
    min_path_closeness_px: int,
) -> list[PatrolQualityMetrics]:
    """Compute per-robot patrol quality metrics including overlap and coverage."""
    coverage_radius = max(1, int(min_path_closeness_px))
    speed_mps = max(1e-6, float(estimated_speed_mps))

    route_masks: dict[int, np.ndarray] = {}
    for section in sections:
        route_masks[section.robot_index] = _route_coverage_mask(
            section.mask.shape[:2],
            patrols.get(section.robot_index, []),
            path_radius_px=coverage_radius,
        )

    metrics: list[PatrolQualityMetrics] = []
    all_indices = [section.robot_index for section in sections]

    for section in sections:
        idx = section.robot_index
        section_region = section.mask > 0
        section_area = int(np.count_nonzero(section_region))

        route = patrols.get(idx, [])
        route_len_px = _route_length_px(route)
        route_len_m = float(route_len_px) * float(resolution_m_per_px)
        cycle_time_s = route_len_m / speed_mps

        route_mask = route_masks[idx] > 0
        covered_in_section = int(np.count_nonzero(section_region & route_mask))
        coverage_percent = 100.0 * covered_in_section / max(1, section_area)

        overlap_pixels = 0
        for other_idx in all_indices:
            if other_idx == idx:
                continue
            overlap_pixels += int(np.count_nonzero(route_mask & (route_masks[other_idx] > 0) & section_region))
        overlap_percent = 100.0 * overlap_pixels / max(1, section_area)

        metrics.append(
            PatrolQualityMetrics(
                robot_index=idx,
                section_area_px=section_area,
                route_points=len(route),
                route_length_px=route_len_px,
                route_length_m=route_len_m,
                estimated_cycle_time_s=cycle_time_s,
                coverage_percent=coverage_percent,
                overlap_percent=overlap_percent,
            )
        )

    return metrics


def draw_route(image_bgr: np.ndarray, route: Sequence[GridPoint], color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """Draw a patrol route polyline on a BGR image and return the updated image."""
    canvas = image_bgr.copy()
    if len(route) >= 2:
        pts = np.array(route, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)
    if route:
        cv2.circle(canvas, route[0], 5, color, -1)
    return canvas


def partition_map_and_plan_patrols(
    map_file: str | Path,
    robot_starts: Sequence[GridPoint],
    sweep_step_px: int = 16,
    min_segment_points: int = 3,
    min_path_closeness_px: int = 8,
    close_loop: bool = False,
    obstacle_clearance_px: int = 0,
    resolution_m_per_px: float = 0.0197,
    estimated_speed_mps: float = 0.22,
) -> tuple[np.ndarray, list[RobotSection], dict[int, list[GridPoint]], list[PatrolQualityMetrics]]:
    """Convenience API to partition a map and generate one patrol route per section."""
    map_manager = MapManager(map_file)
    partitioner = MultiRobotMapPartitioner(
        map_manager.occupancy_grid,
        obstacle_clearance_px=obstacle_clearance_px,
    )
    _, sections, adjusted_starts = partitioner.partition(robot_starts)

    patrols: dict[int, list[GridPoint]] = {}
    for section, start in zip(sections, adjusted_starts):
        planner = SectionPatrolPlanner(section.mask)
        patrols[section.robot_index] = planner.build_patrol_route(
            start=start,
            sweep_step_px=sweep_step_px,
            min_segment_points=min_segment_points,
            min_path_closeness_px=min_path_closeness_px,
            close_loop=close_loop,
        )

    overlay = SectionVisualizer.render(map_manager.occupancy_grid, sections)
    metrics = compute_patrol_quality_metrics(
        sections=sections,
        patrols=patrols,
        resolution_m_per_px=resolution_m_per_px,
        estimated_speed_mps=estimated_speed_mps,
        min_path_closeness_px=min_path_closeness_px,
    )
    return overlay, sections, patrols, metrics


if __name__ == "__main__":
    if __package__ in {None, ""}:
        from interactive_mapper.config import MAP_FILE, MULTI_ROBOT_STARTS, MULTI_ROBOT_SWEEP_STEP_PX, MULTI_ROBOT_MIN_SEGMENT_POINTS, MULTI_ROBOT_MIN_PATH_CLOSENESS_PX, MULTI_ROBOT_CLOSE_LOOP, MULTI_ROBOT_MIN_OBSTACLE_DISTANCE_PX, RESOLUTION_M_PER_PX, MULTI_ROBOT_ESTIMATED_SPEED_MPS  # type: ignore
    else:
        from .config import MAP_FILE, MULTI_ROBOT_STARTS, MULTI_ROBOT_SWEEP_STEP_PX, MULTI_ROBOT_MIN_SEGMENT_POINTS, MULTI_ROBOT_MIN_PATH_CLOSENESS_PX, MULTI_ROBOT_CLOSE_LOOP, MULTI_ROBOT_MIN_OBSTACLE_DISTANCE_PX, RESOLUTION_M_PER_PX, MULTI_ROBOT_ESTIMATED_SPEED_MPS  # type: ignore

    # Example starts; replace with your robot initial positions.
    # starts = [(80, 80), (260, 100), (180, 260)]

    # two-robot example:
    # starts = [(80, 80), (350, 80)]

    # three-robot example:
    starts = [(80, 80), (350, 200), (100, 700)]

    overlay, sections, patrols, metrics = partition_map_and_plan_patrols(
        MAP_FILE,
        starts,
        sweep_step_px=MULTI_ROBOT_SWEEP_STEP_PX,
        min_segment_points=MULTI_ROBOT_MIN_SEGMENT_POINTS,
        min_path_closeness_px=MULTI_ROBOT_MIN_PATH_CLOSENESS_PX,
        close_loop=MULTI_ROBOT_CLOSE_LOOP,
        obstacle_clearance_px=MULTI_ROBOT_MIN_OBSTACLE_DISTANCE_PX,
        resolution_m_per_px=RESOLUTION_M_PER_PX,
        estimated_speed_mps=MULTI_ROBOT_ESTIMATED_SPEED_MPS,

    )

    palette = SectionVisualizer.DEFAULT_COLORS
    viz = overlay.copy()
    for section in sections:
        color = palette[section.robot_index % len(palette)]
        viz = draw_route(viz, patrols.get(section.robot_index, []), color=color, thickness=2)

    for item in metrics:
        print(
            f"R{item.robot_index}: coverage={item.coverage_percent:.1f}% "
            f"overlap={item.overlap_percent:.1f}% "
            f"cycle={item.estimated_cycle_time_s:.1f}s"
        )

    cv2.imshow("Multi-Robot Sections + Patrols", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
