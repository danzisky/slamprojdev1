"""OpenCV-based rendering helpers for localization demos."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .map_data import ChairLocalizationMap
from .particle_filter import ParticleFilterLocalizer
from .types import LocalizationUpdate, Pose2D


def _polar(cx: int, cy: int, r: float, angle: float) -> tuple[int, int]:
    """Return a point `r` pixels from (cx, cy) in direction `angle` radians."""
    return (int(round(cx + r * math.cos(angle))), int(round(cy + r * math.sin(angle))))


@dataclass(frozen=True)
class VisualizationConfig:
    """Rendering settings for the localization demo windows."""

    show_observation_preview: bool = True
    preview_window_name: str = "Chair Observations"
    show_map_preview: bool = True
    map_window_name: str = "Localization Map"
    map_preview_scale: float = 1
    draw_particles: bool = True
    max_particles_to_draw: int = 250
    particle_dot_radius_px: int = 4          # filled circle radius for each particle
    particle_heading_line_px: int = 4        # length of the heading tick from the dot center
    color_particles_by_weight: bool = True   # cool→warm heatmap: high-weight particles glow warm
    draw_detection_overlay_on_map: bool = True
    draw_match_links_on_map: bool = True
    draw_unmatched_predictions: bool = True
    detection_point_radius_px: int = 4
    detection_ray_thickness_px: int = 2
    trajectory_history: int = 150

    # Robot marker
    robot_body_radius_px: int = 10          # half-size of the triangle body
    robot_heading_arrow_px: int = 28        # length of the forward heading arrow
    draw_robot_fov: bool = True             # draw a semi-transparent camera FOV cone
    robot_fov_half_angle_rad: float = math.radians(35.0)  # half-angle of the FOV cone
    robot_fov_range_px: int = 60            # how far the cone extends


class LocalizationVisualizer:
    """Render detector observations and map-space localization overlays."""

    def __init__(self, config: VisualizationConfig | None = None):
        self.config = config or VisualizationConfig()
        self._trajectory: list[Pose2D] = []
        self._update_count: int = 0  # incremented on every real measurement update

    def record_pose(self, pose: Pose2D) -> None:
        self._trajectory.append(pose)
        history_limit = self.config.trajectory_history
        if len(self._trajectory) > history_limit:
            self._trajectory = self._trajectory[-history_limit:]

    def show_update(
        self,
        frame: np.ndarray | None,
        localization_map: ChairLocalizationMap,
        localizer: ParticleFilterLocalizer,
        update: LocalizationUpdate | None,
    ) -> None:
        if update is not None:
            self.record_pose(update.estimate.pose)
            self._update_count += 1

        if self.config.show_observation_preview and frame is not None and update is not None:
            observation_preview = self.render_observation_preview(frame, update)
            cv2.imshow(self.config.preview_window_name, observation_preview)

        if self.config.show_map_preview:
            map_preview = self.render_map_preview(localization_map, localizer, update)
            cv2.imshow(self.config.map_window_name, map_preview)

    def render_observation_preview(
        self,
        frame: np.ndarray,
        update: LocalizationUpdate,
    ) -> np.ndarray:
        preview = frame.copy()
        for observation in update.observations:
            if observation.bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in observation.bbox]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"r={observation.range_m:.2f}m, b={math.degrees(observation.bearing_rad):.1f}deg"
            cv2.putText(
                preview,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return preview

    def render_map_preview(
        self,
        localization_map: ChairLocalizationMap,
        localizer: ParticleFilterLocalizer,
        update: LocalizationUpdate | None,
    ) -> np.ndarray:
        grid = localization_map.occupancy_grid
        preview = np.zeros((localization_map.height_px, localization_map.width_px, 3), dtype=np.uint8)
        preview[grid == 0] = (245, 245, 245)
        preview[grid == 127] = (170, 170, 170)
        preview[grid == 255] = (35, 35, 35)

        if self.config.draw_particles:
            particles = localizer.particle_array()
            if len(particles) > self.config.max_particles_to_draw:
                ranked_indices = np.argsort(particles[:, 3])[::-1][: self.config.max_particles_to_draw]
                particles = particles[ranked_indices]

            dot_r = self.config.particle_dot_radius_px
            line_len = self.config.particle_heading_line_px

            # Weight-based heatmap: cool (blue) = low weight, warm (yellow/red) = high weight.
            # All particles get the same solid blue when coloring is disabled.
            if self.config.color_particles_by_weight and len(particles) > 0:
                w = particles[:, 3].astype(np.float32)
                w_range = w.max() - w.min()
                w_norm = ((w - w.min()) / (w_range + 1e-12) * 255).astype(np.uint8).reshape(-1, 1)
                particle_colors = cv2.applyColorMap(w_norm, cv2.COLORMAP_JET).reshape(-1, 3)
            else:
                particle_colors = np.full((len(particles), 3), [80, 150, 255], dtype=np.uint8)

            for idx, (particle_x, particle_y, particle_h, _) in enumerate(particles):
                px, py = localization_map.world_to_pixel(float(particle_x), float(particle_y))
                if not (0 <= px < localization_map.width_px and 0 <= py < localization_map.height_px):
                    continue
                color = tuple(int(c) for c in particle_colors[idx])
                # Filled centre dot
                cv2.circle(preview, (px, py), dot_r, color, -1, cv2.LINE_AA)
                # Heading tick extending from the dot edge
                tip = _polar(px, py, dot_r + line_len, float(particle_h))
                cv2.line(preview, (px, py), tip, (0, 255, 0), 1, cv2.LINE_AA)

        trajectory_pixels = self._trajectory_pixels(localization_map)
        if len(trajectory_pixels) >= 2:
            cv2.polylines(
                preview,
                [np.asarray(trajectory_pixels, dtype=np.int32)],
                False,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )

        for landmark in localization_map.landmarks:
            px, py = localization_map.world_to_pixel(landmark.x_m, landmark.y_m)
            if 0 <= px < localization_map.width_px and 0 <= py < localization_map.height_px:
                cv2.circle(preview, (px, py), 5, (0, 200, 255), -1)
                cv2.putText(
                    preview,
                    f"L{landmark.landmark_id}",
                    (px + 6, max(12, py - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 120, 180),
                    1,
                    cv2.LINE_AA,
                )

        if update is not None:
            if self.config.draw_detection_overlay_on_map:
                self._draw_detection_overlay_on_map(preview, localization_map, localizer, update)
            self._draw_pose_marker(preview, update.estimate.pose, localization_map, (0, 0, 255), "robot")
            self._draw_status_box(preview, update)
        else:
            # No measurement yet — label the view so the user knows this is the prior
            self._draw_phase_label(preview, "INITIAL SPREAD")

        if self.config.map_preview_scale != 1.0:
            preview = cv2.resize(
                preview,
                dsize=None,
                fx=self.config.map_preview_scale,
                fy=self.config.map_preview_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        return preview

    def close(self) -> None:
        cv2.destroyAllWindows()

    def _trajectory_pixels(self, localization_map: ChairLocalizationMap) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for pose in self._trajectory[-self.config.trajectory_history :]:
            px, py = localization_map.world_to_pixel(pose.x_m, pose.y_m)
            if 0 <= px < localization_map.width_px and 0 <= py < localization_map.height_px:
                points.append((px, py))
        return points

    def _draw_pose_marker(
        self,
        image: np.ndarray,
        pose: Pose2D,
        localization_map: ChairLocalizationMap,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x_px, y_px = localization_map.world_to_pixel(pose.x_m, pose.y_m)
        if not (0 <= x_px < localization_map.width_px and 0 <= y_px < localization_map.height_px):
            return

        h = pose.heading_rad
        r = self.config.robot_body_radius_px
        cfg = self.config

        # --- FOV cone (semi-transparent wedge showing camera view direction) ---
        if cfg.draw_robot_fov:
            fov = cfg.robot_fov_half_angle_rad
            fov_r = cfg.robot_fov_range_px
            num_pts = 20
            fan: list[tuple[int, int]] = [(x_px, y_px)]
            for i in range(num_pts + 1):
                angle = h - fov + (2.0 * fov * i / num_pts)
                fan.append((
                    int(round(x_px + fov_r * math.cos(angle))),
                    int(round(y_px + fov_r * math.sin(angle))),
                ))
            overlay = image.copy()
            cv2.fillPoly(overlay, [np.array(fan, dtype=np.int32)], (100, 220, 100))
            cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)
            # Cone outline
            cv2.line(image, (x_px, y_px), fan[1], (100, 200, 100), 1, cv2.LINE_AA)
            cv2.line(image, (x_px, y_px), fan[-1], (100, 200, 100), 1, cv2.LINE_AA)

        # --- Filled triangle body pointing in heading direction ---
        #   tip: front, rear_l / rear_r: two back corners
        tip    = _polar(x_px, y_px, r,        h)
        rear_l = _polar(x_px, y_px, r * 0.75, h + 2.45)
        rear_r = _polar(x_px, y_px, r * 0.75, h - 2.45)
        pts = np.array([tip, rear_l, rear_r], dtype=np.int32)
        cv2.fillPoly(image, [pts], color)
        cv2.polylines(image, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Heading arrow from body centre forward ---
        arrow_end = _polar(x_px, y_px, cfg.robot_heading_arrow_px, h)
        cv2.arrowedLine(image, (x_px, y_px), arrow_end, color, 2, tipLength=0.25, line_type=cv2.LINE_AA)

        # --- Label ---
        cv2.putText(
            image,
            label,
            (x_px + r + 4, max(12, y_px - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x_px + r + 4, max(12, y_px - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_status_box(self, image: np.ndarray, update: LocalizationUpdate) -> None:
        estimate = update.estimate
        status_lines = [
            f"Update #{self._update_count}",
            f"x={estimate.pose.x_m:.2f}m  y={estimate.pose.y_m:.2f}m",
            f"heading={math.degrees(estimate.pose.heading_rad):.1f}\u00b0",
            f"obs={estimate.observation_count}  matches={estimate.matched_landmarks}",
            f"ess={estimate.effective_sample_size:.1f}",
        ]
        for line_index, text in enumerate(status_lines):
            y = 20 + 18 * line_index
            cv2.putText(
                image,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_detection_overlay_on_map(
        self,
        image: np.ndarray,
        localization_map: ChairLocalizationMap,
        localizer: ParticleFilterLocalizer,
        update: LocalizationUpdate,
    ) -> None:
        """Project image detections to map coordinates and draw map-fit diagnostics."""
        pose = update.estimate.pose
        robot_px, robot_py = localization_map.world_to_pixel(pose.x_m, pose.y_m)
        if not (0 <= robot_px < localization_map.width_px and 0 <= robot_py < localization_map.height_px):
            return

        if not update.observations:
            return

        predictions = localization_map.visible_landmarks(
            pose,
            max_range_m=localizer.measurement_max_range_m,
            half_fov_rad=localizer.camera_half_fov_rad,
        )
        association = localizer.associator.associate(update.observations, predictions)
        matched_by_observation = {obs_idx: pred_idx for obs_idx, pred_idx in association.matches}
        matched_predictions = {pred_idx for _, pred_idx in association.matches}

        for obs_idx, observation in enumerate(update.observations):
            absolute_bearing = pose.heading_rad + observation.bearing_rad
            observed_x_m = pose.x_m + observation.range_m * math.cos(absolute_bearing)
            observed_y_m = pose.y_m + observation.range_m * math.sin(absolute_bearing)
            obs_px, obs_py = localization_map.world_to_pixel(observed_x_m, observed_y_m)

            if not (0 <= obs_px < localization_map.width_px and 0 <= obs_py < localization_map.height_px):
                continue

            matched = obs_idx in matched_by_observation
            detection_color = (70, 220, 70) if matched else (0, 165, 255)

            cv2.line(
                image,
                (robot_px, robot_py),
                (obs_px, obs_py),
                detection_color,
                self.config.detection_ray_thickness_px,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                (obs_px, obs_py),
                self.config.detection_point_radius_px,
                detection_color,
                -1,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                f"D{obs_idx}",
                (obs_px + 4, max(12, obs_py - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"D{obs_idx}",
                (obs_px + 4, max(12, obs_py - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if not self.config.draw_match_links_on_map or not matched:
                continue

            prediction = predictions[matched_by_observation[obs_idx]]
            landmark_px, landmark_py = localization_map.world_to_pixel(
                prediction.landmark.x_m,
                prediction.landmark.y_m,
            )
            if not (0 <= landmark_px < localization_map.width_px and 0 <= landmark_py < localization_map.height_px):
                continue

            cv2.line(image, (obs_px, obs_py), (landmark_px, landmark_py), (255, 0, 255), 1, cv2.LINE_AA)

            fit_error_m = math.hypot(observed_x_m - prediction.landmark.x_m, observed_y_m - prediction.landmark.y_m)
            label_x = (obs_px + landmark_px) // 2
            label_y = (obs_py + landmark_py) // 2
            cv2.putText(
                image,
                f"{fit_error_m:.2f}m",
                (label_x + 3, max(12, label_y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (40, 0, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"{fit_error_m:.2f}m",
                (label_x + 3, max(12, label_y - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (255, 210, 255),
                1,
                cv2.LINE_AA,
            )

        if self.config.draw_unmatched_predictions:
            for prediction_index, prediction in enumerate(predictions):
                if prediction_index in matched_predictions:
                    continue
                px, py = localization_map.world_to_pixel(prediction.landmark.x_m, prediction.landmark.y_m)
                if 0 <= px < localization_map.width_px and 0 <= py < localization_map.height_px:
                    cv2.circle(image, (px, py), 8, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_phase_label(self, image: np.ndarray, text: str) -> None:
        """Draw a prominent phase banner (e.g. 'INITIAL SPREAD') on the preview."""
        cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)