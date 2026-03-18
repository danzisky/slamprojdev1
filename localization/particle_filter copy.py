"""Particle filter localization using chair landmarks and global association."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .association import AssociationResult, JointCompatibilityAssociator
from .detector import ChairObservationDetector
from .map_data import ChairLocalizationMap
from .math_utils import weighted_circular_mean, weighted_circular_std, wrap_to_pi
from .types import (
    ChairObservation,
    LocalizationEstimate,
    LocalizationUpdate,
    MotionCommand,
    Pose2D,
)


class ParticleFilterLocalizer:
    """
    Chair-based particle filter localizer.

    The filter supports two initialization modes:
    - global initialization over free space when initial_pose is None
    - Gaussian initialization around a known pose when initial_pose is provided
    """

    def __init__(
        self,
        localization_map: ChairLocalizationMap,
        detector: Optional[ChairObservationDetector] = None,
        particle_count: int = 800,
        initial_pose: Optional[Pose2D] = None,
        initial_position_std_m: float = 0.25,
        initial_heading_std_rad: float = math.radians(10.0),
        motion_forward_std_m: float = 0.05,
        motion_turn_std_rad: float = math.radians(2.5),
        measurement_max_range_m: Optional[float] = None,
        camera_half_fov_rad: Optional[float] = None,
        enable_range_fov_gating: bool = False,
        resample_threshold: float = 0.55,
        resampling_method: str = "wheel",
        roughening_position_std_m: float = 0.5,
        roughening_heading_std_rad: float = math.radians(3),
        associator: Optional[JointCompatibilityAssociator] = None,
        random_seed: Optional[int] = None,
    ):
        self.map = localization_map
        self.detector = detector
        self.particle_count = int(particle_count)
        self.initial_position_std_m = float(initial_position_std_m)
        self.initial_heading_std_rad = float(initial_heading_std_rad)
        self.motion_forward_std_m = float(motion_forward_std_m)
        self.motion_turn_std_rad = float(motion_turn_std_rad)
        self.measurement_max_range_m = (
            None if measurement_max_range_m is None else float(measurement_max_range_m)
        )
        self.enable_range_fov_gating = bool(enable_range_fov_gating)
        self.resample_threshold = float(resample_threshold)
        self.resampling_method = str(resampling_method).lower().strip()
        if self.resampling_method not in {"systematic", "wheel"}:
            raise ValueError("resampling_method must be 'systematic' or 'wheel'")
        self.roughening_position_std_m = float(roughening_position_std_m)
        self.roughening_heading_std_rad = float(roughening_heading_std_rad)
        self.associator = associator or JointCompatibilityAssociator()
        self.rng = np.random.default_rng(random_seed)

        if (
            self.enable_range_fov_gating
            and camera_half_fov_rad is None
            and detector is not None
            and detector.image_size is not None
        ):
            camera_half_fov_rad = 0.5 * detector.horizontal_fov_rad()
        self.camera_half_fov_rad = camera_half_fov_rad

        self.x_m = np.zeros(self.particle_count, dtype=np.float64)
        self.y_m = np.zeros(self.particle_count, dtype=np.float64)
        self.heading_rad = np.zeros(self.particle_count, dtype=np.float64)
        self.weights = np.full(self.particle_count, 1.0 / self.particle_count, dtype=np.float64)

        self.last_update: Optional[LocalizationUpdate] = None
        self._initialize_particles(initial_pose)

    def _initialize_particles(self, initial_pose: Optional[Pose2D]) -> None:
        if initial_pose is None:
            free_positions = self.map.sample_free_positions(self.particle_count, self.rng)
            self.x_m = free_positions[:, 0]
            self.y_m = free_positions[:, 1]
            self.heading_rad = self.rng.uniform(-math.pi, math.pi, size=self.particle_count)
        else:
            self.x_m, self.y_m, self.heading_rad = self._sample_around_pose(
                initial_pose=initial_pose,
                count=self.particle_count,
                position_std_m=self.initial_position_std_m,
                heading_std_rad=self.initial_heading_std_rad,
            )

        self.weights.fill(1.0 / self.particle_count)

    def _sample_around_pose(
        self,
        initial_pose: Pose2D,
        count: int,
        position_std_m: float,
        heading_std_rad: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a Gaussian cloud around a known pose and repair invalid samples."""
        x_values = initial_pose.x_m + self.rng.normal(0.0, position_std_m, size=count)
        y_values = initial_pose.y_m + self.rng.normal(0.0, position_std_m, size=count)
        heading_values = wrap_to_pi(
            initial_pose.heading_rad + self.rng.normal(0.0, heading_std_rad, size=count)
        )

        self._repair_invalid_positions(
            x_values,
            y_values,
            base_x=np.full(count, initial_pose.x_m, dtype=np.float64),
            base_y=np.full(count, initial_pose.y_m, dtype=np.float64),
            jitter_std_m=position_std_m,
        )
        return x_values, y_values, heading_values

    def _repair_invalid_positions(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        base_x: np.ndarray,
        base_y: np.ndarray,
        jitter_std_m: float,
    ) -> None:
        """Move invalid particles back into free space without collapsing the cloud."""
        invalid_indices = np.flatnonzero(~self.map.are_free(x_values, y_values))
        if invalid_indices.size == 0:
            return

        for _ in range(6):
            if invalid_indices.size == 0:
                break

            x_candidates = base_x[invalid_indices] + self.rng.normal(0.0, jitter_std_m, size=invalid_indices.size)
            y_candidates = base_y[invalid_indices] + self.rng.normal(0.0, jitter_std_m, size=invalid_indices.size)
            valid_candidates = self.map.are_free(x_candidates, y_candidates)
            if np.any(valid_candidates):
                valid_indices = invalid_indices[valid_candidates]
                x_values[valid_indices] = x_candidates[valid_candidates]
                y_values[valid_indices] = y_candidates[valid_candidates]
                invalid_indices = invalid_indices[~valid_candidates]

        if invalid_indices.size > 0:
            global_samples = self.map.sample_free_positions(invalid_indices.size, self.rng)
            x_values[invalid_indices] = global_samples[:, 0]
            y_values[invalid_indices] = global_samples[:, 1]

    def set_initial_pose(
        self,
        pose: Optional[Pose2D],
        position_std_m: Optional[float] = None,
        heading_std_rad: Optional[float] = None,
    ) -> None:
        """Reset the particle cloud using either a known pose or global free-space sampling."""
        if position_std_m is not None:
            self.initial_position_std_m = float(position_std_m)
        if heading_std_rad is not None:
            self.initial_heading_std_rad = float(heading_std_rad)
        self._initialize_particles(pose)

    def predict(self, motion: MotionCommand) -> None:
        """Apply the motion model to every particle."""
        if abs(motion.forward_m) < 1e-9 and abs(motion.turn_rad) < 1e-9:
            return

        forward_samples = motion.forward_m + self.rng.normal(0.0, self.motion_forward_std_m, size=self.particle_count)
        turn_samples = motion.turn_rad + self.rng.normal(0.0, self.motion_turn_std_rad, size=self.particle_count)

        candidate_heading = wrap_to_pi(self.heading_rad + turn_samples)
        candidate_x = self.x_m + forward_samples * np.cos(candidate_heading)
        candidate_y = self.y_m + forward_samples * np.sin(candidate_heading)

        invalid_before_repair = ~self.map.are_free(candidate_x, candidate_y)
        self._repair_invalid_positions(
            candidate_x,
            candidate_y,
            base_x=self.x_m,
            base_y=self.y_m,
            jitter_std_m=max(self.motion_forward_std_m, self.roughening_position_std_m),
        )

        self.x_m = candidate_x
        self.y_m = candidate_y
        self.heading_rad = candidate_heading

        if np.any(invalid_before_repair):
            self.weights[invalid_before_repair] *= 0.85
            self.weights /= np.sum(self.weights)

    def update_from_image(
        self,
        image: np.ndarray,
        motion: Optional[MotionCommand] = None,
        do_resample: bool = True,
    ) -> LocalizationUpdate:
        """Full predict-update cycle using the configured chair detector."""
        if self.detector is None:
            raise ValueError("update_from_image() requires a configured ChairObservationDetector.")

        if self.enable_range_fov_gating and self.camera_half_fov_rad is None:
            self.camera_half_fov_rad = 0.5 * self.detector.horizontal_fov_rad(image.shape[1])

        observations = self.detector.observations_from_image(image)
        return self.update_from_observations(
            observations=observations,
            motion=motion,
            do_resample=do_resample,
        )

    def update_from_observations(
        self,
        observations: Sequence[ChairObservation],
        motion: Optional[MotionCommand] = None,
        do_resample: bool = True,
    ) -> LocalizationUpdate:
        """Update particle weights from externally supplied chair observations."""
        if motion is not None:
            self.predict(motion)

        observations = tuple(observations)
        if not observations:
            estimate = self.estimate_pose(matched_landmarks=0, observation_count=0)
            update = LocalizationUpdate(
                estimate=estimate,
                observations=tuple(),
                matches=tuple(),
                best_particle_index=int(np.argmax(self.weights)),
            )
            self.last_update = update
            return update

        prior_log_weights = np.log(np.clip(self.weights, 1e-300, None))
        posterior_log_weights = np.zeros_like(prior_log_weights)
        association_results: list[AssociationResult] = []

        for particle_index in range(self.particle_count):
            particle_pose = Pose2D(
                x_m=float(self.x_m[particle_index]),
                y_m=float(self.y_m[particle_index]),
                heading_rad=float(self.heading_rad[particle_index]),
            )

            max_range_m = self.measurement_max_range_m if self.enable_range_fov_gating else None
            half_fov_rad = self.camera_half_fov_rad if self.enable_range_fov_gating else None
            predictions = self.map.visible_landmarks(
                particle_pose,
                max_range_m=max_range_m,
                half_fov_rad=half_fov_rad,
            )
            # Associate observations with visible predictions
            result = self.associator.associate(observations, predictions)
            association_results.append(result)
            # Update weight using likelihood directly (not log)
            posterior_log_weights[particle_index] = prior_log_weights[particle_index] + math.log(max(result.likelihood, 1e-300))

        max_log_weight = float(np.max(posterior_log_weights))
        normalized_weights = np.exp(posterior_log_weights - max_log_weight)
        weight_sum = float(np.sum(normalized_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            normalized_weights = np.full(self.particle_count, 1.0 / self.particle_count, dtype=np.float64)
        else:
            normalized_weights /= weight_sum

        self.weights = normalized_weights
        best_particle_index = int(np.argmax(self.weights))
        best_association = association_results[best_particle_index]

        if do_resample and self.effective_sample_size() < self.resample_threshold * self.particle_count:
            self._resample_particles()

        estimate = self.estimate_pose(
            matched_landmarks=len(best_association.matches),
            observation_count=len(observations),
        )
        update = LocalizationUpdate(
            estimate=estimate,
            observations=observations,
            matches=best_association.matches,
            best_particle_index=best_particle_index,
        )
        self.last_update = update
        return update

    def effective_sample_size(self) -> float:
        """Return the standard particle-filter ESS metric."""
        return float(1.0 / np.sum(np.square(np.clip(self.weights, 1e-300, None))))

    def _resample_particles(self) -> None:
        """Resample particles using the configured method."""
        if self.resampling_method == "systematic":
            indices = self._systematic_resample_indices()
        else:
            indices = self._wheel_resample_indices()
        self._apply_resampled_indices(indices)

    def _systematic_resample_indices(self) -> np.ndarray:
        """Return resampled particle indices using systematic resampling."""
        step = 1.0 / self.particle_count
        start = self.rng.uniform(0.0, step)
        positions = start + step * np.arange(self.particle_count)

        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="left")

    def _wheel_resample_indices(self) -> np.ndarray:
        """Return resampled particle indices using the old roulette-wheel method."""
        weights = np.clip(self.weights, 0.0, None)
        max_weight = float(np.max(weights))
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            return self.rng.integers(0, self.particle_count, size=self.particle_count)

        indices = np.empty(self.particle_count, dtype=np.int64)
        index = int(self.rng.integers(0, self.particle_count))
        beta = 0.0

        for sample_index in range(self.particle_count):
            beta += float(self.rng.random()) * 2.0 * max_weight
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % self.particle_count
            indices[sample_index] = index

        return indices

    def _apply_resampled_indices(self, indices: np.ndarray) -> None:
        """Copy, roughen, and repair particles selected by a resampler."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.shape != (self.particle_count,):
            raise ValueError("resampled indices must have shape (particle_count,)")

        resampled_x = self.x_m[indices].copy()
        resampled_y = self.y_m[indices].copy()
        resampled_heading = self.heading_rad[indices].copy()

        resampled_x += self.rng.normal(0.0, self.roughening_position_std_m, size=self.particle_count)
        resampled_y += self.rng.normal(0.0, self.roughening_position_std_m, size=self.particle_count)
        resampled_heading = wrap_to_pi(
            resampled_heading + self.rng.normal(0.0, self.roughening_heading_std_rad, size=self.particle_count)
        )

        self._repair_invalid_positions(
            resampled_x,
            resampled_y,
            base_x=self.x_m[indices],
            base_y=self.y_m[indices],
            jitter_std_m=max(self.roughening_position_std_m, self.motion_forward_std_m),
        )

        self.x_m = resampled_x
        self.y_m = resampled_y
        self.heading_rad = resampled_heading
        self.weights.fill(1.0 / self.particle_count)

    def _systematic_resample(self) -> None:
        """Backward-compatible systematic resampling entry point."""
        self._apply_resampled_indices(self._systematic_resample_indices())

    def estimate_pose(
        self,
        matched_landmarks: int,
        observation_count: int,
    ) -> LocalizationEstimate:
        """Compute a weighted pose and uncertainty estimate from the current cloud."""
        x_mean = float(np.sum(self.x_m * self.weights))
        y_mean = float(np.sum(self.y_m * self.weights))
        heading_mean = weighted_circular_mean(self.heading_rad, self.weights)

        x_std = float(np.sqrt(np.sum(self.weights * np.square(self.x_m - x_mean))))
        y_std = float(np.sqrt(np.sum(self.weights * np.square(self.y_m - y_mean))))
        heading_std = weighted_circular_std(self.heading_rad, self.weights)

        return LocalizationEstimate(
            pose=Pose2D(x_m=x_mean, y_m=y_mean, heading_rad=heading_mean),
            position_std_m=(x_std, y_std),
            heading_std_rad=heading_std,
            effective_sample_size=self.effective_sample_size(),
            matched_landmarks=matched_landmarks,
            observation_count=observation_count,
        )

    def particle_array(self) -> np.ndarray:
        """Return the particle cloud as an Nx4 array of x, y, heading, weight."""
        return np.column_stack((self.x_m, self.y_m, self.heading_rad, self.weights))