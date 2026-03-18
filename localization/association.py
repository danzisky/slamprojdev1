"""Chair-landmark data association using Hungarian or ICP-style matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np

from .math_utils import wrap_to_pi
from .types import ChairObservation, PredictedLandmarkMeasurement

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

if TYPE_CHECKING:  # pragma: no cover
    from .map_data import ChairLocalizationMap


@dataclass(frozen=True)
class AssociationResult:
    """Best assignment found for one particle hypothesis."""

    matches: Tuple[Tuple[int, int], ...]
    unmatched_observations: Tuple[int, ...]
    unmatched_predictions: Tuple[int, ...]
    likelihood: float


class JointCompatibilityAssociator:
    """
    Associate observed chairs with known landmarks using Hungarian or ICP-style matching.
    
    Simplified implementation directly based on particle_filter_localizer.py logic.
    """

    def __init__(
        self,
        default_range_std_m: float = 0.5,
        default_bearing_std_rad: float = math.radians(4.0),
        matching_method: str = "hungarian",
    ):
        self.default_range_std_m = float(default_range_std_m)
        self.default_bearing_std_rad = float(default_bearing_std_rad)
        self.matching_method = matching_method.lower().strip()
        if self.matching_method not in {"hungarian", "icp"}:
            raise ValueError("matching_method must be 'hungarian' or 'icp'")
        self.visible_landmarks = None

    def set_occlusion_context(
        self,
        visible_landmarks: Sequence[Tuple[float, float, int]] | None = None,
    ) -> None:
        """Store visible landmarks for occlusion checking during association."""
        self.visible_landmarks = visible_landmarks

    def associate(
        self,
        observations: Sequence[ChairObservation],
        predictions: Sequence[PredictedLandmarkMeasurement],
    ) -> AssociationResult:
        """
        Return the best assignment according to the configured matching method.
        
        Returns a likelihood value (not log-likelihood) for direct comparison to particle weights.
        """
        observations = tuple(observations)
        predictions = tuple(predictions)
        
        if not observations or not predictions:
            # Empty cases: no assignment possible
            return AssociationResult(
                matches=tuple(),
                unmatched_observations=tuple(range(len(observations))),
                unmatched_predictions=tuple(range(len(predictions))),
                likelihood=1e-300,
            )

        # Build cost matrix: range_err^2 + bearing_err^2 (Mahalanobis-like)
        cost_matrix = self._build_score_matrix(observations, predictions)
        
        # Find best assignment
        matches = self._assign_matches(cost_matrix)
        
        # Compute total match cost
        if matches:
            total_cost = float(sum(cost_matrix[i, j] for i, j in matches))
        else:
            total_cost = 0.0
        
        # Likelihood from assignment cost
        likelihood = math.exp(-0.5 * total_cost)
        
        # Old-code style count mismatch penalty: only when we observed more than predicted visible.
        if len(predictions) < len(observations):
            likelihood *= 0.1 ** (len(observations) - len(predictions))
        
        # Occlusion consistency check (like particle_filter_localizer.py)
        likelihood *= 1/self._occlusion_penalty(observations, predictions, matches)
        
        # Extract unmatched indices
        matched_observations = {i for i, _ in matches}
        matched_predictions = {j for _, j in matches}
        unmatched_observations = tuple(i for i in range(len(observations)) if i not in matched_observations)
        unmatched_predictions = tuple(j for j in range(len(predictions)) if j not in matched_predictions)

        return AssociationResult(
            matches=tuple(sorted(matches)),
            unmatched_observations=unmatched_observations,
            unmatched_predictions=unmatched_predictions,
            likelihood=max(likelihood, 1e-300),
        )

    def _build_score_matrix(
        self,
        observations: Sequence[ChairObservation],
        predictions: Sequence[PredictedLandmarkMeasurement],
    ) -> np.ndarray:
        """Build cost matrix using fixed sigmas (same as particle_filter_localizer.py)."""
        cost_matrix = np.zeros((len(observations), len(predictions)), dtype=np.float64)
        
        sigma_dist = self.default_range_std_m
        sigma_bearing = self.default_bearing_std_rad
        
        for i, observation in enumerate(observations):
            for j, prediction in enumerate(predictions):
                dist_err = (observation.range_m - prediction.range_m) / sigma_dist
                bearing_err = wrap_to_pi(observation.bearing_rad - prediction.bearing_rad) / sigma_bearing
                cost_matrix[i, j] = dist_err ** 2 + bearing_err ** 2
        
        return cost_matrix

    def _assign_matches(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Find best assignment using configured matching method."""
        if cost_matrix.size == 0:
            return []

        if self.matching_method == "icp":
            return self._assign_icp_greedy(cost_matrix)

        if linear_sum_assignment is None:
            return self._assign_icp_greedy(cost_matrix)

        return self._assign_hungarian(cost_matrix)

    def _assign_hungarian(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Hungarian algorithm on rectangular cost matrix (scipy handles it natively)."""
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]

    @staticmethod
    def _assign_icp_greedy(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Greedy ICP-style: repeatedly take lowest-cost remaining pair."""
        working = np.array(cost_matrix, copy=True, dtype=np.float64)
        matches = []
        
        while True:
            if working.size == 0:
                break
            
            # Find lowest cost among finite values
            if not np.any(np.isfinite(working)):
                break
            
            flat_idx = int(np.nanargmin(working))
            obs_idx, pred_idx = np.unravel_index(flat_idx, working.shape)
            
            if not np.isfinite(working[obs_idx, pred_idx]):
                break
            
            matches.append((int(obs_idx), int(pred_idx)))
            
            # Mask matched row and column
            working[obs_idx, :] = np.nan
            working[:, pred_idx] = np.nan
        
        return matches

    def _occlusion_penalty(
        self,
        observations: Sequence[ChairObservation],
        predictions: Sequence[PredictedLandmarkMeasurement],
        matches: Sequence[Tuple[int, int]],
    ) -> float:
        """
        Apply penalty if we match to a far prediction while leaving a closer unmatched one
        in approximately the same direction.
        
        Mimics the occlusion check from particle_filter_localizer.py.
        """
        bearing_tolerance = math.radians(8.6)  # ~8.6 degrees
        penalty = 1.0
        
        if not matches or not predictions:
            return penalty
        
        matched_pred_indices = {pred_idx for _, pred_idx in matches}
        
        for obs_idx, matched_pred_idx in matches:
            matched_pred = predictions[matched_pred_idx]
            matched_dist = matched_pred.range_m
            matched_bearing = matched_pred.bearing_rad
            
            # Check all unmatched predictions
            for unmatched_idx, pred in enumerate(predictions):
                if unmatched_idx in matched_pred_indices:
                    continue
                
                # Is it closer?
                if pred.range_m < matched_dist:
                    # Is it in roughly the same direction?
                    bearing_diff = abs(wrap_to_pi(pred.bearing_rad - matched_bearing))
                    
                    if bearing_diff < bearing_tolerance:
                        # Violation: matched to far while closer unmatched in same direction
                        penalty *= 1.5
        
        return penalty
