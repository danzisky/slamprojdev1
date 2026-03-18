"""Small math helpers shared across the localization package."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle):
    """Wrap an angle or array of angles to [-pi, pi)."""
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if np.isscalar(angle):
        return float(wrapped)
    return wrapped


def weighted_circular_mean(angles, weights):
    """Compute the weighted mean of circular angles."""
    angles_array = np.asarray(angles, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    sin_sum = np.sum(np.sin(angles_array) * weights_array)
    cos_sum = np.sum(np.cos(angles_array) * weights_array)
    return float(np.arctan2(sin_sum, cos_sum))


def weighted_circular_std(angles, weights):
    """Compute a circular standard deviation in radians."""
    angles_array = np.asarray(angles, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights_array))
    if weight_sum <= 0.0:
        return float(np.pi)

    sin_sum = np.sum(np.sin(angles_array) * weights_array)
    cos_sum = np.sum(np.cos(angles_array) * weights_array)
    resultant = np.hypot(sin_sum, cos_sum) / weight_sum
    resultant = float(np.clip(resultant, 1e-9, 1.0))
    return float(np.sqrt(max(0.0, -2.0 * np.log(resultant))))