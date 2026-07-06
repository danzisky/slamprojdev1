"""VL53L5CX rear-mounted ToF depth integration for particle-filter localization.

Sensor geometry
---------------
The VL53L5CX is an 8×8 multi-zone ToF sensor with a **65° diagonal square FoV**,
which corresponds to approximately **45° × 45°** horizontal × vertical FOV
(diagonal = 45° × √2 ≈ 63.6°, close to the 65° spec).

Zone layout in the 8×8 grid (as received from the driver):
  - Column index increases left → right  (horizontal azimuth)
  - Row index increases top → bottom     (vertical elevation)
  - Zone size: 45° / 8 zones = 5.625° per zone

Horizontal angle of zone column `j`  (j = 0..7):
    δ_h = (j − 3.5) × 5.625°
    →  −19.69°  −14.06°  −8.44°  −2.81°  +2.81°  +8.44°  +14.06°  +19.69°

Default "centre strip" zone selection
--------------------------------------
Rows 3 and 4 (the two vertical-centre rows), all 8 columns → **16 zones**.
This keeps horizontal coverage at ±19.69° while restricting vertical elevation to
±2.81° from horizontal.  Useful for a sensor mounted level on the rover.

To approximate the user's suggested ~26 active zones use the wider factory preset::

    config = ToFConfig(zone_mask=ToFConfig.wide_strip_mask())   # 32 zones, rows 2-5

Or supply any custom boolean 8×8 numpy array as ``zone_mask``.

Rear mounting
-------------
The sensor boresight faces **opposite** to the rover's forward heading.
Ray directions from particle (x, y, θ):
    world_angle = θ + π + δ_h      (for each active horizontal zone)

An optional ``sensor_offset_rear_m`` displaces the ray origin backwards along
the rover body axis (positive = further toward the rear).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from .types import ToFFrame

if TYPE_CHECKING:
    from .map_data import ChairLocalizationMap

# ──────────────────────────────────────────────────────────────────────────────
# Sensor constants (VL53L5CX)
# ──────────────────────────────────────────────────────────────────────────────

_ROWS = 8
_COLS = 8
# Azimuth (horizontal) and elevation (vertical) full FoV in degrees.
# Each of the 8 columns/rows subtends ZONE_DEG degrees.
_H_FOV_DEG: float = 45.0
_V_FOV_DEG: float = 45.0
_ZONE_DEG: float = _H_FOV_DEG / _COLS          # 5.625°

# Minimum valid range reported by the sensor (metres).
_MIN_VALID_RANGE_M: float = 0.02


# ──────────────────────────────────────────────────────────────────────────────
# Zone-mask helpers
# ──────────────────────────────────────────────────────────────────────────────

def _centre_strip_mask() -> np.ndarray:
    """Centre 2 rows (rows 3-4), all 8 columns → 16 zones, ±19.69° H, ±2.81° V."""
    mask = np.zeros((_ROWS, _COLS), dtype=bool)
    mask[3:5, :] = True
    return mask


def _middle_strip_mask() -> np.ndarray:
    """Single middle row (row 4), all 8 columns -> 8 zones for 2D horizontal rays."""
    mask = np.zeros((_ROWS, _COLS), dtype=bool)
    mask[4, :] = True
    return mask


def _wide_strip_mask() -> np.ndarray:
    """Centre 4 rows (rows 2-5), all 8 columns → 32 zones, ±19.69° H, ±11.25° V."""
    mask = np.zeros((_ROWS, _COLS), dtype=bool)
    mask[2:6, :] = True
    return mask


def _zone_horizontal_offsets_rad(zone_mask: np.ndarray) -> np.ndarray:
    """Return the horizontal angle offset (rad) for each active zone in *zone_mask*."""
    _, active_cols = np.where(zone_mask)
    return np.deg2rad((active_cols.astype(np.float64) - (_COLS - 1) / 2.0) * _ZONE_DEG)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class ToFConfig:
    """Geometry and noise parameters for the rear-mounted VL53L5CX sensor.

    Parameters
    ----------
    zone_mask:
        Boolean array of shape (8, 8) selecting which zones to use.
        ``None`` → centre strip (rows 3-4, 16 zones).
    h_fov_deg:
        Total horizontal FoV of the sensor in degrees.  Default 45°.
    max_range_m:
        Readings beyond this distance are clamped to this value (metres).
    range_std_m:
        1-σ measurement noise for range comparisons (metres).
    raycast_step_m:
        Step size used by the vectorised raycast (metres).
        Smaller → more accurate but slower.  Default 0.08 m (≈ map resolution).
    sensor_offset_rear_m:
        Distance (metres) from the rover's rotation centre to the sensor,
        measured along the rear direction.  0 = sensor at the pivot point.
    """

    def __init__(
        self,
        zone_mask: Optional[np.ndarray] = None,
        h_fov_deg: float = _H_FOV_DEG,
        max_range_m: float = 3.0,
        range_std_m: float = 0.06,
        raycast_step_m: float = 0.08,
        sensor_offset_rear_m: float = 0.0,
        use_middle_strip_2d: bool = False,
    ) -> None:
        if zone_mask is None:
            zone_mask = _centre_strip_mask()
        zone_mask = np.asarray(zone_mask, dtype=bool)
        if zone_mask.shape != (_ROWS, _COLS):
            raise ValueError(f"zone_mask must be (8, 8), got {zone_mask.shape}")
        if not np.any(zone_mask):
            raise ValueError("zone_mask must have at least one active zone")

        self.zone_mask = zone_mask
        self.h_fov_deg = float(h_fov_deg)
        self.max_range_m = float(max_range_m)
        self.range_std_m = float(range_std_m)
        self.raycast_step_m = float(raycast_step_m)
        self.sensor_offset_rear_m = float(sensor_offset_rear_m)
        self.use_middle_strip_2d = bool(use_middle_strip_2d)

        # Pre-computed per-zone quantities
        active_rows, active_cols = np.where(zone_mask)
        self._active_rows: np.ndarray = active_rows                     # (K,)
        self._active_cols: np.ndarray = active_cols                     # (K,)
        self._n_zones: int = int(active_rows.size)

        zone_deg = h_fov_deg / _COLS
        self._hz_offsets_rad: np.ndarray = np.deg2rad(                  # (K,)
            (active_cols.astype(np.float64) - (_COLS - 1) / 2.0) * zone_deg
        )

    # Convenience factory methods ──────────────────────────────────────────────

    @staticmethod
    def centre_strip_mask() -> np.ndarray:
        """Centre 2 rows, all 8 columns (16 zones, ±19.69° H)."""
        return _centre_strip_mask()

    @staticmethod
    def wide_strip_mask() -> np.ndarray:
        """Centre 4 rows, all 8 columns (32 zones, ±19.69° H, ±11.25° V)."""
        return _wide_strip_mask()

    @staticmethod
    def middle_strip_mask() -> np.ndarray:
        """Single middle row, all 8 columns (8 zones, ±19.69° H)."""
        return _middle_strip_mask()

    def __repr__(self) -> str:
        return (
            f"ToFConfig(n_zones={self._n_zones}, h_fov_deg={self.h_fov_deg}, "
            f"max_range_m={self.max_range_m}, range_std_m={self.range_std_m}, "
            f"use_middle_strip_2d={self.use_middle_strip_2d})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised raycast
# ──────────────────────────────────────────────────────────────────────────────

def _vectorised_raycast(
    occupancy_grid: np.ndarray,
    resolution_m_per_px: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    ray_angles_rad: np.ndarray,
    max_range_m: float,
    step_m: float,
) -> np.ndarray:
    """Return predicted wall distance for N rays, all starting from different poses.

    Parameters
    ----------
    occupancy_grid : ndarray (H, W) uint8
        0 = free, 255 = occupied.
    resolution_m_per_px : float
        Metres per pixel.
    x_m, y_m : ndarray (N,)
        Ray origin coordinates in world frame.
    ray_angles_rad : ndarray (N,)
        Ray direction for each particle (world frame, radians).
    max_range_m : float
        Maximum distance to march before returning max.
    step_m : float
        March step size in metres.

    Returns
    -------
    ndarray (N,) float64
        Predicted distance for each particle ray.
    """
    grid_h, grid_w = occupancy_grid.shape
    n_steps = int(math.ceil(max_range_m / step_m)) + 1

    cos_a = np.cos(ray_angles_rad)
    sin_a = np.sin(ray_angles_rad)

    distances = np.full(len(x_m), max_range_m, dtype=np.float64)
    hit = np.zeros(len(x_m), dtype=bool)

    for s in range(n_steps):
        active = ~hit
        if not np.any(active):
            break
        d = s * step_m

        px_m = x_m[active] + d * cos_a[active]
        py_m = y_m[active] + d * sin_a[active]

        x_px = np.rint(px_m / resolution_m_per_px).astype(np.int32)
        y_px = np.rint(py_m / resolution_m_per_px).astype(np.int32)

        out_of_bounds = (
            (x_px < 0) | (x_px >= grid_w) | (y_px < 0) | (y_px >= grid_h)
        )
        active_indices = np.flatnonzero(active)

        new_hits = out_of_bounds.copy()
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            occupied = occupancy_grid[y_px[in_bounds], x_px[in_bounds]] == 255
            new_hits[in_bounds] = occupied

        hit_global = active_indices[new_hits]
        distances[hit_global] = d
        hit[hit_global] = True

    return distances


# ──────────────────────────────────────────────────────────────────────────────
# Integrator
# ──────────────────────────────────────────────────────────────────────────────

class ToFIntegrator:
    """Compute per-particle log-likelihoods from a VL53L5CX depth frame.

    Usage::

        config = ToFConfig()                    # or pass custom zone_mask/params
        integrator = ToFIntegrator(loc_map, config)

        # Inside the particle filter update loop:
        tof_ll = integrator.compute_log_likelihoods(
            x_m, y_m, headings_rad, tof_frame
        )
        log_weights += tof_weight * tof_ll

    The sensor is assumed to face **rearward** (boresight = heading + π).
    Each active zone contributes an independent Gaussian likelihood:
        log p ∝ -0.5 × ((measured − predicted) / σ)²
    Invalid / out-of-range readings are silently skipped.
    """

    def __init__(
        self,
        localization_map: "ChairLocalizationMap",
        config: Optional[ToFConfig] = None,
    ) -> None:
        self.map = localization_map
        self.config = config or ToFConfig()

    def compute_log_likelihoods(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        headings_rad: np.ndarray,
        tof_frame: "ToFFrame",  # type: ignore[name-defined]
    ) -> np.ndarray:
        """Return shape-(N,) log-likelihood contribution for each particle.

        Parameters
        ----------
        x_m, y_m, headings_rad : ndarray (N,)
            Current particle poses.
        tof_frame : ToFFrame
            Sensor reading containing an (8, 8) distance array in metres.

        Returns
        -------
        ndarray (N,) float64
            Sum of per-zone log-likelihoods.  All zeros if no valid zones.
        """
        cfg = self.config
        ranges = tof_frame.ranges_m          # (8, 8)
        log_ll = np.zeros(len(x_m), dtype=np.float64)

        # Sensor origin: offset rearwards from particle centre.
        rear_angle = headings_rad + math.pi
        if cfg.sensor_offset_rear_m > 0.0:
            sensor_x = x_m + cfg.sensor_offset_rear_m * np.cos(rear_angle)
            sensor_y = y_m + cfg.sensor_offset_rear_m * np.sin(rear_angle)
        else:
            sensor_x = x_m
            sensor_y = y_m

        inv2sigma2 = 0.5 / (cfg.range_std_m ** 2)
        occ = self.map.occupancy_grid
        res = self.map.resolution_m_per_px
        step = cfg.raycast_step_m
        max_r = cfg.max_range_m

        if cfg.use_middle_strip_2d:
            zone_deg = cfg.h_fov_deg / _COLS
            hz_offsets = np.deg2rad((np.arange(_COLS, dtype=np.float64) - (_COLS - 1) / 2.0) * zone_deg)
            for col in range(_COLS):
                # Collapse the two center rows into one 2D horizontal strip measurement.
                measured = float(np.nanmean(ranges[3:5, col]))
                if not math.isfinite(measured) or measured < _MIN_VALID_RANGE_M:
                    continue
                measured = min(measured, max_r)

                ray_angles = rear_angle + hz_offsets[col]
                predicted = _vectorised_raycast(occ, res, sensor_x, sensor_y, ray_angles, max_r, step)
                diff = measured - predicted
                log_ll -= inv2sigma2 * (diff * diff)
            return log_ll

        for k in range(cfg._n_zones):
            row = int(cfg._active_rows[k])
            col = int(cfg._active_cols[k])
            measured = float(ranges[row, col])

            # Skip invalid or out-of-sensor-range readings.
            if not math.isfinite(measured) or measured < _MIN_VALID_RANGE_M:
                continue
            measured = min(measured, max_r)

            ray_angles = rear_angle + cfg._hz_offsets_rad[k]
            predicted = _vectorised_raycast(
                occ, res, sensor_x, sensor_y, ray_angles, max_r, step
            )

            diff = measured - predicted
            log_ll -= inv2sigma2 * (diff * diff)

        return log_ll

    def zone_count(self) -> int:
        """Number of active zones used for scoring."""
        return self.config._n_zones

    def fov_summary(self) -> str:
        """Human-readable summary of the active zone angular coverage."""
        offsets_deg = np.rad2deg(self.config._hz_offsets_rad)
        return (
            f"{self.zone_count()} active zones | "
            f"H azimuth: {offsets_deg.min():.1f}° to {offsets_deg.max():.1f}° "
            f"(boresight = rover rear)"
        )
