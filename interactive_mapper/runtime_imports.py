"""Runtime-safe imports for modules that live outside this package folder."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SLAM_ROVER_DIR = PACKAGE_DIR.parent
METRIC_DEPTH_DIR = SLAM_ROVER_DIR.parent

# Ensure both slam_rover/ and metric_depth/ are importable when running scripts directly.
for import_root in (SLAM_ROVER_DIR, METRIC_DEPTH_DIR):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from sensor_interface import CameraInterface, IMUInterface, SensorFactory
from waveshare_robot_controller import WaveRoverController

__all__ = [
    "CameraInterface",
    "IMUInterface",
    "SensorFactory",
    "WaveRoverController",
]
