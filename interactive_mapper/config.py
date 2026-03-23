"""Constants used to run the interactive mapper without CLI arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# Root of the slam_rover workspace.
SLAM_ROVER_DIR = Path(__file__).resolve().parent.parent

# Map and start pose configuration.
MAP_FILE = SLAM_ROVER_DIR / "inputs" / "classmap.png"
# MAP_FILE = SLAM_ROVER_DIR / "inputs" / "kitchenmap.png"
START_X = 100
START_Y = 100
START_HEADING_DEG = 90.0

# Robot and sensor configuration.
ROBOT_IP = "192.168.137.104"
PHONE_IP = "192.168.1.101"
CAMERA_TYPE: Optional[str] = None  # "usb", "android", "rb3", or None
IMU_TYPE: Optional[str] = "android"     # "android", "rb3", or None
CAMERA_ID = 0

# Planning and control configuration.
RESOLUTION_M_PER_PX = 0.0197
# RESOLUTION_M_PER_PX = 0.01
MIN_WALL_DISTANCE_M = 0.3
SPEED_MULTIPLIER = 1.0
USE_FUSED_INTERNAL_YAW = False
AUTO_EXECUTE_PATH = True
WINDOW_NAME = "Interactive Mapper"
LANDMARK_SETUP_WINDOW_NAME = "Landmark Setup"
LANDMARK_SETUP_MODE = False
LANDMARKS_FILE = SLAM_ROVER_DIR / "inputs" / "landmarks.json"

# Runtime behavior.
CONNECT_ROBOT_ON_START = False
POLL_SENSORS_IN_LOOP = True
RUN_WAIT_MS = 100
