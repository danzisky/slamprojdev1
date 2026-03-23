"""Constants-based launcher for the modular interactive mapper package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    SLAM_ROVER_DIR = PACKAGE_DIR.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from interactive_mapper.config import (  # type: ignore
        AUTO_EXECUTE_PATH,
        CAMERA_ID,
        CAMERA_TYPE,
        CONNECT_ROBOT_ON_START,
        IMU_TYPE,
        LANDMARKS_FILE,
        LANDMARK_SETUP_MODE,
        LANDMARK_SETUP_WINDOW_NAME,
        MAP_FILE,
        MIN_WALL_DISTANCE_M,
        PHONE_IP,
        POLL_SENSORS_IN_LOOP,
        RESOLUTION_M_PER_PX,
        ROBOT_IP,
        RUN_WAIT_MS,
        SPEED_MULTIPLIER,
        START_HEADING_DEG,
        START_X,
        START_Y,
        USE_FUSED_INTERNAL_YAW,
        WINDOW_NAME,
    )
    from interactive_mapper.controller import InteractiveMapper  # type: ignore
    from interactive_mapper.landmark_setup import run_landmark_setup_session  # type: ignore
else:
    from .config import (
        AUTO_EXECUTE_PATH,
        CAMERA_ID,
        CAMERA_TYPE,
        CONNECT_ROBOT_ON_START,
        IMU_TYPE,
        LANDMARKS_FILE,
        LANDMARK_SETUP_MODE,
        LANDMARK_SETUP_WINDOW_NAME,
        MAP_FILE,
        MIN_WALL_DISTANCE_M,
        PHONE_IP,
        POLL_SENSORS_IN_LOOP,
        RESOLUTION_M_PER_PX,
        ROBOT_IP,
        RUN_WAIT_MS,
        SPEED_MULTIPLIER,
        START_HEADING_DEG,
        START_X,
        START_Y,
        USE_FUSED_INTERNAL_YAW,
        WINDOW_NAME,
    )
    from .controller import InteractiveMapper
    from .landmark_setup import run_landmark_setup_session


def run() -> None:
    """Run mapper using values from config.py (no command-line arguments)."""

    if LANDMARK_SETUP_MODE:
        run_landmark_setup_session(
            map_file=MAP_FILE,
            landmarks_file=LANDMARKS_FILE,
            window_name=LANDMARK_SETUP_WINDOW_NAME,
        )
        return

    mapper = InteractiveMapper(
        map_file=MAP_FILE,
        start_pos=(START_X, START_Y, START_HEADING_DEG),
        robot_ip=ROBOT_IP,
        resolution=RESOLUTION_M_PER_PX,
        min_wall_distance=MIN_WALL_DISTANCE_M,
        speed_multiplier=SPEED_MULTIPLIER,
        use_fused_internal_yaw=USE_FUSED_INTERNAL_YAW,
        auto_execute_path=AUTO_EXECUTE_PATH,
        window_name=WINDOW_NAME,
    )

    # Optional camera/IMU attachments are controlled by constants.
    if CAMERA_TYPE is not None:
        camera_kwargs: Dict[str, Any] = {}
        if CAMERA_TYPE == "usb":
            camera_kwargs["camera_id"] = CAMERA_ID
        elif CAMERA_TYPE == "android":
            camera_kwargs["phone_ip"] = PHONE_IP
        mapper.attach_camera(sensor_type=CAMERA_TYPE, start=True, **camera_kwargs)

    if IMU_TYPE is not None:
        imu_kwargs: Dict[str, Any] = {}
        if IMU_TYPE == "android":
            imu_kwargs["listen_port"] = 5555
        mapper.attach_imu(sensor_type=IMU_TYPE, start=True, **imu_kwargs)

    mapper.run(
        connect_robot=CONNECT_ROBOT_ON_START,
        wait_ms=RUN_WAIT_MS,
        poll_sensors=POLL_SENSORS_IN_LOOP,
    )


if __name__ == "__main__":
    run()
