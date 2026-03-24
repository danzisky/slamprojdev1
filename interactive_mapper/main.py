"""Constants-based launcher for the modular interactive mapper package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import cv2

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
        MULTI_ROBOT_CLOSE_LOOP,
        MULTI_ROBOT_ESTIMATED_SPEED_MPS,
        MULTI_ROBOT_MIN_OBSTACLE_DISTANCE_PX,
        MULTI_ROBOT_MIN_PATH_CLOSENESS_PX,
        MULTI_ROBOT_MIN_SEGMENT_POINTS,
        MULTI_ROBOT_OUTPUT_IMAGE,
        MULTI_ROBOT_PATROL_MODE,
        MULTI_ROBOT_SHOW_WINDOW,
        MULTI_ROBOT_STARTS,
        MULTI_ROBOT_SWEEP_STEP_PX,
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
    from interactive_mapper.multi_robot_sections import (  # type: ignore
        SectionVisualizer,
        draw_route,
        partition_map_and_plan_patrols,
    )
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
        MULTI_ROBOT_CLOSE_LOOP,
        MULTI_ROBOT_ESTIMATED_SPEED_MPS,
        MULTI_ROBOT_MIN_OBSTACLE_DISTANCE_PX,
        MULTI_ROBOT_MIN_PATH_CLOSENESS_PX,
        MULTI_ROBOT_MIN_SEGMENT_POINTS,
        MULTI_ROBOT_OUTPUT_IMAGE,
        MULTI_ROBOT_PATROL_MODE,
        MULTI_ROBOT_SHOW_WINDOW,
        MULTI_ROBOT_STARTS,
        MULTI_ROBOT_SWEEP_STEP_PX,
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
    from .multi_robot_sections import SectionVisualizer, draw_route, partition_map_and_plan_patrols


def run() -> None:
    """Run mapper using values from config.py (no command-line arguments)."""

    if LANDMARK_SETUP_MODE:
        run_landmark_setup_session(
            map_file=MAP_FILE,
            landmarks_file=LANDMARKS_FILE,
            window_name=LANDMARK_SETUP_WINDOW_NAME,
        )
        return

    if MULTI_ROBOT_PATROL_MODE:
        overlay, sections, patrols, metrics = partition_map_and_plan_patrols(
            map_file=MAP_FILE,
            robot_starts=MULTI_ROBOT_STARTS,
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

        print("\n" + "=" * 60)
        print("MULTI-ROBOT SECTION PATROL ASSIGNMENT")
        print("=" * 60)
        for item in metrics:
            print(
                f"R{item.robot_index}: "
                f"coverage={item.coverage_percent:.1f}% "
                f"overlap={item.overlap_percent:.1f}% "
                f"length={item.route_length_m:.2f}m "
                f"cycle={item.estimated_cycle_time_s:.1f}s "
                f"waypoints={item.route_points}"
            )
        print("=" * 60)

        output_path = Path(MULTI_ROBOT_OUTPUT_IMAGE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), viz)
        print(f"Saved multi-robot patrol overlay to: {output_path}")

        if MULTI_ROBOT_SHOW_WINDOW:
            cv2.imshow("Multi-Robot Sections + Patrols", viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
