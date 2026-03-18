"""Minimal demo entry point for the chair-based particle filter localizer."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import cv2


if __package__ in {None, ""}:
    SLAM_ROVER_DIR = Path(__file__).resolve().parent.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from localization.detector import ChairObservationDetector
    from localization.map_data import ChairLocalizationMap
    from localization.particle_filter import ParticleFilterLocalizer
    from localization.types import MotionCommand, Pose2D
    from localization.visualization import LocalizationVisualizer, VisualizationConfig
else:
    from .detector import ChairObservationDetector
    from .map_data import ChairLocalizationMap
    from .particle_filter import ParticleFilterLocalizer
    from .types import MotionCommand, Pose2D
    from .visualization import LocalizationVisualizer, VisualizationConfig

from sensor_interface import USBCamera


LOCALIZATION_DIR = Path(__file__).resolve().parent
SLAM_ROVER_DIR = LOCALIZATION_DIR.parent

# File inputs
MAP_PATH = SLAM_ROVER_DIR / "inputs" / "classmap.png"
LANDMARKS_PATH = SLAM_ROVER_DIR / "inputs" / "landmarks.json"
TEST_IMAGE_PATH = SLAM_ROVER_DIR / "inputs" / "localize_imgs" / "prime1.jpg"
USE_TEST_IMAGE = True
MAP_RESOLUTION_M_PER_PX = 0.0197

# Camera configuration
CAMERA_ID = 0
FX = 589.54200724
FY = 589.80048532
CX = 328.93066342
CY = 200.86625768 * 0.85

# Particle filter configuration
PARTICLE_COUNT = 1000
LOAD_DEFAULT_DETECTOR = True
CAPTURE_RETRY_COUNT = 20
CAPTURE_RETRY_DELAY_MS = 50

# Set to None for global initialization over free space.
KNOWN_INITIAL_POSE: Optional[Pose2D] = Pose2D(
    x_m=2,
    y_m=2,
    heading_rad=math.radians(00.0),
)
KNOWN_INITIAL_POSE: Optional[Pose2D] = None

# Set a non-zero motion command when the robot has moved since the previous update.
MOTION_COMMAND = MotionCommand(forward_m=0.0, turn_rad=0.0)

# Visualization
VISUALIZATION_CONFIG = VisualizationConfig(
    show_observation_preview=True,
    preview_window_name="Chair Observations",
    show_map_preview=True,
    map_window_name="Localization Map",
    map_preview_scale=1.5,
    draw_particles=True,
    max_particles_to_draw=1000,
    trajectory_history=150,
)
LIVE_VIEW_WAIT_MS = 1
# In test-image mode: how many filter update cycles to run on the same detections.
# Each cycle lets you watch the particles contract toward the most likely robot position.
CONVERGENCE_ITERATIONS = 5
# Milliseconds to pause between convergence steps (set to 0 to require a keypress each step).
UPDATE_STEP_DELAY_MS = 1500


def validate_demo_configuration() -> None:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Map image not found: {MAP_PATH}")
    if not LANDMARKS_PATH.exists():
        raise FileNotFoundError(f"Landmarks file not found: {LANDMARKS_PATH}")
    if USE_TEST_IMAGE and not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")


def print_localization_result(estimate, update_index: int) -> None:
    print(f"Update {update_index}")
    print(
        f"  Pose: x={estimate.pose.x_m:.2f} m, y={estimate.pose.y_m:.2f} m, "
        f"heading={math.degrees(estimate.pose.heading_rad):.1f} deg"
    )
    print(f"  Position std: ({estimate.position_std_m[0]:.2f}, {estimate.position_std_m[1]:.2f}) m")
    print(f"  Heading std: {math.degrees(estimate.heading_std_rad):.1f} deg")
    print(f"  Observations: {estimate.observation_count}")
    print(f"  Matched landmarks: {estimate.matched_landmarks}")
    print(f"  ESS: {estimate.effective_sample_size:.1f}")


def main() -> None:
    validate_demo_configuration()

    localization_map = ChairLocalizationMap.from_files(
        map_path=str(MAP_PATH),
        landmarks_path=str(LANDMARKS_PATH),
        resolution_m_per_px=MAP_RESOLUTION_M_PER_PX,
    )
    detector = ChairObservationDetector(
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        load_default_detector=LOAD_DEFAULT_DETECTOR,
    )
    localizer = ParticleFilterLocalizer(
        localization_map=localization_map,
        detector=detector,
        particle_count=PARTICLE_COUNT,
        initial_pose=KNOWN_INITIAL_POSE,
    )
    visualizer = LocalizationVisualizer(config=VISUALIZATION_CONFIG)

    camera = None
    if not USE_TEST_IMAGE:
        camera = USBCamera(camera_id=CAMERA_ID)
        if not camera.start():
            raise SystemExit("Failed to start the camera.")

    try:
        update_index = 0

        if USE_TEST_IMAGE:
            frame = cv2.imread(str(TEST_IMAGE_PATH))
            if frame is None:
                raise SystemExit(f"Failed to load test image: {TEST_IMAGE_PATH}")

            # --- Phase 0: show the prior (initial particle spread) ---
            print("Showing initial particle spread. Press any key to begin update cycles...")
            visualizer.show_update(None, localization_map, localizer, None)
            cv2.waitKey(0)

            # Detect chairs once and reuse observations across all convergence cycles.
            # This lets you watch the filter contract from a global prior to a confident estimate.
            observations = detector.observations_from_image(frame)
            print(f"Detected {len(observations)} chair observation(s). "
                  f"Running {CONVERGENCE_ITERATIONS} convergence cycle(s)...")

            for iteration in range(CONVERGENCE_ITERATIONS):
                # Apply the motion command only during the first cycle
                motion = MOTION_COMMAND if iteration == 0 else None
                update = localizer.update_from_observations(observations, motion=motion)
                update_index += 1
                print_localization_result(update.estimate, update_index)
                visualizer.show_update(frame, localization_map, localizer, update)

                delay = UPDATE_STEP_DELAY_MS if UPDATE_STEP_DELAY_MS > 0 else 0
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q")):
                    break

            # Hold the final result until the user closes the window
            if VISUALIZATION_CONFIG.show_observation_preview or VISUALIZATION_CONFIG.show_map_preview:
                cv2.waitKey(0)
        else:
            # Show the initial spread briefly before the live update loop
            print("Showing initial particle spread...")
            visualizer.show_update(None, localization_map, localizer, None)
            cv2.waitKey(1500)

            while True:
                frame = None
                for _ in range(CAPTURE_RETRY_COUNT):
                    frame = camera.get_frame()
                    if frame is not None:
                        break
                    cv2.waitKey(CAPTURE_RETRY_DELAY_MS)

                if frame is None:
                    raise SystemExit("No frame received from the camera.")

                update = localizer.update_from_image(frame, motion=MOTION_COMMAND)
                update_index += 1

                print_localization_result(update.estimate, update_index)
                visualizer.show_update(frame, localization_map, localizer, update)

                key = cv2.waitKey(LIVE_VIEW_WAIT_MS) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        if camera is not None:
            camera.stop()
        visualizer.close()


if __name__ == "__main__":
    main()