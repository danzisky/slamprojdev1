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
else:
    from .detector import ChairObservationDetector
    from .map_data import ChairLocalizationMap
    from .particle_filter import ParticleFilterLocalizer
    from .types import MotionCommand, Pose2D

from sensor_interface import USBCamera


LOCALIZATION_DIR = Path(__file__).resolve().parent
SLAM_ROVER_DIR = LOCALIZATION_DIR.parent

# File inputs
MAP_PATH = SLAM_ROVER_DIR / "inputs" / "classmap.png"
LANDMARKS_PATH = SLAM_ROVER_DIR / "inputs" / "landmarks.json"
MAP_RESOLUTION_M_PER_PX = None

# Camera configuration
CAMERA_ID = 0
FX = 589.54200724
FY = 589.80048532
CX = 328.93066342
CY = 200.86625768 * 0.85

# Particle filter configuration
PARTICLE_COUNT = 800
LOAD_DEFAULT_DETECTOR = True
CAPTURE_RETRY_COUNT = 20
CAPTURE_RETRY_DELAY_MS = 50

# Set to None for global initialization over free space.
KNOWN_INITIAL_POSE: Optional[Pose2D] = None

# Example known-pose initialization:
# KNOWN_INITIAL_POSE = Pose2D(
#     x_m=1.2,
#     y_m=0.8,
#     heading_rad=math.radians(30.0),
# )

# Set a non-zero motion command when the robot has moved since the previous update.
MOTION_COMMAND = MotionCommand(forward_m=0.0, turn_rad=0.0)

# Visualization
SHOW_OBSERVATION_PREVIEW = True
PREVIEW_WINDOW_NAME = "Chair Observations"


def validate_demo_configuration() -> None:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Map image not found: {MAP_PATH}")
    if not LANDMARKS_PATH.exists():
        raise FileNotFoundError(f"Landmarks file not found: {LANDMARKS_PATH}")


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

    camera = USBCamera(camera_id=CAMERA_ID)
    if not camera.start():
        raise SystemExit("Failed to start the camera.")

    try:
        frame = None
        for _ in range(CAPTURE_RETRY_COUNT):
            frame = camera.get_frame()
            if frame is not None:
                break
            cv2.waitKey(CAPTURE_RETRY_DELAY_MS)

        if frame is None:
            raise SystemExit("No frame received from the camera.")

        update = localizer.update_from_image(frame, motion=MOTION_COMMAND)
        estimate = update.estimate

        print("Localization result")
        print(
            f"  Pose: x={estimate.pose.x_m:.2f} m, y={estimate.pose.y_m:.2f} m, "
            f"heading={math.degrees(estimate.pose.heading_rad):.1f} deg"
        )
        print(f"  Position std: ({estimate.position_std_m[0]:.2f}, {estimate.position_std_m[1]:.2f}) m")
        print(f"  Heading std: {math.degrees(estimate.heading_std_rad):.1f} deg")
        print(f"  Observations: {estimate.observation_count}")
        print(f"  Matched landmarks: {estimate.matched_landmarks}")
        print(f"  ESS: {estimate.effective_sample_size:.1f}")

        if SHOW_OBSERVATION_PREVIEW and update.observations:
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

            cv2.imshow(PREVIEW_WINDOW_NAME, preview)
            cv2.waitKey(0)
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()