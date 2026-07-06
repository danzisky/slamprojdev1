"""Minimal demo entry point for the chair-based particle filter localizer."""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


if __package__ in {None, ""}:
    SLAM_ROVER_DIR = Path(__file__).resolve().parent.parent
    if str(SLAM_ROVER_DIR) not in sys.path:
        sys.path.insert(0, str(SLAM_ROVER_DIR))

    from localization.detector import ChairObservationDetector
    from localization.map_data import ChairLocalizationMap
    from localization.particle_filter import ParticleFilterLocalizer
    from localization.tof import ToFConfig
    from localization.types import MotionCommand, Pose2D, ToFFrame
    from localization.visualization import LocalizationVisualizer, VisualizationConfig
else:
    from .detector import ChairObservationDetector
    from .map_data import ChairLocalizationMap
    from .particle_filter import ParticleFilterLocalizer
    from .tof import ToFConfig
    from .types import MotionCommand, Pose2D, ToFFrame
    from .visualization import LocalizationVisualizer, VisualizationConfig

from sensor_interface import USBCamera


LOCALIZATION_DIR = Path(__file__).resolve().parent
SLAM_ROVER_DIR = LOCALIZATION_DIR.parent

# File inputs
MAP_PATH = SLAM_ROVER_DIR / "inputs" / "classmap.png"
LANDMARKS_PATH = SLAM_ROVER_DIR / "inputs" / "landmarks.json"
TEST_IMAGE_PATH = SLAM_ROVER_DIR / "inputs" / "localize_imgs" / "prime.jpg"
USE_TEST_IMAGE = False
MAP_RESOLUTION_M_PER_PX = 0.0197

def convert_angle_180_to_360(angle):
    """
    Converts an angle from the range [-180, 180] to [0, 360].
    
    Args:
        angle (float or int): The input angle in degrees.
        
    Returns:
        float: The converted angle in degrees [0, 360).
    """
    # Use modulo 360 to handle angles outside the standard range 
    # (e.g., 540 degrees becomes 180 degrees).
    # Adding 360 ensures positive results for negative inputs before the modulo.

    if (angle < 0):
        angle = -angle
    else:
        angle = 360 - angle
    normalized_angle = angle % 360
    return normalized_angle

# Frame sequence mode: process multiple images taken at different headings
# Set to True to use frame sequences instead of single image or live camera
USE_FRAME_SEQUENCE = False
# List of (frame_path_relative_to_inputs, heading_in_degrees) tuples
# Example: images taken around a point at 0°, 90°, 180°, 270°
FRAME_SEQUENCE = [
    ("images_surroundings/min_88.jpeg", ((360 - convert_angle_180_to_360(-83)) -83 - 100) % 360),
    ("images_surroundings/min_141.jpeg", ((360 - convert_angle_180_to_360(-141)) -83 - 100) % 360),
    ("images_surroundings/plus_157.jpeg", ((360 - convert_angle_180_to_360(157)) -83 - 100) % 360),
]

# Camera configuration
CAMERA_ID = 0
FX = 589.54200724
FY = 589.80048532
CX = 328.93066342
CY = 200.86625768 * 0.85

# ===========================
# ESP32 device configuration
# ===========================
# Set ESPCAM_ENABLED=True to use the ESP32 camera module over HTTP instead of
# a local USB camera.  Set the IP address printed on the ESP32 serial console
# at boot (e.g. "Connected. IP: 192.168.4.1").
ESPCAM_ENABLED = True
ESPCAM_IP = "192.168.137.245"        # <-- change to your rover's IP
ESPCAM_CAPTURE_PORT = 80          # main HTTP server port on the ESP32
ESPCAM_TIMEOUT_S = 3.0            # per-request timeout in seconds

# Set ESPCAM_TOF_ENABLED=True to also fetch VL53L5CX readings from the ESP32
# (/distance endpoint) and fuse them into the particle filter.
ESPCAM_TOF_ENABLED = True
TOF_ZONE_PRESET = "middle"        # "middle" = single horizontal strip (8 rays), "centre" or "wide"
TOF_MAX_RANGE_M = 3.0
TOF_RANGE_STD_M = 0.06
TOF_SENSOR_OFFSET_REAR_M = 0.12  # metres from rover pivot centre to the rear sensor
TOF_WEIGHT = 0.5                  # blend weight: scales ToF log-likelihood vs. landmark camera

# Particle filter configuration
PARTICLE_COUNT = 5000
LOAD_DEFAULT_DETECTOR = True
CAPTURE_RETRY_COUNT = 20
CAPTURE_RETRY_DELAY_MS = 50

# Set to None for global initialization over free space.
KNOWN_INITIAL_POSE: Optional[Pose2D] = Pose2D(
    x_m=4,
    y_m=2,
    heading_rad=math.radians(90),
)
# KNOWN_INITIAL_POSE: Optional[Pose2D] = None

# Set a non-zero motion command when the robot has moved since the previous update.
MOTION_COMMAND = MotionCommand(forward_m=0.0, turn_rad=0.0)

# Visualization
VISUALIZATION_CONFIG = VisualizationConfig(
    show_observation_preview=True,
    preview_window_name="Chair Observations",
    show_map_preview=True,
    map_window_name="Localization Map",
    map_preview_scale=1,
    draw_particles=True,
    max_particles_to_draw=PARTICLE_COUNT // 2,
    trajectory_history=150,
)
LIVE_VIEW_WAIT_MS = 1
# In test-image mode: how many filter update cycles to run on the same detections.
# Each cycle lets you watch the particles contract toward the most likely robot position.
CONVERGENCE_ITERATIONS = 5
# Milliseconds to pause between convergence steps (set to 0 to require a keypress each step).
UPDATE_STEP_DELAY_MS = 1500


class EspCamSource:
    """HTTP client for the XIAO ESP32S3 camera/ToF sensor node.

    Connects to the ESP32's HTTP server and provides:
    - ``capture_frame()``  -- fetches ``/capture`` and decodes the JPEG as a BGR frame.
    - ``read_tof()``       -- fetches ``/distance`` and returns a :class:`ToFFrame`.

    Set ``ESPCAM_IP`` in the config section above to the IP printed on the
    ESP32 serial console after it connects to Wi-Fi.
    """

    def __init__(self, ip: str, port: int = 80, timeout_s: float = 3.0) -> None:
        self._base = f"http://{ip}:{port}"
        self._timeout = timeout_s

    @staticmethod
    def _center_strip_values_m(grid_m: np.ndarray) -> np.ndarray:
        """Return flattened center-strip values (rows 3-4) after orientation correction."""
        return grid_m[3:5, :].reshape(-1)

    def capture_frame(self) -> Optional[np.ndarray]:
        """GET /capture → decode JPEG → BGR ndarray, or None on failure."""
        try:
            with urllib.request.urlopen(
                f"{self._base}/capture", timeout=self._timeout
            ) as resp:
                data = resp.read()
        except Exception as exc:
            print(f"[EspCam] capture failed: {exc}")
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("[EspCam] JPEG decode failed")
            return None
        # Camera is inverted on the y-axis; flip to correct orientation.
        frame = cv2.flip(frame, 0)
        return frame

    def read_tof(self) -> Optional[ToFFrame]:
        """GET /distance → ToFFrame with distances in metres, or None on failure."""
        try:
            with urllib.request.urlopen(
                f"{self._base}/distance", timeout=self._timeout
            ) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            print(f"[EspCam] distance read failed: {exc}")
            return None
        if not payload.get("ok"):
            print(f"[EspCam] distance error: {payload.get('error')}")
            return None
        grid_mm = payload.get("grid")
        if grid_mm is None or len(grid_mm) != 8:
            print("[EspCam] unexpected grid shape")
            return None
        arr = np.array(grid_mm, dtype=np.float32) / 1000.0  # mm -> m
        # ToF module is mounted upside down; rotate to rover frame.
        arr = np.rot90(arr, 2)
        arr[arr <= 0.0] = float("nan")                       # zero = no return

        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            center_mm = float(np.nanmean(arr[3:5, 3:5]) * 1000.0)
            strip_mm = np.round(self._center_strip_values_m(arr) * 1000.0, 0).astype(np.int32)
            print(
                "[EspCam ToF] "
                f"min={float(np.nanmin(valid) * 1000.0):.0f} mm, "
                f"max={float(np.nanmax(valid) * 1000.0):.0f} mm, "
                f"center={center_mm:.0f} mm, "
                f"strip_mm={strip_mm.tolist()}"
            )
        else:
            print("[EspCam ToF] no valid zones in current frame")

        return ToFFrame(arr)


def load_frame_sequence() -> list:
    """Load frames from the configured sequence. Returns list of (frame, heading_rad, heading_deg) tuples."""
    frames = []
    for frame_rel_path, heading_deg in FRAME_SEQUENCE:
        frame_path = SLAM_ROVER_DIR / frame_rel_path
        if not frame_path.exists():
            print(f"⚠️  Frame not found: {frame_path}")
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️  Failed to load frame: {frame_path}")
            continue
        heading_rad = math.radians(heading_deg)
        frames.append((frame, heading_rad, heading_deg))
    return frames


def validate_demo_configuration() -> None:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Map image not found: {MAP_PATH}")
    if not LANDMARKS_PATH.exists():
        raise FileNotFoundError(f"Landmarks file not found: {LANDMARKS_PATH}")
    if USE_TEST_IMAGE and not USE_FRAME_SEQUENCE and not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_IMAGE_PATH}")
    if USE_FRAME_SEQUENCE and not FRAME_SEQUENCE:
        raise ValueError("FRAME_SEQUENCE is empty but USE_FRAME_SEQUENCE=True")


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
    tof_config = None
    if ESPCAM_TOF_ENABLED:
        preset = TOF_ZONE_PRESET.lower().strip()
        if preset == "middle":
            zone_mask = ToFConfig.middle_strip_mask()
            use_middle_strip_2d = True
        elif preset == "centre":
            zone_mask = ToFConfig.centre_strip_mask()
            use_middle_strip_2d = False
        else:
            zone_mask = ToFConfig.wide_strip_mask()
            use_middle_strip_2d = False
        tof_config = ToFConfig(
            zone_mask=zone_mask,
            max_range_m=TOF_MAX_RANGE_M,
            range_std_m=TOF_RANGE_STD_M,
            sensor_offset_rear_m=TOF_SENSOR_OFFSET_REAR_M,
            use_middle_strip_2d=use_middle_strip_2d,
        )
        print(f"[ToF] enabled -- {tof_config}")

    localizer = ParticleFilterLocalizer(
        localization_map=localization_map,
        detector=detector,
        particle_count=PARTICLE_COUNT,
        initial_pose=KNOWN_INITIAL_POSE,
        tof_config=tof_config,
        tof_weight=TOF_WEIGHT,
    )
    visualizer = LocalizationVisualizer(config=VISUALIZATION_CONFIG)

    camera = None
    esp_cam = None
    if not USE_TEST_IMAGE and not USE_FRAME_SEQUENCE:
        if ESPCAM_ENABLED:
            esp_cam = EspCamSource(ESPCAM_IP, ESPCAM_CAPTURE_PORT, ESPCAM_TIMEOUT_S)
            print(f"[EspCam] using ESP32 camera at http://{ESPCAM_IP}:{ESPCAM_CAPTURE_PORT}")
            if ESPCAM_TOF_ENABLED:
                print("[EspCam] ToF depth fusion enabled")
        else:
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
                # motion = MOTION_COMMAND if iteration == 0 else None
                motion = MOTION_COMMAND
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
        elif USE_FRAME_SEQUENCE:
            frame_sequence = load_frame_sequence()
            if not frame_sequence:
                raise SystemExit("Frame sequence is empty after loading.")
            
            # Show the initial particle spread
            print("Showing initial particle spread for frame sequence mode. Press any key to begin...")
            visualizer.show_update(None, localization_map, localizer, None)
            cv2.waitKey(0)
            
            # Track current heading in radians (supports both Pose2D and legacy tuple/list)
            if KNOWN_INITIAL_POSE is None:
                current_heading_rad = 0.0
            elif hasattr(KNOWN_INITIAL_POSE, "heading_rad"):
                current_heading_rad = float(KNOWN_INITIAL_POSE.heading_rad)
            elif isinstance(KNOWN_INITIAL_POSE, (tuple, list)) and len(KNOWN_INITIAL_POSE) >= 3:
                current_heading_rad = float(KNOWN_INITIAL_POSE[2])
            else:
                raise TypeError("KNOWN_INITIAL_POSE must be Pose2D, tuple/list(x, y, heading), or None")
            
            for frame_idx, (frame, target_heading_rad, target_heading_deg) in enumerate(frame_sequence):
                # Calculate the turn needed to reach this frame's heading
                angle_diff = target_heading_rad - current_heading_rad
                # Normalize angle difference to [-pi, pi]
                turn_rad = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
                
                # Create motion command for the turn
                motion = MotionCommand(forward_m=0.0, turn_rad=turn_rad)
                
                # Detect observations from this frame
                observations = detector.observations_from_image(frame)
                print(f"Frame {frame_idx} (heading={target_heading_deg:.1f}°): "
                      f"detected {len(observations)} observation(s), "
                      f"motion turn={math.degrees(turn_rad):.1f}°")
                
                # Update particle filter with motion and observations
                update = localizer.update_from_observations(observations, motion=motion)
                update_index += 1
                print_localization_result(update.estimate, update_index)
                
                # Visualize the update
                visualizer.show_update(frame, localization_map, localizer, update)
                
                # Update heading for next iteration
                current_heading_rad = target_heading_rad
                
                # Allow user to step through or exit
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
                    if esp_cam is not None:
                        frame = esp_cam.capture_frame()
                    else:
                        frame = camera.get_frame()
                    if frame is not None:
                        break
                    cv2.waitKey(CAPTURE_RETRY_DELAY_MS)

                if frame is None:
                    raise SystemExit("No frame received from the camera.")

                tof_frame = None
                if esp_cam is not None and ESPCAM_TOF_ENABLED:
                    tof_frame = esp_cam.read_tof()

                update = localizer.update_from_image(
                    frame, motion=MOTION_COMMAND, tof_frame=tof_frame
                )
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