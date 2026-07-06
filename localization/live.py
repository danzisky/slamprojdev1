"""ESP32-only live localization runner for slam_rover."""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


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
    from rover_controller import WaveRoverController
else:
    from .detector import ChairObservationDetector
    from .map_data import ChairLocalizationMap
    from .particle_filter import ParticleFilterLocalizer
    from .tof import ToFConfig
    from .types import MotionCommand, Pose2D, ToFFrame
    from .visualization import LocalizationVisualizer, VisualizationConfig


LOCALIZATION_DIR = Path(__file__).resolve().parent
SLAM_ROVER_DIR = LOCALIZATION_DIR.parent
METRIC_DEPTH_DIR = SLAM_ROVER_DIR.parent

# MAP_PATH = SLAM_ROVER_DIR / "inputs" / "classmap.png"
# LANDMARKS_PATH = SLAM_ROVER_DIR / "inputs" / "landmarks.json"
# MAP_RESOLUTION_M_PER_PX = 0.0197
MAP_PATH = SLAM_ROVER_DIR / "inputs" / "kitchenmap.png"
LANDMARKS_PATH = SLAM_ROVER_DIR / "inputs" / "landmarks_kitchen.json"
MAP_RESOLUTION_M_PER_PX = 0.01

# DepthAnythingV2 configuration
DEPTH_MODEL_VARIANT = "vitl"  # "vits" | "vitb" | "vitl"
DEPTH_MODEL_MAX_DEPTH_M = 10.0

# Camera intrinsics
# FX = 589.54200724
# FY = 589.80048532
# CX = 328.93066342
# CY = 200.86625768 * 0.85
FX = 277
FY = 277
CX = 160
CY = 120 # * 0.85

# ESP32 camera + ToF endpoint
ESPCAM_IP = "192.168.137.167"
ESPCAM_CAPTURE_PORT = 80
ESPCAM_TIMEOUT_S = 3.0

# ToF fusion configuration
ESPCAM_TOF_ENABLED = True
TOF_ZONE_PRESET = "middle"  # "middle" | "centre" | "wide"
TOF_MAX_RANGE_M = 3.0
TOF_RANGE_STD_M = 0.06
TOF_SENSOR_OFFSET_REAR_M = 0.12
TOF_WEIGHT = 0.5

# Filter + runtime configuration
PARTICLE_COUNT = 1000
LOAD_DEFAULT_DETECTOR = True
CAPTURE_RETRY_COUNT = 20
CAPTURE_RETRY_DELAY_MS = 50
# KNOWN_INITIAL_POSE: Optional[Pose2D] = Pose2D(x_m=4, y_m=2, heading_rad=math.radians(90))
KNOWN_INITIAL_POSE: Optional[Pose2D] = None
MOTION_COMMAND = MotionCommand(forward_m=0.0, turn_rad=0.0)
LIVE_VIEW_WAIT_MS = 2000
ANGLE_DELTA = math.radians(30)
ANGLE_RANGE = math.radians(270)
# +1 for clockwise/right sweep, -1 for counter-clockwise/left sweep.
SCAN_TURN_DIRECTION = 1

ROVER_IP = "192.168.137.73"

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


def _resolve_depth_checkpoint(model_variant: str) -> Path:
    checkpoint_name = f"depth_anything_v2_metric_hypersim_{model_variant}.pth"
    candidates = [
        METRIC_DEPTH_DIR / "checkpoints" / checkpoint_name,
        METRIC_DEPTH_DIR.parent / "checkpoints" / checkpoint_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "DepthAnythingV2 checkpoint not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def build_depth_model():
    if str(METRIC_DEPTH_DIR) not in sys.path:
        sys.path.insert(0, str(METRIC_DEPTH_DIR))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    if DEPTH_MODEL_VARIANT not in model_configs:
        raise ValueError(f"Unsupported DEPTH_MODEL_VARIANT: {DEPTH_MODEL_VARIANT}")

    checkpoint_path = _resolve_depth_checkpoint(DEPTH_MODEL_VARIANT)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthAnythingV2(**{**model_configs[DEPTH_MODEL_VARIANT], "max_depth": DEPTH_MODEL_MAX_DEPTH_M})
    try:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"[Depth] loaded {DEPTH_MODEL_VARIANT} on {device}: {checkpoint_path}")
    return model


class EspCamSource:
    """HTTP client for the ESP32 camera + VL53L5CX node."""

    def __init__(self, ip: str, port: int = 80, timeout_s: float = 3.0) -> None:
        self._base = f"http://{ip}:{port}"
        self._timeout = timeout_s

    @staticmethod
    def _center_strip_values_m(grid_m: np.ndarray) -> np.ndarray:
        return grid_m[3:5, :].reshape(-1)

    def capture_frame(self) -> Optional[np.ndarray]:
        """GET /capture -> decode JPEG -> BGR ndarray, or None on failure."""
        try:
            with urllib.request.urlopen(f"{self._base}/capture", timeout=self._timeout) as resp:
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
        """GET /distance -> ToFFrame in metres, or None on failure."""
        try:
            with urllib.request.urlopen(f"{self._base}/distance", timeout=self._timeout) as resp:
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

        arr = np.array(grid_mm, dtype=np.float32) / 1000.0
        # ToF mounted upside down.
        arr = np.rot90(arr, 2)
        arr[arr <= 0.0] = float("nan")

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


def validate_configuration() -> None:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Map image not found: {MAP_PATH}")
    if not LANDMARKS_PATH.exists():
        raise FileNotFoundError(f"Landmarks file not found: {LANDMARKS_PATH}")
    if ANGLE_DELTA <= 0:
        raise ValueError("ANGLE_DELTA must be > 0")
    if ANGLE_RANGE <= 0:
        raise ValueError("ANGLE_RANGE must be > 0")
    if SCAN_TURN_DIRECTION not in (-1, 1):
        raise ValueError("SCAN_TURN_DIRECTION must be either -1 or +1")


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


def build_tof_config() -> Optional[ToFConfig]:
    if not ESPCAM_TOF_ENABLED:
        return None

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

    return ToFConfig(
        zone_mask=zone_mask,
        max_range_m=TOF_MAX_RANGE_M,
        range_std_m=TOF_RANGE_STD_M,
        sensor_offset_rear_m=TOF_SENSOR_OFFSET_REAR_M,
        use_middle_strip_2d=use_middle_strip_2d,
    )


def main() -> None:
    validate_configuration()

    depth_model = build_depth_model()

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
        depth_model=depth_model,
        load_default_detector=LOAD_DEFAULT_DETECTOR,
        detection_threshold=0.5,
    )
    tof_config = build_tof_config()
    if tof_config is not None:
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

    esp_cam = EspCamSource(ESPCAM_IP, ESPCAM_CAPTURE_PORT, ESPCAM_TIMEOUT_S)
    print(f"[EspCam] using ESP32 camera at http://{ESPCAM_IP}:{ESPCAM_CAPTURE_PORT}")
    if ESPCAM_TOF_ENABLED:
        print("[EspCam] ToF depth fusion enabled")

    """ Localization Logic:
        Capture frame at regular angles, extract chair observations, update particle filter, visualize results.
    """

    rover_controller = WaveRoverController(robot_ip=ROVER_IP)
    if not rover_controller.connect():
        raise SystemExit(f"Failed to connect rover controller at {ROVER_IP}")


    try:
        update_index = 0

        print("Showing initial particle spread...")
        visualizer.show_update(None, localization_map, localizer, None)
        cv2.waitKey(1200)

        step_angle_rad = float(ANGLE_DELTA) * float(SCAN_TURN_DIRECTION)
        step_angle_deg = math.degrees(step_angle_rad)
        steps_per_sweep = max(1, int(round(abs(ANGLE_RANGE / ANGLE_DELTA))))

        print(
            f"[Scan] interval={abs(step_angle_deg):.1f}deg, "
            f"range={math.degrees(ANGLE_RANGE):.1f}deg, "
            f"steps={steps_per_sweep + 1}, direction={'right' if SCAN_TURN_DIRECTION > 0 else 'left'}"
        )

        while True:
            # One localization sweep: capture at initial heading + each turn step.
            for step_idx in range(steps_per_sweep + 1):
                motion = MOTION_COMMAND
                if step_idx > 0:
                    rover_orientation_before_move = rover_controller.get_current_heading()
                    print(f"[Scan] turning {step_angle_deg:.1f}deg (step {step_idx}/{steps_per_sweep})")
                    # rover_controller.turn_certain_degrees(
                    #     degrees=step_angle_deg,
                    #     max_speed=0.27,
                    #     min_speed=0.23,
                    #     timeout=20,
                    #     tolerance=2.5,
                    # )

                    rover_orientation_after_move = rover_controller.get_current_heading()
                    actual_turn = (rover_orientation_after_move - rover_orientation_before_move) % 360

                    motion = MotionCommand(
                        forward_m=float(MOTION_COMMAND.forward_m),
                        turn_rad=float(MOTION_COMMAND.turn_rad) - math.radians(actual_turn),
                    )

                frame = None
                for _ in range(CAPTURE_RETRY_COUNT):
                    frame = esp_cam.capture_frame()
                    if frame is not None:
                        break
                    cv2.waitKey(CAPTURE_RETRY_DELAY_MS)

                if frame is None:
                    raise SystemExit("No frame received from ESP32 camera.")

                time.sleep(0.1)  # small delay to allow port to free up after capture
            
                tof_frame = esp_cam.read_tof() if ESPCAM_TOF_ENABLED else None
                update = localizer.update_from_image(frame, motion=motion, tof_frame=tof_frame)
                update_index += 1

                print_localization_result(update.estimate, update_index)
                visualizer.show_update(frame, localization_map, localizer, update)

                key = cv2.waitKey(LIVE_VIEW_WAIT_MS) & 0xFF
                if key in (27, ord("q")):
                    return

            # Re-center heading to sweep again around approximately the same central direction.
            recenter_deg = -step_angle_deg * steps_per_sweep
            print(f"[Scan] recentering by {recenter_deg:.1f}deg")
            rover_controller.turn_certain_degrees(
                degrees=recenter_deg,
                max_speed=0.27,
                min_speed=0.23,
                timeout=25,
                tolerance=3.0,
            )
    finally:
        rover_controller.disconnect()
        visualizer.close()


if __name__ == "__main__":
    main()
