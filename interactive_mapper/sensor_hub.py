"""Sensor attachment and latest-snapshot management."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .runtime_imports import CameraInterface, IMUInterface, SensorFactory, WaveRoverController
from .types import SensorCallback, SensorSnapshot


class SensorHub:
    """Coordinates camera/IMU receivers and keeps a thread-safe latest snapshot."""

    def __init__(self, robot: WaveRoverController) -> None:
        self.robot = robot
        self.camera: Optional[CameraInterface] = None
        self.external_imu: Optional[IMUInterface] = None

        self._sensor_lock = threading.Lock()
        self._sensor_snapshot = SensorSnapshot(timestamp=time.time())
        self._sensor_callback: Optional[SensorCallback] = None

    def register_callback(self, callback: Optional[SensorCallback]) -> None:
        """Register a callback that receives each updated snapshot."""
        self._sensor_callback = callback

    def attach_camera(self, sensor_type: str = "usb", start: bool = False, **kwargs: Any) -> CameraInterface:
        """Create and optionally start a camera receiver."""
        self.camera = SensorFactory.create_camera(sensor_type=sensor_type, **kwargs)
        if start:
            self.camera.start()
        return self.camera

    def attach_imu(self, sensor_type: str = "android", start: bool = False, **kwargs: Any) -> IMUInterface:
        """Create and optionally start an IMU receiver."""
        self.external_imu = SensorFactory.create_imu(sensor_type=sensor_type, **kwargs)
        self.robot.external_imu = self.external_imu
        if start:
            self.external_imu.start()
        return self.external_imu

    def start_receivers(self) -> None:
        """Start any currently attached receivers."""
        if self.camera is not None:
            self.camera.start()
        if self.external_imu is not None:
            self.external_imu.start()

    def stop_receivers(self) -> None:
        """Stop any currently attached receivers."""
        if self.camera is not None:
            self.camera.stop()
        if self.external_imu is not None:
            self.external_imu.stop()

    def receive_sensor_data(
        self,
        frame: Optional[np.ndarray] = None,
        imu_data: Optional[object] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SensorSnapshot:
        """Push externally collected sensor data into mapper state."""
        snapshot = SensorSnapshot(
            timestamp=time.time(),
            frame=frame.copy() if frame is not None else None,
            imu_data=imu_data,
            metadata=dict(metadata or {}),
        )
        with self._sensor_lock:
            self._sensor_snapshot = snapshot

        if self._sensor_callback is not None:
            self._sensor_callback(snapshot)
        return snapshot

    def poll_receivers(self, include_robot_imu: bool = False) -> SensorSnapshot:
        """Poll attached receivers and store the result as the latest snapshot."""
        frame = self.camera.get_frame() if self.camera is not None else None
        imu_data: Optional[object] = None
        metadata: Dict[str, Any] = {}

        if self.external_imu is not None:
            imu_data = self.external_imu.get_imu_data()
            metadata["imu_source"] = "external"

        # This path lets you inspect robot IMU telemetry even without external IMU.
        if include_robot_imu:
            robot_imu = self.robot.get_imu_data(retries=1)
            if robot_imu is not None:
                metadata["robot_imu"] = robot_imu
                if imu_data is None:
                    imu_data = robot_imu

        if frame is not None:
            metadata["frame_shape"] = tuple(frame.shape)

        return self.receive_sensor_data(frame=frame, imu_data=imu_data, metadata=metadata)

    def get_snapshot(self) -> SensorSnapshot:
        """Return a copy of the most recent snapshot."""
        with self._sensor_lock:
            snapshot = self._sensor_snapshot
            frame = snapshot.frame.copy() if snapshot.frame is not None else None
            metadata = dict(snapshot.metadata)

        return SensorSnapshot(
            timestamp=snapshot.timestamp,
            frame=frame,
            imu_data=snapshot.imu_data,
            metadata=metadata,
        )
