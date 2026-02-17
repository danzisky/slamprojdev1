"""
Abstract sensor interfaces with implementations for:
- Android phone (DroidCam + HyperIMU)
- RB3 board (CSI camera + onboard IMU)
- USB camera + MPU6050 IMU

This allows easy swapping between different sensor sources
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
import socket
import struct
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class IMUData:
    """Standard IMU data structure"""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    orient_roll: float = 0.0
    orient_pitch: float = 0.0
    orient_yaw: float = 0.0

class CameraInterface(ABC):
    """Abstract camera interface"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start camera stream"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop camera stream"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is open"""
        pass

class IMUInterface(ABC):
    """Abstract IMU interface"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start IMU streaming"""
        pass
    
    @abstractmethod
    def get_imu_data(self) -> Optional[IMUData]:
        """Get latest IMU data"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop IMU streaming"""
        pass

# ============================================
# ANDROID PHONE IMPLEMENTATIONS
# ============================================

class DroidCamCamera(CameraInterface):
    """Camera implementation using DroidCam from Android phone"""
    
    def __init__(self, phone_ip='192.168.1.101', port=4747, quality='high'):
        """
        Initialize DroidCam camera
        
        Args:
            phone_ip: Android phone IP address
            port: DroidCam port (default: 4747)
            quality: 'low', 'medium', 'high'
        """
        self.phone_ip = phone_ip
        self.port = port
        self.quality = quality
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.streaming = False
        self.stream_thread = None
        
    def start(self) -> bool:
        """Start DroidCam stream"""
        
        # DroidCam URL format: http://phone_ip:4747/video
        if self.quality == 'high':
            url = f"http://{self.phone_ip}:{self.port}/video"
        elif self.quality == 'medium':
            url = f"http://{self.phone_ip}:{self.port}/video?640x480"
        else:  # low
            url = f"http://{self.phone_ip}:{self.port}/video?320x240"
        
        self.cap = cv2.VideoCapture(url)
        
        if not self.cap.isOpened():
            print(f"Failed to connect to DroidCam at {url}")
            print("Make sure DroidCam is running on your phone")
            return False
        
        print(f"Connected to DroidCam at {url}")
        
        # Start background capture thread
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stream_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self.streaming:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print("Warning: Failed to read frame from DroidCam")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Stop camera stream"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is open"""
        return self.cap is not None and self.cap.isOpened()

class HyperIMU(IMUInterface):
    """IMU implementation using HyperIMU app from Android phone"""
    
    def __init__(self, listen_port=5555):
        """
        Initialize HyperIMU receiver
        
        Args:
            listen_port: Port to listen for UDP packets (configure in HyperIMU app)
        """
        self.listen_port = listen_port
        self.sock = None
        self.latest_imu = None
        self.imu_lock = threading.Lock()
        self.reading = False
        self.read_thread = None
        
    def start(self) -> bool:
        """Start IMU data reception"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', self.listen_port))
            self.sock.settimeout(1.0)
            print(f"Listening for HyperIMU data on port {self.listen_port}")
            print(f"Configure HyperIMU app to send UDP to PC_IP:{self.listen_port}")
            
            self.reading = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to start HyperIMU receiver: {e}")
            return False
    
    def _read_loop(self):
        """Background thread for IMU data reception"""
        while self.reading:
            try:
                data, addr = self.sock.recvfrom(1024)
                
                # HyperIMU can send in different formats
                # Try CSV text format first (most common)
                try:
                    text = data.decode('ascii').strip()
                    values = [float(v) for v in text.split(',')]
                    
                    # Create IMU data based on what we received
                    if len(values) >= 3:
                        # Has at least accelerometer data
                        imu_data = IMUData(
                            timestamp=time.time(),  # Use system time since CSV doesn't include it
                            accel_x=values[0],
                            accel_y=values[1],
                            accel_z=values[2],
                            gyro_x=values[3] if len(values) > 3 else 0.0,
                            gyro_y=values[4] if len(values) > 4 else 0.0,
                            gyro_z=values[5] if len(values) > 5 else 0.0,
                            mag_x=values[6] if len(values) > 6 else 0.0,
                            mag_y=values[7] if len(values) > 7 else 0.0,
                            mag_z=values[8] if len(values) > 8 else 0.0,
                            orient_yaw=values[9] if len(values) > 9 else 0.0, # refrernce orientation to robot
                            orient_roll=values[10] if len(values) > 10 else 0.0,
                            orient_pitch=values[11] if len(values) > 11 else 0.0
                        )
                        
                        with self.imu_lock:
                            self.latest_imu = imu_data
                        continue
                except (UnicodeDecodeError, ValueError):
                    # Not CSV format, try binary
                    pass
                
                # Try binary format (old HyperIMU versions)
                # Format: timestamp(double) + 9 floats (accel_xyz, gyro_xyz, mag_xyz)
                if len(data) >= 44:  # 8 + 9*4 bytes
                    # Unpack binary data
                    timestamp = struct.unpack('d', data[0:8])[0]
                    values = struct.unpack('9f', data[8:44])
                    
                    imu_data = IMUData(
                        timestamp=timestamp,
                        accel_x=values[0],
                        accel_y=values[1],
                        accel_z=values[2],
                        gyro_x=values[3],
                        gyro_y=values[4],
                        gyro_z=values[5],
                        mag_x=values[6],
                        mag_y=values[7],
                        mag_z=values[8]
                    )
                    
                    with self.imu_lock:
                        self.latest_imu = imu_data
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.reading:  # Only print if not stopping
                    print(f"HyperIMU read error: {e}")
    
    def get_imu_data(self) -> Optional[IMUData]:
        """Get latest IMU data"""
        with self.imu_lock:
            return self.latest_imu
    
    def stop(self):
        """Stop IMU reading"""
        self.reading = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()

# ============================================
# RB3 BOARD IMPLEMENTATIONS
# ============================================

class RB3Camera(CameraInterface):
    """Camera implementation for RB3 board CSI camera via RTSP"""
    
    def __init__(self, rb3_ip='192.168.1.100', rtsp_port=8554):
        self.rb3_ip = rb3_ip
        self.rtsp_port = rtsp_port
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.streaming = False
        self.stream_thread = None
    
    def start(self) -> bool:
        """Start RTSP stream from RB3"""
        rtsp_url = f"rtsp://{self.rb3_ip}:{self.rtsp_port}/camera"
        
        self.cap = cv2.VideoCapture(rtsp_url)
        
        if not self.cap.isOpened():
            print(f"Failed to connect to RB3 camera at {rtsp_url}")
            return False
        
        print(f"Connected to RB3 camera at {rtsp_url}")
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stream_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Background capture loop"""
        while self.streaming:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Stop camera"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is open"""
        return self.cap is not None and self.cap.isOpened()

class RB3IMU(IMUInterface):
    """IMU implementation for RB3 board"""
    
    def __init__(self, rb3_ip='192.168.1.100', imu_port=5001):
        self.rb3_ip = rb3_ip
        self.imu_port = imu_port
        self.sock = None
        self.latest_imu = None
        self.imu_lock = threading.Lock()
        self.reading = False
        self.read_thread = None
    
    def start(self) -> bool:
        """Start IMU data reception from RB3"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('0.0.0.0', self.imu_port))
            self.sock.settimeout(1.0)
            print(f"Listening for RB3 IMU data on port {self.imu_port}")
            
            self.reading = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            return True
        except Exception as e:
            print(f"Failed to start RB3 IMU receiver: {e}")
            return False
    
    def _read_loop(self):
        """Background IMU reading loop"""
        while self.reading:
            try:
                data, addr = self.sock.recvfrom(1024)
                
                # Try JSON first
                try:
                    imu_json = json.loads(data.decode())
                    imu_data = IMUData(
                        timestamp=imu_json.get('timestamp', time.time()),
                        accel_x=imu_json.get('accel_x', 0.0),
                        accel_y=imu_json.get('accel_y', 0.0),
                        accel_z=imu_json.get('accel_z', 0.0),
                        gyro_x=imu_json.get('gyro_x', 0.0),
                        gyro_y=imu_json.get('gyro_y', 0.0),
                        gyro_z=imu_json.get('gyro_z', 0.0),
                        mag_x=imu_json.get('mag_x', 0.0),
                        mag_y=imu_json.get('mag_y', 0.0),
                        mag_z=imu_json.get('mag_z', 0.0)
                    )
                    
                    with self.imu_lock:
                        self.latest_imu = imu_data
                        
                except json.JSONDecodeError:
                    # Try binary format
                    if len(data) >= 36:
                        values = struct.unpack('9f', data[:36])
                        imu_data = IMUData(
                            timestamp=time.time(),
                            accel_x=values[0],
                            accel_y=values[1],
                            accel_z=values[2],
                            gyro_x=values[3],
                            gyro_y=values[4],
                            gyro_z=values[5],
                            mag_x=values[6],
                            mag_y=values[7],
                            mag_z=values[8]
                        )
                        
                        with self.imu_lock:
                            self.latest_imu = imu_data
                            
            except socket.timeout:
                continue
            except Exception as e:
                if self.reading:
                    print(f"RB3 IMU read error: {e}")
    
    def get_imu_data(self) -> Optional[IMUData]:
        """Get latest IMU data"""
        with self.imu_lock:
            return self.latest_imu
    
    def stop(self):
        """Stop IMU reading"""
        self.reading = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()

# ============================================
# USB CAMERA IMPLEMENTATION
# ============================================

class USBCamera(CameraInterface):
    """Simple USB camera implementation"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.streaming = False
        self.stream_thread = None
    
    def start(self) -> bool:
        """Start USB camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Failed to open USB camera {self.camera_id}")
            return False
        
        print(f"Opened USB camera {self.camera_id}")
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stream_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Background capture loop"""
        while self.streaming:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Stop camera"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is open"""
        return self.cap is not None and self.cap.isOpened()

# ============================================
# SENSOR FACTORY
# ============================================

class SensorFactory:
    """Factory to create sensor instances based on configuration"""
    
    @staticmethod
    def create_camera(sensor_type='android', **kwargs) -> CameraInterface:
        """
        Create camera instance
        
        Args:
            sensor_type: 'android', 'rb3', or 'usb'
            **kwargs: sensor-specific arguments
        """
        if sensor_type == 'android':
            return DroidCamCamera(**kwargs)
        elif sensor_type == 'rb3':
            return RB3Camera(**kwargs)
        elif sensor_type == 'usb':
            return USBCamera(**kwargs)
        else:
            raise ValueError(f"Unknown camera type: {sensor_type}")
    
    @staticmethod
    def create_imu(sensor_type='android', **kwargs) -> IMUInterface:
        """
        Create IMU instance
        
        Args:
            sensor_type: 'android' or 'rb3'
            **kwargs: sensor-specific arguments
        """
        if sensor_type == 'android':
            return HyperIMU(**kwargs)
        elif sensor_type == 'rb3':
            return RB3IMU(**kwargs)
        else:
            raise ValueError(f"Unknown IMU type: {sensor_type}")

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test sensor interfaces')
    parser.add_argument('--camera', type=str, default='android',
                        choices=['android', 'rb3', 'usb'],
                        help='Camera type')
    parser.add_argument('--imu', type=str, default='android',
                        choices=['android', 'rb3'],
                        help='IMU type')
    parser.add_argument('--phone-ip', type=str, default='192.168.1.101',
                        help='Android phone IP (for DroidCam)')
    parser.add_argument('--rb3-ip', type=str, default='192.168.1.100',
                        help='RB3 board IP')
    parser.add_argument('--imu-port', type=int, default=5555,
                        help='Port for IMU data')
    
    args = parser.parse_args()
    
    # Create sensors based on arguments
    if args.camera == 'android':
        camera = SensorFactory.create_camera('android', phone_ip=args.phone_ip)
    elif args.camera == 'rb3':
        camera = SensorFactory.create_camera('rb3', rb3_ip=args.rb3_ip)
    else:
        camera = SensorFactory.create_camera('usb', camera_id=0)
    
    if args.imu == 'android':
        imu = SensorFactory.create_imu('android', listen_port=args.imu_port)
    else:
        imu = SensorFactory.create_imu('rb3', rb3_ip=args.rb3_ip)
    
    # Start sensors
    print("Starting camera...")
    if not camera.start():
        print("Failed to start camera!")
        exit(1)
    
    print("Starting IMU...")
    if not imu.start():
        print("Warning: IMU not available")
    
    # Test loop
    try:
        while True:
            # Get camera frame
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow('Camera', frame)
            
            # Get IMU data
            imu_data = imu.get_imu_data()
            if imu_data is not None:
                print(f"IMU - Accel: ({imu_data.accel_x:.2f}, {imu_data.accel_y:.2f}, {imu_data.accel_z:.2f}), "
                      f"Gyro: ({imu_data.gyro_x:.2f}, {imu_data.gyro_y:.2f}, {imu_data.gyro_z:.2f})")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.stop()
        imu.stop()
        cv2.destroyAllWindows()
        print("Sensors stopped")