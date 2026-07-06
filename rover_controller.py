"""
WaveShare WAVE ROVER Robot Controller
Integrates with frontier exploration for autonomous mapping
Uses HTTP/JSON communication as per WaveShare documentation
"""

from dataclasses import dataclass
import requests
import json
import time
import threading
import numpy as np
from typing import Optional

# Import the interfaces from the correct module
from sensor_interface import IMUInterface, IMUData


@dataclass
class WaveRoverIMUData:
    """IMU data from WaveRover's built-in MPU9250 sensor"""
    # Orientation (Euler angles in degrees)
    roll: float          # Roll angle
    pitch: float         # Pitch angle  
    yaw: float           # Yaw/heading angle (0-360°)
    
    # Acceleration (m/s²)
    accel_x: float
    accel_y: float
    accel_z: float
    
    # Gyroscope (°/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Magnetometer (μT - microtesla)
    mag_x: float
    mag_y: float
    mag_z: float
    
    # Temperature (°C)
    temperature: float
    
    raw_yaw: float       # Raw yaw from the sensor for reference
    timestamp: float

class WaveRoverController:
    def __init__(self, robot_ip='192.168.4.1', speed_multiplier=1.0, external_imu: Optional[IMUInterface] = None):
        """
        Initialize WaveShare WAVE ROVER controller.

        Args:
            robot_ip: Robot's IP address.
            speed_multiplier: Global speed multiplier (0.0 to 1.0).
            external_imu: An optional external IMU object that conforms to IMUInterface.
        """
        print("Initializing WaveRoverController with IP:", robot_ip)
        self.robot_ip = robot_ip
        self.base_url = f"http://{robot_ip}/js?json="
        self.connected = False
        self.LOOP_DELAY_SECONDS = 0.1  # rover keeps commands alive for 3 seconds, so we can use 1Hz command rate not to overwhelm it
        self.speed_multiplier = float(np.clip(speed_multiplier, 0.0, 1.0))
        
        # Robot physical parameters
        self.wheel_radius = 0.04  # meters
        self.wheel_base = 0.13    # meters
        
        # Speed calibration based on y = 0.851x - 0.052
        self.velocity_slope = 0.851  # m/s per speed unit (the 'm' in y=mx+c)
        self.velocity_intercept = -0.052 # m/s (the 'c' in y=mx+c)
        
        # Robot state
        self.current_speed = 0
        self.current_angle = 0
        
        # IMU state
        self.external_imu = external_imu
        self.last_imu_data = None
        self.heading_offset_deg = 0.0 # Offset to align IMU yaw with map coordinates
        
        # State for integrated yaw (for both internal and external IMU)
        self.integrated_yaw = 0.0
        self.last_imu_time = None
        self.yaw_initialized = False

        # Background IMU polling thread (used for integrated internal IMU mode)
        self._imu_poll_interval_s = 0.05
        self._imu_thread = None
        self._imu_stop_event = threading.Event()
        self._imu_lock = threading.Lock()
        self._latest_internal_imu = None

        if self.external_imu:
            print("✅ Using external IMU.")
        else:
            print("🤖 Using built-in WaveRover IMU with gyro integration (CMD 126).")

    def connect(self):
        """Test connection to the robot"""
        try:
            # Try to get status or send a harmless command
            test_command = {"T": 1, "L": 0, "R": 0}
            response = self._send_http_command(test_command)
            
            if response:
                self.connected = True
                print(f"Connected to WAVE ROVER at {self.robot_ip}")
                self._start_imu_thread_if_needed()
                return True
            else:
                print(f"Failed to connect to WAVE ROVER at {self.robot_ip}")
                return False
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.connected:
            self._stop_imu_thread()
            self.stop()
            self.connected = False
            print("Disconnected from robot")

    def _start_imu_thread_if_needed(self):
        """Start background polling for integrated internal IMU mode."""
        if self.external_imu is not None:
            return
        if self._imu_thread is not None and self._imu_thread.is_alive():
            return

        self._imu_stop_event.clear()
        self._imu_thread = threading.Thread(target=self._imu_worker_loop, daemon=True)
        self._imu_thread.start()
        print("✅ Started background integrated IMU thread (CMD 126).")

    def _stop_imu_thread(self):
        """Stop background IMU polling thread if running."""
        if self._imu_thread is None:
            return
        self._imu_stop_event.set()
        self._imu_thread.join(timeout=1.0)
        self._imu_thread = None

    def _imu_worker_loop(self):
        """Continuously poll CMD 126 and maintain integrated yaw in background."""
        while not self._imu_stop_event.is_set() and self.connected:
            try:
                command = {"T": 126}
                response = self._send_http_command(command)
                if response is None:
                    time.sleep(self._imu_poll_interval_s)
                    continue

                data = json.loads(response)
                current_time = time.time()
                imu_data = self._build_integrated_internal_imu(data, current_time, apply_offset=True)

                with self._imu_lock:
                    self._latest_internal_imu = imu_data
                    self.last_imu_data = imu_data
            except Exception:
                pass

            time.sleep(self._imu_poll_interval_s)

    def _build_integrated_internal_imu(self, data: dict, current_time: float, apply_offset: bool) -> WaveRoverIMUData:
        """Build one integrated internal-IMU sample from CMD 126 payload."""
        mg_to_mps2 = 0.00981
        raw_yaw = float(data.get('y', 0))

        dt = current_time - (self.last_imu_time or current_time)
        self.last_imu_time = current_time

        if not self.yaw_initialized:
            self.integrated_yaw = self._normalize_angle_360(raw_yaw)
            self.yaw_initialized = True
        else:
            self.integrated_yaw += float(data.get('gz', 0)) * dt

        raw_integrated_yaw = self._normalize_angle_360(self.integrated_yaw)
        corrected_yaw = self._normalize_angle_360(raw_integrated_yaw + self.heading_offset_deg) if apply_offset else raw_integrated_yaw

        return WaveRoverIMUData(
            roll=float(data.get('r', 0)),
            pitch=float(data.get('p', 0)),
            yaw=corrected_yaw,
            raw_yaw=raw_integrated_yaw,
            accel_x=float(data.get('ax', 0)) * mg_to_mps2,
            accel_y=float(data.get('ay', 0)) * mg_to_mps2,
            accel_z=float(data.get('az', 0)) * mg_to_mps2,
            gyro_x=float(data.get('gx', 0)),
            gyro_y=float(data.get('gy', 0)),
            gyro_z=float(data.get('gz', 0)),
            mag_x=float(data.get('mx', 0)),
            mag_y=float(data.get('my', 0)),
            mag_z=float(data.get('mz', 0)),
            temperature=float(data.get('temp', 0)),
            timestamp=current_time
        )
    
    def _send_http_command(self, command_dict):
        """Send JSON command via HTTP GET request"""
        try:
            command_json = json.dumps(command_dict)
            url = self.base_url + command_json
            # print(f"  → HTTP: {url}")  # Debug: uncomment to see actual commands
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"HTTP error: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("Command timeout")
            return None
        except Exception as e:
            print(f"Failed to send command: {e}")
            return None
    
    def send_command(self, command_dict):
        """Send JSON command to robot (wrapper for compatibility)"""
        result = self._send_http_command(command_dict)
        return result is not None
    
    def set_speed_multiplier(self, multiplier):
        """Set global speed multiplier (0.0 to 1.0)"""
        self.speed_multiplier = float(np.clip(multiplier, 0.0, 1.0))
        print(f"Speed multiplier set to: {self.speed_multiplier}")
    
    def move(self, left_speed, right_speed, stop_after=None):
        """
        Move robot with individual wheel speeds (CMD_SPEED_CTRL)
        
        Args:
            left_speed: -0.5 to 0.5 (negative = backward, positive = forward)
            right_speed: -0.5 to 0.5 (negative = backward, positive = forward)
            stop_after: Optional duration in seconds to stop after moving
        """
        left_speed = self._get_speed_ctrl(left_speed)
        right_speed = self._get_speed_ctrl(right_speed)
        
        command = {
            "T": 1,  # CMD_SPEED_CTRL
            "L": left_speed,
            "R": right_speed
        }

        # print(f"Moving: left_speed={left_speed}, right_speed={right_speed}")
        
        self.current_speed = (left_speed + right_speed) / 2
        
        result = self.send_command(command)
        
        if stop_after: self._timed_stop(stop_after)
        
        return result
    
    def stop(self):
        """Stop the robot"""
        command = {"T": 1, "L": 0.0, "R": 0.0}
        return self.send_command(command)
    
    def _timed_stop(self, duration):
        """Helper to stop after a certain duration"""
        time.sleep(duration)
        self.stop()
    
    def forward(self, speed=0.3, stop_after=None):
        """Move forward at specified speed (0.0 to 0.5)"""
        speed = self._get_speed_ctrl(speed)
        command = {"T": 1, "L": speed, "R": speed}
        result = self.send_command(command)
        
        if stop_after: self._timed_stop(stop_after)
        
        return result
    
    def backward(self, speed=0.3, stop_after=None):
        """Move backward at specified speed (0.0 to 0.5)"""
        speed = self._get_speed_ctrl(speed)
        command = {"T": 1, "L": -speed, "R": -speed}
        result = self.send_command(command)
        
        if stop_after: self._timed_stop(stop_after)
        
        return result
    
    def turn_left(self, speed=0.3):
        """Turn left by rotating (left wheel backward, right wheel forward)"""
        speed = self._get_speed_ctrl(speed)
        command = {"T": 1, "L": -speed, "R": speed}
        return self.send_command(command)
    
    def turn_right(self, speed=0.3):
        """Turn right by rotating (left wheel forward, right wheel backward)"""
        speed = self._get_speed_ctrl(speed)
        command = {"T": 1, "L": speed, "R": -speed}
        return self.send_command(command)
    
    def _get_speed_ctrl(self, speed):
        """Convert speed to a value suitable for the command."""
        return float(np.clip(speed, -0.5, 0.5)) * self.speed_multiplier

    def _get_velocity_from_speed_ctrl(self, speed):
        """Calculate expected velocity in m/s from a given speed setting."""
        speed_value = self._get_speed_ctrl(speed) * 2 # Scale back to -1.0 to 1.0 range for calculation
        expected_velocity = (self.velocity_slope * abs(speed_value)) + self.velocity_intercept
        return max(0.0, expected_velocity)  # Ensure non-negative velocity

    def _normalize_angle(self, angle):
        """Normalize an angle to the range [-180, 180] degrees."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _normalize_angle_360(self, angle):
        """Normalize an angle to the range [0, 360) degrees."""
        return (angle + 360) % 360

    def turn_certain_degrees(self, degrees, max_speed=0.37, min_speed=0.24, timeout=20, tolerance = 2.5):
        """
        Turn a precise number of degrees at a fixed speed using IMU feedback.
        This function does NOT use timed durations and relies solely on the IMU.

        Args:
            degrees (float): The angle to turn. Positive for right, negative for left.
            speed (float): The fixed rotation speed (0.0 to 0.5).
            timeout (int): Maximum seconds to attempt the turn before giving up.
            tolerance (float): Tolerance for target angular degree value
        """
        print(f"🤖 Attempting precise turn of {degrees:.1f}° at fixed speed...")

        # 1. Get initial state and calculate the absolute target heading
        initial_heading = self.get_current_heading(retries=3)
        if initial_heading is None:
            # FIX: Do not use a timed fallback. If there's no IMU, we cannot be precise.
            print("❌ Cannot perform precise IMU turn: No IMU data available.")
            return

        start_time = time.time()
        target_heading = self._normalize_angle_360(initial_heading + degrees)

        # PD state
        initial_error = self._normalize_angle(target_heading - initial_heading)
        last_error = initial_error
        last_time = start_time

        # Slowdown profile near target to reduce overshoot/chatter.
        slowdown_start_deg = max(12.0, 4.0 * float(tolerance))
        near_stop_deg = 1.5 * float(tolerance)

        print(f"  Initial: {initial_heading:.1f}°, Target: {target_heading:.1f}°")

        while time.time() - start_time < timeout:
            current_heading = self.get_current_heading(retries=1)
            if current_heading is None:
                print("\n⚠️ Lost IMU data during turn. Stopping.")
                break

            current_time = time.time()
            dt = current_time - last_time

            if dt <= 0: # Avoid division by zero if loop is too fast
                time.sleep(self.LOOP_DELAY_SECONDS)
                continue

            # Correctly calculate shortest path error to the target
            error = self._normalize_angle(target_heading - current_heading)

            # Check for completion within the fault tolerance
            if abs(error) < tolerance: # Tolerance of tolerance degrees
                self.stop()
                time.sleep(0.3)
                # verify again to ensure stability
                verify_heading = self.get_current_heading(retries=1)
                if verify_heading is None:
                    print("\n⚠️ IMU unavailable during final verification. Stopping at tolerance.")
                    break
                error = self._normalize_angle(target_heading - verify_heading)
                if abs(error) < tolerance:
                    print(f"\n✅ Turn complete. Final heading: {verify_heading:.1f}°")
                    break

            
            # Determine direction based on the shortest path and turn at fixed speed
            kp = 0.004
            kd = 0.005
            # kp = 0.0065
            # kd = 0.08
            derivative = (error - last_error) / dt
            last_error = error
            last_time = current_time
            
            # PD output determines the speed and direction
            output = (kp * error) + (kd * derivative)
            # print("Output", output, "\n")

            abs_error = abs(error)
            if abs_error < slowdown_start_deg:
                # Taper the allowed max speed as we approach target.
                scaled_max = min_speed + (max_speed - min_speed) * (abs_error / slowdown_start_deg)
            else:
                scaled_max = max_speed

            # Keep a minimum command when far, but allow very low command near stop band
            # so the robot can settle instead of oscillating around tolerance.
            lower_bound = 0.0 if abs_error < near_stop_deg else min_speed
            wheel_speed = float(np.clip(abs(output), lower_bound, scaled_max))

            # Pulse-drive then stop. Keeping this non-blocking in move() prevents
            # long dead-time and lets heading updates track each pulse better.
            pulse_s = 0.08
            settle_s = 0.1

            # Positive error means we need to increase heading toward target.
            # On this platform, (L+, R-) increases yaw and (L-, R+) decreases yaw.
            if error > 0:
                self.move(-wheel_speed, wheel_speed)  # Turn left / decrease yaw
            else:
                self.move(wheel_speed, -wheel_speed)  # Turn right / increase yaw

            time.sleep(pulse_s)
            self.stop()
            time.sleep(settle_s)
            
            print(f"  Current: {current_heading:.1f}°, Error: {error:.1f}°  ", end='\r')
        
        else: # This 'else' belongs to the 'while' loop, runs if the loop finishes without 'break'
            print(f"\n⚠️ Turn timed out after {timeout} seconds. Stopping.")

        self.stop()
        print("")

    def calibrate_heading(self, initial_map_heading_deg):
        """
        Calibrates the IMU heading by calculating the offset between the
        robot's current physical heading and the desired initial heading on the map.

        Args:
            initial_map_heading_deg (float): The heading (in degrees) that the robot
                                             is assumed to have on the map at startup.
        """
        print(f"🤖 Calibrating heading. Desired map heading: {initial_map_heading_deg:.1f}°")
        
        # Get the raw, uncorrected heading from the IMU
        initial_physical_heading = self.get_imu_data(apply_offset=False)
        
        if initial_physical_heading is None:
            print("❌ Heading calibration failed: Could not get initial IMU data.")
            return

        physical_yaw = initial_physical_heading.yaw
        self.heading_offset_deg = self._normalize_angle(initial_map_heading_deg - physical_yaw)
        
        print(f"  Initial physical IMU yaw: {physical_yaw:.1f}°")
        print(f"  Calculated heading offset: {self.heading_offset_deg:.1f}°")
        
        # Verify
        corrected_heading = self.get_current_heading()
        print(f"✅ Calibration complete. Corrected heading is now: {corrected_heading:.1f}°")

    def get_current_heading(self, retries=3, apply_offset=True) -> Optional[float]:
        """
        Utility to get the current yaw/heading in degrees [0, 360).
        
        Args:
            retries (int): Number of times to attempt to get IMU data.
            apply_offset (bool): If True, applies the calibrated heading offset.
            
        Returns:
            Optional[float]: The current heading in degrees, or None if unavailable.
        """
        imu_data = self.get_imu_data(retries=retries, apply_offset=apply_offset)
        if imu_data:
            return imu_data.yaw
        return None

    def get_imu_data(self, retries=3, apply_offset=True):
        data = self._get_imu_data(retries, apply_offset)
        self.last_imu_data = data
        return data
    def _get_imu_data(self, retries=3, apply_offset=True) -> Optional[WaveRoverIMUData]:
        """
        Retrieve IMU data, prioritizing the external IMU if available.
        
        Args:
            retries (int): Number of times to attempt to get IMU data.
            apply_offset (bool): If True, applies the calibrated heading offset.
        """
        if self.external_imu:
            # --- Use External IMU (e.g., HyperIMU) ---
            # The external IMU provides stable, fused orientation data. Use it directly.
            ext_data = self.external_imu.get_imu_data()
            # print(ext_data)
            if ext_data is None:
                return None

            # Adapt external IMUData to the WaveRoverIMUData format.
            # No integration needed, as the phone's sensor fusion is superior.
            
            raw_yaw = (360 - ext_data.orient_yaw) # 0-360 yaw from the phone, inverted.
            corrected_yaw = self._normalize_angle_360(raw_yaw + self.heading_offset_deg) if apply_offset else raw_yaw

            return WaveRoverIMUData(
                roll=ext_data.orient_roll,
                pitch=ext_data.orient_pitch,
                yaw=corrected_yaw,
                raw_yaw=raw_yaw,
                accel_x=ext_data.accel_x,
                accel_y=ext_data.accel_y,
                accel_z=ext_data.accel_z,
                gyro_x=np.degrees(ext_data.gyro_x),
                gyro_y=np.degrees(ext_data.gyro_y),
                gyro_z=np.degrees(ext_data.gyro_z),
                mag_x=ext_data.mag_x,
                mag_y=ext_data.mag_y,
                mag_z=ext_data.mag_z,
                temperature=0, # Not provided by HyperIMU
                timestamp=ext_data.timestamp
            )
        else:
            # --- Use Built-in WaveRover IMU ---
            # --- Use Gyro Integration (CMD 126) ---
            # Prefer the continuously-updated background sample.
            for attempt in range(retries):
                with self._imu_lock:
                    cached = self._latest_internal_imu
                if cached is not None:
                    yaw_value = cached.yaw if apply_offset else cached.raw_yaw
                    return WaveRoverIMUData(
                        roll=cached.roll,
                        pitch=cached.pitch,
                        yaw=yaw_value,
                        raw_yaw=cached.raw_yaw,
                        accel_x=cached.accel_x,
                        accel_y=cached.accel_y,
                        accel_z=cached.accel_z,
                        gyro_x=cached.gyro_x,
                        gyro_y=cached.gyro_y,
                        gyro_z=cached.gyro_z,
                        mag_x=cached.mag_x,
                        mag_y=cached.mag_y,
                        mag_z=cached.mag_z,
                        temperature=cached.temperature,
                        timestamp=cached.timestamp,
                    )
                if attempt < retries - 1:
                    time.sleep(0.05)

            # Fallback: synchronous read if thread cache is not ready yet.
            for attempt in range(retries):
                try:
                    command = {"T": 126}
                    response = self._send_http_command(command)
                    
                    if response is None:
                        if attempt < retries - 1: time.sleep(0.1); continue
                        return None

                    data = json.loads(response)
                    current_time = time.time()
                    imu_sample_corrected = self._build_integrated_internal_imu(
                        data,
                        current_time,
                        apply_offset=True,
                    )
                    with self._imu_lock:
                        # Keep cache updated even in fallback path.
                        self._latest_internal_imu = imu_sample_corrected

                    if apply_offset:
                        return imu_sample_corrected

                    return WaveRoverIMUData(
                        roll=imu_sample_corrected.roll,
                        pitch=imu_sample_corrected.pitch,
                        yaw=imu_sample_corrected.raw_yaw,
                        raw_yaw=imu_sample_corrected.raw_yaw,
                        accel_x=imu_sample_corrected.accel_x,
                        accel_y=imu_sample_corrected.accel_y,
                        accel_z=imu_sample_corrected.accel_z,
                        gyro_x=imu_sample_corrected.gyro_x,
                        gyro_y=imu_sample_corrected.gyro_y,
                        gyro_z=imu_sample_corrected.gyro_z,
                        mag_x=imu_sample_corrected.mag_x,
                        mag_y=imu_sample_corrected.mag_y,
                        mag_z=imu_sample_corrected.mag_z,
                        temperature=imu_sample_corrected.temperature,
                        timestamp=imu_sample_corrected.timestamp,
                    )
                except Exception as e:
                    if attempt < retries - 1:
                        print(f"IMU (integrated) retry {attempt+1}/{retries}: {e}")
                        time.sleep(0.1)
                        continue
                    print(f"Failed to get integrated IMU data after {retries} attempts: {e}")
                    return None
    
    def move_to(self, target_pos, current_pos, move_speed=0.4):
        """
        Moves the robot to a target position using a 'turn-then-move' strategy.
        This is a blocking operation.

        Args:
            target_pos (list or tuple): [x, y] target in meters.
            current_pos (list or tuple): [x, y] current robot position in meters.
            move_speed (float): The speed setting for forward movement (0.1 to 0.5).
        
        Returns:
            np.array: The estimated movement vector [dx, dy, dtheta] that was executed.
        """
        print(f"🤖 Executing move_to from {current_pos} to {target_pos}")
        
        # --- 1. Calculate Distance and Angle to Target ---
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance_to_target = np.sqrt(dx**2 + dy**2)

        if distance_to_target < 0.05: # If target is within 5cm, consider it reached.
            print("  Target is too close, skipping move.")
            return np.array([0, 0, 0])

        # Desired heading in degrees. np.arctan2 returns radians from -pi to pi.
        # We convert it to degrees [0, 360] to match the IMU's frame of reference.
        target_heading_rad = np.arctan2(dy, dx)
        target_heading_deg = np.degrees(target_heading_rad)
        # Normalize to 0-360
        target_heading_deg = self._normalize_angle_360(target_heading_deg)

        # --- 2. Turn Towards Target ---
        current_heading = self.get_current_heading()
        if current_heading is None:
            print("❌ Cannot execute move_to: No IMU data available.")
            return np.array([0, 0, 0])

        # Calculate the shortest angle to turn (from -180 to 180)
        turn_angle_deg = self._normalize_angle(target_heading_deg - current_heading)
        
        print(f"  Distance: {distance_to_target:.2f}m, Current Heading: {current_heading:.1f}°, Target Heading: {target_heading_deg:.1f}°")

        # Only turn if the required angle is significant (e.g., > 5 degrees)
        if abs(turn_angle_deg) > 5.0:
            print(f"  Turning by {turn_angle_deg:.1f}°...")
            self.turn_certain_degrees(degrees=turn_angle_deg, max_speed=0.4, min_speed=0.3, timeout=15)
        else:
            print("  Already facing target, no turn needed.")
    
        # --- 3. Move Forward to Target ---
        # Calculate expected velocity using our calibration formula\

        expected_velocity = self._get_velocity_from_speed_ctrl(move_speed)

        if expected_velocity <= 0:
            print(f"❌ Cannot move: Speed ctrl {move_speed} is too low to produce movement.")
            return np.array([0, 0, 0])

        # Calculate time needed to cover the distance
        travel_time = distance_to_target / expected_velocity
        
        print(f"  Moving forward for {travel_time:.2f} seconds at speed {move_speed} (exp. velocity: {expected_velocity:.2f} m/s)...")
        
        start_time = time.time()

        while time.time() - start_time < (travel_time * 0.9):  # Stop slightly early to avoid overshoot
            self.forward(move_speed)
            time.sleep(0.1)
        # time.sleep(travel_time)
        self.stop()
        
        print("  Move complete.")

        # Return the vector representing the intended movement
        turn_angle_rad = np.radians(turn_angle_deg)
        final_heading_rad = np.radians(self._normalize_angle_360(current_heading + turn_angle_deg))
        dx_final = distance_to_target * np.cos(final_heading_rad)
        dy_final = distance_to_target * np.sin(final_heading_rad)
        
        return np.array([dx_final, dy_final, turn_angle_rad])