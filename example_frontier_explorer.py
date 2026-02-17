"""
DroidCam + WaveRover example for frontier exploration.
"""

from sensor_interface import DroidCamCamera, HyperIMU, USBCamera
from waveshare_robot_controller import WaveRoverController
from frontier_explorer import FrontierExplorer
from mapper import visualize_occupancy_grid
import cv2


def main():
    # camera = DroidCamCamera(phone_ip="192.168.1.101", port=4747, quality="high")
    camera = USBCamera(camera_id=1)
    if not camera.start():
        raise SystemExit("Failed to start camera")

    # external_imu = HyperIMU()
    # robot = WaveRoverController(robot_ip="192.168.137.122", external_imu=external_imu)
    robot = WaveRoverController(robot_ip="192.168.137.122", use_fused_internal_yaw=False)
    robot.connect()

    frame = None
    for _ in range(20):
        frame = camera.get_frame()
        if frame is not None:
            break
        cv2.waitKey(50)

    if frame is None:
        camera.stop()
        raise SystemExit("No frame from camera")

    h, w = frame.shape[:2]
    intrinsics = {
        "fx": 470.4 * 1.5,
        "fy": 470.4 * 1.5,
        "cx": w / 2.0,
        "cy": h / 2.0,
    }

    explorer = FrontierExplorer(robot=robot, camera=camera, intrinsics=intrinsics)
    scans = explorer.scan_n_times(num_scans=5, total_angle_deg=180.0)
    combined_grid = explorer.build_combined_map(scans, debug=True)

    vis = visualize_occupancy_grid(combined_grid)

    cv2.imshow("Combined Occupancy Grid", vis)
    
    scans_vis = []
    for scan in scans:  
        img_vis = scan["image"].copy()
        cv2.putText(img_vis, f"Angle: {scan['angle_deg']:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        scans_vis.append(img_vis)
        
    combined_vis = cv2.hconcat(scans_vis)
    cv2.imshow("Scan Images", combined_vis)
    cv2.waitKey(0)

    # if combined_grid is not None:
    #     goal = explorer.select_best_frontier(combined_grid, criteria="closest")
    #     if goal is not None:
    #         explorer.navigate_to_frontier(goal)

    camera.stop()
    robot.disconnect()


if __name__ == "__main__":
    main()
