import cv2
import numpy as np
import os
import sys
import torch

# Ensure we can import mapper 
# (assuming this script is in d:\schproj\Depth-Anything-V2\metric_depth\slam_rover\)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mapper import (
    _trim_3d_pointcloud,
    get_model,
    image_to_3d_pointcloud,
    pointcloud_to_occupancy_grid,
    _process_occupancy_grid,
    visualize_occupancy_grid
)

from frontier_utils import (
    find_frontiers,
    draw_frontiers,
    extract_frontier_goals,
    draw_frontier_goals
)

def warp_to_combined(occupancy_grid, angle_deg, combined_shape, anchor=None):
    """
    Rotate an occupancy grid around the robot anchor, then place it into the combined grid.

    Args:
        occupancy_grid: 2D array with values -1 (unknown), 0 (free), 1 (occupied)
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)
        combined_shape: (height, width) of the combined grid
        anchor: (x, y) point in occupancy_grid to rotate around; defaults to robot origin (top-center)
    Returns:
        Rotated-and-placed grid with shape combined_shape.
    """
    height, width = occupancy_grid.shape
    if anchor is None:
        anchor = (width / 2.0, 0.0)

    combined_h, combined_w = combined_shape
    combined_center = (combined_w / 2.0, combined_h / 2.0)

    rot_mat = cv2.getRotationMatrix2D(anchor, angle_deg, 1.0)
    rot_mat[0, 2] += combined_center[0] - anchor[0]
    rot_mat[1, 2] += combined_center[1] - anchor[1]

    # Warp directly into combined-grid coordinates.
    return cv2.warpAffine(
        occupancy_grid,
        rot_mat,
        (combined_w, combined_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1
    )

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

def main():
    # 1. Define Data
    # Dictionary of "Image Path" -> Rotation Angle (degrees)
    # 0 degrees = Facing forward (Negative Z in current mapper config?)
    # Adjust paths to your actual images
    
    base_path = "D:/schproj/Depth-Anything-V2/metric_depth/slam_rover"
    
    # Using the same image for demonstration if multiple distinct ones aren't available
    # In a real scenario, use different images
    img_path = os.path.join(base_path, "test_image.jpeg") # Ensure this exists
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found. Using placeholder.")
        # Create a dummy image if needed or just fail
    
    # image_data = {
    #     # Image Path : Angle (Yaw)
    #     img_path: 0,         # Front
    #     img_path: 45,        # 45 deg right (simulated)
    #     img_path: 90,        # 90 deg right
    #     img_path: -45,       # 45 deg left
    # }
    
    # Since keys must be unique, let's use a list of tuples instead for simulation
    # (Using the same image multiple times to simulate a panoramic sweep)
    scan_sequence = [
        # ('images_surroundings/min_83.jpeg', convert_angle_180_to_360(-83)),
        # ('images_surroundings/min_88.jpeg', convert_angle_180_to_360(-88)),
        ('images_surroundings/min_141.jpeg', convert_angle_180_to_360(-141)),
        # ('images_surroundings/plus_149.jpeg', convert_angle_180_to_360(149)),
        ('images_surroundings/plus_157.jpeg', convert_angle_180_to_360(157)),
    ]
    
    # 2. Load Model
    print("Loading Model...")
    model = get_model(encoder='vitl', dataset='hypersim', max_depth=20)
    
    # 3. Process
    print("Starting Processing Loop...")

    # Camera Intrinsics (Approximate or calibrated)
    dummy_img = cv2.imread(img_path)
    if dummy_img is None:
        print("Error: Could not read image.")
        return
    h, w = dummy_img.shape[:2]
    intrinsics = {
        'fx': 470.4 * 1.5,
        'fy': 470.4 * 1.5,
        'cx': w / 2.0,
        'cy': h / 2.0
    }

    grid_size = (10.0, 15.0)
    resolution = 0.05
    height_range = (-0.5, 2.0)
    obstacle_threshold = 0.8

    # Initialize combined grid (centered, fixed size)
    grid_w = int(np.ceil(grid_size[0] / resolution))
    grid_h = int(np.ceil(grid_size[1] / resolution))
    padding_factor = 2
    combined_grid = np.full((grid_h * padding_factor, grid_w * padding_factor), -1, dtype=np.int8)

    for i, (path, angle) in enumerate(scan_sequence):
        # angle = -angle  # Negate if your rotation direction is opposite
        print(f"Processing Image {i+1}/{len(scan_sequence)} | Angle: {angle}°")

        # Load Image
        path = os.path.join(base_path, path)
        rgb = cv2.imread(path)
        if rgb is None:
            print(f"Warning: Could not read {path}")
            continue

        # Get Point Cloud
        points = image_to_3d_pointcloud(rgb, intrinsics, model=model)

        pointcloud_trimmed = _trim_3d_pointcloud(points, 20.0)
        # Preserve robot origin by using forward-positive Z with fixed grid bounds.
        pointcloud_trimmed[:, 2] = -pointcloud_trimmed[:, 2]
        pointcloud_trimmed[:, 0] = -pointcloud_trimmed[:, 0] # Flip X to match right-positive convention

        # Convert Point Cloud -> Occupancy Grid (camera frame)
        grid = pointcloud_to_occupancy_grid(
            pointcloud=pointcloud_trimmed,
            grid_size=grid_size,
            grid_resolution=resolution,
            height_range=height_range,
            obstacle_threshold=obstacle_threshold,
            use_data_bounds=False
        )

        # Process (Refine)
        processed_grid = _process_occupancy_grid(grid)

        # Visualize per-frame raw and processed grids (mirrors example_usage)
        raw_vis = visualize_occupancy_grid(grid)
        processed_vis = visualize_occupancy_grid(processed_grid)
        combined_vis = np.hstack([raw_vis, processed_vis])
        cv2.imshow('Raw (Left) vs Processed (Right)', combined_vis)

        # Rotate around robot origin (top-center) and place directly into combined grid
        anchor = (processed_grid.shape[1] / 2.0, 0.0)
        warped_grid = warp_to_combined(processed_grid, angle, combined_grid.shape, anchor=anchor)

        # Update combined grid with new observations (occupied overrides free)
        occupied_mask = (warped_grid == 1)
        free_mask = (warped_grid == 0)
        combined_grid[occupied_mask] = 1
        combined_grid[free_mask & (combined_grid != 1)] = 0

        # Visualize
        vis = visualize_occupancy_grid(combined_grid)

        # Add text
        cv2.putText(vis, f"Frame: {i+1} Angle: {angle}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Known Cells: {np.sum(combined_grid != -1)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw Robot Center
        center_x = vis.shape[1] // 2
        center_y = vis.shape[0] // 2
        cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.imshow("Global Map (Robot Centered)", vis)
        key = cv2.waitKey(1000)
        if key == 27:
            break
    
    frontiers = find_frontiers(combined_grid)
    vis = draw_frontiers(combined_grid, frontiers)
    cv2.imshow("Global Map with Frontiers", vis)

    goals = extract_frontier_goals(frontiers, min_region_size=20)
    vis = draw_frontier_goals(vis, goals)
    cv2.imshow("Frontiers + Goals", vis)
    key = cv2.waitKey(1000)

    print("Finished. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
