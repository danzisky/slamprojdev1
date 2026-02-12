import cv2
import numpy as np
import os
import sys
import torch

# Ensure we can import mapper 
# (assuming this script is in d:\schproj\Depth-Anything-V2\metric_depth\slam_rover\)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mapper import (
    get_model,
    image_to_3d_pointcloud,
    _process_occupancy_grid,
    visualize_occupancy_grid
)

def rotate_pointcloud_y(points, angle_deg):
    """
    Rotate pointcloud around the Y axis (height axis).
    points: Nx3 array (X, Y, Z)
    angle_deg: Rotation angle in degrees (positive = counter-clockwise?)
    """
    if len(points) == 0:
        return points
    
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix for Y-axis
    # | cos  0  sin |
    # |  0   1   0  |
    # | -sin 0  cos |
    
    # Standard rotation (check handedness)
    # x' = x*cos + z*sin
    # y' = y
    # z' = -x*sin + z*cos
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    x_new = x * cos_a + z * sin_a
    y_new = y
    z_new = -x * sin_a + z * cos_a
    
    return np.stack([x_new, y_new, z_new], axis=-1)

def generate_centered_grid(points, grid_size_meters=40.0, resolution=0.05, height_range=(-0.5, 2.0), obstacle_threshold=0.1):
    """
    Generate an occupancy grid centered at (0,0) with fixed size.
    grid_size_meters: Width and Height of the area covered (square)
    """
    if len(points) == 0:
        dim = int(grid_size_meters / resolution)
        return np.full((dim, dim), -1, dtype=np.int8)

    half_size = grid_size_meters / 2.0
    x_min, x_max = -half_size, half_size
    z_min, z_max = -half_size, half_size
    
    grid_dim = int(np.ceil(grid_size_meters / resolution))
    
    # Extract
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Filter bounds
    in_bounds = (x >= x_min) & (x < x_max) & (z >= z_min) & (z < z_max)
    in_height = (y >= height_range[0]) & (y <= height_range[1])
    
    valid_mask = in_bounds & in_height
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return np.full((grid_dim, grid_dim), -1, dtype=np.int8)
    
    vx = valid_points[:, 0]
    vy = valid_points[:, 1]
    vz = valid_points[:, 2]
    
    # Grid indices
    # (0,0) in grid corresponds to (x_min, z_min)
    ix = ((vx - x_min) / resolution).astype(int)
    iz = ((vz - z_min) / resolution).astype(int)
    
    ix = np.clip(ix, 0, grid_dim - 1)
    iz = np.clip(iz, 0, grid_dim - 1)
    
    # Max height map
    flat_idx = iz * grid_dim + ix
    height_map_flat = np.full(grid_dim * grid_dim, -np.inf, dtype=np.float32)
    np.maximum.at(height_map_flat, flat_idx, vy)
    
    height_map = height_map_flat.reshape(grid_dim, grid_dim)
    
    # Occupancy
    occupancy = np.full((grid_dim, grid_dim), -1, dtype=np.int8)
    
    # Visited
    visited_flat = np.zeros(grid_dim * grid_dim, dtype=bool)
    visited_flat[flat_idx] = True
    visited = visited_flat.reshape(grid_dim, grid_dim)
    
    # Set values
    is_obstacle = height_map > obstacle_threshold
    occupancy[visited] = 0
    occupancy[is_obstacle] = 1
    occupancy[~visited] = -1
    
    return occupancy

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
        # ('images_surroundings/min_83.jpeg', -83),
        # ('images_surroundings/min_88.jpeg', -88),
        ('images_surroundings/min_141.jpeg', -141),
        # ('images_surroundings/plus_149.jpeg', 149),
        ('images_surroundings/plus_157.jpeg', 157),
    ]
    
    # 2. Load Model
    print("Loading Model...")
    model = get_model(encoder='vitl', dataset='hypersim', max_depth=20)
    
    # 3. Process
    accumulated_points = []
    
    print("Starting Processing Loop...")
    
    # Camera Intrinsics (Approximate or calibrated)
    # Note: These should match your camera. Using defaults from example_usage.py
    # But need image size first.
    
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

    grid_size_m = 30.0 # 30x30 meter area
    
    for i, (path, angle) in enumerate(scan_sequence):
        angle = -angle # Negate if your rotation direction is opposite
        print(f"Processing Image {i+1}/{len(scan_sequence)} | Angle: {angle}°")
        
        # Load Image
        path = os.path.join(base_path, path) # Ensure these images exist
        rgb = cv2.imread(path)
        if rgb is None: continue
        
        # Get Point Cloud
        points = image_to_3d_pointcloud(rgb, intrinsics, model=model)
        
        # Rotate Point Cloud
        points_rotated = rotate_pointcloud_y(points, angle)
        
        # Add to accumulator
        # (Inefficient for massive clouds, but fine for demo)
        if i == 0:
            accumulated_points = points_rotated
        else:
            accumulated_points = np.concatenate([accumulated_points, points_rotated], axis=0)
            
        print(f"  Total points: {len(accumulated_points)}")
        
        # Generate Centered Grid
        grid = generate_centered_grid(accumulated_points, grid_size_meters=grid_size_m, resolution=0.05)
        
        # Process (Refine)
        processed_grid = _process_occupancy_grid(grid)
        
        # Visualize
        vis = visualize_occupancy_grid(processed_grid)
        
        # Add text
        cv2.putText(vis, f"Frame: {i+1} Angle: {angle}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Points: {len(accumulated_points)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw Robot Center
        center_x = vis.shape[1] // 2
        center_y = vis.shape[0] // 2
        cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1) # Red dot at robot
        
        cv2.imshow("Global Map (Robot Centered)", vis)
        key = cv2.waitKey(1000) # Show for 1 second
        if key == 27: # ESC
            break
            
    print("Finished. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
