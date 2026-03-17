"""
Example usage of mapper.py functions for image-to-pointcloud and occupancy grid generation.

This demonstrates the modular pipeline:
1. Load DepthAnythingV2 model
2. Convert RGB image to 3D point cloud (metric depth)
3. Convert point cloud to occupancy grid
4. Process and visualize the occupancy grid
"""

import cv2
import numpy as np
from mapper import (
    get_model, 
    image_to_3d_pointcloud, 
    _trim_3d_pointcloud,
    pointcloud_to_occupancy_grid, 
    _process_occupancy_grid,
    visualize_occupancy_grid,
    save_pointcloud_to_ply,
    _align_pointcloud_horizontally
)

def read_ply(filepath):
    """Read PLY file and extract points"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find header end
    header_end = 0
    vertex_count = 0
    for i, line in enumerate(lines):
        if 'element vertex' in line:
            vertex_count = int(line.split()[-1])
        if 'end_header' in line:
            header_end = i + 1
            break
    
    # Read vertex data
    points = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
    
    return np.array(points)

def main():
    # === STEP 1: Load Model (single time) ===
    print("=" * 60)
    print("STEP 1: Loading DepthAnythingV2 Model")
    print("=" * 60)
    
    # For indoor scenes, use 'hypersim' dataset with max_depth=20
    # For outdoor scenes, use 'vkitti' dataset with max_depth=80
    model = get_model(
        # encoder='vits',           # Use small encoder for faster inference (testing)
        encoder='vitl',           # Use large encoder for better quality
        dataset='hypersim',       # Indoor model
        max_depth=20              # Maximum depth for indoor
    )
    
    # === STEP 2: Process Image to Point Cloud ===
    print("\n" + "=" * 60)
    print("STEP 2: Converting RGB Image to 3D Point Cloud")
    print("=" * 60)
    
    # Load image
    # image_path = './test_image.jpeg'  # Replace with your image
    # image_path = './images_surroundings/plus_157.jpeg'  # Replace with your image
    image_path = './inputs/2.jpg'  # Replace with your image
    rgb_image = cv2.imread(image_path)
    # cap = cv2.VideoCapture(1)
    # ret, rgb_image = cap.read()
    # cap.release()
    
    if rgb_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image shape: {rgb_image.shape}")
    
    # Define camera intrinsics (adjust based on your camera)
    """ camera_intrinsics = {
        'fx': 470.4 * 1.5,    # Focal length X
        'fy': 470.4 * 1.5,    # Focal length Y
        'cx': rgb_image.shape[1] / 2.0,  # Principal point X
        'cy': rgb_image.shape[0] / 2.0   # Principal point Y
    } """
    """ camera_intrinsics = {
        "fx": 1332.00718,
        "fy": 1332.22300,
        "cx": 969.027259,
        "cy": 452.073112,
    } """
    camera_intrinsics = {
        "fx": 589.54200724,
        "fy": 589.80048532,
        "cx": 328.93066342,
        "cy": 200.86625768 * 0.85,
    }
    
    # Convert image to point cloud using DepthAnythingV2
    # Option A: Pass model directly (faster if processing multiple images)
    pointcloud = image_to_3d_pointcloud(
        rgb_image=rgb_image,
        camera_intrinsics=camera_intrinsics,
        model=model  # Pass pre-loaded model
    )
    # flip z to make forward-positive, and flip x to make right-positive
    # pointcloud[:, 2] = -pointcloud[:, 2]
    # pointcloud = _align_pointcloud_horizontally(pointcloud)

    # pointcloud = read_ply('./test_pointcloud.ply')  # Replace with your PLY file path
    
    # Option B: Auto-load model if not provided
    # pointcloud = image_to_3d_pointcloud(
    #     rgb_image=rgb_image,
    #     camera_intrinsics=camera_intrinsics,
    #     encoder='vitl',
    #     dataset='hypersim',
    #     max_depth=20
    # )
    # flip z to make forward-positive, and flip x to make right-positive
    # pointcloud[:, 2] = -pointcloud[:, 2]
    # pointcloud[:, 0] = -pointcloud[:, 0]

    print(f"Generated point cloud with {len(pointcloud)} points")
    print(f"Point cloud shape: {pointcloud.shape}")
    print(f"Point cloud bounds:")
    print(f"  X: [{pointcloud[:, 0].min():.2f}, {pointcloud[:, 0].max():.2f}]")
    print(f"  Y: [{pointcloud[:, 1].min():.2f}, {pointcloud[:, 1].max():.2f}]")
    print(f"  Z: [{pointcloud[:, 2].min():.2f}, {pointcloud[:, 2].max():.2f}]")
    
    # === STEP 3: Trim Point Cloud (Optional) ===
    print("\n" + "=" * 60)
    print("STEP 3: Trimming Point Cloud (Optional)")
    print("=" * 60)
    
    max_distance = 7.0  # Keep only points within 7 meters
    # pointcloud_trimmed = _trim_3d_pointcloud(pointcloud, max_distance)
    pointcloud_trimmed = pointcloud.copy()  # Skip trimming for now
    # pointcloud_trimmed = _align_pointcloud_horizontally(pointcloud)
    print(f"After trimming: {len(pointcloud_trimmed)} points remain")
    
    # === STEP 4: Convert Point Cloud to Occupancy Grid ===
    print("\n" + "=" * 60)
    print("STEP 4: Converting Point Cloud to Occupancy Grid")
    print("=" * 60)
    
    # Define grid parameters
    grid_size = (10.0, 15.0)        # (width, height) in meters
    grid_resolution = 0.05          # 5cm per cell
    height_range = (-0.5, 2.0)      # Consider obstacles from -0.5m to 2.0m
    # obstacle_threshold = 0.1        # Height threshold for obstacles
    obstacle_threshold = (0.4, 0.6) # Optional: use (min_height, max_height) for more precise classification
    
    occupancy_grid = pointcloud_to_occupancy_grid(
        pointcloud=pointcloud_trimmed,
        grid_size=grid_size,
        grid_resolution=grid_resolution,
        height_range=height_range,
        obstacle_threshold=obstacle_threshold
    )
    
    print(f"Occupancy grid shape: {occupancy_grid.shape}")
    print(f"Grid statistics:")
    print(f"  Free cells: {np.sum(occupancy_grid == 0)}")
    print(f"  Occupied cells: {np.sum(occupancy_grid == 1)}")
    print(f"  Unknown cells: {np.sum(occupancy_grid == -1)}")
    
    # === STEP 5: Process Occupancy Grid (Noise Removal & Gap Filling) ===
    print("\n" + "=" * 60)
    print("STEP 5: Processing Occupancy Grid (Morphological Operations)")
    print("=" * 60)
    
    # processed_grid = _process_occupancy_grid(occupancy_grid)
    processed_grid = occupancy_grid
    
    print(f"Processed grid statistics:")
    print(f"  Free cells: {np.sum(processed_grid == 0)}")
    print(f"  Occupied cells: {np.sum(processed_grid == 1)}")
    print(f"  Unknown cells: {np.sum(processed_grid == -1)}")
    
    # === STEP 6: Visualize ===
    print("\n" + "=" * 60)
    print("STEP 6: Visualizing Occupancy Grids")
    print("=" * 60)
    
    # Visualize raw grid
    raw_vis = visualize_occupancy_grid(occupancy_grid)
    cv2.imshow('Raw Occupancy Grid', raw_vis)
    
    # Visualize processed grid
    processed_vis = visualize_occupancy_grid(processed_grid)
    cv2.imshow('Processed Occupancy Grid', processed_vis)
    
    # Side-by-side comparison
    combined = np.hstack([raw_vis, processed_vis])
    cv2.imshow('Raw (Left) vs Processed (Right)', combined)
    
    print("Press any key to close visualization...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # === STEP 7: Save Results (Optional) ===
    print("\n" + "=" * 60)
    print("STEP 7: Saving Results")
    print("=" * 60)
    
    cv2.imwrite('raw_occupancy_grid.png', raw_vis)
    cv2.imwrite('processed_occupancy_grid.png', processed_vis)
    # np.save('pointcloud.npy', pointcloud_trimmed)
    # np.save('occupancy_grid.npy', processed_grid)

    save_pointcloud_to_ply(pointcloud_trimmed, 'ex_u_pointcloud.ply')
    
    print("Results saved:")
    print("  - raw_occupancy_grid.png")
    # print("  - processed_occupancy_grid.png")
    # print("  - pointcloud.npy")
    # print("  - occupancy_grid.npy")
    print("  - pointcloud.ply")
    
    print("\n✓ Pipeline completed successfully!")

if __name__ == '__main__':
    main()
