import cv2
import torch
import numpy as np
import path
import os
import sys

# setting path
# sys.path.append('../../metric_depth')
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from depth_anything_v2.dpt import DepthAnythingV2

# --- Model Loading ---
def get_model(encoder='vitl', dataset='hypersim', max_depth=20, checkpoint_path=None):
    """
    Load and setup the DepthAnythingV2 metric depth model.
    
    Args:
        encoder (str): Model encoder size. Options: 'vits', 'vitb', 'vitl'
        dataset (str): Dataset type. Options: 'hypersim' (indoor), 'vkitti' (outdoor)
        max_depth (float): Maximum depth value. 20 for indoor, 80 for outdoor.
        checkpoint_path (str): Path to checkpoint. If None, auto-construct from encoder/dataset.
    
    Returns:
        torch.nn.Module: Loaded DepthAnythingV2 model in eval mode on appropriate device.
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Model configuration
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    if encoder not in model_configs:
        raise ValueError(f"Encoder '{encoder}' not supported. Choose from: {list(model_configs.keys())}")
    
    # Initialize model
    # model = DepthAnythingV2(**{**model_configs[encoder]})
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = f'../checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
    
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    
    print(f"Model loaded on device: {device}")
    return model

def _process_occupancy_grid(occupancy_grid):
    """
    Process the raw occupancy grid to clean up noise and solidify obstacles.
    
    Args:
        occupancy_grid (np.ndarray): 2D array where 0=free, 1=occupied, -1=unknown.
    Returns:
        np.ndarray: Processed occupancy grid (0=free, 1=occupied, -1=unknown).
    """
    # Convert to format suitable for CV2 operations
    # 255: Occupied (1), 0: Free (0), 128: Unknown (-1)
    grid_cv = np.full_like(occupancy_grid, 128, dtype=np.uint8)
    grid_cv[occupancy_grid == 0] = 0
    grid_cv[occupancy_grid == 1] = 255

    obstacle_mask = (grid_cv == 255).astype(np.uint8)
    freespace_mask = (grid_cv == 0).astype(np.uint8)

    # --- 1. Process Obstacles ---
    # close gaps using rectangular kernels (longer in the forward direction)
    kernel_longitudinal = np.ones((5, 2), np.uint8)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel_longitudinal, iterations=2)
    
    # Remove noise (small specks)
    kernel_noise = np.ones((2, 1), np.uint8)
    obstacle_mask = cv2.erode(obstacle_mask, kernel_noise, iterations=6)
    
    # Solidify (connect sparse closer points)
    kernel_solid = np.ones((4, 2), np.uint8)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel_solid, iterations=2)

    # Remove noise again
    kernel_noise = np.ones((2, 2), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel_noise, iterations=2)
    
    # --- 2. Process Freespace ---
    # First dilate to close small gaps
    kernel_dilate = np.ones((3, 3), np.uint8)
    freespace_mask = cv2.dilate(freespace_mask, kernel_dilate, iterations=2)

    # Remove small specks
    kernel_noise = np.ones((3, 3), np.uint8)
    freespace_mask = cv2.morphologyEx(freespace_mask, cv2.MORPH_OPEN, kernel_noise, iterations=2)

    # Then erode to restore original size
    kernel_erode = np.ones((3, 3), np.uint8)
    freespace_mask = cv2.erode(freespace_mask, kernel_erode, iterations=2)

    # Close gaps using rectangular top-down kernels
    kernel_vertical = np.ones((10, 2), np.uint8)
    freespace_mask = cv2.morphologyEx(freespace_mask, cv2.MORPH_CLOSE, kernel_vertical, iterations=2)

    # --- Reconstruct refined grid ---
    refined_grid = np.full_like(occupancy_grid, -1)  # Initialize as unknown
    refined_grid[freespace_mask > 0] = 0             # Set freespace
    refined_grid[obstacle_mask > 0] = 1              # Set obstacles (priority)

    # Eliminate obstacles not contacting freespace
    refined_grid = _eliminate_obstacles_not_contacting_freespace(refined_grid)

    # thicken obstacles again after elimination
    obstacle_mask = (refined_grid == 1).astype(np.uint8)
    kernel_thicken = np.ones((2, 2), np.uint8)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel_thicken, iterations=1)
    refined_grid[obstacle_mask > 0] = 1

    return refined_grid

def _eliminate_obstacles_not_contacting_freespace(occupancy_grid):
    """
    Eliminate obstacles that are not in contact with any free space (4-connectivity).
    Args:
        occupancy_grid (np.ndarray): 2D array where 0=free, 1=occupied, -1=unknown.
    Returns:
        np.ndarray: Updated occupancy grid with isolated obstacles removed.
    """

    # Method 1: Use dilation to find obstacles that contact free space, then keep only those
    # Create a mask for occupied cells
    obstacle_mask = (occupancy_grid == 1).astype(np.uint8)
    
    # Create a mask for free cells
    freespace_mask = (occupancy_grid == 0).astype(np.uint8)
    
    # Dilate freespace to find neighboring obstacles
    kernel = np.ones((3, 3), np.uint8)
    dilated_freespace = cv2.dilate(freespace_mask, kernel, iterations=1)
    
    # Find obstacles that are in contact with free space
    contact_mask = (dilated_freespace > 0) & (obstacle_mask > 0)
    
    # Update occupancy grid: keep only obstacles that contact free space
    updated_grid = np.copy(occupancy_grid)
    updated_grid[obstacle_mask > 0] = np.where(contact_mask[obstacle_mask > 0], 1, -1)  # Keep if contact, else unknown

    # Method. 2: For each occupied cell, check if any 4-neighbor is free. If not, set to unknown.
    """ updated_grid = np.copy(occupancy_grid)
    obstacle_mask = (occupancy_grid == 1)
    free_mask = (occupancy_grid == 0)
    h, w = occupancy_grid.shape
    # For each obstacle pixel, check if any 4-neighbor is free
    for y in range(h):
        for x in range(w):
            if obstacle_mask[y, x]:
                # 4-connectivity neighbors
                neighbors = [
                    (y-1, x), (y+1, x), (y, x-1), (y, x+1)
                ]
                touches_free = False
                for ny, nx in neighbors:
                    if 0 <= ny < h and 0 <= nx < w:
                        if free_mask[ny, nx]:
                            touches_free = True
                            break
                if not touches_free:
                    updated_grid[y, x] = -1 """
    return updated_grid

def _trim_3d_pointcloud(pointcloud, max_distance):
    """
    Trim the 3D point cloud to remove points beyond a maximum distance.
    
    Args:
        pointcloud (np.ndarray): Nx3 array of 3D points.
        max_distance (float): Maximum distance from the origin to keep points.

    Returns:
        np.ndarray: Trimmed point cloud.
    """
    if len(pointcloud) == 0:
        return pointcloud
        
    # Calculate Euclidean distance from origin (0,0,0) for each point
    distances = np.linalg.norm(pointcloud, axis=1)
    
    # Filter points within max_distance
    mask = distances <= max_distance
    return pointcloud[mask]

def image_to_3d_pointcloud(rgb_image, camera_intrinsics, model=None, 
                          encoder='vitl', dataset='hypersim', max_depth=20):
    """
    Convert an RGB image to a 3D point cloud using DepthAnythingV2 metric depth estimation.
    
    Pipeline:
    1. Use DepthAnythingV2 to estimate metric depth from RGB image
    2. Back-project depth map to 3D points using camera intrinsics
    3. Return Nx3 point cloud array
    
    Args:
        rgb_image (np.ndarray): Input RGB image (HxWx3). BGR if from cv2.imread, convert if needed.
        camera_intrinsics (dict): Camera intrinsics with keys: 'fx', 'fy', 'cx', 'cy'
        model (torch.nn.Module): Pre-loaded DepthAnythingV2 model. If None, loads one automatically.
        encoder (str): Encoder type if model is None. Options: 'vits', 'vitb', 'vitl'
        dataset (str): Dataset type if model is None. Options: 'hypersim', 'vkitti'
        max_depth (float): Max depth value if model is None. 20 for indoor, 80 for outdoor.
    
    Returns:
        np.ndarray: Nx3 array of 3D points in camera frame (X right, Y down, Z forward).
    """
    # Load model if not provided
    if model is None:
        print("No model provided. Loading DepthAnythingV2 model...")
        model = get_model(encoder=encoder, dataset=dataset, max_depth=max_depth)
    
    # Ensure model is in eval mode
    model.eval()
    
    depth = model.infer_image(rgb_image)  # Returns HxW depth in meters
    # Infer depth map in meters (HxW array)
    # with torch.no_grad():
    #     depth = model.infer_image(rgb_image)  # Returns HxW depth in meters
    
    # Get image dimensions
    h, w = depth.shape
    
    # Extract camera intrinsics
    fx = camera_intrinsics.get('fx')
    fy = camera_intrinsics.get('fy')
    cx = camera_intrinsics.get('cx', w / 2.0)
    cy = camera_intrinsics.get('cy', h / 2.0)
    # fx = fy = 470.4 * 1.5  # focal length
    cx, cy = w / 2, h / 2  # principal point at image center
    
    # Create pixel grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Back-project to 3D with inverted Y and Z axes
    X = (x - cx) * depth / fx
    Y = -(y - cy) * depth / fy
    Z = -depth
    
    # Stack into point cloud (N, 3)
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    # Filter out invalid points
    valid_mask = (np.abs(Z.reshape(-1)) > 0) & (np.abs(Z.reshape(-1)) < max_depth)
    points = points[valid_mask]

    # save as PLY file
    with open('pointcloud.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for p in points:
            f.write(f'{p[0]} {p[1]} {p[2]}\n')
    
    return points

def pointcloud_to_occupancy_grid(pointcloud, grid_size, grid_resolution, height_range=(-0.5, 2.0), obstacle_threshold=0.1, use_data_bounds=True, padding=0.5):
    """
    Convert a 3D point cloud to a 2D occupancy grid.
    
    Args:
        pointcloud (np.ndarray): Nx3 array of 3D points in the camera frame (X right, Y down, Z forward).
        grid_size (tuple): (width, height) of the occupancy grid in meters. Used only when use_data_bounds=False.
        grid_resolution (float): Size of each grid cell in meters.
        height_range (tuple): (min, max) height to consider for obstacles/freespace (relative to camera Y).
        obstacle_threshold (float): Height threshold to consider a cell occupied.
        use_data_bounds (bool): If True, compute grid bounds from point cloud data (mirrors pointcloud_to_occupancy_grid.py).
        padding (float): Padding (meters) to add around data bounds when use_data_bounds=True.
    
    Returns:
        np.ndarray: 2D array where 0=free, 1=occupied, -1=unknown.
    """
    if len(pointcloud) == 0:
        if use_data_bounds:
            return np.zeros((0, 0), dtype=np.int8)
        grid_width_m, grid_height_m = grid_size
        grid_w = int(np.ceil(grid_width_m / grid_resolution))
        grid_h = int(np.ceil(grid_height_m / grid_resolution))
        return np.full((grid_h, grid_w), -1, dtype=np.int8)

    # Extract coordinates
    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]  # Y is height/vertical (downwards in camera frame)
    z_points = pointcloud[:, 2]

    # Filter points within height range (Y axis)
    in_height_mask = (y_points >= height_range[0]) & (y_points <= height_range[1])
    filtered_points = pointcloud[in_height_mask]

    if len(filtered_points) == 0:
        if use_data_bounds:
            return np.zeros((0, 0), dtype=np.int8)
        grid_width_m, grid_height_m = grid_size
        grid_w = int(np.ceil(grid_width_m / grid_resolution))
        grid_h = int(np.ceil(grid_height_m / grid_resolution))
        return np.full((grid_h, grid_w), -1, dtype=np.int8)

    valid_x = filtered_points[:, 0]
    valid_z = filtered_points[:, 2]
    valid_y = filtered_points[:, 1]

    if use_data_bounds:
        x_min, x_max = valid_x.min(), valid_x.max()
        z_min, z_max = valid_z.min(), valid_z.max()

        # Add padding around the data bounds
        x_min -= padding
        x_max += padding
        z_min -= padding
        z_max += padding

        grid_w = int(np.ceil((x_max - x_min) / grid_resolution))
        grid_h = int(np.ceil((z_max - z_min) / grid_resolution))
    else:
        grid_width_m, grid_height_m = grid_size
        grid_w = int(np.ceil(grid_width_m / grid_resolution))
        grid_h = int(np.ceil(grid_height_m / grid_resolution))

        # Define grid bounds (centered X, positive Z)
        x_min = -grid_width_m / 2.0
        x_max = grid_width_m / 2.0
        z_min = 0.0
        z_max = grid_height_m

        # Filter points within grid bounds (X and Z)
        in_bounds_mask = (valid_x >= x_min) & (valid_x < x_max) & (valid_z >= z_min) & (valid_z < z_max)
        valid_x = valid_x[in_bounds_mask]
        valid_z = valid_z[in_bounds_mask]
        valid_y = valid_y[in_bounds_mask]

        if len(valid_x) == 0:
            return np.full((grid_h, grid_w), -1, dtype=np.int8)

    # Initialize grid
    occupancy_grid = np.full((grid_h, grid_w), -1, dtype=np.int8)

    # Convert coordinates to grid indices
    grid_x = ((valid_x - x_min) / grid_resolution).astype(int)
    grid_z = ((valid_z - z_min) / grid_resolution).astype(int)

    # Clip indices to be safe
    grid_x = np.clip(grid_x, 0, grid_w - 1)
    grid_z = np.clip(grid_z, 0, grid_h - 1)

    # Flatten indices for fast processing
    flat_indices = grid_z * grid_w + grid_x

    # Fill grid with maximum height per cell
    # Use np.maximum.at for vectorized processing (much faster than loop)
    # Initialize with -inf to correctly capture negative heights (if any)
    # Note: pointcloud_to_occupancy_grid.py used zeros initialization which clamps negative max heights to 0.
    # We use -inf here to be more mathematically correct for max height finding, 
    # but the subsequent threshold check handles occupancy logic similarly.
    height_grid_flat = np.full(grid_h * grid_w, -np.inf, dtype=np.float32)
    np.maximum.at(height_grid_flat, flat_indices, valid_y)
    height_grid = height_grid_flat.reshape(grid_h, grid_w)

    # Mark occupied cells (cells with obstacles above threshold)
    occupancy_grid[height_grid > obstacle_threshold] = 1
    occupancy_grid[height_grid <= obstacle_threshold] = 0

    # Mark cells with no data as unknown
    visited_flat = np.zeros(grid_h * grid_w, dtype=bool)
    visited_flat[flat_indices] = True
    visited = visited_flat.reshape(grid_h, grid_w)
    
    occupancy_grid[~visited] = -1

    return occupancy_grid

def visualize_occupancy_grid(occupancy_grid):
    """
    Visualize the occupancy grid using OpenCV.

    Args:
        occupancy_grid (np.ndarray): 2D array with probability values in [0, 1]
            or discrete values -1 (unknown), 0 (free), 1 (occupied).
    Returns:
        np.ndarray: Color image visualizing the occupancy grid (BGR).
    """
    height, width = occupancy_grid.shape

    grid_min = float(np.min(occupancy_grid))
    grid_max = float(np.max(occupancy_grid))
    if grid_min >= 0.0 and grid_max <= 1.0:
        prob = np.clip(occupancy_grid.astype(np.float32), 0.0, 1.0)
        intensity = (1.0 - prob) * 255.0
        gray = intensity.astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    vis_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Colors (BGR)
    COLOR_UNKNOWN = [128, 128, 128]  # Gray
    COLOR_FREE = [255, 255, 255]     # White
    COLOR_OCCUPIED = [0, 0, 0]       # Black

    vis_image[occupancy_grid == -1] = COLOR_UNKNOWN
    vis_image[occupancy_grid == 0] = COLOR_FREE
    vis_image[occupancy_grid == 1] = COLOR_OCCUPIED

    return vis_image