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

import cv2
import numpy as np
import os
import sys

# Ensure we can import mapper 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mapper import (
    get_model,
    image_to_3d_pointcloud,
    _process_occupancy_grid,
    visualize_occupancy_grid,
    pointcloud_to_occupancy_grid
)

def warp_local_grid_to_global(local_grid, global_shape, angle_deg, resolution=0.05):
    """
    Warp the local occupancy grid (robot at top-center, facing down)
    into the global grid frame (robot at center).
    
    local_grid: The occupancy grid of the single view.
    global_shape: (H, W) of the global map.
    angle_deg: Rotation of the robot for this view (0 = World Forward/Up).
    """
    
    h_local, w_local = local_grid.shape
    h_global, w_global = global_shape
    
    # Pivot in Local Grid (Robot Position) -> Top Center
    # Since grid_x = (x - x_min)/res, x_min = -W/2. 
    # At x=0, grid_x = (0 - (-W/2))/res = W/2/res = w_local/2.
    # grid_z = z/res. At z=0, grid_z = 0.
    pivot_local = (w_local // 2, 0)
    
    # Pivot in Global Grid (Robot Position) -> Center
    pivot_global = (w_global // 2, h_global // 2)
    
    # Rotation Angle
    # Local grid faces DOWN (Positive Y in image coords).
    # We want Angle=0 to face UP (Negative Y in image coords).
    # So base rotation is 180 degrees.
    # Heading (angle_deg): Positive is usually Right (CW) or Left (CCW)?
    # Standard math: CCW. 
    # Robot convention: usually CCW (X forward, Y left) or CW (Compass).
    # Assuming positive angle = Turn Right (CW) for now based on user hint "rotate on their bottoms".
    # If 0 is Up, 90 is Right.
    # In Image coords (Y down), +90 rotation is CW? No, CV rotation is usually CCW around origin (X right, Y down).
    # X -> Y is 90 deg. (Right -> Down).
    # We want Up (0) -> Right (90). That's 90 deg CW.
    # Rotation matrix in OpenCV is CCW.
    # So we want -angle for CW turn.
    # Base: Down (Local) -> Up (Global). 180 deg.
    # Total = 180 - angle.
    rotation_angle = 180 - angle_deg
    
    # Create Transform Matrix manually
    # 1. Translate Local Pivot to Origin
    # 2. Rotate
    # 3. Translate Origin to Global Pivot
    
    rad = np.deg2rad(rotation_angle)
    c = np.cos(rad)
    s = np.sin(rad)
    
    # Rotation Matrix (around origin)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # Translation 1 (Local Pivot -> Origin)
    T1 = np.array([
        [1, 0, -pivot_local[0]],
        [0, 1, -pivot_local[1]],
        [0, 0, 1]
    ])
    
    # Translation 2 (Origin -> Global Pivot)
    T2 = np.array([
        [1, 0, pivot_global[0]],
        [0, 1, pivot_global[1]],
        [0, 0, 1]
    ])
    
    M_combined = T2 @ R @ T1
    M_cv = M_combined[:2, :]
    
    # Map -1 to 127 for warping. 0->255 (Free), 1->0 (Occupied)? 
    # Or keep standard: 0=Free, 255=Occupied, 127=Unknown.
    grid_uint8 = np.full_like(local_grid, 127, dtype=np.uint8)
    grid_uint8[local_grid == 0] = 255 # Free as White
    grid_uint8[local_grid == 1] = 0   # Occupied as Black
    
    warped = cv2.warpAffine(
        grid_uint8, 
        M_cv, 
        (w_global, h_global), 
        flags=cv2.INTER_NEAREST, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=127 # Unknown
    )
    
    # Convert back to -1, 0, 1
    # 127 -> -1 (Unknown)
    # 255 -> 0 (Free)
    # 0 -> 1 (Occupied)
    warped_conv = np.full(global_shape, -1, dtype=np.int8)
    warped_conv[warped == 255] = 0
    warped_conv[warped == 0] = 1
    
    return warped_conv

def main():
    base_path = "D:/schproj/Depth-Anything-V2/metric_depth/slam_rover"
    img_path = os.path.join(base_path, "test_image.jpeg")
    
    # Use images if available, else re-use same image with different angles
    scan_sequence = [
        # (Image Path, Angle in Degrees)
        # 0 = Facing Forward/Up
        # +90 = Facing Right
        # -90 = Facing Left
        (img_path, 0),
        (img_path, 45),
        (img_path, 90),
        (img_path, 135),
        (img_path, 180),
        (img_path, -45),
        (img_path, -90)
    ]
    
    print("Loading Model...")
    model = get_model(encoder='vitl', dataset='hypersim', max_depth=20)
    
    # Setup Global Grid (60x60m)
    global_size_m = 60.0
    resolution = 0.05
    global_dim = int(np.ceil(global_size_m / resolution))
    
    # 0 = Free, 1 = Occupied, -1 = Unknown
    combined_grid = np.full((global_dim, global_dim), -1, dtype=np.int8)
    
    # Intrinsic Setup (Once, assumes same camera)
    dummy = cv2.imread(img_path)
    if dummy is None:
        print(f"Error reading {img_path}")
        return
    h, w = dummy.shape[:2]
    # Adjust intrinsics to match your setup
    intrinsics = { 'fx': 470.4 * 1.5, 'fy': 470.4 * 1.5, 'cx': w/2, 'cy': h/2 }
    
    print("Starting Processing Loop...")
    
    for i, (path, angle) in enumerate(scan_sequence):
        real_path = os.path.join(base_path, path) if not os.path.exists(path) else path
        if not os.path.exists(real_path): real_path = img_path # Fallback for demo
        
        rgb = cv2.imread(real_path)
        if rgb is None: continue
        
        print(f"Frame {i}: Processing Angle {angle} deg")
        
        # 1. Get Points (returns -Z depth)
        points = image_to_3d_pointcloud(rgb, intrinsics, model=model)
        
        # 2. Fix Z for Local Grid Gen (Make Z positive)
        # We need positive Z for the fixed-bounds grid generation to works as expected (0..Max)
        points[:, 2] = -points[:, 2] 
        
        # 3. Generate Local Occupancy Grid
        # Use Fixed Bounds so we know exactly where the robot is (Top Center)
        local_w_m = 30.0 # Wide enough for the cone
        local_h_m = 20.0 # Max depth coverage
        
        local_grid = pointcloud_to_occupancy_grid(
            points,
            grid_size=(local_w_m, local_h_m),
            grid_resolution=resolution,
            use_data_bounds=False, # Important: Fixed frame
            height_range=(-0.5, 2.0),
            obstacle_threshold=0.5
        )
        
        # 4. Process (Refine) Local Grid
        local_processed = _process_occupancy_grid(local_grid)
        
        # 5. Rotate and Map to Global
        warped_grid = warp_local_grid_to_global(local_processed, (global_dim, global_dim), angle)
        
        # 6. Merge
        # Occupied overwrites everything.
        # Free overwrites Unknown (but not Occupied).
        
        occupied_mask = (warped_grid == 1)
        free_mask = (warped_grid == 0)
        
        combined_grid[occupied_mask] = 1
        # Set free if not already occupied
        combined_grid[free_mask & (combined_grid != 1)] = 0
        
        # 7. Visualize
        vis = visualize_occupancy_grid(combined_grid)
        
        # Draw Robot and Direction
        cx, cy = global_dim // 2, global_dim // 2
        # Angle 0 is Up (-90 deg in CV space)
        rad = np.deg2rad(angle - 90) 
        end_x = int(cx + 40 * np.cos(rad))
        end_y = int(cy + 40 * np.sin(rad))
        
        cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 0, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        
        cv2.putText(vis, f"Frame {i} Angle {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Global Map", vis)
        if cv2.waitKey(250) == 27: break # Wait 250ms

    print("Finished. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
