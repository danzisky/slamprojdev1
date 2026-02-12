# Mapper Module Documentation

## Overview

The `mapper.py` module provides a modular pipeline for converting RGB images to 3D point clouds and generating occupancy grids for robotic mapping and navigation. It uses **DepthAnythingV2** for metric depth estimation.

## Key Components

### 1. `get_model()` - Model Loading

Loads and initializes the DepthAnythingV2 metric depth model.

```python
model = get_model(
    encoder='vitl',           # 'vits', 'vitb', 'vitl'
    dataset='hypersim',       # 'hypersim' (indoor) or 'vkitti' (outdoor)
    max_depth=20,             # 20 for indoor, 80 for outdoor
    checkpoint_path=None      # Auto-construct if None
)
```

**Parameters:**
- `encoder`: Model size. Larger = better quality but slower
- `dataset`: Determines the pre-trained weights (indoor/outdoor)
- `max_depth`: Maximum depth value the model can estimate
- `checkpoint_path`: Custom path to checkpoint (optional)

**Returns:** DepthAnythingV2 model in eval mode on appropriate device (CUDA/MPS/CPU)

---

### 2. `image_to_3d_pointcloud()` - RGB to Point Cloud

Converts an RGB image to a 3D point cloud using DepthAnythingV2 metric depth.

```python
pointcloud = image_to_3d_pointcloud(
    rgb_image=img,                    # HxWx3 RGB/BGR image
    camera_intrinsics={
        'fx': 470.4,                  # Focal length X
        'fy': 470.4,                  # Focal length Y
        'cx': width / 2.0,            # Principal point X
        'cy': height / 2.0            # Principal point Y
    },
    model=model,                      # Pre-loaded model (optional)
    encoder='vitl',                   # Used only if model=None
    dataset='hypersim',               # Used only if model=None
    max_depth=20                      # Used only if model=None
)
```

**Pipeline:**
1. Depth estimation: DepthAnythingV2 infers HxW depth map (in meters)
2. Back-projection: Uses camera intrinsics to convert depth to 3D
3. Filtering: Removes points with zero/invalid depth

**Returns:** Nx3 numpy array of 3D points in camera frame

**Coordinate Frame:**
- X-axis: Right
- Y-axis: Down
- Z-axis: Forward (depth)

---

### 3. `_trim_3d_pointcloud()` - Point Cloud Filtering

Removes points beyond a maximum distance (useful for nearby mapping).

```python
trimmed_cloud = _trim_3d_pointcloud(pointcloud, max_distance=20.0)
```

**Parameters:**
- `pointcloud`: Nx3 array
- `max_distance`: Maximum Euclidean distance from origin

**Returns:** Filtered Nx3 array

---

### 4. `pointcloud_to_occupancy_grid()` - Point Cloud to Grid

Converts 3D point cloud to a 2D occupancy grid representation.

```python
grid = pointcloud_to_occupancy_grid(
    pointcloud=pointcloud,
    grid_size=(10.0, 15.0),          # (width, height) in meters
    grid_resolution=0.05,             # 5cm per cell
    height_range=(-0.5, 2.0),        # Height filtering
    obstacle_threshold=0.1            # Height for occupancy
)
```

**Parameters:**
- `grid_size`: (width, height) in meters, centered at origin
- `grid_resolution`: Cell size in meters
- `height_range`: (min, max) height to consider (Y-axis)
- `obstacle_threshold`: Height threshold for marking cells as occupied

**Returns:** HxW array where:
- `0` = Free space
- `1` = Occupied
- `-1` = Unknown (not observed)

**Grid Layout:**
- X-axis: Centered (from -width/2 to +width/2)
- Z-axis: Forward (from 0 to height)
- Row 0: Z=0 (closest)
- Row H-1: Z=max (farthest)

---

### 5. `_process_occupancy_grid()` - Grid Post-Processing

Cleans up occupancy grid using morphological operations:
- **Obstacles**: Noise removal + solidification
- **Freespace**: Gap closing with vertical kernels

```python
processed_grid = _process_occupancy_grid(raw_grid)
```

**Operations:**
1. Obstacle processing:
   - Erosion to remove noise
   - Dilation to solidify scattered points
   - Opening to clean remaining noise

2. Freespace processing:
   - Dilation to close small gaps
   - Opening to remove noise
   - Erosion to restore size
   - Vertical closing to fill vertical stripes

**Returns:** Processed grid with same format as input

---

### 6. `visualize_occupancy_grid()` - Grid Visualization

Creates a color visualization of occupancy grid.

```python
img = visualize_occupancy_grid(grid)
cv2.imshow('Occupancy Grid', img)
```

**Color Mapping (BGR):**
- White (255,255,255): Free space
- Black (0,0,0): Obstacles
- Gray (128,128,128): Unknown

**Returns:** HxWx3 BGR image suitable for cv2 display/saving

---

## Usage Pattern

### Option A: Reuse Model for Multiple Images (Recommended)

```python
# Load model once
model = get_model(encoder='vitl', dataset='hypersim', max_depth=20)

# Process multiple images
for image_path in image_paths:
    img = cv2.imread(image_path)
    
    # Pass same model instance
    cloud = image_to_3d_pointcloud(img, camera_intrinsics, model=model)
    
    # Process cloud
    cloud = _trim_3d_pointcloud(cloud, max_distance=20)
    grid = pointcloud_to_occupancy_grid(cloud, grid_size=(10, 15), grid_resolution=0.05)
    grid = _process_occupancy_grid(grid)
    
    # Visualize
    vis = visualize_occupancy_grid(grid)
    cv2.imwrite(f'{image_path.stem}_grid.png', vis)
```

### Option B: Auto-Load Model (Simple, Slower)

```python
# Model loads automatically if not provided
cloud = image_to_3d_pointcloud(
    img, 
    camera_intrinsics,
    encoder='vitl', 
    dataset='hypersim'
)
```

---

## Camera Intrinsics

Camera intrinsics are crucial for accurate back-projection. Format:

```python
camera_intrinsics = {
    'fx': 470.4,      # Focal length in X direction (pixels)
    'fy': 470.4,      # Focal length in Y direction (pixels)
    'cx': 320.0,      # Principal point X (pixels)
    'cy': 240.0       # Principal point Y (pixels)
}
```

**Obtaining intrinsics:**
- **From camera calibration**: Use OpenCV calibration tools
- **Default assumption**: Principal point at image center (cx=w/2, cy=h/2)
- **Example focal lengths**: 470-500 pixels (depends on resolution and actual camera)

---

## Grid Parameters Guide

### Resolution
- `0.02` (2cm): Fine-grained mapping, slower processing
- `0.05` (5cm): Good balance (recommended)
- `0.1` (10cm): Coarse mapping, faster processing

### Grid Size
For a 3m x 4m room with 0.05m resolution:
```python
grid_size = (3.0, 4.0)
# Results in 60x80 cell grid
```

### Height Range
Filters points by Y coordinate. For indoor robot:
```python
height_range = (-0.5, 2.0)  # Include ground-level to human-height obstacles
```

### Obstacle Threshold
Height above which points are considered obstacles:
```python
obstacle_threshold = 0.1  # 10cm above ground
```

---

## Performance Considerations

### Model Selection
| Encoder | Speed | Quality | Memory | Recommended For |
|---------|-------|---------|--------|-----------------|
| vits    | Fast  | Lower   | 24.8M  | Real-time on CPU |
| vitb    | Medium| Medium  | 97.5M  | Balanced |
| vitl    | Slow  | Highest | 335.3M | Best quality |

### Speed Tips
1. **Reuse model** for multiple images
2. **Use vits** encoder for real-time applications
3. **Increase grid resolution** (0.1 instead of 0.02) for faster processing
4. **Use GPU** (CUDA/MPS) when available

---

## Examples

See `example_usage.py` for complete working example including:
- Model loading
- Image-to-pointcloud conversion
- Point cloud trimming
- Occupancy grid generation
- Grid processing
- Visualization and saving

---

## Dependencies

```
numpy
opencv-python (cv2)
torch
depth-anything-v2  (from source in parent directory)
```

---

## Coordinate System Conventions

**Camera Frame (output of `image_to_3d_pointcloud`):**
```
       Y (down)
       |
       *----X (right)
      /
     / Z (forward/depth)
```

**Grid Frame (output of `pointcloud_to_occupancy_grid`):**
```
  Z (distance)
  ^
  |    Row 0: Z=0
  |  +---------+
  |  |  Grid  |
  |  +---------+
  |    Row H-1: Z=max
  +--------> X (centered)
```

---

## Troubleshooting

### Model Load Error
- Check checkpoint path: `checkpoints/depth_anything_v2_metric_<dataset>_<encoder>.pth`
- Download model from [GitHub releases](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)

### Point Cloud Issues
- Check camera intrinsics accuracy
- Verify image format (BGR from cv2, not RGB)
- Check depth map bounds match max_depth parameter

### Grid Artifacts
- Adjust `height_range` for your scene
- Tune `obstacle_threshold` based on clutter
- Increase resolution if grid is too coarse
