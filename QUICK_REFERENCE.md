# Mapper Quick Reference

## Function Signatures

```python
# Model Management
model = get_model(encoder='vitl', dataset='hypersim', max_depth=20, checkpoint_path=None)

# Image → Point Cloud
cloud = image_to_3d_pointcloud(rgb_image, camera_intrinsics, model=None, 
                                encoder='vitl', dataset='hypersim', max_depth=20)

# Point Cloud Filtering
trimmed_cloud = _trim_3d_pointcloud(pointcloud, max_distance)

# Point Cloud → Grid
grid = pointcloud_to_occupancy_grid(pointcloud, grid_size, grid_resolution, 
                                    height_range=(-0.5, 2.0), obstacle_threshold=0.1)

# Grid Processing
processed = _process_occupancy_grid(occupancy_grid)

# Visualization
img = visualize_occupancy_grid(occupancy_grid)
```

---

## Common Parameters

### Camera Intrinsics
```python
camera_intrinsics = {
    'fx': 470.4,                    # Focal length X (pixels)
    'fy': 470.4,                    # Focal length Y (pixels)
    'cx': img_width / 2.0,          # Principal point X (pixels)
    'cy': img_height / 2.0          # Principal point Y (pixels)
}
```

### Grid Parameters
```python
grid_size = (10.0, 15.0)           # (width, height) in meters
grid_resolution = 0.05              # Cell size in meters (5cm)
height_range = (-0.5, 2.0)         # Min/max height to consider (meters)
obstacle_threshold = 0.1            # Height for occupancy (meters)
```

---

## Common Recipes

### 1. Single Image to Occupancy Grid
```python
model = get_model()
img = cv2.imread('image.jpg')
cloud = image_to_3d_pointcloud(img, camera_intrinsics, model=model)
grid = pointcloud_to_occupancy_grid(cloud, (10, 15), 0.05)
vis = visualize_occupancy_grid(grid)
```

### 2. Batch Processing
```python
model = get_model()  # Load once
for img_path in image_paths:
    img = cv2.imread(img_path)
    cloud = image_to_3d_pointcloud(img, camera_intrinsics, model=model)
    # ... process cloud
```

### 3. With Cleanup
```python
cloud = image_to_3d_pointcloud(img, intrinsics, model=model)
cloud = _trim_3d_pointcloud(cloud, max_distance=15)  # Remove far points
grid = pointcloud_to_occupancy_grid(cloud, (10, 15), 0.05)
grid = _process_occupancy_grid(grid)  # Clean up
```

### 4. Real-time Fast Processing
```python
model = get_model(encoder='vits')  # Fast model
cloud = image_to_3d_pointcloud(img, intrinsics, model=model)
grid = pointcloud_to_occupancy_grid(cloud, (10, 15), 0.1)  # Coarser resolution
```

### 5. High-Quality Mapping
```python
model = get_model(encoder='vitl')  # Best quality
cloud = image_to_3d_pointcloud(img, intrinsics, model=model)
grid = pointcloud_to_occupancy_grid(cloud, (10, 15), 0.02)  # Fine resolution
grid = _process_occupancy_grid(grid)
```

---

## Return Formats

### Point Cloud
```python
cloud.shape = (N, 3)
# N = number of points
# 3 = [X, Y, Z] in camera frame
```

### Occupancy Grid
```python
grid.shape = (height_cells, width_cells)
grid values:
  0 = free space
  1 = occupied
  -1 = unknown
```

### Visualization
```python
vis.shape = (height_cells, width_cells, 3)
# BGR image, can use cv2.imshow() or cv2.imwrite()
```

---

## Model Configurations

| Config | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| vits   | ⚡⚡⚡  | ⭐      | Real-time on CPU |
| vitb   | ⚡⚡   | ⭐⭐   | Balanced |
| vitl   | ⚡    | ⭐⭐⭐ | Best quality, GPU recommended |

| Dataset | Scene | max_depth |
|---------|-------|-----------|
| hypersim | Indoor | 20 |
| vkitti | Outdoor | 80 |

---

## Grid Resolution Guide

| Resolution | Cell Size | Use Case |
|------------|-----------|----------|
| 0.02 | 2cm | Fine mapping, slower |
| 0.05 | 5cm | **Recommended** |
| 0.1 | 10cm | Fast mapping, coarse |
| 0.2 | 20cm | Very fast, rough |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Check checkpoint path in `checkpoints/` |
| Empty point cloud | Check image quality and camera intrinsics |
| Grid all unknown | Adjust height_range or obstacle_threshold |
| GPU out of memory | Use smaller encoder (vits) or reduce resolution |
| Slow inference | Use vits encoder or smaller input size |

---

## Integration Example

```python
from slam_rover.mapper import *

# Initialize
self.model = get_model(encoder='vitl', dataset='hypersim')
self.camera_intrinsics = {...}

# Main loop
def process_frame(self, rgb_image):
    # Generate map
    cloud = image_to_3d_pointcloud(rgb_image, self.camera_intrinsics, self.model)
    grid = pointcloud_to_occupancy_grid(cloud, (10, 15), 0.05, 
                                       height_range=(-0.5, 2.0),
                                       obstacle_threshold=0.15)
    grid = _process_occupancy_grid(grid)
    
    # Update global map
    self.update_map(grid)
    
    # Visualize
    vis = visualize_occupancy_grid(grid)
    return vis
```

---

## Resources

- **Full Docs**: See `MAPPER_README.md`
- **Example**: See `example_usage.py`
- **GitHub**: https://github.com/DepthAnything/Depth-Anything-V2
- **Paper**: https://arxiv.org/abs/2406.09414
