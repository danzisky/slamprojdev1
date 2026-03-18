# Localization Package

This package provides a chair-based particle filter localizer for slam_rover.

Key points:
- Global initialization: if no initial pose is known, particles are sampled across free space.
- Pose-hint initialization: if an initial pose is known, particles are sampled around it with a configurable spread.
- Chair matching: ambiguous chair observations are associated with landmarks using gated joint compatibility branch-and-bound rather than greedy nearest-neighbor matching.
- Detector optionality: the localizer can run either from direct chair observations or from images through the detector wrapper.

Main modules:
- `map_data.py`: occupancy map loading, free-space sampling, visibility checks, and raycasting.
- `detector.py`: chair detection plus depth-to-range conversion.
- `association.py`: joint compatibility data association.
- `particle_filter.py`: motion model, measurement update, resampling, and pose estimation.

Expected landmarks JSON:

```json
{
  "resolution": 0.0197,
  "landmarks": [[64.0, 191.0], [274.0, 31.0]]
}
```

The landmark coordinates are interpreted in map pixels and converted to meters using the provided resolution.

Example usage from the `slam_rover` directory:

```bash
python -m localization.demo --map ..\rover\live\classmap.png --landmarks ..\landmarks.json --camera-id 1 --fx 589.54200724 --fy 589.80048532 --cx 328.93066342 --cy 170.73631903
```

If you already have detections from another model, skip the built-in detector and call `update_from_observations()` directly with `ChairObservation` objects.