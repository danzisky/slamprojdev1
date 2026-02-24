def extract_frontier_goals(frontier_mask, min_region_size=10):
    """
    For each contiguous frontier region above a minimum size, select a goal point (centroid).
    Args:
        frontier_mask (np.ndarray): Boolean mask of frontier cells.
        min_region_size (int): Minimum number of pixels for a region to be considered a goal.
    Returns:
        List of (y, x) tuples: goal points (row, col) in image coordinates.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask.astype(np.uint8), connectivity=8)
    goals = []
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            # Centroid is (x, y) float; convert to int (row, col)
            cy, cx = int(round(centroids[i][1])), int(round(centroids[i][0]))
            goals.append((cy, cx))
    return goals

def draw_frontier_goals(vis, goals, color=(0,255,255), radius=4, thickness=2):
    """
    Draw goal points on a visualization image.
    Args:
        vis (np.ndarray): BGR image to draw on.
        goals (list): List of (y, x) tuples.
        color (tuple): BGR color for goals.
        radius (int): Circle radius.
        thickness (int): Circle thickness.
    Returns:
        np.ndarray: Image with goals drawn.
    """
    for (y, x) in goals:
        cv2.circle(vis, (x, y), radius, color, thickness)
    return vis
import numpy as np
import cv2


def find_frontiers(occupancy_grid, unknown_band=0.1):
    """
    Find frontier cells in an occupancy grid.
    A frontier cell is a free cell adjacent (4- or 8-connectivity) to unknown space.

    Args:
        occupancy_grid (np.ndarray): 2D array with probability values in [0, 1]
            or discrete values -1 (unknown), 0 (free), 1 (occupied).
        unknown_band (float): Probability band around 0.5 treated as unknown.
    Returns:
        np.ndarray: Boolean mask of frontier cells (True = frontier).
    """
    grid_min = float(np.min(occupancy_grid))
    grid_max = float(np.max(occupancy_grid))
    if grid_min >= 0.0 and grid_max <= 1.0:
        unknown_mask = np.abs(occupancy_grid - 0.5) <= unknown_band
        free_mask = occupancy_grid < (0.5 - unknown_band)
    else:
        free_mask = (occupancy_grid == 0)
        unknown_mask = (occupancy_grid == -1)
    # 8-connectivity kernel
    kernel = np.ones((3, 3), np.uint8)
    # Dilate unknowns: cells adjacent to unknowns become 1
    unknown_dilated = cv2.dilate(unknown_mask.astype(np.uint8), kernel, iterations=1)
    # A frontier is a free cell that touches unknown
    frontier_mask = free_mask & (unknown_dilated > 0)
    return frontier_mask


def draw_frontiers(occupancy_grid, frontier_mask, color=(0, 0, 255)):
    """
    Overlay frontier cells on a visualization of the occupancy grid.
    Args:
        occupancy_grid (np.ndarray): 2D array where 0=free, 1=occupied, -1=unknown.
        frontier_mask (np.ndarray): Boolean mask of frontier cells.
        color (tuple): BGR color for frontiers.
    Returns:
        np.ndarray: BGR image with frontiers highlighted.
    """
    from mapper import visualize_occupancy_grid
    vis = visualize_occupancy_grid(occupancy_grid)
    vis[frontier_mask] = color
    return vis


if __name__ == "__main__":
    # Example usage: load a grid, find and display frontiers
    import sys
    if len(sys.argv) < 2:
        print("Usage: python frontier_utils.py <occupancy_grid.npy>")
        sys.exit(1)
    grid = np.load(sys.argv[1])
    frontiers = find_frontiers(grid)
    vis = draw_frontiers(grid, frontiers)
    goals = extract_frontier_goals(frontiers, min_region_size=20)
    vis = draw_frontier_goals(vis, goals)
    cv2.imshow("Frontiers + Goals", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
