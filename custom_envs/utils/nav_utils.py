"""Navigation utility functions — coordinate transforms, path smoothing, etc."""

import numpy as np


def world_to_grid(wx: float, wy: float, origin: tuple, resolution: float):
    """Convert world coordinates (meters) to grid indices.

    Args:
        wx, wy:   world position in meters.
        origin:   (ox, oy) world position of grid cell (0, 0).
        resolution: meters per grid cell.

    Returns:
        (row, col) int grid index.
    """
    col = int((wx - origin[0]) / resolution)
    row = int((wy - origin[1]) / resolution)
    return (row, col)


def grid_to_world(row: int, col: int, origin: tuple, resolution: float):
    """Convert grid indices to world coordinates (cell centre)."""
    wx = origin[0] + (col + 0.5) * resolution
    wy = origin[1] + (row + 0.5) * resolution
    return (wx, wy)


def world_to_body(wx: float, wy: float, robot_pos: tuple):
    """Convert a world point to robot body frame.

    Args:
        wx, wy:     world point (m).
        robot_pos:  (rx, ry, ryaw) robot pose in world frame.

    Returns:
        (dx_body, dy_body) in robot local frame.
    """
    rx, ry, ryaw = robot_pos
    dx = wx - rx
    dy = wy - ry
    cos_yaw = np.cos(ryaw)
    sin_yaw = np.sin(ryaw)
    dx_body = cos_yaw * dx + sin_yaw * dy
    dy_body = -sin_yaw * dx + cos_yaw * dy
    return (dx_body, dy_body)


def smooth_path(path: list, window_size: int = 3):
    """Moving-average smooth a list of (x, y) waypoints.

    Args:
        path: list of (x, y) waypoints.
        window_size: averaging window (odd preferred).

    Returns:
        Smoothed list of same length.
    """
    if len(path) < window_size:
        return path
    smoothed = []
    half = window_size // 2
    for i in range(len(path)):
        start = max(0, i - half)
        end = min(len(path), i + half + 1)
        xs = [p[0] for p in path[start:end]]
        ys = [p[1] for p in path[start:end]]
        smoothed.append((np.mean(xs), np.mean(ys)))
    return smoothed


def is_goal_reached(robot_pos: tuple, goal_pos: tuple, threshold: float = 0.3):
    """Check whether robot is within threshold of the goal."""
    dx = robot_pos[0] - goal_pos[0]
    dy = robot_pos[1] - goal_pos[1]
    return (dx * dx + dy * dy) < (threshold * threshold)


def euler_from_quat(quat):
    """Extract yaw from a [w, x, y, z] quaternion (Isaac Sim convention)."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))
