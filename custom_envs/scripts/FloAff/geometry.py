"""SE(2), table-edge and camera-pose geometry."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def unit_xy(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError(f"expected a 2-vector, got {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("zero-length direction")
    return vector / norm


def left_tangent(outward_xy: Iterable[float]) -> np.ndarray:
    outward = unit_xy(outward_xy)
    return np.array([-outward[1], outward[0]], dtype=np.float64)


def angle_of(vector_xy: Iterable[float]) -> float:
    vector = unit_xy(vector_xy)
    return math.atan2(float(vector[1]), float(vector[0]))


def angular_distance(first: float, second: float) -> float:
    return abs(wrap_angle(float(first) - float(second)))


def angular_distance_deg(first: float, second: float) -> float:
    return math.degrees(angular_distance(first, second))


def joint1_towards_target(base_pose: Iterable[float], target_xy: Iterable[float]) -> float:
    bx, by, byaw = (float(value) for value in base_pose)
    tx, ty = (float(value) for value in target_xy)
    return wrap_angle(math.atan2(ty - by, tx - bx) - byaw)


def base_yaw_parallel_to_edge(outward_xy: Iterable[float]) -> float:
    """Select the requested +90-degree/current branch for the robot long axis."""
    return wrap_angle(angle_of(outward_xy) - math.pi / 2.0)


def edge_coordinates(
    point_xy: Iterable[float],
    target_xy: Iterable[float],
    outward_xy: Iterable[float],
) -> tuple[float, float]:
    delta = np.asarray(tuple(point_xy), dtype=np.float64) - np.asarray(
        tuple(target_xy), dtype=np.float64
    )
    outward = unit_xy(outward_xy)
    tangent = left_tangent(outward)
    return float(delta @ outward), float(delta @ tangent)


def edge_point_world(
    target_xy: Iterable[float],
    outward_xy: Iterable[float],
    outward_m: float,
    tangent_m: float,
) -> np.ndarray:
    target = np.asarray(tuple(target_xy), dtype=np.float64)
    outward = unit_xy(outward_xy)
    tangent = left_tangent(outward)
    return target + float(outward_m) * outward + float(tangent_m) * tangent


def observation_camera_xy(
    target_xy: Iterable[float],
    outward_xy: Iterable[float],
    camera_range_m: float,
    alpha_rad: float,
) -> np.ndarray:
    outward = unit_xy(outward_xy)
    tangent = left_tangent(outward)
    direction = math.cos(alpha_rad) * outward + math.sin(alpha_rad) * tangent
    return np.asarray(tuple(target_xy), dtype=np.float64) + float(camera_range_m) * direction


def camera_yaw_towards_target(
    camera_xy: Iterable[float], target_xy: Iterable[float], yaw_offset_rad: float = 0.0
) -> float:
    camera = np.asarray(tuple(camera_xy), dtype=np.float64)
    target = np.asarray(tuple(target_xy), dtype=np.float64)
    delta = target - camera
    return wrap_angle(math.atan2(float(delta[1]), float(delta[0])) + yaw_offset_rad)


def base_pose_for_camera_pose(
    camera_xy_world: Iterable[float],
    camera_yaw_world: float,
    camera_xy_in_base: Iterable[float],
) -> np.ndarray:
    """Recover planar base pose from a desired base-fixed camera pose."""
    camera_xy = np.asarray(tuple(camera_xy_world), dtype=np.float64)
    offset = np.asarray(tuple(camera_xy_in_base), dtype=np.float64)
    c, s = math.cos(camera_yaw_world), math.sin(camera_yaw_world)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    base_xy = camera_xy - rotation @ offset
    return np.array([base_xy[0], base_xy[1], wrap_angle(camera_yaw_world)], dtype=np.float64)


def observation_metrics(
    camera_xy: Iterable[float], target_xy: Iterable[float], outward_xy: Iterable[float]
) -> tuple[float, float]:
    camera = np.asarray(tuple(camera_xy), dtype=np.float64)
    target = np.asarray(tuple(target_xy), dtype=np.float64)
    delta = camera - target
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        raise ValueError("camera and target positions coincide")
    alpha = wrap_angle(angle_of(delta) - angle_of(outward_xy))
    return distance, alpha


def corner_family_is_exclusive(
    camera_xy: Iterable[float],
    target_xy: Iterable[float],
    selected_outward_xy: Iterable[float],
    other_outward_xy: Iterable[float],
    selected_max_deg: float = 35.0,
    other_min_deg: float = 55.0,
) -> bool:
    direction = np.asarray(tuple(camera_xy), dtype=np.float64) - np.asarray(
        tuple(target_xy), dtype=np.float64
    )
    theta = angle_of(direction)
    selected = angular_distance_deg(theta, angle_of(selected_outward_xy))
    other = angular_distance_deg(theta, angle_of(other_outward_xy))
    return selected <= float(selected_max_deg) + 1e-9 and other >= float(other_min_deg) - 1e-9


def rotate_world_xy_to_frame(values_xy: np.ndarray, frame_x_world: Iterable[float]) -> np.ndarray:
    """Express world XY vectors in a frame whose +X is ``frame_x_world``."""
    values = np.asarray(values_xy, dtype=np.float64)
    x_axis = unit_xy(frame_x_world)
    y_axis = left_tangent(x_axis)
    rotation_world_frame = np.stack([x_axis, y_axis], axis=1)
    return values @ rotation_world_frame


def outward_world_to_base(outward_xy_world: Iterable[float], base_yaw_world: float) -> np.ndarray:
    c, s = math.cos(base_yaw_world), math.sin(base_yaw_world)
    rotation_world_base = np.array([[c, -s], [s, c]], dtype=np.float64)
    return unit_xy(outward_xy_world) @ rotation_world_base
