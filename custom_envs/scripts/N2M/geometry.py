"""Small, explicit SE(2)/SE(3) helpers used by the N2M collector."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return rpy_to_quat_wxyz(0.0, 0.0, yaw)


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert intrinsic XYZ roll/pitch/yaw to MuJoCo ``(w,x,y,z)``."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def se2_to_matrix(pose: Iterable[float]) -> np.ndarray:
    x, y, yaw = (float(v) for v in pose)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c, -s, 0.0, x],
         [s,  c, 0.0, y],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def matrix_to_se2(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.array(
        [transform[0, 3], transform[1, 3],
         math.atan2(transform[1, 0], transform[0, 0])],
        dtype=np.float64,
    )


def relative_se2(origin_world: Iterable[float], target_world: Iterable[float]) -> np.ndarray:
    return matrix_to_se2(
        np.linalg.inv(se2_to_matrix(origin_world)) @ se2_to_matrix(target_world)
    )


def transform_se2(origin_world: Iterable[float], local_pose: Iterable[float]) -> np.ndarray:
    return matrix_to_se2(se2_to_matrix(origin_world) @ se2_to_matrix(local_pose))


def joint1_towards_cube(base_pose: Iterable[float], cube_world: Iterable[float]) -> float:
    bx, by, byaw = (float(v) for v in base_pose)
    cube = np.asarray(cube_world, dtype=np.float64)
    return wrap_angle(math.atan2(cube[1] - by, cube[0] - bx) - byaw)


def angular_error(actual: float, target: float) -> float:
    return wrap_angle(float(actual) - float(target))
