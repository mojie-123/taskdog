"""Pinhole unprojection and MuJoCo frame conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_vertical_fov(
        cls, width: int, height: int, fovy_degrees: float
    ) -> "CameraIntrinsics":
        focal = 0.5 * float(height) / math.tan(math.radians(fovy_degrees) / 2.0)
        return cls(
            width=int(width),
            height=int(height),
            fx=focal,
            fy=focal,
            cx=(float(width) - 1.0) / 2.0,
            cy=(float(height) - 1.0) / 2.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def intrinsics_from_mujoco(model, camera_name: str, width: int, height: int) -> CameraIntrinsics:
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera not found: {camera_name}")
    return CameraIntrinsics.from_vertical_fov(
        width, height, float(model.cam_fovy[camera_id])
    )


def depth_to_optical_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    expected = (intrinsics.height, intrinsics.width)
    if depth.shape != expected:
        raise ValueError(f"depth shape {depth.shape} does not match {expected}")
    valid = np.isfinite(depth) & (depth > 0.0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != expected:
            raise ValueError(f"mask shape {mask.shape} does not match {expected}")
        valid &= mask
    rows, columns = np.nonzero(valid)
    z = depth[rows, columns]
    x = (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    return (
        np.stack([x, y, z], axis=1),
        (rows * intrinsics.width + columns).astype(np.int64),
    )


def optical_to_mujoco_camera(points_optical: np.ndarray) -> np.ndarray:
    points = np.asarray(points_optical, dtype=np.float64).reshape(-1, 3)
    return np.column_stack([points[:, 0], -points[:, 1], -points[:, 2]])


def camera_points_to_frame(
    points_optical: np.ndarray,
    camera_position_world: Iterable[float],
    camera_rotation_world: np.ndarray,
    frame_position_world: Iterable[float],
    frame_rotation_world: np.ndarray,
) -> np.ndarray:
    points_camera = optical_to_mujoco_camera(points_optical)
    camera_rotation = np.asarray(camera_rotation_world, dtype=np.float64).reshape(3, 3)
    camera_position = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
    frame_rotation = np.asarray(frame_rotation_world, dtype=np.float64).reshape(3, 3)
    frame_position = np.asarray(frame_position_world, dtype=np.float64).reshape(3)
    points_world = points_camera @ camera_rotation.T + camera_position
    return (points_world - frame_position) @ frame_rotation


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) == 0 or voxel_size <= 0.0:
        return np.arange(len(points), dtype=np.int64)
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first).astype(np.int64, copy=False)


def fixed_count_sample(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly ``count`` points plus a source-index vector.

    Sparse classes are sampled with replacement instead of zero padded, so no
    artificial origin points are introduced into the semantic cloud.
    """
    points = np.asarray(points)
    count = int(count)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if count <= 0:
        raise ValueError("count must be positive")
    if len(points) == 0:
        raise ValueError("cannot sample an empty point cloud")
    indices = rng.choice(len(points), size=count, replace=len(points) < count)
    return points[indices], indices.astype(np.int64)
