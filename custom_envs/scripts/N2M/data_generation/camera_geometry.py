"""Camera projection helpers with explicit MuJoCo/base frame conventions.

MuJoCo's camera frame uses local ``-Z`` as the viewing direction and local
``+Y`` as image up.  The public helpers below expose the usual optical frame:
``+X`` right, ``+Y`` down, ``+Z`` forward.

MuJoCo 相机参数 + 深度图
→ optical 相机坐标系点云
→ MuJoCo 相机坐标系
→ 世界坐标系
→ 机器人底盘坐标系
→ 裁剪和体素降采样

MuJoCo camera:
    +X 向右
    +Y 向上
    -Z 向前

Optical camera:
    +X 向右
    +Y 向下
    +Z 向前
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:   # 内参
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_vertical_fov(
        cls, width: int, height: int, fovy_degrees: float
    ) -> "CameraIntrinsics":   # 计算焦距，返回相机内参
        fy = 0.5 * float(height) / math.tan(math.radians(fovy_degrees) / 2.0)
        return cls(
            width=int(width),
            height=int(height),
            fx=fy,
            fy=fy,
            cx=(float(width) - 1.0) / 2.0,
            cy=(float(height) - 1.0) / 2.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def intrinsics_from_mujoco(model, camera_name: str, width: int, height: int):   # 从 MuJoCo 模型中读取指定相机的 FOV，并生成 CameraIntrinsics
    """Read a fixed-fovy MuJoCo camera and return pinhole intrinsics."""
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera not found: {camera_name}")
    return CameraIntrinsics.from_vertical_fov(
        width=width,
        height=height,
        fovy_degrees=float(model.cam_fovy[camera_id]),
    )


def depth_to_optical_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:   # 从深度图生成点云
    """Unproject metric image-plane depth into an optical XYZ point cloud.

    Returns ``(points, flat_pixel_indices)`` so colors or segmentation can be
    gathered using exactly the same validity mask.
    """
    depth = np.asarray(depth, dtype=np.float64)
    expected = (intrinsics.height, intrinsics.width)
    if depth.shape != expected:
        raise ValueError(f"depth shape {depth.shape} does not match {expected}")

    valid = np.isfinite(depth) & (depth > 0.0)   # 有效深度mask
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != expected:
            raise ValueError(f"valid_mask shape {mask.shape} does not match {expected}")
        valid &= mask

    rows, cols = np.nonzero(valid)   # 有效像素坐标
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    points = np.stack((x, y, z), axis=1)   # 点云
    flat_indices = rows * intrinsics.width + cols   # 每个点来自深度图中的哪个像素
    return points, flat_indices.astype(np.int64)


def optical_to_mujoco_camera(points_optical: np.ndarray) -> np.ndarray:
    """Convert optical ``right/down/forward`` to MuJoCo camera local XYZ.

    optical X → MuJoCo X
    optical Y → MuJoCo -Y
    optical Z → MuJoCo -Z"""
    points = np.asarray(points_optical, dtype=np.float64).reshape(-1, 3)
    return np.column_stack((points[:, 0], -points[:, 1], -points[:, 2]))


def camera_points_to_frame(
    points_optical: np.ndarray,
    camera_position_world: Iterable[float],
    camera_rotation_world: np.ndarray,
    frame_position_world: Iterable[float],
    frame_rotation_world: np.ndarray,
) -> np.ndarray:
    """Transform optical points through world into an arbitrary target frame.
    
    optical相机系
    → MuJoCo相机局部系
    → world
    → 任意目标frame
    
    注意这里最后两行的坐标变换，学习！
    """
    points_camera = optical_to_mujoco_camera(points_optical)
    camera_rotation_world = np.asarray(camera_rotation_world, dtype=np.float64).reshape(3, 3)
    camera_position_world = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
    frame_rotation_world = np.asarray(frame_rotation_world, dtype=np.float64).reshape(3, 3)  # 目标frame一般是base_link
    frame_position_world = np.asarray(frame_position_world, dtype=np.float64).reshape(3)
    points_world = points_camera @ camera_rotation_world.T + camera_position_world   # 相机点转换到world系
    return (points_world - frame_position_world) @ frame_rotation_world


def voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:   # 执行降采样
    """Deterministically keep the first point/color in every occupied voxel."""
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("xyz and rgb must both have shape (N, 3)")
    first = voxel_downsample_indices(xyz, voxel_size)
    return xyz[first], rgb[first]


def voxel_downsample_indices(xyz: np.ndarray, voxel_size: float) -> np.ndarray:   # 把空间划分成固定大小的小立方体，每个体素只保留一个点
    """Return the deterministic representative index of every occupied voxel.

    Keeping the indices separate lets callers propagate semantic quality-control
    flags (for example, whether a point belongs to the cube) through the exact
    same downsampling operation without storing segmentation in the training
    cloud.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if len(xyz) == 0 or voxel_size <= 0.0:
        return np.arange(len(xyz), dtype=np.int64)
    keys = np.floor(xyz / float(voxel_size)).astype(np.int64)   # 计算各点都在哪个体素
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first).astype(np.int64, copy=False)


def crop_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    minimum_xyz: Iterable[float],
    maximum_xyz: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = crop_point_mask(xyz, minimum_xyz, maximum_xyz)
    return xyz[mask], rgb[mask]


def crop_point_mask(
    xyz: np.ndarray,
    minimum_xyz: Iterable[float],
    maximum_xyz: Iterable[float],
) -> np.ndarray:   # 保留指定三维长方体范围内的点
    """Return the mask used for an axis-aligned point-cloud crop."""
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    minimum = np.asarray(minimum_xyz, dtype=np.float64).reshape(3)
    maximum = np.asarray(maximum_xyz, dtype=np.float64).reshape(3)
    if np.any(minimum >= maximum):
        raise ValueError("every crop minimum must be smaller than its maximum")
    return np.all((xyz >= minimum) & (xyz <= maximum), axis=1)
