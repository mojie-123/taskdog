"""Capture one deployment-time N2M observation from ``front_realsense``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from custom_envs.scripts.N2M.data_generation.camera_geometry import (
    camera_points_to_frame,
    crop_point_mask,
    depth_to_optical_points,
    intrinsics_from_mujoco,
    voxel_downsample_indices,
)
from custom_envs.scripts.N2M.data_generation.capture_observations import (
    _geom_ids_for_bodies,
    _render_rgb_depth_segmentation,
    _segmentation_mask,
    _subtree_body_ids,
)


class LiveObservationError(RuntimeError):
    def __init__(self, reason: str, **metrics):
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics


@dataclass(frozen=True)
class LiveObservationConfig:
    camera_name: str = "front_realsense"
    target_body: str = "cube"
    min_target_pixels: int = 150
    min_target_points: int = 15
    minimum_point_count: int = 8192
    max_camera_distance_m: float = 1.7
    max_depth_m: float = 3.0
    voxel_size_m: float = 0.01
    crop_min: tuple[float, float, float] = (-1.0, -2.0, -1.0)
    crop_max: tuple[float, float, float] = (2.5, 2.0, 1.5)


@dataclass
class LiveObservation:
    xyz_base: np.ndarray
    rgb: np.ndarray
    observation_pose_world: list[float]
    target_pose_world: list[float]
    target_centroid_base: np.ndarray
    metrics: dict
    image_rgb: np.ndarray = field(repr=False)
    image_depth: np.ndarray = field(repr=False)
    target_mask: np.ndarray = field(repr=False)


def _yaw_from_quaternion_wxyz(quaternion: Iterable[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def capture_live_observation(
    env,
    config: LiveObservationConfig | None = None,
) -> LiveObservation:
    """Apply the training collector's geometry and screening to the live scene."""
    import mujoco

    cfg = config or LiveObservationConfig()
    model, data = env.model, env.data
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name
    )
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, cfg.target_body
    )
    if min(camera_id, base_id, target_id) < 0:
        raise LiveObservationError(
            "missing_camera_base_or_target",
            camera=cfg.camera_name,
            target=cfg.target_body,
        )

    rgb, depth, segmentation = _render_rgb_depth_segmentation(env, cfg.camera_name)
    height, width = depth.shape
    intrinsics = intrinsics_from_mujoco(model, cfg.camera_name, width, height)
    robot_geoms = _geom_ids_for_bodies(model, _subtree_body_ids(model, "base_link"))
    target_geoms = _geom_ids_for_bodies(
        model, _subtree_body_ids(model, cfg.target_body)
    )
    robot_mask = _segmentation_mask(segmentation, robot_geoms)
    target_mask = _segmentation_mask(segmentation, target_geoms)

    camera_position = data.cam_xpos[camera_id].astype(np.float64).copy()
    target_position = data.xpos[target_id].astype(np.float64).copy()
    target_quaternion = data.xquat[target_id].astype(np.float64).copy()
    camera_distance = float(np.linalg.norm(target_position - camera_position))
    metrics = {
        "target_visible_pixels": int(target_mask.sum()),
        "target_camera_distance_m": camera_distance,
    }
    if camera_distance > float(cfg.max_camera_distance_m):
        raise LiveObservationError("target_too_far", **metrics)
    if metrics["target_visible_pixels"] < int(cfg.min_target_pixels):
        raise LiveObservationError("target_too_few_pixels", **metrics)

    valid = (
        (~robot_mask)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth <= float(cfg.max_depth_m))
    )
    optical_points, flat_indices = depth_to_optical_points(depth, intrinsics, valid)
    point_rgb = rgb.reshape(-1, 3)[flat_indices]
    target_flags = target_mask.reshape(-1)[flat_indices]
    xyz_base = camera_points_to_frame(
        optical_points,
        camera_position,
        data.cam_xmat[camera_id],
        data.xpos[base_id],
        data.xmat[base_id],
    )

    keep = crop_point_mask(xyz_base, cfg.crop_min, cfg.crop_max)
    xyz_base = xyz_base[keep]
    point_rgb = point_rgb[keep]
    target_flags = target_flags[keep]
    keep = voxel_downsample_indices(xyz_base, float(cfg.voxel_size_m))
    xyz_base = xyz_base[keep]
    point_rgb = point_rgb[keep]
    target_flags = target_flags[keep]

    metrics.update(
        point_count=int(len(xyz_base)),
        target_points_after_voxel=int(target_flags.sum()),
    )
    if len(xyz_base) < int(cfg.minimum_point_count):
        raise LiveObservationError("too_few_cloud_points", **metrics)
    if int(target_flags.sum()) < int(cfg.min_target_points):
        raise LiveObservationError("too_few_target_points_after_voxel", **metrics)

    target_points = xyz_base[target_flags]
    target_centroid_base = np.median(target_points, axis=0)
    base_position = data.xpos[base_id].astype(np.float64).copy()
    base_quaternion = data.xquat[base_id].astype(np.float64).copy()
    observation_pose = [
        float(base_position[0]),
        float(base_position[1]),
        _yaw_from_quaternion_wxyz(base_quaternion),
    ]
    return LiveObservation(
        xyz_base=xyz_base.astype(np.float32),
        rgb=point_rgb.astype(np.uint8),
        observation_pose_world=observation_pose,
        target_pose_world=np.concatenate(
            (target_position, target_quaternion)
        ).astype(float).tolist(),
        target_centroid_base=target_centroid_base.astype(np.float64),
        metrics=metrics,
        image_rgb=rgb,
        image_depth=depth,
        target_mask=target_mask,
    )

