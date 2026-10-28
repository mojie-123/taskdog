"""Lighting-independent target/anchor point clouds from MuJoCo segmentation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from custom_envs.scripts.FloAff.simulation.camera_geometry import (
    CameraIntrinsics,
    camera_points_to_frame,
    depth_to_optical_points,
    voxel_downsample_indices,
)
from custom_envs.scripts.FloAff.simulation.collision_checks import body_subtree_ids


def geom_ids_for_body_subtree(model, root_body_name: str) -> np.ndarray:
    bodies = body_subtree_ids(model, root_body_name)
    return np.flatnonzero(np.isin(np.asarray(model.geom_bodyid), list(bodies))).astype(np.int32)


def segmentation_mask(segmentation: np.ndarray, geom_ids: Iterable[int]) -> np.ndarray:
    import mujoco

    ids = np.asarray(tuple(geom_ids), dtype=np.int32)
    return np.isin(segmentation[..., 0], ids) & (
        segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )


def render_rgb_depth_segmentation(env, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    renderer = mujoco.Renderer(env.model, height=env._cam_h, width=env._cam_w)
    try:
        renderer.update_scene(env.data, camera=camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    if segmentation.shape != (*depth.shape, 2):
        raise RuntimeError(f"unexpected segmentation shape: {segmentation.shape}")
    return rgb, depth, segmentation


def camera_offset_in_base(env, camera_name: str, base_body_name: str) -> np.ndarray:
    import mujoco

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if camera_id < 0 or base_id < 0:
        raise ValueError("camera or base body not found")
    return (
        env.data.cam_xpos[camera_id] - env.data.xpos[base_id]
    ) @ env.data.xmat[base_id].reshape(3, 3)


def _masked_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth_m: float,
) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth_m))
    points, _ = depth_to_optical_points(depth, intrinsics, valid)
    return points


def extract_semantic_clouds(
    env,
    *,
    camera_name: str,
    base_body_name: str,
    target_body_name: str,
    anchor_body_name: str,
    intrinsics: CameraIntrinsics,
    max_depth_m: float,
    voxel_size_m: float,
    anchor_max_behind_target_m: float,
) -> dict:
    import mujoco

    rgb, depth, segmentation = render_rgb_depth_segmentation(env, camera_name)
    target_mask = segmentation_mask(
        segmentation, geom_ids_for_body_subtree(env.model, target_body_name)
    )
    anchor_mask = segmentation_mask(
        segmentation, geom_ids_for_body_subtree(env.model, anchor_body_name)
    )
    target_optical = _masked_points(depth, target_mask, intrinsics, max_depth_m)
    anchor_optical = _masked_points(depth, anchor_mask, intrinsics, max_depth_m)

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if min(camera_id, base_id, target_id) < 0:
        raise ValueError("camera/base/target not found")

    camera_position = env.data.cam_xpos[camera_id].astype(np.float64).copy()
    target_position = env.data.xpos[target_id].astype(np.float64).copy()
    target_camera_distance = float(np.linalg.norm(target_position - camera_position))
    if len(anchor_optical):
        anchor_ranges = np.linalg.norm(anchor_optical, axis=1)
        anchor_optical = anchor_optical[
            anchor_ranges <= target_camera_distance + float(anchor_max_behind_target_m)
        ]

    transform_args = (
        camera_position,
        env.data.cam_xmat[camera_id],
        env.data.xpos[base_id],
        env.data.xmat[base_id],
    )
    target_base = camera_points_to_frame(target_optical, *transform_args)
    anchor_base = camera_points_to_frame(anchor_optical, *transform_args)
    target_base = target_base[voxel_downsample_indices(target_base, voxel_size_m)]
    anchor_base = anchor_base[voxel_downsample_indices(anchor_base, voxel_size_m)]
    target_center_base = (
        target_position - env.data.xpos[base_id]
    ) @ env.data.xmat[base_id].reshape(3, 3)
    return {
        "rgb": rgb,
        "depth": depth,
        "target_mask": target_mask,
        "anchor_mask": anchor_mask,
        "target_xyz_base": target_base.astype(np.float32),
        "anchor_xyz_base": anchor_base.astype(np.float32),
        "target_center_base": target_center_base.astype(np.float32),
        "camera_position_world": camera_position,
        "camera_rotation_world": env.data.cam_xmat[camera_id].reshape(3, 3).copy(),
        "target_camera_distance_m": target_camera_distance,
    }
