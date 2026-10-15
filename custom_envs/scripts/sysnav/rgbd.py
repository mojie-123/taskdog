"""MuJoCo RGB-D capture and mask-to-world localization."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from custom_envs.scripts.N2M.data_generation.camera_geometry import (
    CameraIntrinsics,
    camera_points_to_frame,
    depth_to_optical_points,
    intrinsics_from_mujoco,
)
from custom_envs.scripts.map.room_map_io import RoomMap

from .config import PerceptionConfig
from .types import Detection2D, LocalizedDetection


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray = field(repr=False, compare=False)
    depth: np.ndarray = field(repr=False, compare=False)
    intrinsics: CameraIntrinsics
    camera_position_world: np.ndarray = field(repr=False, compare=False)
    camera_rotation_world: np.ndarray = field(repr=False, compare=False)
    camera_name: str
    timestamp_s: float

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        if rgb.shape != (*depth.shape, 3):
            raise ValueError("RGB and depth shapes disagree")
        if depth.shape != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError("frame shape disagrees with camera intrinsics")
        object.__setattr__(self, "rgb", rgb.astype(np.uint8, copy=False))
        object.__setattr__(self, "depth", depth.astype(np.float32, copy=False))
        object.__setattr__(
            self,
            "camera_position_world",
            np.asarray(self.camera_position_world, dtype=np.float64).reshape(3),
        )
        object.__setattr__(
            self,
            "camera_rotation_world",
            np.asarray(self.camera_rotation_world, dtype=np.float64).reshape(3, 3),
        )


def capture_rgbd_frame(env, camera_name: str = "front_realsense") -> RGBDFrame:
    """Capture public RGB/depth plus the known camera calibration and pose."""

    import mujoco

    camera_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera_name)
    )
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera not found: {camera_name}")
    rgb, depth = env.render_camera(camera_name)
    intrinsics = intrinsics_from_mujoco(
        env.model, camera_name, int(rgb.shape[1]), int(rgb.shape[0])
    )
    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        camera_position_world=env.data.cam_xpos[camera_id].copy(),
        camera_rotation_world=env.data.cam_xmat[camera_id].copy(),
        camera_name=str(camera_name),
        timestamp_s=time.time(),
    )


class DepthLocalizer:
    """Project SAM2 masks through metric depth into the map/world frame."""

    def __init__(self, config: PerceptionConfig, room_map: RoomMap):
        self.config = config
        self.room_map = room_map

    def localize(
        self, frame: RGBDFrame, detection: Detection2D
    ) -> LocalizedDetection | None:
        if detection.mask.shape != frame.depth.shape:
            raise ValueError("detection mask and RGB-D frame shapes disagree")
        valid = (
            detection.mask
            & np.isfinite(frame.depth)
            & (frame.depth > 0.0)
            & (frame.depth <= self.config.max_depth_m)
        )
        optical, _ = depth_to_optical_points(frame.depth, frame.intrinsics, valid)
        if len(optical) < self.config.min_localization_points:
            return None
        world = camera_points_to_frame(
            optical,
            frame.camera_position_world,
            frame.camera_rotation_world,
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )
        # SAM masks occasionally include a thin background rim.  A radial MAD
        # filter retains the compact object surface without assuming its shape.
        median = np.median(world, axis=0)
        radial = np.linalg.norm(world - median, axis=1)
        radial_median = float(np.median(radial))
        mad = float(np.median(np.abs(radial - radial_median)))
        threshold = radial_median + max(0.03, 4.0 * 1.4826 * mad)
        filtered = world[radial <= threshold]
        if len(filtered) < self.config.min_localization_points:
            filtered = world
        position = np.median(filtered, axis=0)
        minimum = np.percentile(filtered, 2.0, axis=0)
        maximum = np.percentile(filtered, 98.0, axis=0)
        room_id = self.room_map.room_id_at_world(position[0], position[1])
        return LocalizedDetection(
            detection=detection,
            position_world=position,
            bbox_min_world=minimum,
            bbox_max_world=maximum,
            point_count=len(filtered),
            room_id=room_id,
        )

    def localize_all(
        self, frame: RGBDFrame, detections: list[Detection2D]
    ) -> list[LocalizedDetection]:
        result = []
        for detection in detections:
            localized = self.localize(frame, detection)
            if localized is not None:
                result.append(localized)
        return result
