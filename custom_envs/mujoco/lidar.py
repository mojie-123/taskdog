"""MuJoCo ray-cast implementation of the project's Mid-360 LiDAR.

The default scan pattern mirrors ``get_mid360_lidar_pattern()``: 40 vertical
channels, a 59 degree vertical field of view, and 1 degree horizontal
resolution (14,400 rays total).  Results are returned in world coordinates so
both simulators can share OccupancyGrid.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np


@dataclass(frozen=True)
class LidarScan:
    """One world-frame LiDAR scan."""

    origin_w: np.ndarray
    directions_w: np.ndarray
    ranges: np.ndarray
    hits_w: np.ndarray
    geom_ids: np.ndarray
    valid_mask: np.ndarray


class MuJoCoMid360:
    """Batched MuJoCo ray caster matching the IsaacLab Mid-360 setup."""

    def __init__(
        self,
        model: mujoco.MjModel,
        base_body_name: str = "base_link",
        offset: Sequence[float] = (0.30, 0.0, 0.55),
        channels: int = 16,    # 40
        vertical_fov_deg: Sequence[float] = (-7.0, 52.0),
        horizontal_resolution_deg: float = 2.0,    # 1.0
        max_distance: float = 70.0,
        geomgroup: Optional[Sequence[int]] = None,
        robot_raycast_group: int = 4,
    ) -> None:
        self.model = model
        self.offset = np.asarray(offset, dtype=np.float64)
        if self.offset.shape != (3,):
            raise ValueError(f"offset must have shape (3,), got {self.offset.shape}")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if horizontal_resolution_deg <= 0.0:
            raise ValueError("horizontal_resolution_deg must be positive")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")

        self.channels = int(channels)
        self.vertical_fov_deg = tuple(map(float, vertical_fov_deg))
        if len(self.vertical_fov_deg) != 2:
            raise ValueError("vertical_fov_deg must contain (min, max)")
        self.horizontal_resolution_deg = float(horizontal_resolution_deg)
        self.max_distance = float(max_distance)

        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if self.base_body_id < 0:
            raise ValueError(f"MuJoCo body {base_body_name!r} was not found")

        self.directions_b = self._make_pattern()
        self.num_rays = int(self.directions_b.shape[0])
        self._geom_ids = np.full(self.num_rays, -1, dtype=np.int32)
        self._ranges = np.full(self.num_rays, -1.0, dtype=np.float64)

        # MuJoCo's bodyexclude excludes only one body, not the articulated
        # descendants.  Put every collidable geom in the robot subtree into a
        # ray-only group.  geom_group affects visualization/raycast filtering,
        # not contact dynamics.  Visual robot geoms stay in group 1.
        self.robot_geom_ids = self._isolate_robot_collision_geoms(
            robot_raycast_group)

        if geomgroup is None:
            # Scene/table/object collision geoms use groups 0 and 3.  Robot
            # visuals use group 1 and robot collision geoms were moved to 4.
            geomgroup = (1, 0, 0, 1, 0, 0)
        self.geomgroup = np.asarray(geomgroup, dtype=np.uint8)
        if self.geomgroup.shape != (6,):
            raise ValueError(
                f"geomgroup must contain six entries, got {self.geomgroup.shape}")

    def _make_pattern(self) -> np.ndarray:
        elevation = np.deg2rad(np.linspace(
            self.vertical_fov_deg[0],
            self.vertical_fov_deg[1],
            self.channels,
            dtype=np.float64,
        ))
        azimuth = np.deg2rad(np.arange(
            -180.0,
            180.0,
            self.horizontal_resolution_deg,
            dtype=np.float64,
        ))
        elev_grid, az_grid = np.meshgrid(elevation, azimuth, indexing="ij")
        directions = np.stack(
            [
                np.cos(elev_grid) * np.cos(az_grid),
                np.cos(elev_grid) * np.sin(az_grid),
                np.sin(elev_grid),
            ],
            axis=-1,
        ).reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return np.ascontiguousarray(directions, dtype=np.float64)

    def _isolate_robot_collision_geoms(self, target_group: int) -> np.ndarray:
        if not 0 <= int(target_group) <= 5:
            raise ValueError("robot_raycast_group must be in [0, 5]")

        robot_body = np.zeros(self.model.nbody, dtype=bool)
        for body_id in range(self.model.nbody):
            ancestor = body_id
            while ancestor > 0 and ancestor != self.base_body_id:
                ancestor = int(self.model.body_parentid[ancestor])
            robot_body[body_id] = ancestor == self.base_body_id

        body_ids = self.model.geom_bodyid.astype(np.int32, copy=False)
        collidable = ((self.model.geom_contype != 0)
                      | (self.model.geom_conaffinity != 0))
        geom_ids = np.flatnonzero(robot_body[body_ids] & collidable)
        self.model.geom_group[geom_ids] = int(target_group)
        return geom_ids.astype(np.int32, copy=False)

    def sensor_pose_world(self, data: mujoco.MjData):
        """Return ``(position, rotation)`` of the LiDAR in world frame."""
        rotation = data.xmat[self.base_body_id].reshape(3, 3).copy()
        origin = (data.xpos[self.base_body_id]
                  + rotation @ self.offset).astype(np.float64, copy=False)
        return origin.copy(), rotation

    def scan(self, data: mujoco.MjData) -> LidarScan:
        """Cast one scan and return finite hits in world coordinates."""
        origin_w, rotation_wb = self.sensor_pose_world(data)
        directions_w = np.ascontiguousarray(
            self.directions_b @ rotation_wb.T,
            dtype=np.float64,
        )
        self._geom_ids.fill(-1)
        self._ranges.fill(-1.0)
        mujoco.mj_multiRay(
            m=self.model,
            d=data,
            pnt=origin_w,
            vec=directions_w.ravel(),
            geomgroup=self.geomgroup,
            flg_static=1,
            bodyexclude=-1,
            geomid=self._geom_ids,
            dist=self._ranges,
            normal=None,
            nray=self.num_rays,
            cutoff=self.max_distance,
        )
        valid = ((self._ranges >= 0.0)
                 & np.isfinite(self._ranges)
                 & (self._ranges <= self.max_distance))
        hits_w = (origin_w
                  + directions_w[valid] * self._ranges[valid, None])
        return LidarScan(
            origin_w=origin_w,
            directions_w=directions_w,
            ranges=self._ranges.copy(),
            hits_w=np.ascontiguousarray(hits_w),
            geom_ids=self._geom_ids.copy(),
            valid_mask=valid.copy(),
        )
