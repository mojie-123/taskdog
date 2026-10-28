"""Safe close-up viewpoint selection for provisional SysNav targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from custom_envs.scripts.map.nav2_map_io import Nav2Map
from custom_envs.scripts.map.room_map_io import RoomMap


def _wrap_angle(value: np.ndarray | float):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TargetReobservationPose:
    """One collision-free base pose aimed at a provisional object."""

    pose: tuple[float, float, float]
    grid_cell: tuple[int, int]
    target_distance_m: float
    travel_distance_m: float
    bearing_change_deg: float


def plan_target_reobservation_pose(
    room_map: RoomMap,
    nav_map: Nav2Map,
    room_id: str,
    target_position_world: np.ndarray,
    current_pose_world: np.ndarray,
    *,
    preferred_standoff_m: float,
    min_standoff_m: float,
    max_standoff_m: float,
) -> TargetReobservationPose | None:
    """Choose a safe nearby cell, preferring the side that saw the object.

    The room-map searchable mask already includes the configured clearance
    from obstacles.  Intersecting it with Nav2 free space therefore filters
    both the robot footprint and static obstacles.  We intentionally do not
    apply a 2-D line-of-sight test: an object on a tabletop lies inside the
    table's occupied footprint even though the elevated front camera can see
    it perfectly well.
    """

    preferred = float(preferred_standoff_m)
    minimum = float(min_standoff_m)
    maximum = float(max_standoff_m)
    if not 0.0 < minimum <= preferred <= maximum:
        raise ValueError(
            "reobservation standoff must satisfy 0 < min <= preferred <= max"
        )
    if room_map.room_region_id_map.shape != nav_map.free_mask.shape:
        raise ValueError("room and Nav2 rasters have different shapes")

    target = np.asarray(target_position_world, dtype=np.float64).reshape(-1)
    current = np.asarray(current_pose_world, dtype=np.float64).reshape(-1)
    if target.size < 2 or current.size < 2:
        raise ValueError("target and current pose must contain x and y")

    safe = room_map.searchable_mask(room_id) & nav_map.free_mask
    rows, cols = np.nonzero(safe)
    if len(rows) == 0:
        return None

    x = nav_map.origin[0] + (cols.astype(np.float64) + 0.5) * nav_map.resolution
    y = (
        nav_map.origin[1]
        + (nav_map.height - 1 - rows.astype(np.float64) + 0.5)
        * nav_map.resolution
    )
    candidate_xy = np.column_stack((x, y))
    target_xy = target[:2]
    current_xy = current[:2]
    target_delta = candidate_xy - target_xy
    target_distance = np.linalg.norm(target_delta, axis=1)
    valid = (target_distance >= minimum) & (target_distance <= maximum)
    if not bool(valid.any()):
        return None

    rows = rows[valid]
    cols = cols[valid]
    candidate_xy = candidate_xy[valid]
    target_delta = target_delta[valid]
    target_distance = target_distance[valid]
    travel_distance = np.linalg.norm(candidate_xy - current_xy, axis=1)

    # Keep the successful line of sight whenever possible.  This also avoids
    # walking around a table simply because another annulus cell is a few
    # centimetres closer to the nominal standoff distance.
    current_from_target = current_xy - target_xy
    if float(np.linalg.norm(current_from_target)) > 1e-6:
        current_bearing = math.atan2(
            float(current_from_target[1]), float(current_from_target[0])
        )
        candidate_bearing = np.arctan2(target_delta[:, 1], target_delta[:, 0])
        bearing_change = np.abs(_wrap_angle(candidate_bearing - current_bearing))
    else:
        bearing_change = np.zeros(len(candidate_xy), dtype=np.float64)

    # Standoff accuracy dominates.  Travel and side changes are only
    # tie-breakers among similarly useful close-up viewpoints.
    score = (
        4.0 * np.abs(target_distance - preferred)
        + 0.20 * travel_distance
        + 0.35 * bearing_change
    )
    index = int(np.argmin(score))
    selected = candidate_xy[index]
    yaw = math.atan2(
        float(target_xy[1] - selected[1]),
        float(target_xy[0] - selected[0]),
    )
    return TargetReobservationPose(
        pose=(float(selected[0]), float(selected[1]), float(yaw)),
        grid_cell=(int(rows[index]), int(cols[index])),
        target_distance_m=float(target_distance[index]),
        travel_distance_m=float(travel_distance[index]),
        bearing_change_deg=math.degrees(float(bearing_change[index])),
    )
