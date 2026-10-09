"""Replaceable target-position providers for online N2M navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from custom_envs.scripts.N2M.geometry import wrap_angle


@dataclass(frozen=True)
class TargetEstimate:
    """One target estimate in the MuJoCo/Nav2 world frame."""

    position_world: np.ndarray
    quaternion_world_wxyz: np.ndarray
    source: str

    def pose_wxyz(self) -> list[float]:
        return np.concatenate(
            (self.position_world, self.quaternion_world_wxyz)
        ).astype(float).tolist()


class TargetProvider(Protocol):
    def estimate(self, env, object_name: str) -> TargetEstimate:
        """Return the latest target estimate in world coordinates."""


class MuJoCoTruthTargetProvider:
    """Read a named MuJoCo body pose.

    This is the deliberately privileged provider used by the first integrated
    simulation.  A detector/tracker can replace it later without changing the
    navigation state machine.
    """

    def estimate(self, env, object_name: str = "cube") -> TargetEstimate:
        import mujoco

        body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, str(object_name)
        )
        if body_id < 0:
            raise ValueError(f"target body not found: {object_name}")
        return TargetEstimate(
            position_world=env.data.xpos[body_id].astype(np.float64).copy(),
            quaternion_world_wxyz=env.data.xquat[body_id].astype(np.float64).copy(),
            source="mujoco_truth",
        )


@dataclass(frozen=True)
class NavigationTargetMetrics:
    horizontal_distance_m: float
    bearing_error_rad: float


def navigation_target_metrics(
    robot_pose_world: np.ndarray | list[float],
    target_position_world: np.ndarray | list[float],
) -> NavigationTargetMetrics:
    """Return planar target distance and target bearing in the base frame."""
    robot = np.asarray(robot_pose_world, dtype=np.float64).reshape(-1)
    target = np.asarray(target_position_world, dtype=np.float64).reshape(-1)
    if robot.size < 3 or target.size < 2:
        raise ValueError("robot pose must be [x,y,yaw] and target must contain x,y")
    delta = target[:2] - robot[:2]
    distance = float(np.linalg.norm(delta))
    bearing_world = math.atan2(float(delta[1]), float(delta[0]))
    return NavigationTargetMetrics(
        horizontal_distance_m=distance,
        bearing_error_rad=wrap_angle(bearing_world - float(robot[2])),
    )


def standoff_goal(
    robot_xy: np.ndarray | list[float],
    target_xy: np.ndarray | list[float],
    distance_m: float,
) -> np.ndarray:
    """Project the on-table target to a floor-reachable observation waypoint."""
    if float(distance_m) <= 0.0:
        raise ValueError("standoff distance must be positive")
    robot = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    toward = target - robot
    norm = float(np.linalg.norm(toward))
    if norm < 1e-9:
        raise ValueError("cannot infer an approach direction at the target position")
    return target - toward / norm * float(distance_m)


def observation_gate(
    metrics: NavigationTargetMetrics,
    *,
    min_distance_m: float,
    max_distance_m: float,
    max_bearing_rad: float,
) -> bool:
    """Check whether a stopped base pose lies in the online O capture band."""
    return bool(
        float(min_distance_m)
        <= metrics.horizontal_distance_m
        <= float(max_distance_m)
        and abs(metrics.bearing_error_rad) <= float(max_bearing_rad)
    )

