"""Build and verify observation-frame manipulation-pose labels."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from custom_envs.scripts.N2M.geometry import relative_se2, transform_se2, angular_error


def target_in_observation_frame(
    observation_pose_world: Iterable[float], manipulation_pose_world: Iterable[float]
) -> np.ndarray:
    return relative_se2(observation_pose_world, manipulation_pose_world).astype(np.float32)


def assert_target_round_trip(
    observation_pose_world: Iterable[float],
    manipulation_pose_world: Iterable[float],
    target_observation: Iterable[float],
    atol: float = 1e-6,
) -> None:
    expected = np.asarray(manipulation_pose_world, dtype=np.float64)
    rebuilt = transform_se2(observation_pose_world, target_observation)
    if not np.allclose(rebuilt[:2], expected[:2], atol=atol):
        raise ValueError(f"SE(2) label round trip failed: {rebuilt} != {expected}")
    if abs(angular_error(rebuilt[2], expected[2])) > atol:
        raise ValueError(f"SE(2) yaw round trip failed: {rebuilt[2]} != {expected[2]}")

