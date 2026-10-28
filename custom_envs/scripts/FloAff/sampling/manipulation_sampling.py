"""Continuous manipulation-pose (M) proposals."""

from __future__ import annotations

import numpy as np

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.FloAff.data_generation.common import (
    manipulation_count_for_placement,
)
from custom_envs.scripts.FloAff.geometry import (
    base_yaw_parallel_to_edge,
    edge_point_world,
    joint1_towards_target,
)
from custom_envs.scripts.FloAff.schemas import InstanceRecord, ManipulationCandidate
from custom_envs.scripts.FloAff.sampling.common import latin_hypercube, scale_unit


def sample_manipulation_candidates(
    instance: InstanceRecord,
    config: DataGenerationConfig,
    seed: int,
) -> list[ManipulationCandidate]:
    count = manipulation_count_for_placement(config, instance.placement_id)
    rng = np.random.default_rng(int(seed))
    unit = latin_hypercube(count, 2, rng)
    outward_values = scale_unit(
        unit[:, 0],
        config.manipulation.outward_min_m,
        config.manipulation.outward_max_m,
    )
    tangent_values = scale_unit(
        unit[:, 1],
        config.manipulation.tangent_min_m,
        config.manipulation.tangent_max_m,
    )
    target_xy = instance.target_pose_world[:2]
    base_yaw = base_yaw_parallel_to_edge(instance.outward_xy_world)
    rows: list[ManipulationCandidate] = []
    for index, (outward_m, tangent_m) in enumerate(zip(outward_values, tangent_values)):
        base_xy = edge_point_world(
            target_xy,
            instance.outward_xy_world,
            float(outward_m),
            float(tangent_m),
        )
        base_pose = (float(base_xy[0]), float(base_xy[1]), float(base_yaw))
        rows.append(
            ManipulationCandidate(
                candidate_id=f"{instance.instance_id}__m{index:04d}",
                instance_id=instance.instance_id,
                placement_id=instance.placement_id,
                scene_id=instance.instance_id,
                category=instance.category,
                approach_edge=instance.approach_edge,
                cube_pose=instance.target_pose_world,
                base_pose=base_pose,
                joint1_target=joint1_towards_target(base_pose, target_xy),
                proposal={
                    "outward_m": float(outward_m),
                    "tangent_m": float(tangent_m),
                    "base_yaw_rad": float(base_yaw),
                },
            )
        )
    return rows
