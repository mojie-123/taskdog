"""Camera-centric observation-pose (O) proposals with fixed strata."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.FloAff.data_generation.common import (
    observation_count_for_placement,
)
from custom_envs.scripts.FloAff.geometry import (
    base_pose_for_camera_pose,
    camera_yaw_towards_target,
    corner_family_is_exclusive,
    observation_camera_xy,
)
from custom_envs.scripts.FloAff.schemas import InstanceRecord, ObservationProposal
from custom_envs.scripts.FloAff.sampling.common import latin_hypercube, scale_unit
from custom_envs.scripts.FloAff.sampling.placement_sampling import EDGE_OUTWARDS


def observation_quotas(instance: InstanceRecord, config: DataGenerationConfig) -> dict[str, int]:
    profile = config.observations.profile(instance.observation_profile)
    return profile.quotas(
        observation_count_for_placement(config, instance.placement_id)
    )


def propose_observation_poses(
    instance: InstanceRecord,
    config: DataGenerationConfig,
    camera_xy_in_base: Iterable[float],
    seed: int,
    *,
    strata: Iterable[str] = ("core", "negative_tail", "positive_tail"),
    batch_index: int = 0,
) -> list[ObservationProposal]:
    """Build a deterministic proposal pool for one or more strata.

    Batch zero preserves the original proposal IDs.  Positive batch indices
    create disjoint IDs so a resumed capture can append supplemental samples
    without changing or retrying the original pool.
    """
    profile = config.observations.profile(instance.observation_profile)
    quotas = observation_quotas(instance, config)
    rng = np.random.default_rng(int(seed))
    target_xy = instance.target_pose_world[:2]
    rows: list[ObservationProposal] = []
    selected_strata = tuple(str(value) for value in strata)
    unknown = set(selected_strata) - set(quotas)
    if unknown:
        raise KeyError(f"unknown observation strata: {sorted(unknown)}")
    if int(batch_index) < 0:
        raise ValueError("batch_index must be non-negative")
    batch_tag = "" if int(batch_index) == 0 else f"_b{int(batch_index):02d}"
    for stratum in selected_strata:
        desired = quotas[stratum]
        pool_size = max(desired, desired * int(config.observations.proposal_multiplier))
        unit = latin_hypercube(pool_size, 3, rng)
        alpha_min, alpha_max = profile.bounds(stratum)
        ranges = scale_unit(
            unit[:, 0],
            config.observations.camera_range_min_m,
            config.observations.camera_range_max_m,
        )
        alphas = scale_unit(unit[:, 1], alpha_min, alpha_max)
        yaw_offsets = scale_unit(
            unit[:, 2],
            config.observations.yaw_offset_min_deg,
            config.observations.yaw_offset_max_deg,
        )
        for index, (camera_range, alpha_deg, yaw_offset_deg) in enumerate(
            zip(ranges, alphas, yaw_offsets)
        ):
            camera_xy = observation_camera_xy(
                target_xy,
                instance.outward_xy_world,
                float(camera_range),
                math.radians(float(alpha_deg)),
            )
            if instance.category == "corner":
                other_edge = instance.corner_other_edge
                if other_edge is None:
                    raise ValueError(f"corner instance lacks sibling edge: {instance.instance_id}")
                if not corner_family_is_exclusive(
                    camera_xy,
                    target_xy,
                    instance.outward_xy_world,
                    EDGE_OUTWARDS[other_edge],
                ):
                    continue
            camera_yaw = camera_yaw_towards_target(
                camera_xy,
                target_xy,
                math.radians(float(yaw_offset_deg)),
            )
            base_pose = base_pose_for_camera_pose(
                camera_xy,
                camera_yaw,
                camera_xy_in_base,
            )
            rows.append(
                ObservationProposal(
                    proposal_id=(
                        f"{instance.instance_id}__o_{stratum}{batch_tag}_{index:05d}"
                    ),
                    instance_id=instance.instance_id,
                    placement_id=instance.placement_id,
                    approach_edge=instance.approach_edge,
                    stratum=stratum,
                    camera_range_m=float(camera_range),
                    alpha_deg=float(alpha_deg),
                    yaw_offset_deg=float(yaw_offset_deg),
                    desired_camera_pose_world=(
                        float(camera_xy[0]),
                        float(camera_xy[1]),
                        float(camera_yaw),
                    ),
                    base_pose_world=(
                        float(base_pose[0]),
                        float(base_pose[1]),
                        float(base_pose[2]),
                    ),
                )
            )
    rng.shuffle(rows)
    # Preserve random order within each stratum while making stratum iteration stable.
    order = {"core": 0, "negative_tail": 1, "positive_tail": 2}
    rows.sort(key=lambda item: order[item.stratum])
    return rows
