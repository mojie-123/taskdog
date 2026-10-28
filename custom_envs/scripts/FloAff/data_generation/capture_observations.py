#!/usr/bin/env python
"""Capture the configured number of valid camera-centric O samples."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
import os
from pathlib import Path

import numpy as np

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    load_instances,
    output_root,
    stable_seed,
)
from custom_envs.scripts.FloAff.geometry import (
    corner_family_is_exclusive,
    outward_world_to_base,
)
from custom_envs.scripts.FloAff.manifests import append_jsonl, read_jsonl
from custom_envs.scripts.FloAff.sampling.observation_sampling import (
    observation_quotas,
    propose_observation_poses,
)
from custom_envs.scripts.FloAff.sampling.placement_sampling import EDGE_OUTWARDS
from custom_envs.scripts.FloAff.simulation.camera_geometry import intrinsics_from_mujoco
from custom_envs.scripts.FloAff.simulation.collision_checks import (
    actual_camera_metrics,
    robot_environment_contacts,
)
from custom_envs.scripts.FloAff.simulation.mujoco_state import place_observation_state
from custom_envs.scripts.FloAff.simulation.mujoco_truth import (
    camera_offset_in_base,
    extract_semantic_clouds,
)


STRATA = ("core", "negative_tail", "positive_tail")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument(
        "--max-supplemental-batches",
        type=int,
        default=8,
        help=(
            "maximum deterministic extra proposal pools per deficient stratum "
            "after the original pool is exhausted (default: 8)"
        ),
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser


def _save_preview(path: Path, rgb, target_mask, anchor_mask) -> None:
    from PIL import Image

    image = np.asarray(rgb, dtype=np.float32).copy()
    target_color = np.array([255.0, 55.0, 55.0], dtype=np.float32)
    anchor_color = np.array([50.0, 170.0, 255.0], dtype=np.float32)
    image[target_mask] = 0.25 * image[target_mask] + 0.75 * target_color
    image[anchor_mask] = 0.45 * image[anchor_mask] + 0.55 * anchor_color
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def _actual_base_yaw(env, base_body_name: str) -> float:
    import mujoco

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if body_id < 0:
        raise ValueError(f"base body not found: {base_body_name}")
    rotation = env.data.xmat[body_id].reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _reject(
    path: Path,
    proposal,
    reason: str,
    metrics: dict,
    details: dict | None = None,
) -> None:
    append_jsonl(
        path,
        {
            "schema_version": 1,
            **proposal.to_dict(),
            "observation_id": proposal.proposal_id,
            "reason": reason,
            "metrics": metrics,
            "details": details or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _proposal_stream(
    instance,
    config,
    camera_xy_in_base,
    stratum: str,
    initial_proposals,
    max_supplemental_batches: int,
):
    """Yield the original pool followed by deterministic, disjoint retries."""
    yield from initial_proposals
    for batch_index in range(1, int(max_supplemental_batches) + 1):
        seed = stable_seed(
            config.seed,
            f"O:{instance.instance_id}:{stratum}:supplement:{batch_index}",
        )
        proposals = propose_observation_poses(
            instance,
            config,
            camera_xy_in_base,
            seed,
            strata=(stratum,),
            batch_index=batch_index,
        )
        print(
            f"[FloAff][O] {instance.instance_id}/{stratum} "
            f"supplemental_batch={batch_index}/{max_supplemental_batches} "
            f"pool={len(proposals)}",
            flush=True,
        )
        yield from proposals


def main() -> int:
    args = build_parser().parse_args()
    if args.max_supplemental_batches < 0:
        raise ValueError("--max-supplemental-batches must be non-negative")
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    instances = load_instances(root, args.instance)
    observation_dir = root / "observations"
    cloud_dir = observation_dir / "clouds"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = observation_dir / "manifest.jsonl"
    rejected_path = observation_dir / "rejected.jsonl"
    if not args.resume and (manifest_path.exists() or rejected_path.exists()):
        raise FileExistsError(
            f"{observation_dir} already contains capture results; pass --resume"
        )

    accepted_rows = read_jsonl(manifest_path) if args.resume else []
    accepted_ids = {row["observation_id"] for row in accepted_rows}
    accepted_counts = Counter(
        (row["instance_id"], row["stratum"]) for row in accepted_rows
    )
    rejected_ids = (
        set() if args.retry_rejected else {row["observation_id"] for row in read_jsonl(rejected_path)}
    )

    if not args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from custom_envs.mujoco.env import MuJoCoEnv

    env = MuJoCoEnv(
        camera_width=config.scene.camera_width,
        camera_height=config.scene.camera_height,
        render=args.render,
        scene_path=str(Path(config.scene.scene_xml).resolve()),
    )
    try:
        intrinsics = intrinsics_from_mujoco(
            env.model,
            config.scene.camera_name,
            config.scene.camera_width,
            config.scene.camera_height,
        )
        camera_offset = camera_offset_in_base(
            env, config.scene.camera_name, config.scene.robot_root_body
        )
        print(f"[FloAff][O] camera offset in base={np.round(camera_offset, 5).tolist()}")
        for instance in instances:
            quotas = observation_quotas(instance, config)
            proposals = propose_observation_poses(
                instance,
                config,
                camera_offset[:2],
                stable_seed(config.seed, f"O:{instance.instance_id}"),
            )
            by_stratum = {
                stratum: [row for row in proposals if row.stratum == stratum]
                for stratum in STRATA
            }
            profile = config.observations.profile(instance.observation_profile)
            for stratum in STRATA:
                target_count = quotas[stratum]
                print(
                    f"[FloAff][O] {instance.instance_id}/{stratum} "
                    f"accepted={accepted_counts[(instance.instance_id, stratum)]}/{target_count} "
                    f"pool={len(by_stratum[stratum])}",
                    flush=True,
                )
                for proposal in _proposal_stream(
                    instance,
                    config,
                    camera_offset[:2],
                    stratum,
                    by_stratum[stratum],
                    args.max_supplemental_batches,
                ):
                    if accepted_counts[(instance.instance_id, stratum)] >= target_count:
                        break
                    if proposal.proposal_id in accepted_ids or proposal.proposal_id in rejected_ids:
                        continue

                    metrics: dict = {}
                    place_observation_state(
                        env,
                        instance.target_pose_world,
                        proposal.base_pose_world,
                        target_freejoint=config.scene.target_freejoint,
                        robot_root_z=config.scene.robot_root_z,
                    )
                    contacts = robot_environment_contacts(
                        env, config.scene.robot_root_body
                    )
                    if contacts:
                        _reject(rejected_path, proposal, "collision", metrics, {"contacts": contacts})
                        rejected_ids.add(proposal.proposal_id)
                        continue

                    actual = actual_camera_metrics(
                        env,
                        config.scene.camera_name,
                        config.scene.target_body,
                        instance.outward_xy_world,
                    )
                    metrics.update(actual)
                    lower, upper = profile.bounds(stratum)
                    if not (
                        config.observations.camera_range_min_m - 1e-3
                        <= actual["camera_range_xy_m"]
                        <= config.observations.camera_range_max_m + 1e-3
                    ):
                        _reject(rejected_path, proposal, "actual_camera_range", metrics)
                        rejected_ids.add(proposal.proposal_id)
                        continue
                    if not lower - 0.1 <= actual["alpha_deg"] <= upper + 0.1:
                        _reject(rejected_path, proposal, "actual_alpha_stratum", metrics)
                        rejected_ids.add(proposal.proposal_id)
                        continue
                    if instance.category == "corner":
                        import mujoco

                        camera_id = mujoco.mj_name2id(
                            env.model, mujoco.mjtObj.mjOBJ_CAMERA, config.scene.camera_name
                        )
                        target_id = mujoco.mj_name2id(
                            env.model, mujoco.mjtObj.mjOBJ_BODY, config.scene.target_body
                        )
                        other_edge = instance.corner_other_edge
                        if other_edge is None:
                            raise ValueError(
                                f"corner instance lacks sibling edge: {instance.instance_id}"
                            )
                        exclusive = corner_family_is_exclusive(
                            env.data.cam_xpos[camera_id, :2],
                            env.data.xpos[target_id, :2],
                            instance.outward_xy_world,
                            EDGE_OUTWARDS[other_edge],
                        )
                        if not exclusive:
                            _reject(rejected_path, proposal, "corner_family_overlap", metrics)
                            rejected_ids.add(proposal.proposal_id)
                            continue

                    clouds = extract_semantic_clouds(
                        env,
                        camera_name=config.scene.camera_name,
                        base_body_name=config.scene.robot_root_body,
                        target_body_name=config.scene.target_body,
                        anchor_body_name=config.scene.anchor_body,
                        intrinsics=intrinsics,
                        max_depth_m=config.point_cloud.max_depth_m,
                        voxel_size_m=config.point_cloud.voxel_size_m,
                        anchor_max_behind_target_m=config.point_cloud.anchor_max_behind_target_m,
                    )
                    target_pixels = int(clouds["target_mask"].sum())
                    target_points = int(len(clouds["target_xyz_base"]))
                    anchor_points = int(len(clouds["anchor_xyz_base"]))
                    metrics.update(
                        target_pixels=target_pixels,
                        target_points=target_points,
                        anchor_points=anchor_points,
                        target_camera_distance_3d_m=float(clouds["target_camera_distance_m"]),
                    )
                    if target_pixels < config.point_cloud.min_target_pixels:
                        reason = "too_few_target_pixels"
                    elif target_points < config.point_cloud.min_target_points:
                        reason = "too_few_target_points"
                    elif anchor_points < config.point_cloud.min_anchor_points:
                        reason = "too_few_anchor_points"
                    else:
                        reason = None
                    if reason is not None:
                        _reject(rejected_path, proposal, reason, metrics)
                        rejected_ids.add(proposal.proposal_id)
                        continue

                    base_yaw = _actual_base_yaw(env, config.scene.robot_root_body)
                    axis_base = outward_world_to_base(
                        instance.outward_xy_world, base_yaw
                    ).astype(np.float32)
                    cloud_relative = Path("clouds") / f"{proposal.proposal_id}.npz"
                    np.savez_compressed(
                        observation_dir / cloud_relative,
                        target_xyz_base=clouds["target_xyz_base"],
                        anchor_xyz_base=clouds["anchor_xyz_base"],
                        target_center_base=clouds["target_center_base"],
                        axis_outward_gt_base=axis_base,
                        floor_z_base=np.float32(-config.scene.robot_root_z),
                    )
                    if args.save_previews:
                        preview_relative = Path("previews") / f"{proposal.proposal_id}.png"
                        _save_preview(
                            observation_dir / preview_relative,
                            clouds["rgb"],
                            clouds["target_mask"],
                            clouds["anchor_mask"],
                        )
                    else:
                        preview_relative = None
                    row = {
                        "schema_version": 1,
                        "observation_id": proposal.proposal_id,
                        "instance_id": instance.instance_id,
                        "placement_id": instance.placement_id,
                        "category": instance.category,
                        "approach_edge": instance.approach_edge,
                        "stratum": proposal.stratum,
                        "proposal": {
                            "camera_range_m": proposal.camera_range_m,
                            "alpha_deg": proposal.alpha_deg,
                            "yaw_offset_deg": proposal.yaw_offset_deg,
                        },
                        "base_pose_world": list(proposal.base_pose_world),
                        "camera_pose_world": {
                            "position": clouds["camera_position_world"].tolist(),
                            "rotation": clouds["camera_rotation_world"].tolist(),
                        },
                        "target_pose_world": list(instance.target_pose_world),
                        "outward_xy_world": list(instance.outward_xy_world),
                        "axis_outward_gt_base": axis_base.tolist(),
                        "cloud_file": str(cloud_relative),
                        "preview_file": None if preview_relative is None else str(preview_relative),
                        "metrics": metrics,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(manifest_path, row)
                    accepted_ids.add(proposal.proposal_id)
                    accepted_counts[(instance.instance_id, stratum)] += 1
                    print(
                        f"[FloAff][O] accept {proposal.proposal_id} "
                        f"quota={accepted_counts[(instance.instance_id, stratum)]}/{target_count}",
                        flush=True,
                    )

                achieved = accepted_counts[(instance.instance_id, stratum)]
                if achieved < target_count and not args.allow_shortfall:
                    raise RuntimeError(
                        f"proposal pools exhausted for {instance.instance_id}/{stratum}: "
                        f"{achieved}/{target_count} after the original pool and "
                        f"{args.max_supplemental_batches} supplemental batches; "
                        "inspect rejected.jsonl or increase --max-supplemental-batches"
                    )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
