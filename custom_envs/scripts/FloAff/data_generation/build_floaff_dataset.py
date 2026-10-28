#!/usr/bin/env python
"""Build FloAff samples using frozen, cached AnchorAxisNet predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.affordance_labels import (
    instance_candidate_arrays,
    kernel_affordance_labels,
    query_grid,
)
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    manipulation_count_for_placement,
    output_root,
    stable_seed,
)
from custom_envs.scripts.FloAff.geometry import rotate_world_xy_to_frame, unit_xy
from custom_envs.scripts.FloAff.manifests import read_json, read_jsonl, write_jsonl
from custom_envs.scripts.FloAff.simulation.camera_geometry import fixed_count_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--axis-predictions",
        type=Path,
        required=True,
        help="JSONL rows with observation_id and axis_xy_base",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        choices=["axis_train", "axis_eval_floaff_train", "transfer_test"],
    )
    parser.add_argument("--allow-incomplete-m", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _rotate_xyz(points: np.ndarray, axis_xy: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    result[:, :2] = rotate_world_xy_to_frame(result[:, :2], axis_xy).astype(np.float32)
    return result


def _candidate_coordinates_in_predicted_frame(
    observation: dict,
    candidates: list[dict],
    summaries_by_id: dict[str, dict],
    predicted_axis_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in candidates if row["instance_id"] == observation["instance_id"]]
    target_xy_world = np.asarray(observation["target_pose_world"][:2], dtype=np.float64)
    base_yaw = float(observation["base_pose_world"][2])
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    rotation_base_world = np.array([[c, -s], [s, c]], dtype=np.float64)
    coordinates, rates = [], []
    for row in selected:
        summary = summaries_by_id.get(row["candidate_id"])
        if summary is None:
            continue
        delta_world = np.asarray(row["base_pose"][:2], dtype=np.float64) - target_xy_world
        delta_base = delta_world @ rotation_base_world
        coordinate = rotate_world_xy_to_frame(delta_base[None, :], predicted_axis_base)[0]
        coordinates.append(coordinate)
        rates.append(float(summary["success_rate"]))
    return np.asarray(coordinates).reshape(-1, 2), np.asarray(rates, dtype=np.float64)


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    split = read_json(root / "splits.json")
    roles = set(args.role or ["axis_eval_floaff_train"])
    observation_roles = {
        observation_id: role
        for role, ids in split["observations"].items()
        if role in roles
        for observation_id in ids
    }
    observations = {
        row["observation_id"]: row
        for row in read_jsonl(root / "observations" / "manifest.jsonl")
    }
    predictions = {
        row["observation_id"]: row for row in read_jsonl(args.axis_predictions)
    }
    candidates = read_jsonl(root / "manipulation" / "candidates.jsonl")
    summaries = {
        row["candidate_id"]: row
        for row in read_jsonl(root / "manipulation" / "summaries.jsonl")
    }
    processed_dir = root / "processed" / "floaff"
    if processed_dir.exists() and not args.overwrite and (processed_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"{processed_dir} already exists; pass --overwrite")
    samples_dir = processed_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    queries_xy = query_grid(config.affordance).astype(np.float32)
    manifest = []
    for observation_id, role in sorted(observation_roles.items()):
        observation = observations.get(observation_id)
        prediction = predictions.get(observation_id)
        if observation is None or prediction is None:
            raise KeyError(f"missing observation or AxisNet prediction for {observation_id}")
        axis_key = "axis_xy_base" if "axis_xy_base" in prediction else "axis_outward_pred_base"
        predicted_axis = unit_xy(prediction[axis_key])
        cloud = np.load(root / "observations" / observation["cloud_file"])
        target_center = cloud["target_center_base"].astype(np.float32)
        target = cloud["target_xyz_base"].astype(np.float32) - target_center[None, :]
        anchor = cloud["anchor_xyz_base"].astype(np.float32) - target_center[None, :]
        target = _rotate_xyz(target, predicted_axis)
        anchor = _rotate_xyz(anchor, predicted_axis)
        rng = np.random.default_rng(stable_seed(config.seed, f"floaff:{observation_id}"))
        target, _ = fixed_count_sample(target, config.point_cloud.target_sample_points, rng)
        anchor, _ = fixed_count_sample(anchor, config.point_cloud.anchor_sample_points, rng)
        target_features = np.column_stack(
            [target, np.ones(len(target), dtype=np.float32), np.zeros(len(target), dtype=np.float32)]
        )
        anchor_features = np.column_stack(
            [anchor, np.zeros(len(anchor), dtype=np.float32), np.ones(len(anchor), dtype=np.float32)]
        )
        scene_points = np.concatenate([target_features, anchor_features], axis=0).astype(np.float32)
        order = rng.permutation(len(scene_points))
        scene_points = scene_points[order]

        candidate_xy, rates = _candidate_coordinates_in_predicted_frame(
            observation, candidates, summaries, predicted_axis
        )
        expected_m = manipulation_count_for_placement(
            config, observation["placement_id"]
        )
        if len(candidate_xy) != expected_m and not args.allow_incomplete_m:
            raise RuntimeError(
                f"{observation['instance_id']} has {len(candidate_xy)}/{expected_m} M labels"
            )
        labels, support, nearest = kernel_affordance_labels(
            queries_xy,
            candidate_xy,
            rates,
            sigma_m=config.affordance.kernel_sigma_m,
            support_radius_m=config.affordance.support_radius_m,
        )
        query_z = float(cloud["floor_z_base"] - target_center[2])
        query_xyz = np.column_stack(
            [queries_xy, np.full(len(queries_xy), query_z, dtype=np.float32)]
        ).astype(np.float32)
        relative = Path("samples") / f"{observation_id}.npz"
        np.savez_compressed(
            processed_dir / relative,
            scene_points=scene_points,
            query_xyz=query_xyz,
            affordance=labels,
            label_support_mask=support,
            nearest_candidate_m=nearest,
            axis_outward_pred_base=predicted_axis.astype(np.float32),
            axis_outward_gt_base=cloud["axis_outward_gt_base"].astype(np.float32),
        )
        manifest.append(
            {
                "schema_version": 1,
                "observation_id": observation_id,
                "instance_id": observation["instance_id"],
                "placement_id": observation["placement_id"],
                "category": observation["category"],
                "approach_edge": observation["approach_edge"],
                "split_role": role,
                "sample_file": str(relative),
                "axis_confidence": prediction.get("confidence"),
                "supported_query_count": int(support.sum()),
            }
        )
    np.random.default_rng(stable_seed(config.seed, "floaff-manifest")).shuffle(manifest)
    write_jsonl(processed_dir / "manifest.jsonl", manifest)
    print(f"[FloAff] wrote {len(manifest)} FloAff samples to {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
