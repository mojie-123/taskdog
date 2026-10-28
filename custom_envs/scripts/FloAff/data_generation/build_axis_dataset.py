#!/usr/bin/env python
"""Build fixed-size visible-anchor samples for AnchorAxisNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    output_root,
    stable_seed,
)
from custom_envs.scripts.FloAff.manifests import read_json, read_jsonl, write_jsonl
from custom_envs.scripts.FloAff.simulation.camera_geometry import fixed_count_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--role",
        action="append",
        choices=["axis_train", "axis_eval_floaff_train", "transfer_test"],
        default=[],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    split = read_json(root / "splits.json")
    roles = set(args.role or split["observations"].keys())
    observation_roles = {
        observation_id: role
        for role, ids in split["observations"].items()
        if role in roles
        for observation_id in ids
    }
    observations = read_jsonl(root / "observations" / "manifest.jsonl")
    processed_dir = root / "processed" / "axis"
    samples_dir = processed_dir / "samples"
    if processed_dir.exists() and not args.overwrite and (processed_dir / "manifest.jsonl").exists():
        raise FileExistsError(f"{processed_dir} already exists; pass --overwrite")
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in observations:
        role = observation_roles.get(row["observation_id"])
        if role is None:
            continue
        cloud = np.load(root / "observations" / row["cloud_file"])
        anchor = cloud["anchor_xyz_base"].astype(np.float32)
        center = cloud["target_center_base"].astype(np.float32)
        centered = anchor - center[None, :]
        rng = np.random.default_rng(stable_seed(config.seed, f"axis:{row['observation_id']}"))
        sampled, source_indices = fixed_count_sample(
            centered, config.point_cloud.axis_anchor_points, rng
        )
        axis = cloud["axis_outward_gt_base"].astype(np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        relative = Path("samples") / f"{row['observation_id']}.npz"
        np.savez_compressed(
            processed_dir / relative,
            anchor_xyz_centered_base=sampled.astype(np.float32),
            axis_outward_gt_base=axis,
            source_indices=source_indices,
        )
        manifest.append(
            {
                "schema_version": 1,
                "observation_id": row["observation_id"],
                "placement_id": row["placement_id"],
                "instance_id": row["instance_id"],
                "category": row["category"],
                "approach_edge": row["approach_edge"],
                "split_role": role,
                "sample_file": str(relative),
            }
        )
    np.random.default_rng(stable_seed(config.seed, "axis-manifest")).shuffle(manifest)
    write_jsonl(processed_dir / "manifest.jsonl", manifest)
    print(f"[FloAff] wrote {len(manifest)} AnchorAxisNet samples to {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
