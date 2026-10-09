#!/usr/bin/env python3
"""Run structural and geometric checks on an assembled N2M dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from custom_envs.scripts.N2M.data_generation.pair_geometry import assert_target_round_trip
from custom_envs.scripts.N2M.manifests import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--minimum-points", type=int, default=8192)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    observations = read_jsonl(dataset_dir / "observations.jsonl")
    pairs = read_jsonl(dataset_dir / "pairs.jsonl")
    if not observations or not pairs:
        raise RuntimeError("dataset must contain non-empty observations.jsonl and pairs.jsonl")

    observations_by_id = {row["observation_id"]: row for row in observations}
    split_for_observation: dict[str, set[str]] = defaultdict(set)
    point_counts = []
    for observation in observations:
        cloud_path = dataset_dir / observation["cloud_file"]
        if not cloud_path.exists():
            raise FileNotFoundError(cloud_path)
        with np.load(cloud_path) as cloud:
            xyz, rgb = cloud["xyz"], cloud["rgb"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
            raise ValueError(f"{cloud_path}: expected xyz/rgb shape (N,3)")
        if len(xyz) < args.minimum_points:
            raise ValueError(f"{cloud_path}: only {len(xyz)} points")
        if not np.isfinite(xyz).all():
            raise ValueError(f"{cloud_path}: non-finite XYZ values")
        point_counts.append(len(xyz))

    for pair in pairs:
        observation = observations_by_id.get(pair["observation_id"])
        if observation is None:
            raise KeyError(f"pair references unknown observation: {pair['observation_id']}")
        if pair["cloud_file"] != observation["cloud_file"]:
            raise ValueError(f"pair {pair['pair_id']} has inconsistent cloud_file")
        assert_target_round_trip(
            pair["observation_pose_world"],
            pair["manipulation_pose_world"],
            pair["target_pose_observation"],
        )
        split_for_observation[pair["observation_id"]].add(pair["split"])
    leakage = {key: value for key, value in split_for_observation.items() if len(value) != 1}
    if leakage:
        raise ValueError(f"observation leakage across splits: {leakage}")

    print(f"[N2M] observations={len(observations)} pairs={len(pairs)}")
    print(
        f"[N2M] point_count min/median/max="
        f"{min(point_counts)}/{int(np.median(point_counts))}/{max(point_counts)}"
    )
    print(f"[N2M] split pairs={dict(Counter(row['split'] for row in pairs))}")
    print(f"[N2M] scenes={dict(Counter(row['scene_id'] for row in pairs))}")
    print("[N2M] dataset validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

