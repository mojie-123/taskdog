#!/usr/bin/env python3
"""Join captured observations with successful manipulation poses.

The script accepts either the future canonical manipulation-pose manifest or
the current ``success_poses.jsonl`` files.  It never random-splits pair rows:
all labels sharing one observation are assigned to the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from custom_envs.scripts.N2M.data_generation.pair_geometry import (
    assert_target_round_trip,
    target_in_observation_frame,
)
from custom_envs.scripts.N2M.manifests import read_jsonl, write_json, write_jsonl


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _normalise_success_rows(paths: list[Path]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("stable_success") is False:
                continue
            if row.get("decision") not in (
                None,
                "success_threshold_reached_early",
                "success_threshold_reached_at_limit",
                "success",
            ):
                continue
            required = ("scene_id", "approach_edge", "base_pose")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}: success row is missing {missing}: {row}")
            pose = [float(value) for value in row["base_pose"][:3]]
            key = (
                row["scene_id"],
                row["approach_edge"],
                *(round(value, 6) for value in pose),
            )
            normalised = dict(row)
            normalised["base_pose"] = pose
            normalised["m_uid"] = (
                row.get("m_uid")
                or row.get("candidate_id")
                or f"{row['scene_id']}__{row['approach_edge']}__"
                f"{hashlib.sha1(str(key).encode()).hexdigest()[:10]}"
            )
            normalised["source_manifest"] = str(path.resolve())
            unique[key] = normalised
    return list(unique.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--success-manifest", type=Path, action="append", required=True)
    parser.add_argument("--max-local-distance", type=float, default=2.0)
    parser.add_argument("--max-local-yaw-deg", type=float, default=180.0)
    parser.add_argument(
        "--allow-cross-edge",
        action="store_true",
        help="pair a corner observation with successful M from its other edge",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0 - args.train_ratio):
        raise ValueError("--val-ratio must be >=0 and train+val must be <1")
    dataset_dir = args.dataset_dir.resolve()
    observations = read_jsonl(dataset_dir / "observations.jsonl")
    if not observations:
        raise RuntimeError(f"no observations in {dataset_dir / 'observations.jsonl'}")
    successes = _normalise_success_rows(args.success_manifest)
    if not successes:
        raise RuntimeError("no successful manipulation poses were found")

    successes_by_scene: dict[str, list[dict]] = {}
    for row in successes:
        successes_by_scene.setdefault(row["scene_id"], []).append(row)

    pairs: list[dict] = []
    observations_with_pairs: set[str] = set()
    rejected_distance = rejected_yaw = rejected_edge = 0
    for observation in observations:
        observation_id = observation["observation_id"]
        observation_pose = observation["observation_pose_world"]
        for manipulation in successes_by_scene.get(observation["scene_id"], []):
            if (
                not args.allow_cross_edge
                and manipulation["approach_edge"] != observation["approach_edge"]
            ):
                rejected_edge += 1
                continue
            target = target_in_observation_frame(observation_pose, manipulation["base_pose"])
            distance = float(np.linalg.norm(target[:2]))
            if distance > args.max_local_distance:
                rejected_distance += 1
                continue
            if abs(float(target[2])) > math.radians(args.max_local_yaw_deg):
                rejected_yaw += 1
                continue
            assert_target_round_trip(observation_pose, manipulation["base_pose"], target)
            pairs.append(
                {
                    "schema_version": 1,
                    "pair_id": f"{observation_id}__{manipulation['m_uid']}",
                    "observation_id": observation_id,
                    "cloud_file": observation["cloud_file"],
                    "m_uid": manipulation["m_uid"],
                    "scene_id": observation["scene_id"],
                    "approach_edge": manipulation["approach_edge"],
                    "target_pose_observation": target.tolist(),
                    "observation_pose_world": observation_pose,
                    "manipulation_pose_world": manipulation["base_pose"],
                    "split_group": observation_id,
                }
            )
            observations_with_pairs.add(observation_id)

    if not pairs:
        raise RuntimeError("all observation/manipulation-pose pairs were filtered")

    split_groups = {"train": [], "val": [], "test": []}
    threshold_val = args.train_ratio + args.val_ratio
    for observation_id in sorted(observations_with_pairs):
        fraction = _stable_fraction(observation_id, args.seed)
        split = "train" if fraction < args.train_ratio else "val" if fraction < threshold_val else "test"
        split_groups[split].append(observation_id)
    split_by_observation = {
        observation_id: split
        for split, observation_ids in split_groups.items()
        for observation_id in observation_ids
    }
    for row in pairs:
        row["split"] = split_by_observation[row["observation_id"]]

    write_jsonl(dataset_dir / "manipulation_poses.jsonl", successes)
    write_jsonl(dataset_dir / "pairs.jsonl", pairs)
    splits_dir = dataset_dir / "splits"
    for split, observation_ids in split_groups.items():
        write_json(splits_dir / f"{split}.json", {"observation_ids": observation_ids})
    split_pair_counts = {
        split: sum(row["split"] == split for row in pairs) for split in split_groups
    }
    stats = {
        "observation_count": len(observations),
        "observation_count_with_pairs": len(observations_with_pairs),
        "manipulation_pose_count": len(successes),
        "pair_count": len(pairs),
        "split_observation_counts": {key: len(value) for key, value in split_groups.items()},
        "split_pair_counts": split_pair_counts,
        "filtered": {
            "edge_mismatch": rejected_edge,
            "local_distance": rejected_distance,
            "local_yaw": rejected_yaw,
        },
    }
    write_json(dataset_dir / "dataset.json", {"schema_version": 1, **stats})
    print(f"[N2M] built {len(pairs)} pairs from {len(observations_with_pairs)} observations")
    print(f"[N2M] split pair counts: {split_pair_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
