#!/usr/bin/env python
"""Create placement-level splits without leaking sibling O/instances."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    output_root,
    stable_seed,
)
from custom_envs.scripts.FloAff.manifests import read_jsonl, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--transfer-placement",
        action="append",
        default=[],
        help="placement id reserved for final transfer_test; may be repeated",
    )
    return parser


def build_split(config, placements: list[dict], instances: list[dict], observations: list[dict]) -> dict:
    transfer = set()
    # Explicit IDs and source regions prefixed with transfer_ are both supported.
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in placements:
        if row.get("split_role") == "transfer_test" or str(row.get("source_region", "")).startswith("transfer_"):
            transfer.add(row["placement_id"])
        else:
            by_category[row["category"]].append(row["placement_id"])

    roles: dict[str, str] = {placement_id: "transfer_test" for placement_id in transfer}
    for category, placement_ids in sorted(by_category.items()):
        placement_ids = sorted(placement_ids)
        rng = np.random.default_rng(stable_seed(config.splits.seed, f"split:{category}"))
        rng.shuffle(placement_ids)
        train_count = int(round(len(placement_ids) * config.splits.axis_train_fraction))
        if len(placement_ids) >= 2:
            train_count = min(max(train_count, 1), len(placement_ids) - 1)
        for index, placement_id in enumerate(placement_ids):
            roles[placement_id] = (
                "axis_train" if index < train_count else "axis_eval_floaff_train"
            )

    instances_by_role: dict[str, list[str]] = defaultdict(list)
    for row in instances:
        instances_by_role[roles[row["placement_id"]]].append(row["instance_id"])
    observations_by_role: dict[str, list[str]] = defaultdict(list)
    for row in observations:
        observations_by_role[roles[row["placement_id"]]].append(row["observation_id"])
    return {
        "schema_version": 1,
        "seed": config.splits.seed,
        "axis_train_fraction": config.splits.axis_train_fraction,
        "placement_roles": dict(sorted(roles.items())),
        "instances": {
            role: sorted(values) for role, values in sorted(instances_by_role.items())
        },
        "observations": {
            role: sorted(values) for role, values in sorted(observations_by_role.items())
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    placements = read_jsonl(root / "placements.jsonl")
    instances = read_jsonl(root / "instances.jsonl")
    observations = read_jsonl(root / "observations" / "manifest.jsonl")
    requested_transfer = set(args.transfer_placement)
    known = {row["placement_id"] for row in placements}
    missing = requested_transfer - known
    if missing:
        raise KeyError(f"unknown transfer placements: {sorted(missing)}")
    for row in placements:
        if row["placement_id"] in requested_transfer:
            row["split_role"] = "transfer_test"
    split = build_split(config, placements, instances, observations)
    write_json(root / "splits.json", split)
    counts = defaultdict(int)
    for role in split["placement_roles"].values():
        counts[role] += 1
    print(f"[FloAff] placement split: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
