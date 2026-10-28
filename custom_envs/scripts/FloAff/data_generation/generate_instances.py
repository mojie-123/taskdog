#!/usr/bin/env python
"""Generate continuous target placements and edge-conditioned instances."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    output_root,
)
from custom_envs.scripts.FloAff.manifests import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from custom_envs.scripts.FloAff.sampling.placement_sampling import (
    generate_placements_and_instances,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse compatible placement manifests when they already exist",
    )
    mode.add_argument(
        "--extend-transfer",
        action="store_true",
        help=(
            "append only transfer_* placements/instances to a compatible existing "
            "dataset and update its configuration snapshot"
        ),
    )
    mode.add_argument(
        "--extend-development",
        action="store_true",
        help=(
            "append only long-edge development placements/instances to a compatible "
            "existing dataset and update its configuration snapshot"
        ),
    )
    return parser


_TRANSFER_EXTENSION_PATHS = (
    ("placements", "transfer_short_count"),
    ("placements", "transfer_short_edge"),
    ("placements", "transfer_corner_count_per_region"),
    ("placements", "transfer_corner_edges"),
    ("placements", "transfer_long_count_per_edge"),
    ("placements", "transfer_long_edges"),
    ("manipulation", "transfer_candidates_per_instance"),
    ("observations", "transfer_valid_per_instance"),
)

_DEVELOPMENT_EXTENSION_PATHS = (
    ("placements", "long_count"),
    ("placements", "long_edges"),
    ("observations", "long_edge_train"),
)


def _without_transfer_extension_values(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("output_dir", None)
    for section, key in _TRANSFER_EXTENSION_PATHS:
        nested = result.get(section)
        if isinstance(nested, dict):
            nested.pop(key, None)
    return result


def _without_development_extension_values(value: dict) -> dict:
    result = deepcopy(value)
    result.pop("output_dir", None)
    for section, key in _DEVELOPMENT_EXTENSION_PATHS:
        nested = result.get(section)
        if isinstance(nested, dict):
            nested.pop(key, None)
    return result


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _extend_transfer_manifests(
    root: Path, config, args: argparse.Namespace
) -> tuple[int, int]:
    placements_path = root / "placements.jsonl"
    instances_path = root / "instances.jsonl"
    config_path = root / "dataset_config.json"
    required = (placements_path, instances_path, config_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "--extend-transfer requires an existing complete manifest set; missing "
            + ", ".join(missing)
        )

    snapshot = read_json(config_path)
    saved_config = dict(snapshot["config"])
    current_config = config.to_dict()
    if _canonical(_without_transfer_extension_values(saved_config)) != _canonical(
        _without_transfer_extension_values(current_config)
    ):
        raise ValueError(
            "existing dataset differs in fields other than the permitted transfer "
            "extension settings"
        )

    generated_placements, generated_instances = generate_placements_and_instances(config)
    generated_placement_rows = [row.to_dict() for row in generated_placements]
    generated_instance_rows = [row.to_dict() for row in generated_instances]
    generated_placement_by_id = {
        row["placement_id"]: row for row in generated_placement_rows
    }
    generated_instance_by_id = {row["instance_id"]: row for row in generated_instance_rows}
    existing_placements = read_jsonl(placements_path)
    existing_instances = read_jsonl(instances_path)

    for kind, existing, generated, id_key in (
        ("placement", existing_placements, generated_placement_by_id, "placement_id"),
        ("instance", existing_instances, generated_instance_by_id, "instance_id"),
    ):
        for row in existing:
            identifier = row[id_key]
            expected = generated.get(identifier)
            if expected is None:
                raise ValueError(
                    f"existing {kind} {identifier} is absent from the extended configuration"
                )
            if _canonical(row) != _canonical(expected):
                raise ValueError(
                    f"existing {kind} {identifier} would change; refusing transfer extension"
                )

    old_placement_ids = {row["placement_id"] for row in existing_placements}
    old_instance_ids = {row["instance_id"] for row in existing_instances}
    new_placements = [
        row for row in generated_placement_rows if row["placement_id"] not in old_placement_ids
    ]
    new_instances = [
        row for row in generated_instance_rows if row["instance_id"] not in old_instance_ids
    ]
    if any(not str(row["source_region"]).startswith("transfer_") for row in new_placements):
        raise ValueError("--extend-transfer attempted to add a non-transfer placement")
    if any(not str(row["placement_id"]).startswith("transfer_") for row in new_instances):
        raise ValueError("--extend-transfer attempted to add a non-transfer instance")

    write_jsonl(placements_path, generated_placement_rows)
    write_jsonl(instances_path, generated_instance_rows)
    snapshot.update(
        {
            "source_config": None if args.config is None else str(args.config.resolve()),
            "config": current_config,
            "placement_count": len(generated_placement_rows),
            "instance_count": len(generated_instance_rows),
            "transfer_extended_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(config_path, snapshot)
    return len(new_placements), len(new_instances)


def _extend_development_manifests(
    root: Path, config, args: argparse.Namespace
) -> tuple[int, int]:
    """Append new long-edge development rows without touching collected rows."""
    placements_path = root / "placements.jsonl"
    instances_path = root / "instances.jsonl"
    config_path = root / "dataset_config.json"
    required = (placements_path, instances_path, config_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "--extend-development requires an existing complete manifest set; missing "
            + ", ".join(missing)
        )

    snapshot = read_json(config_path)
    saved_config = dict(snapshot["config"])
    current_config = config.to_dict()
    if _canonical(_without_development_extension_values(saved_config)) != _canonical(
        _without_development_extension_values(current_config)
    ):
        raise ValueError(
            "existing dataset differs in fields other than the permitted long-edge "
            "development extension settings"
        )

    generated_placements, generated_instances = generate_placements_and_instances(config)
    generated_placement_rows = [row.to_dict() for row in generated_placements]
    generated_instance_rows = [row.to_dict() for row in generated_instances]
    generated_placement_by_id = {
        row["placement_id"]: row for row in generated_placement_rows
    }
    generated_instance_by_id = {row["instance_id"]: row for row in generated_instance_rows}
    existing_placements = read_jsonl(placements_path)
    existing_instances = read_jsonl(instances_path)

    for kind, existing, generated, id_key in (
        ("placement", existing_placements, generated_placement_by_id, "placement_id"),
        ("instance", existing_instances, generated_instance_by_id, "instance_id"),
    ):
        for row in existing:
            identifier = row[id_key]
            expected = generated.get(identifier)
            if expected is None:
                raise ValueError(
                    f"existing {kind} {identifier} is absent from the extended configuration"
                )
            if _canonical(row) != _canonical(expected):
                raise ValueError(
                    f"existing {kind} {identifier} would change; refusing development extension"
                )

    old_placement_ids = {row["placement_id"] for row in existing_placements}
    old_instance_ids = {row["instance_id"] for row in existing_instances}
    new_placements = [
        row for row in generated_placement_rows if row["placement_id"] not in old_placement_ids
    ]
    new_instances = [
        row for row in generated_instance_rows if row["instance_id"] not in old_instance_ids
    ]
    if any(
        row["category"] != "long_edge"
        or not str(row["source_region"]).startswith("long_")
        or str(row["source_region"]).startswith("transfer_")
        for row in new_placements
    ):
        raise ValueError(
            "--extend-development attempted to add something other than a long-edge "
            "development placement"
        )
    new_placement_ids = {row["placement_id"] for row in new_placements}
    if any(row["placement_id"] not in new_placement_ids for row in new_instances):
        raise ValueError("--extend-development produced an unexpected instance")

    write_jsonl(placements_path, generated_placement_rows)
    write_jsonl(instances_path, generated_instance_rows)
    snapshot.update(
        {
            "source_config": None if args.config is None else str(args.config.resolve()),
            "config": current_config,
            "placement_count": len(generated_placement_rows),
            "instance_count": len(generated_instance_rows),
            "development_extended_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(config_path, snapshot)
    return len(new_placements), len(new_instances)


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    placements_path = root / "placements.jsonl"
    instances_path = root / "instances.jsonl"
    config_path = root / "dataset_config.json"
    existing = (placements_path.exists(), instances_path.exists(), config_path.exists())
    if args.extend_development:
        added_placements, added_instances = _extend_development_manifests(
            root, config, args
        )
        print(
            f"[FloAff] development extension ready in {root}: "
            f"added {added_placements} placements and {added_instances} instances"
        )
        return 0
    if args.extend_transfer:
        added_placements, added_instances = _extend_transfer_manifests(root, config, args)
        print(
            f"[FloAff] transfer extension ready in {root}: "
            f"added {added_placements} placements and {added_instances} instances"
        )
        return 0
    if args.resume and any(existing):
        if not all(existing):
            raise FileNotFoundError(
                f"{root} contains an incomplete manifest set; expected "
                "placements.jsonl, instances.jsonl, and dataset_config.json"
            )
        ensure_dataset_config_compatible(root, config)
        print(f"[FloAff] reusing compatible placement manifests in {root}")
        return 0
    if not args.overwrite and any(existing):
        raise FileExistsError(f"{root} already contains placement manifests; pass --overwrite")
    placements, instances = generate_placements_and_instances(config)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(placements_path, (row.to_dict() for row in placements))
    write_jsonl(instances_path, (row.to_dict() for row in instances))
    write_json(
        root / "dataset_config.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_config": None if args.config is None else str(args.config.resolve()),
            "config": config.to_dict(),
            "placement_count": len(placements),
            "instance_count": len(instances),
        },
    )
    print(f"[FloAff] wrote {len(placements)} placements and {len(instances)} instances to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
