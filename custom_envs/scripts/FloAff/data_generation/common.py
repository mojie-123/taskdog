"""Shared helpers for data-generation commands."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.FloAff.manifests import read_json, read_jsonl
from custom_envs.scripts.FloAff.schemas import InstanceRecord


def stable_seed(master_seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{int(master_seed)}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def output_root(config: DataGenerationConfig, override: Path | None = None) -> Path:
    return Path(override if override is not None else config.output_dir).resolve()


def is_transfer_placement(placement_id: str) -> bool:
    """Return whether a placement belongs to the held-out transfer split."""
    return str(placement_id).startswith("transfer_")


def manipulation_count_for_placement(
    config: DataGenerationConfig, placement_id: str
) -> int:
    if is_transfer_placement(placement_id):
        return int(config.manipulation.transfer_candidates_per_instance)
    return int(config.manipulation.candidates_per_instance)


def observation_count_for_placement(
    config: DataGenerationConfig, placement_id: str
) -> int:
    if is_transfer_placement(placement_id):
        return int(config.observations.transfer_valid_per_instance)
    return int(config.observations.valid_per_instance)


def ensure_dataset_config_compatible(root: Path, config: DataGenerationConfig) -> None:
    """Reject accidental mixing of data generated with different geometry."""
    path = Path(root) / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset configuration snapshot: {path}")
    saved = dict(read_json(path)["config"])
    current = dict(config.to_dict())
    # The physical dataset is independent of where the run directory is mounted.
    saved.pop("output_dir", None)
    current.pop("output_dir", None)
    saved_text = json.dumps(saved, sort_keys=True, separators=(",", ":"))
    current_text = json.dumps(current, sort_keys=True, separators=(",", ":"))
    if saved_text != current_text:
        raise ValueError(
            "the requested configuration differs from dataset_config.json; "
            "regenerate instances in a new output directory"
        )


def instance_from_dict(row: dict) -> InstanceRecord:
    return InstanceRecord(
        instance_id=row["instance_id"],
        placement_id=row["placement_id"],
        category=row["category"],
        approach_edge=row["approach_edge"],
        outward_xy_world=tuple(row["outward_xy_world"]),
        tangent_xy_world=tuple(row["tangent_xy_world"]),
        target_pose_world=tuple(row["target_pose_world"]),
        observation_profile=row["observation_profile"],
        corner_other_edge=row.get("corner_other_edge"),
        schema_version=int(row.get("schema_version", 1)),
    )


def load_instances(root: Path, selected: Iterable[str] | None = None) -> list[InstanceRecord]:
    rows = [instance_from_dict(row) for row in read_jsonl(root / "instances.jsonl")]
    if selected:
        wanted = set(selected)
        rows = [row for row in rows if row.instance_id in wanted]
        missing = wanted - {row.instance_id for row in rows}
        if missing:
            raise KeyError(f"unknown instance ids: {sorted(missing)}")
    return rows


def plain(value):
    return asdict(value) if is_dataclass(value) else value
