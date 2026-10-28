#!/usr/bin/env python
"""Validate FloAff manifests, quotas, geometry and split isolation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    manipulation_count_for_placement,
    observation_count_for_placement,
    output_root,
)
from custom_envs.scripts.FloAff.geometry import corner_family_is_exclusive
from custom_envs.scripts.FloAff.manifests import read_json, read_jsonl, write_json
from custom_envs.scripts.FloAff.schemas import ValidationReport
from custom_envs.scripts.FloAff.sampling.placement_sampling import EDGE_OUTWARDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def validate(root: Path, config, require_complete: bool = False) -> ValidationReport:
    report = ValidationReport()
    placements = read_jsonl(root / "placements.jsonl")
    instances = read_jsonl(root / "instances.jsonl")
    observations = read_jsonl(root / "observations" / "manifest.jsonl")
    candidates = read_jsonl(root / "manipulation" / "candidates.jsonl")
    summaries = read_jsonl(root / "manipulation" / "summaries.jsonl")
    report.statistics.update(
        placements=len(placements),
        instances=len(instances),
        observations=len(observations),
        manipulation_candidates=len(candidates),
        manipulation_summaries=len(summaries),
    )
    placement_ids = [row["placement_id"] for row in placements]
    instance_ids = [row["instance_id"] for row in instances]
    if len(placement_ids) != len(set(placement_ids)):
        report.errors.append("duplicate placement_id")
    if len(instance_ids) != len(set(instance_ids)):
        report.errors.append("duplicate instance_id")
    placement_set, instance_set = set(placement_ids), set(instance_ids)
    for instance in instances:
        if instance["placement_id"] not in placement_set:
            report.errors.append(f"instance references missing placement: {instance['instance_id']}")

    observation_ids = [row["observation_id"] for row in observations]
    if len(observation_ids) != len(set(observation_ids)):
        report.errors.append("duplicate observation_id")
    quota_counts = Counter((row["instance_id"], row["stratum"]) for row in observations)
    instance_by_id = {row["instance_id"]: row for row in instances}
    for row in observations:
        if row["instance_id"] not in instance_set:
            report.errors.append(f"observation references missing instance: {row['observation_id']}")
        cloud = root / "observations" / row["cloud_file"]
        if not cloud.is_file():
            report.errors.append(f"missing cloud: {cloud}")
        metrics = row.get("metrics", {})
        alpha = float(metrics.get("alpha_deg", row["proposal"]["alpha_deg"]))
        instance = instance_by_id.get(row["instance_id"])
        if instance:
            profile = config.observations.profile(instance["observation_profile"])
            lower, upper = profile.bounds(row["stratum"])
            if not lower - 0.2 <= alpha <= upper + 0.2:
                report.errors.append(f"O outside stratum: {row['observation_id']} alpha={alpha}")
            if instance["category"] == "corner":
                other = instance.get("corner_other_edge")
                if other is None:
                    report.errors.append(
                        f"corner instance lacks sibling edge: {instance['instance_id']}"
                    )
                    continue
                if not corner_family_is_exclusive(
                    row["camera_pose_world"]["position"][:2],
                    row["target_pose_world"][:2],
                    instance["outward_xy_world"],
                    EDGE_OUTWARDS[other],
                ):
                    report.errors.append(f"corner O family overlap: {row['observation_id']}")
    if require_complete:
        for instance in instances:
            quotas = config.observations.profile(instance["observation_profile"]).quotas(
                observation_count_for_placement(config, instance["placement_id"])
            )
            for stratum, expected in quotas.items():
                actual = quota_counts[(instance["instance_id"], stratum)]
                if actual != expected:
                    report.errors.append(
                        f"O quota {instance['instance_id']}/{stratum}: {actual}/{expected}"
                    )

    candidate_counts = Counter(row["instance_id"] for row in candidates)
    if require_complete:
        for instance in instances:
            instance_id = instance["instance_id"]
            actual = candidate_counts[instance_id]
            expected = manipulation_count_for_placement(
                config, instance["placement_id"]
            )
            if actual != expected:
                report.errors.append(f"M candidate count {instance_id}: {actual}/{expected}")
    summary_ids = [row["candidate_id"] for row in summaries]
    if len(summary_ids) != len(set(summary_ids)):
        report.errors.append("duplicate M summary candidate_id")
    if require_complete:
        missing = {row["candidate_id"] for row in candidates} - set(summary_ids)
        if missing:
            report.errors.append(f"{len(missing)} M candidates have no completed summary")

    split_path = root / "splits.json"
    if split_path.exists():
        split = read_json(split_path)
        placement_roles = split["placement_roles"]
        if set(placement_roles) != placement_set:
            report.errors.append("splits.json does not cover exactly all placements")
        roles_by_observation: dict[str, set[str]] = defaultdict(set)
        for role, ids in split.get("observations", {}).items():
            for observation_id in ids:
                roles_by_observation[observation_id].add(role)
        leaked = [key for key, roles in roles_by_observation.items() if len(roles) > 1]
        if leaked:
            report.errors.append(f"observation split leakage: {leaked[:5]}")
    elif require_complete:
        report.errors.append("missing splits.json")
    return report


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    report = validate(root, config, args.require_complete)
    report_path = args.report or (root / "validation_report.json")
    write_json(report_path, report.to_dict())
    print(f"[FloAff] validation ok={report.ok} stats={report.statistics}")
    for item in report.errors:
        print(f"[ERROR] {item}")
    for item in report.warnings:
        print(f"[WARN] {item}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
