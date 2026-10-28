#!/usr/bin/env python
"""Turn empirical M success rates into dense ground-query affordance fields."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from custom_envs.scripts.FloAff.config import AffordanceLabelConfig, load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    manipulation_count_for_placement,
    output_root,
)
from custom_envs.scripts.FloAff.manifests import read_jsonl, write_jsonl


def inclusive_axis(minimum: float, maximum: float, resolution: float) -> np.ndarray:
    count = int(round((float(maximum) - float(minimum)) / float(resolution))) + 1
    return np.linspace(float(minimum), float(maximum), count, dtype=np.float64)


def query_grid(config: AffordanceLabelConfig) -> np.ndarray:
    outward = inclusive_axis(config.outward_min_m, config.outward_max_m, config.resolution_m)
    tangent = inclusive_axis(config.tangent_min_m, config.tangent_max_m, config.resolution_m)
    xx, yy = np.meshgrid(outward, tangent, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def kernel_affordance_labels(
    query_xy: np.ndarray,
    candidate_xy: np.ndarray,
    success_rates: np.ndarray,
    *,
    sigma_m: float,
    support_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    candidates = np.asarray(candidate_xy, dtype=np.float64).reshape(-1, 2)
    rates = np.asarray(success_rates, dtype=np.float64).reshape(-1)
    if len(candidates) == 0 or len(candidates) != len(rates):
        raise ValueError("candidate coordinates and rates must be non-empty and aligned")
    if sigma_m <= 0.0 or support_radius_m <= 0.0:
        raise ValueError("kernel sigma and support radius must be positive")
    distances_sq = np.sum((query[:, None, :] - candidates[None, :, :]) ** 2, axis=2)
    weights = np.exp(-0.5 * distances_sq / float(sigma_m) ** 2)
    weight_sum = np.sum(weights, axis=1)
    labels = (weights @ rates) / np.maximum(weight_sum, 1e-12)
    nearest = np.sqrt(np.min(distances_sq, axis=1))
    support = nearest <= float(support_radius_m)
    return labels.astype(np.float32), support.astype(bool), nearest.astype(np.float32)


def instance_candidate_arrays(
    instance_id: str,
    candidates: list[dict],
    summaries_by_id: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in candidates if row["instance_id"] == instance_id]
    coordinates, rates, ids = [], [], []
    for row in selected:
        summary = summaries_by_id.get(row["candidate_id"])
        if summary is None:
            continue
        proposal = row["proposal"]
        coordinates.append([proposal["outward_m"], proposal["tangent_m"]])
        rates.append(float(summary["success_rate"]))
        ids.append(row["candidate_id"])
    return (
        np.asarray(coordinates, dtype=np.float64).reshape(-1, 2),
        np.asarray(rates, dtype=np.float64),
        ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    instances = read_jsonl(root / "instances.jsonl")
    candidates = read_jsonl(root / "manipulation" / "candidates.jsonl")
    summaries = {
        row["candidate_id"]: row
        for row in read_jsonl(root / "manipulation" / "summaries.jsonl")
    }
    field_dir = root / "affordance_fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    grid = query_grid(config.affordance)
    manifest = []
    for instance in instances:
        expected = manipulation_count_for_placement(
            config, instance["placement_id"]
        )
        candidate_xy, success_rates, candidate_ids = instance_candidate_arrays(
            instance["instance_id"], candidates, summaries
        )
        if len(candidate_xy) != expected and not args.allow_incomplete:
            raise RuntimeError(
                f"{instance['instance_id']} has {len(candidate_xy)}/{expected} completed M labels"
            )
        if len(candidate_xy) == 0:
            continue
        labels, support, nearest = kernel_affordance_labels(
            grid,
            candidate_xy,
            success_rates,
            sigma_m=config.affordance.kernel_sigma_m,
            support_radius_m=config.affordance.support_radius_m,
        )
        relative = Path(f"{instance['instance_id']}.npz")
        np.savez_compressed(
            field_dir / relative,
            query_xy=grid.astype(np.float32),
            labels=labels,
            label_support_mask=support,
            nearest_candidate_m=nearest,
            candidate_xy=candidate_xy.astype(np.float32),
            candidate_success_rate=success_rates.astype(np.float32),
            candidate_ids=np.asarray(candidate_ids),
        )
        manifest.append(
            {
                "schema_version": 1,
                "instance_id": instance["instance_id"],
                "placement_id": instance["placement_id"],
                "field_file": str(relative),
                "candidate_count": len(candidate_xy),
                "query_count": len(grid),
                "supported_query_count": int(support.sum()),
            }
        )
    write_jsonl(field_dir / "manifest.jsonl", manifest)
    print(f"[FloAff] wrote {len(manifest)} GT affordance fields to {field_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
