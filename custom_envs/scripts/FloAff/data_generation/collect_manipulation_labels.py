#!/usr/bin/env python
"""Collect empirical manipulation success labels for sampled M poses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    load_instances,
    output_root,
    stable_seed,
)
from custom_envs.scripts.FloAff.manifests import append_jsonl, read_jsonl, write_jsonl
from custom_envs.scripts.FloAff.sampling.manipulation_sampling import (
    sample_manipulation_candidates,
)
from custom_envs.scripts.FloAff.simulation.grasp_episode import FloAffGraspRunner


DETERMINISTIC_FAILURE_STAGES = {"BASE_PLACED", "ARM_INIT", "OPERATOR", "INTERNAL"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance", action="append", default=[])
    parser.add_argument("--grasp-checkpoint", required=False)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser


def _result_row(result, candidate: dict, phase: str) -> dict:
    row = result.to_dict()
    row.update(
        {
            "instance_id": candidate["instance_id"],
            "placement_id": candidate["placement_id"],
            "approach_edge": candidate["approach_edge"],
            "phase": phase,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return row


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    instances = load_instances(root, args.instance)
    manipulation_dir = root / "manipulation"
    manipulation_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = manipulation_dir / "candidates.jsonl"
    rollouts_path = manipulation_dir / "rollouts.jsonl"
    summaries_path = manipulation_dir / "summaries.jsonl"
    if not args.resume and not args.dry_run and (rollouts_path.exists() or summaries_path.exists()):
        raise FileExistsError(
            f"{manipulation_dir} already contains rollout results; pass --resume"
        )

    all_candidates = []
    for instance in instances:
        all_candidates.extend(
            candidate.to_dict()
            for candidate in sample_manipulation_candidates(
                instance, config, stable_seed(config.seed, f"M:{instance.instance_id}")
            )
        )
    if args.max_candidates is not None:
        all_candidates = all_candidates[: max(0, int(args.max_candidates))]
    existing_candidates = {row["candidate_id"]: row for row in read_jsonl(candidates_path)}
    existing_candidates.update({row["candidate_id"]: row for row in all_candidates})
    write_jsonl(candidates_path, sorted(existing_candidates.values(), key=lambda row: row["candidate_id"]))
    if args.dry_run:
        print(f"[FloAff] generated {len(all_candidates)} M candidates at {candidates_path}")
        return 0
    if not args.grasp_checkpoint:
        raise ValueError("--grasp-checkpoint is required unless --dry-run is used")

    summarized = {row["candidate_id"] for row in read_jsonl(summaries_path)} if args.resume else set()
    with FloAffGraspRunner(
        config,
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        debug_dir=manipulation_dir / "debug",
    ) as runner:
        for candidate in all_candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id in summarized:
                print(f"[FloAff][M] resume skip {candidate_id}")
                continue
            results = []
            initial_attempts = int(config.manipulation.initial_attempts)
            for attempt in range(initial_attempts):
                result = runner.run(candidate, attempt)
                results.append(result)
                append_jsonl(rollouts_path, _result_row(result, candidate, "initial"))
                if attempt == 0 and result.failure_stage in DETERMINISTIC_FAILURE_STAGES:
                    break

            deterministic = (
                len(results) == 1
                and results[0].failure_stage in DETERMINISTIC_FAILURE_STAGES
            )
            success_count = sum(result.success is True for result in results)
            if not deterministic and 0 < success_count < len(results):
                start = len(results)
                for attempt in range(
                    start, start + int(config.manipulation.ambiguous_extra_attempts)
                ):
                    result = runner.run(candidate, attempt)
                    results.append(result)
                    append_jsonl(rollouts_path, _result_row(result, candidate, "ambiguous_extra"))

            success_count = sum(result.success is True for result in results)
            summary = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "instance_id": candidate["instance_id"],
                "placement_id": candidate["placement_id"],
                "category": candidate["category"],
                "approach_edge": candidate["approach_edge"],
                "base_pose": candidate["base_pose"],
                "cube_pose": candidate["cube_pose"],
                "proposal": candidate["proposal"],
                "attempt_count": len(results),
                "success_count": int(success_count),
                "success_rate": float(success_count / max(len(results), 1)),
                "deterministic_rejection": deterministic,
                "first_failure_stage": results[0].failure_stage if results else None,
                "first_failure_reason": results[0].failure_reason if results else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(summaries_path, summary)
            print(
                f"[FloAff][M] {candidate_id} success={success_count}/{len(results)} "
                f"deterministic={deterministic}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
