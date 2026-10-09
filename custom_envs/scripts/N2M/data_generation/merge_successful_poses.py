#!/usr/bin/env python3
"""Merge the completed d30 searches into one audited manipulation-pose set.

The original search directories are read-only inputs.  ``short_left`` is a
special case: the old full run contributes yaw ``0,+/-22.5`` degrees and the
supplement contributes ``+/-11.25,+/-30`` degrees.  The obsolete ``+/-45``
candidates are recorded as exclusions and never enter the canonical set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

from custom_envs.scripts.N2M.config import PROJECT_ROOT
from custom_envs.scripts.N2M.manifests import read_json, read_jsonl, write_json, write_jsonl


DATA_ROOT = PROJECT_ROOT / "custom_envs" / "scripts" / "N2M" / "data"
DEFAULT_OUTPUT = DATA_ROOT / "manipulation_pose_dataset" / "d30_v1"
REFINED_YAWS = (0.0, -11.25, 11.25, -22.5, 22.5, -30.0, 30.0)
NORMAL_OFFSETS = (0.0, -0.05, -0.10, -0.15)
TANGENT_OFFSETS = (0.0, -0.125, 0.125, -0.25, 0.25)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    directory: str
    included_yaws: tuple[float, ...]
    expected_candidates: int


SOURCES = (
    SourceSpec(
        "short_left_core",
        "collect_short_left_full",
        (0.0, -22.5, 22.5),
        100,
    ),
    SourceSpec(
        "short_left_yaw_supplement",
        "collect_short_left_yaw_supplement",
        (-11.25, 11.25, -30.0, 30.0),
        80,
    ),
    SourceSpec("short_right_refined", "collect_short_right_refined", REFINED_YAWS, 140),
    SourceSpec("long_bottom_refined", "collect_long_bottom_refined", REFINED_YAWS, 140),
    SourceSpec("long_top_refined", "collect_long_top_refined", REFINED_YAWS, 140),
    SourceSpec("corner_lb_refined", "collect_corner_lb_refined", REFINED_YAWS, 280),
    SourceSpec("corner_rt_refined", "collect_corner_rt_refined", REFINED_YAWS, 280),
)


EDGE_ORDER = {
    ("short_left", "left"): 0,
    ("short_right", "right"): 1,
    ("long_bottom", "bottom"): 2,
    ("long_top", "top"): 3,
    ("corner_lb", "left"): 4,
    ("corner_lb", "bottom"): 5,
    ("corner_rt", "right"): 6,
    ("corner_rt", "top"): 7,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_by_id(rows: list[dict], label: str, path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not candidate_id:
            raise ValueError(f"{path}: {label} row has no candidate_id")
        if candidate_id in result:
            raise ValueError(f"{path}: duplicate {label} candidate_id={candidate_id}")
        result[candidate_id] = row
    return result


def _grid_key(candidate: dict) -> tuple:
    perturbation = candidate["perturbation"]
    return (
        candidate["scene_id"],
        candidate["approach_edge"],
        round(float(perturbation["normal_m"]), 6),
        round(float(perturbation["tangent_m"]), 6),
        round(float(perturbation["yaw_deg"]), 6),
    )


def _signed_token(prefix: str, value: float, scale: int, width: int) -> str:
    sign = "p" if value >= 0.0 else "m"
    integer = int(round(abs(float(value)) * scale))
    return f"{prefix}{sign}{integer:0{width}d}"


def _manipulation_uid(candidate: dict) -> str:
    perturbation = candidate["perturbation"]
    return "__".join(
        (
            "d30",
            candidate["scene_id"],
            candidate["approach_edge"],
            _signed_token("n", perturbation["normal_m"], 1000, 3),
            _signed_token("t", perturbation["tangent_m"], 1000, 3),
            _signed_token("y", perturbation["yaw_deg"], 100, 4),
        )
    )


def _assert_pose_equal(first, second, label: str, atol: float = 1e-9) -> None:
    if len(first) != len(second) or any(abs(float(a) - float(b)) > atol for a, b in zip(first, second)):
        raise ValueError(f"{label} mismatch: {first} != {second}")


def _audit_source(spec: SourceSpec, data_root: Path):
    directory = data_root / spec.directory
    paths = {
        "candidates": directory / "candidates.jsonl",
        "summaries": directory / "candidate_summaries.jsonl",
        "successes": directory / "success_poses.jsonl",
        "rollouts": directory / "manipulation_rollouts.jsonl",
        "config": directory / "run_config.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{spec.source_id}: missing files: {missing}")
    candidate_rows = read_jsonl(paths["candidates"])
    summary_rows = read_jsonl(paths["summaries"])
    success_rows = read_jsonl(paths["successes"])
    rollout_rows = read_jsonl(paths["rollouts"])
    if len(candidate_rows) != spec.expected_candidates:
        raise ValueError(
            f"{spec.source_id}: {len(candidate_rows)}/{spec.expected_candidates} candidates"
        )
    candidates = _unique_by_id(candidate_rows, "candidate", paths["candidates"])
    summaries = _unique_by_id(summary_rows, "summary", paths["summaries"])
    successes = _unique_by_id(success_rows, "success", paths["successes"])
    if set(candidates) != set(summaries):
        raise ValueError(f"{spec.source_id}: candidate/summary ID mismatch")
    stable_ids = {key for key, row in summaries.items() if row["stable_success"]}
    if stable_ids != set(successes):
        raise ValueError(f"{spec.source_id}: summary/success manifest mismatch")

    rollouts: dict[str, list[dict]] = defaultdict(list)
    for rollout in rollout_rows:
        if rollout["candidate_id"] not in candidates:
            raise ValueError(f"{spec.source_id}: rollout references unknown candidate")
        rollouts[rollout["candidate_id"]].append(rollout)
    for candidate_id, summary in summaries.items():
        episodes = rollouts[candidate_id]
        actual_successes = sum(bool(row["success"]) for row in episodes)
        if len(episodes) != int(summary["attempt_count"]):
            raise ValueError(f"{spec.source_id}/{candidate_id}: attempt count mismatch")
        if actual_successes != int(summary["success_count"]):
            raise ValueError(f"{spec.source_id}/{candidate_id}: success count mismatch")
        expected_stable = actual_successes >= int(summary["required_successes"])
        if bool(summary["stable_success"]) != expected_stable:
            raise ValueError(f"{spec.source_id}/{candidate_id}: stable decision mismatch")
        _assert_pose_equal(candidates[candidate_id]["base_pose"], summary["base_pose"], "base pose")
        _assert_pose_equal(candidates[candidate_id]["cube_pose"], summary["cube_pose"], "cube pose")

    run_config = read_json(paths["config"])
    hashes = {name: _sha256(path) for name, path in paths.items()}
    source_report = {
        "source_id": spec.source_id,
        "directory": str(directory.relative_to(PROJECT_ROOT)),
        "included_yaw_offsets_deg": list(spec.included_yaws),
        "raw_candidate_count": len(candidate_rows),
        "raw_stable_success_count": len(success_rows),
        "rollout_count": len(rollout_rows),
        "run_config": run_config,
        "sha256": hashes,
    }
    return candidates, summaries, successes, source_report


def _sort_key(row: dict) -> tuple:
    perturbation = row["perturbation"]
    return (
        EDGE_ORDER[(row["scene_id"], row["approach_edge"])],
        NORMAL_OFFSETS.index(float(perturbation["normal_m"])),
        TANGENT_OFFSETS.index(float(perturbation["tangent_m"])),
        REFINED_YAWS.index(float(perturbation["yaw_deg"])),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} is not empty; pass --force to regenerate")

    canonical_candidates: dict[tuple, tuple[dict, dict, str]] = {}
    excluded: list[dict] = []
    source_reports: list[dict] = []
    for spec in SOURCES:
        candidates, summaries, _, source_report = _audit_source(spec, data_root)
        source_reports.append(source_report)
        allowed = {round(value, 6) for value in spec.included_yaws}
        for candidate_id, candidate in candidates.items():
            summary = summaries[candidate_id]
            yaw = round(float(candidate["perturbation"]["yaw_deg"]), 6)
            if yaw not in allowed:
                excluded.append(
                    {
                        "source_id": spec.source_id,
                        "source_candidate_id": candidate_id,
                        "scene_id": candidate["scene_id"],
                        "approach_edge": candidate["approach_edge"],
                        "perturbation": candidate["perturbation"],
                        "stable_success": bool(summary["stable_success"]),
                        "reason": "yaw_not_in_final_refined_grid",
                    }
                )
                continue
            key = _grid_key(candidate)
            if key in canonical_candidates:
                raise ValueError(f"duplicate canonical grid key: {key}")
            canonical_candidates[key] = (candidate, summary, spec.source_id)

    expected_grid = {
        (normal, tangent, yaw)
        for normal in NORMAL_OFFSETS
        for tangent in TANGENT_OFFSETS
        for yaw in REFINED_YAWS
    }
    for scene_edge in EDGE_ORDER:
        actual = {
            key[2:]
            for key in canonical_candidates
            if key[:2] == scene_edge
        }
        if actual != expected_grid:
            raise ValueError(
                f"{scene_edge}: incomplete refined grid; "
                f"missing={sorted(expected_grid - actual)}, extra={sorted(actual - expected_grid)}"
            )
    if len(canonical_candidates) != 1120:
        raise ValueError(f"canonical candidate count is {len(canonical_candidates)}, expected 1120")
    if any(row["stable_success"] for row in excluded):
        raise ValueError("an excluded old-grid candidate is a stable success")

    successful_poses: list[dict] = []
    edge_stats = defaultdict(lambda: {"candidate_count": 0, "stable_success_count": 0})
    for candidate, summary, source_id in canonical_candidates.values():
        edge = (candidate["scene_id"], candidate["approach_edge"])
        edge_stats[edge]["candidate_count"] += 1
        edge_stats[edge]["stable_success_count"] += int(summary["stable_success"])
        if not summary["stable_success"]:
            continue
        base_pose = [float(value) for value in candidate["base_pose"][:3]]
        cube_pose = [float(value) for value in candidate["cube_pose"]]
        if not all(math.isfinite(value) for value in (*base_pose, *cube_pose)):
            raise ValueError(f"non-finite pose for {candidate['candidate_id']}")
        successful_poses.append(
            {
                "schema_version": 1,
                "dataset_id": "d30_v1",
                "m_uid": _manipulation_uid(candidate),
                "stable_success": True,
                "scene_id": candidate["scene_id"],
                "category": candidate["category"],
                "approach_edge": candidate["approach_edge"],
                "cube_edge_distance_m": 0.30,
                "cube_pose": cube_pose,
                "cube_pose_world": cube_pose,
                "base_pose": base_pose,
                "manipulation_pose_world": base_pose,
                "joint1_target": float(candidate["joint1_target"]),
                "perturbation": {
                    key: float(value) for key, value in candidate["perturbation"].items()
                },
                "success_count": int(summary["success_count"]),
                "attempt_count": int(summary["attempt_count"]),
                "max_attempts": int(summary["max_attempts"]),
                "required_successes": int(summary["required_successes"]),
                "decision": summary["decision"],
                "source": {
                    "source_id": source_id,
                    "candidate_id": candidate["candidate_id"],
                },
            }
        )

    successful_poses.sort(key=_sort_key)
    uids = [row["m_uid"] for row in successful_poses]
    duplicates = [uid for uid, count in Counter(uids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate manipulation-pose UIDs: {duplicates}")
    if len(successful_poses) != 291:
        raise ValueError(f"stable success count is {len(successful_poses)}, expected 291")

    output_dir.mkdir(parents=True, exist_ok=True)
    successful_path = output_dir / "successful_poses.jsonl"
    write_jsonl(successful_path, successful_poses)
    write_jsonl(output_dir / "excluded_from_refined_grid.jsonl", excluded)
    write_json(output_dir / "source_runs.json", source_reports)
    statistics = []
    for edge in sorted(EDGE_ORDER, key=EDGE_ORDER.get):
        row = {"scene_id": edge[0], "approach_edge": edge[1], **edge_stats[edge]}
        row["stable_success_fraction"] = (
            row["stable_success_count"] / row["candidate_count"]
        )
        statistics.append(row)
    dataset_info = {
        "schema_version": 1,
        "dataset_id": "d30_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cube_edge_distance_m": 0.30,
        "verification_policy": {
            "maximum_attempts": 5,
            "required_successes": 2,
            "early_stop_after_required_successes": True,
        },
        "candidate_grid": {
            "normal_offsets_m": list(NORMAL_OFFSETS),
            "tangent_offsets_m": list(TANGENT_OFFSETS),
            "yaw_offsets_deg": list(REFINED_YAWS),
        },
        "canonical_candidate_count": len(canonical_candidates),
        "stable_success_count": len(successful_poses),
        "excluded_old_grid_candidate_count": len(excluded),
        "scene_edge_statistics": statistics,
        "successful_poses_sha256": _sha256(successful_path),
    }
    write_json(output_dir / "dataset_info.json", dataset_info)
    print(
        f"[N2M] canonical d30 merge complete: candidates={len(canonical_candidates)} "
        f"stable_successes={len(successful_poses)} excluded={len(excluded)}"
    )
    for row in statistics:
        print(
            f"[N2M] {row['scene_id']}/{row['approach_edge']}: "
            f"{row['stable_success_count']}/{row['candidate_count']} stable"
        )
    print(f"[N2M] output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

