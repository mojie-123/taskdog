#!/usr/bin/env python3
"""End-to-end N2M test-split evaluation with real MuJoCo manipulation.

The evaluator first loads the N2M checkpoint once and writes one deterministic
prediction for every held-out observation.  It then releases the network/GPU
memory and launches ``adaptive_n2m_recovery.py`` headlessly for each
observation.  Each recovery process gets at most ``--max-m-candidates`` M poses
from that observation's one initial prediction.

Running manipulation in child processes isolates native MuJoCo/GL/ROS cleanup
faults.  A complete per-observation ``summary.json`` is authoritative even if
the child later exits abnormally while destroying native resources.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.config import PROJECT_ROOT
from custom_envs.scripts.N2M.geometry import angular_error
from custom_envs.scripts.N2M.manifests import (
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
)


DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"
ADAPTIVE_SCRIPT = Path(__file__).with_name("adaptive_n2m_recovery.py")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_if_complete(path: Path) -> dict | None:
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _last_rows_by_observation(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        observation_id = row.get("observation_id")
        if observation_id:
            rows[str(observation_id)] = row
    return rows


def _sample_seed(base_seed: int, observation_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{observation_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _select_observations(
    dataset_dir: Path,
    split: str,
    requested_ids: list[str],
    limit: int,
) -> tuple[list[dict], dict]:
    split_path = dataset_dir / "splits" / f"{split}.json"
    split_payload = read_json(split_path)
    split_ids = [str(value) for value in split_payload.get("observation_ids", [])]
    if not split_ids:
        raise RuntimeError(f"split contains no observations: {split_path}")
    observations = {
        str(row["observation_id"]): row
        for row in read_jsonl(dataset_dir / "observations.jsonl")
    }
    missing = [observation_id for observation_id in split_ids if observation_id not in observations]
    if missing:
        raise RuntimeError(f"{len(missing)} split observations are absent from observations.jsonl")

    selected_ids = split_ids
    if requested_ids:
        outside = [value for value in requested_ids if value not in set(split_ids)]
        if outside:
            raise ValueError(f"requested observations are not in split={split!r}: {outside}")
        selected_ids = requested_ids
    if limit:
        selected_ids = selected_ids[:limit]
    rows = [observations[observation_id] for observation_id in selected_ids]
    metadata = {
        "split": split,
        "split_path": str(split_path.resolve()),
        "full_split_observation_count": len(split_ids),
        "selected_observation_count": len(rows),
        "scene_edge_counts": dict(
            sorted(Counter(f"{row['scene_id']}/{row['approach_edge']}" for row in rows).items())
        ),
    }
    return rows, metadata


def _prediction_is_reusable(
    path: Path,
    checkpoint: Path,
    observation_id: str,
    num_samples: int,
) -> bool:
    prediction = _read_json_if_complete(path)
    if prediction is None:
        return False
    try:
        return (
            prediction.get("observation_id") == observation_id
            and Path(prediction.get("checkpoint", "")).resolve() == checkpoint.resolve()
            and len(prediction.get("modes", [])) >= 1
            and len(prediction.get("samples", [])) == num_samples
        )
    except (OSError, TypeError):
        return False


def generate_predictions(
    checkpoint_path: Path,
    dataset_dir: Path,
    observations: list[dict],
    predictions_dir: Path,
    *,
    device_name: str,
    num_samples: int,
    base_seed: int,
    split: str,
    resume: bool,
    inference_manifest: Path,
) -> None:
    """Load N2M once and create a deterministic prediction per observation."""
    pending = []
    for observation in observations:
        path = predictions_dir / f"{observation['observation_id']}.json"
        if not (resume and _prediction_is_reusable(
            path, checkpoint_path, observation["observation_id"], num_samples
        )):
            pending.append(observation)
    if not pending:
        print("[N2M][EVAL_INFER] all predictions already exist", flush=True)
        return

    import torch
    from custom_envs.scripts.N2M.learning.infer import _point_tensor, _pose_rows
    from custom_envs.scripts.N2M.vendor_n2m import N2MNet

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = copy.deepcopy(config["model"])
    model_config["encoder"]["ckpt"] = None
    model = N2MNet(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    point_count = int(config["dataset"].get("num_points", 8192))
    predictions_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[N2M][EVAL_INFER] loading complete; generating {len(pending)} predictions",
        flush=True,
    )

    try:
        for index, observation in enumerate(pending, start=1):
            started = time.monotonic()
            observation_id = str(observation["observation_id"])
            seed = _sample_seed(base_seed, observation_id)
            points = _point_tensor(
                dataset_dir / observation["cloud_file"], point_count, device
            )
            with torch.no_grad():
                means, covariance, weights = model(points)
                generator = torch.Generator(device=device).manual_seed(seed)
                components = torch.multinomial(
                    weights[0], num_samples, replacement=True, generator=generator
                )
                noise = torch.randn(
                    num_samples, 3, device=device, generator=generator,
                    dtype=means.dtype,
                )
                selected_means = means[0][components]
                selected_covariance = covariance[0][components]
                cholesky = torch.linalg.cholesky(selected_covariance)
                samples = selected_means + torch.bmm(
                    cholesky, noise.unsqueeze(-1)
                ).squeeze(-1)
            means_np = means[0].detach().cpu().numpy()
            samples_np = samples.detach().cpu().numpy()
            prediction = {
                "schema_version": 1,
                "checkpoint": str(checkpoint_path.resolve()),
                "observation_id": observation_id,
                "observation_pose_world": observation["observation_pose_world"],
                "evaluation_split": split,
                "sample_seed": seed,
                "modes": [
                    {
                        **pose,
                        "weight": float(weights[0, mode_index].detach().cpu()),
                        "covariance": covariance[0, mode_index].detach().cpu().tolist(),
                    }
                    for mode_index, pose in enumerate(
                        _pose_rows(means_np, observation["observation_pose_world"])
                    )
                ],
                "samples": _pose_rows(samples_np, observation["observation_pose_world"]),
            }
            output_path = predictions_dir / f"{observation_id}.json"
            write_json(output_path, prediction)
            elapsed = time.monotonic() - started
            append_jsonl(
                inference_manifest,
                {
                    "created_at": _utc_now(),
                    "observation_id": observation_id,
                    "prediction": str(output_path.resolve()),
                    "sample_seed": seed,
                    "num_samples": num_samples,
                    "elapsed_s": elapsed,
                },
            )
            print(
                f"[N2M][EVAL_INFER] {index}/{len(pending)} {observation_id} "
                f"time={elapsed:.2f}s",
                flush=True,
            )
            del points, means, covariance, weights, samples
    finally:
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[N2M][EVAL_INFER] model released", flush=True)


def _next_run_dir(output_dir: Path, observation_id: str) -> Path:
    parent = output_dir / "runs" / observation_id
    parent.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_", 1)[1])
        for path in parent.glob("attempt_*")
        if path.is_dir() and path.name.split("_", 1)[1].isdigit()
    ]
    index = max(existing, default=-1) + 1
    return parent / f"attempt_{index:04d}"


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run_recovery_child(command: list[str], log_path: Path, timeout_s: float) -> tuple[int | None, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_s if timeout_s > 0.0 else None)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            return_code = process.returncode
        except KeyboardInterrupt:
            _terminate_process_group(process)
            raise
    return return_code, timed_out, time.monotonic() - started


def _last_failure(m_summaries: list[dict], attempts: list[dict]) -> str | None:
    for row in reversed(m_summaries):
        if row.get("last_failure"):
            return str(row["last_failure"])
    for row in reversed(attempts):
        if row.get("failure_stage"):
            return f"{row.get('failure_stage')}:{row.get('failure_reason')}"
    return None


def extract_result(
    observation: dict,
    prediction_path: Path,
    run_dir: Path,
    *,
    return_code: int | None,
    timed_out: bool,
    wall_time_s: float,
) -> dict:
    summary = _read_json_if_complete(run_dir / "summary.json")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    m_summaries = read_jsonl(run_dir / "m_summaries.jsonl")
    successful_rows = [row for row in m_summaries if row.get("status") == "success"]
    if successful_rows:
        successful_m_rank = int(successful_rows[-1]["queue_index"])
    else:
        successful_attempts = [row for row in attempts if row.get("success") is True]
        successful_m_rank = (
            int(successful_attempts[-1]["queue_index"])
            if successful_attempts else None
        )

    if timed_out:
        status = "timeout"
    elif summary is None:
        status = "missing_summary"
    else:
        status = "valid"
    outcome = summary.get("outcome") if summary else status
    success = bool(summary and summary.get("success") is True)
    expected_exit = 0 if success else 1 if outcome == "candidate_queue_exhausted" else 130 if outcome in {
        "operator_quit", "keyboard_interrupt"
    } else 2
    abnormal_exit_after_summary = bool(
        summary is not None and return_code is not None and return_code != expected_exit
    )
    failure_counts = Counter(
        f"{row.get('failure_stage')}:{row.get('failure_reason')}"
        for row in attempts
        if row.get("success") is not True and row.get("failure_stage")
    )
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "observation_id": observation["observation_id"],
        "scene_id": observation["scene_id"],
        "approach_edge": observation["approach_edge"],
        "prediction": str(prediction_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "evaluation_status": status,
        "outcome": outcome,
        "success": success,
        "successful_m_rank_zero_based": successful_m_rank,
        "successful_m_number_one_based": (
            successful_m_rank + 1 if successful_m_rank is not None else None
        ),
        "attempt_count": int(summary.get("attempt_count", len(attempts))) if summary else len(attempts),
        "attempted_m_count": int(summary.get("attempted_m_count", 0)) if summary else 0,
        "collision_rejected_m_count": int(summary.get("collision_rejected_m_count", 0)) if summary else 0,
        "last_failure": _last_failure(m_summaries, attempts),
        "attempt_failure_counts": dict(sorted(failure_counts.items())),
        "process_return_code": return_code,
        "abnormal_exit_after_summary": abnormal_exit_after_summary,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "initial_cube_pose": summary.get("initial_cube_pose") if summary else None,
        "final_cube_pose": summary.get("final_cube_pose") if summary else None,
    }


def _group_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("evaluation_status") == "valid"]
    success = [row for row in valid if row.get("success") is True]
    return {
        "total": len(rows),
        "valid": len(valid),
        "successes": len(success),
        "success_rate_valid": len(success) / len(valid) if valid else None,
        "success_rate_all": len(success) / len(rows) if rows else None,
        "mean_attempts": float(np.mean([row["attempt_count"] for row in valid])) if valid else None,
        "mean_attempted_m": float(np.mean([row["attempted_m_count"] for row in valid])) if valid else None,
    }


def build_report(
    rows: list[dict],
    planned_observations: list[dict],
    config: dict,
    *,
    stopped_early_reason: str | None = None,
) -> dict:
    by_id = {str(row["observation_id"]): row for row in rows}
    ordered_rows = [
        by_id[str(observation["observation_id"])]
        for observation in planned_observations
        if str(observation["observation_id"]) in by_id
    ]
    valid = [row for row in ordered_rows if row.get("evaluation_status") == "valid"]
    successes = [row for row in valid if row.get("success") is True]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ordered_rows:
        grouped[(row["scene_id"], row["approach_edge"])].append(row)
    failure_counts: Counter[str] = Counter()
    last_failure_counts: Counter[str] = Counter()
    for row in ordered_rows:
        failure_counts.update(row.get("attempt_failure_counts", {}))
        if not row.get("success") and row.get("last_failure"):
            last_failure_counts[str(row["last_failure"])] += 1

    max_m = int(config["max_m_candidates"])
    cumulative_success = {}
    for one_based in range(1, max_m + 1):
        count = sum(
            row.get("successful_m_number_one_based") is not None
            and int(row["successful_m_number_one_based"]) <= one_based
            for row in successes
        )
        cumulative_success[str(one_based)] = {
            "successes": count,
            "rate_of_valid": count / len(valid) if valid else None,
            "rate_of_planned": count / len(planned_observations) if planned_observations else None,
        }
    return {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "split": config["split"],
        "planned_observation_count": len(planned_observations),
        "evaluated_observation_count": len(ordered_rows),
        "pending_observation_count": len(planned_observations) - len(ordered_rows),
        "valid_result_count": len(valid),
        "invalid_result_count": len(ordered_rows) - len(valid),
        "success_count": len(successes),
        "success_rate_valid": len(successes) / len(valid) if valid else None,
        "success_rate_planned": len(successes) / len(planned_observations) if planned_observations else None,
        "cumulative_success_by_m_number": cumulative_success,
        "outcome_counts": dict(sorted(Counter(str(row.get("outcome")) for row in ordered_rows).items())),
        "evaluation_status_counts": dict(sorted(Counter(str(row.get("evaluation_status")) for row in ordered_rows).items())),
        "abnormal_exit_after_summary_count": sum(bool(row.get("abnormal_exit_after_summary")) for row in ordered_rows),
        "attempt_count_total": sum(int(row.get("attempt_count", 0)) for row in ordered_rows),
        "attempted_m_count_total": sum(int(row.get("attempted_m_count", 0)) for row in ordered_rows),
        "collision_rejected_m_count_total": sum(int(row.get("collision_rejected_m_count", 0)) for row in ordered_rows),
        "wall_time_s_total": sum(float(row.get("wall_time_s", 0.0)) for row in ordered_rows),
        "attempt_failure_counts": dict(sorted(failure_counts.items())),
        "unsuccessful_last_failure_counts": dict(sorted(last_failure_counts.items())),
        "by_scene_edge": {
            f"{scene_id}/{edge}": _group_summary(group_rows)
            for (scene_id, edge), group_rows in sorted(grouped.items())
        },
        "stopped_early_reason": stopped_early_reason,
        "config": config,
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def write_reports(
    output_dir: Path,
    rows: list[dict],
    planned_observations: list[dict],
    config: dict,
    *,
    stopped_early_reason: str | None = None,
) -> dict:
    report = build_report(
        rows, planned_observations, config, stopped_early_reason=stopped_early_reason
    )
    write_json(output_dir / "report.json", report)

    by_id = {str(row["observation_id"]): row for row in rows}
    ordered = [
        by_id[str(observation["observation_id"])]
        for observation in planned_observations
        if str(observation["observation_id"]) in by_id
    ]
    csv_fields = [
        "observation_id", "scene_id", "approach_edge", "evaluation_status",
        "outcome", "success", "successful_m_number_one_based", "attempt_count",
        "attempted_m_count", "collision_rejected_m_count", "last_failure",
        "process_return_code", "abnormal_exit_after_summary", "timed_out",
        "wall_time_s", "prediction", "run_dir",
    ]
    csv_path = output_dir / "report_observations.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(csv_path)

    group_path = output_dir / "report_by_scene_edge.csv"
    temporary = group_path.with_name(group_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "scene_edge", "total", "valid", "successes", "success_rate_valid",
            "success_rate_all", "mean_attempts", "mean_attempted_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, value in report["by_scene_edge"].items():
            writer.writerow({"scene_edge": key, **value})
    temporary.replace(group_path)

    lines = [
        "# N2M MuJoCo test-set evaluation",
        "",
        f"- Updated: `{report['updated_at']}`",
        f"- Split: `{report['split']}`",
        f"- Planned observations: **{report['planned_observation_count']}**",
        f"- Evaluated: **{report['evaluated_observation_count']}**",
        f"- Valid results: **{report['valid_result_count']}**",
        f"- Successful LIFT: **{report['success_count']}**",
        f"- Success rate (valid): **{_percent(report['success_rate_valid'])}**",
        f"- Success rate (all planned): **{_percent(report['success_rate_planned'])}**",
        f"- Total grasp attempts: **{report['attempt_count_total']}**",
        f"- Native cleanup anomalies after summary: **{report['abnormal_exit_after_summary_count']}**",
    ]
    if stopped_early_reason:
        lines.extend(("", f"> Evaluation stopped early: `{stopped_early_reason}`"))
    lines.extend(("", "## Cumulative success by allowed M", "", "| Max M | Successes | Valid rate | Planned rate |", "|---:|---:|---:|---:|"))
    for number, value in report["cumulative_success_by_m_number"].items():
        lines.append(
            f"| {number} | {value['successes']} | {_percent(value['rate_of_valid'])} | "
            f"{_percent(value['rate_of_planned'])} |"
        )
    lines.extend(("", "## Results by scene and edge", "", "| Scene/edge | Valid | Success | Valid rate | Mean attempts |", "|---|---:|---:|---:|---:|"))
    for key, value in report["by_scene_edge"].items():
        mean_attempts = "N/A" if value["mean_attempts"] is None else f"{value['mean_attempts']:.2f}"
        lines.append(
            f"| {key} | {value['valid']}/{value['total']} | {value['successes']} | "
            f"{_percent(value['success_rate_valid'])} | {mean_attempts} |"
        )
    lines.extend(("", "## Attempt failure reasons", ""))
    if report["attempt_failure_counts"]:
        lines.extend(("| Failure | Count |", "|---|---:|"))
        for reason, count in sorted(
            report["attempt_failure_counts"].items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No failed attempts recorded yet.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grasp-checkpoint", required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-m-candidates", type=int, default=3)
    parser.add_argument("--attempts-per-m", type=int, default=3)
    parser.add_argument("--min-m-xy", type=float, default=0.08)
    parser.add_argument("--min-m-yaw-deg", type=float, default=10.0)
    parser.add_argument("--observation-timeout-s", type=float, default=3600.0)
    parser.add_argument("--max-consecutive-system-errors", type=int, default=3)
    parser.add_argument("--observation-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.max_m_candidates < 1:
        raise SystemExit("--max-m-candidates must be >= 1")
    if args.attempts_per_m < 1:
        raise SystemExit("--attempts-per-m must be >= 1")
    if args.limit < 0 or args.max_consecutive_system_errors < 0:
        raise SystemExit("--limit and --max-consecutive-system-errors must be >= 0")

    checkpoint_path = args.checkpoint.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = output_dir / "evaluation_config.json"
    if config_path.exists() and not (args.resume or args.report_only):
        raise SystemExit(f"output already contains an evaluation; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, split_metadata = _select_observations(
        dataset_dir, args.split, args.observation_id, args.limit
    )
    config = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "pairs_sha256": _sha256(dataset_dir / "pairs.jsonl"),
        "split_sha256": _sha256(dataset_dir / "splits" / f"{args.split}.json"),
        "output_dir": str(output_dir),
        "scene": str(args.scene.resolve()),
        "grasp_checkpoint": str(Path(args.grasp_checkpoint).resolve()),
        "split": args.split,
        "device": args.device,
        "num_samples": args.num_samples,
        "max_m_candidates": args.max_m_candidates,
        "attempts_per_m": args.attempts_per_m,
        "min_m_xy": args.min_m_xy,
        "min_m_yaw_deg": args.min_m_yaw_deg,
        "observation_timeout_s": args.observation_timeout_s,
        "max_consecutive_system_errors": args.max_consecutive_system_errors,
        "sample_seed": 42,
        "headless": True,
        "selected_observation_ids": [row["observation_id"] for row in observations],
        "split_metadata": split_metadata,
    }
    if config_path.exists():
        existing = read_json(config_path)
        immutable_keys = (
            "checkpoint_sha256", "pairs_sha256", "split_sha256", "split",
            "max_m_candidates", "attempts_per_m", "num_samples",
            "selected_observation_ids",
        )
        mismatches = [key for key in immutable_keys if existing.get(key) != config.get(key)]
        if mismatches and not args.report_only:
            raise SystemExit(f"resume config mismatch for keys: {mismatches}")
        config = existing
    else:
        write_json(config_path, config)
    write_json(
        output_dir / "test_observations.json",
        {"metadata": split_metadata, "observations": observations},
    )

    results_path = output_dir / "results.jsonl"
    existing_results = _last_rows_by_observation(results_path)
    if args.report_only:
        report = write_reports(output_dir, list(existing_results.values()), observations, config)
        print(f"[N2M][EVAL_REPORT] {output_dir / 'report.md'}", flush=True)
        return 0 if report["pending_observation_count"] == 0 else 1

    predictions_dir = output_dir / "predictions"
    base_seed = int(config.get("sample_seed", 42))
    generate_predictions(
        checkpoint_path,
        dataset_dir,
        observations,
        predictions_dir,
        device_name=args.device,
        num_samples=args.num_samples,
        base_seed=base_seed,
        split=args.split,
        resume=args.resume,
        inference_manifest=output_dir / "inference_manifest.jsonl",
    )
    if args.inference_only:
        write_reports(output_dir, list(existing_results.values()), observations, config)
        print("[N2M][EVAL] inference-only stage complete", flush=True)
        return 0

    consecutive_system_errors = 0
    stopped_early_reason = None
    for index, observation in enumerate(observations, start=1):
        observation_id = str(observation["observation_id"])
        if (
            args.resume
            and observation_id in existing_results
            and existing_results[observation_id].get("evaluation_status") == "valid"
        ):
            print(
                f"[N2M][EVAL_SKIP] {index}/{len(observations)} {observation_id} "
                f"outcome={existing_results[observation_id].get('outcome')}",
                flush=True,
            )
            continue
        prediction_path = predictions_dir / f"{observation_id}.json"
        run_dir = _next_run_dir(output_dir, observation_id)
        command = [
            sys.executable,
            str(ADAPTIVE_SCRIPT),
            "--prediction", str(prediction_path),
            "--dataset-dir", str(dataset_dir),
            "--scene", str(args.scene.resolve()),
            "--grasp-checkpoint", str(Path(args.grasp_checkpoint).resolve()),
            "--attempts-per-m", str(args.attempts_per_m),
            "--max-m-candidates", str(args.max_m_candidates),
            "--min-m-xy", str(args.min_m_xy),
            "--min-m-yaw-deg", str(args.min_m_yaw_deg),
            "--output-dir", str(run_dir),
            "--headless",
        ]
        print(
            f"[N2M][EVAL_START] {index}/{len(observations)} {observation_id} "
            f"scene={observation['scene_id']}/{observation['approach_edge']}",
            flush=True,
        )
        return_code, timed_out, wall_time = _run_recovery_child(
            command, run_dir / "run.log", args.observation_timeout_s
        )
        row = extract_result(
            observation,
            prediction_path,
            run_dir,
            return_code=return_code,
            timed_out=timed_out,
            wall_time_s=wall_time,
        )
        append_jsonl(results_path, row)
        existing_results[observation_id] = row
        system_error = (
            row["evaluation_status"] != "valid"
            or row["outcome"] in {"fatal_failure", "transition_failed", "retraction_failed"}
        )
        consecutive_system_errors = consecutive_system_errors + 1 if system_error else 0
        report = write_reports(
            output_dir, list(existing_results.values()), observations, config
        )
        print(
            f"[N2M][EVAL_DONE] {index}/{len(observations)} {observation_id} "
            f"success={row['success']} outcome={row['outcome']} "
            f"m={row['successful_m_number_one_based']} attempts={row['attempt_count']} "
            f"time={wall_time:.1f}s aggregate="
            f"{report['success_count']}/{report['valid_result_count']}",
            flush=True,
        )
        if (
            args.max_consecutive_system_errors
            and consecutive_system_errors >= args.max_consecutive_system_errors
        ):
            stopped_early_reason = (
                f"{consecutive_system_errors} consecutive system errors; "
                "inspect the latest run.log and MoveIt service"
            )
            print(f"[N2M][EVAL_STOP] {stopped_early_reason}", flush=True)
            break

    report = write_reports(
        output_dir,
        list(existing_results.values()),
        observations,
        config,
        stopped_early_reason=stopped_early_reason,
    )
    print(
        f"[N2M][EVAL_SUMMARY] success={report['success_count']}/"
        f"{report['valid_result_count']} pending={report['pending_observation_count']} "
        f"report={output_dir / 'report.md'}",
        flush=True,
    )
    return 0 if report["pending_observation_count"] == 0 and not stopped_early_reason else 2


if __name__ == "__main__":
    raise SystemExit(main())
