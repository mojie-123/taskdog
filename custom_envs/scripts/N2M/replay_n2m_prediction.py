#!/usr/bin/env python3
"""Replay one N2M-predicted manipulation pose in the validated MuJoCo task.

The input is the JSON produced by ``learning/infer.py``.  By default the
highest-weight Gaussian mean is checked and replayed through ARM_INIT.  A
specific Gaussian mode or saved random sample can also be selected.  Full
ARM_INIT -> LIFT validation uses the same runner and success criteria as the
manipulation-pose data collector.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.config import EpisodeConfig, PROJECT_ROOT
from custom_envs.scripts.N2M.geometry import (
    angular_error,
    joint1_towards_cube,
    transform_se2,
)
from custom_envs.scripts.N2M.interactive import InteractiveController
from custom_envs.scripts.N2M.manifests import (
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
)
from custom_envs.scripts.N2M.manipulation_episode import ManipulationEpisodeRunner
from custom_envs.scripts.N2M.mujoco_state import (
    place_episode_state,
    robot_environment_contacts,
    robot_table_contacts,
)


DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_pose(row: dict, key: str) -> list[float]:
    pose = [float(value) for value in row[key]]
    if len(pose) != 3 or not np.isfinite(pose).all():
        raise ValueError(f"{key} must contain three finite values, got {pose}")
    return pose


def _observation_for_prediction(dataset_dir: Path, prediction: dict) -> dict:
    observation_id = prediction.get("observation_id")
    rows = {
        row["observation_id"]: row
        for row in read_jsonl(dataset_dir / "observations.jsonl")
    }
    if observation_id not in rows:
        raise KeyError(
            f"prediction observation {observation_id!r} is absent from "
            f"{dataset_dir / 'observations.jsonl'}"
        )
    observation = rows[observation_id]
    predicted_observation_pose = np.asarray(
        prediction.get("observation_pose_world", []), dtype=np.float64
    )
    saved_observation_pose = np.asarray(
        observation["observation_pose_world"], dtype=np.float64
    )
    if predicted_observation_pose.shape != (3,):
        raise ValueError("prediction has no valid observation_pose_world")
    position_error = float(
        np.linalg.norm(predicted_observation_pose[:2] - saved_observation_pose[:2])
    )
    yaw_error = abs(
        angular_error(predicted_observation_pose[2], saved_observation_pose[2])
    )
    if position_error > 1e-6 or yaw_error > 1e-6:
        raise ValueError(
            "prediction and dataset observation poses differ: "
            f"position={position_error:.6g}m yaw={math.degrees(yaw_error):.6g}deg"
        )
    return observation


def _select_prediction_pose(
    prediction: dict,
    *,
    mode_index: int | None = None,
    sample_index: int | None = None,
) -> tuple[str, int, dict]:
    if mode_index is not None and sample_index is not None:
        raise ValueError("mode_index and sample_index are mutually exclusive")
    if sample_index is not None:
        samples = prediction.get("samples", [])
        if not 0 <= sample_index < len(samples):
            raise IndexError(
                f"sample index {sample_index} outside [0, {len(samples)})"
            )
        return "sample", int(sample_index), samples[sample_index]

    modes = prediction.get("modes", [])
    if not modes:
        raise ValueError("prediction contains no Gaussian modes")
    if mode_index is None:
        mode_index = max(
            range(len(modes)), key=lambda index: float(modes[index].get("weight", 0.0))
        )
    if not 0 <= mode_index < len(modes):
        raise IndexError(f"mode index {mode_index} outside [0, {len(modes)})")
    return "mode", int(mode_index), modes[mode_index]


def build_prediction_candidate(
    prediction: dict,
    observation: dict,
    *,
    mode_index: int | None = None,
    sample_index: int | None = None,
) -> tuple[dict, dict]:
    selection_kind, selection_index, selected = _select_prediction_pose(
        prediction, mode_index=mode_index, sample_index=sample_index
    )
    relative_pose = _finite_pose(selected, "pose_observation")
    world_pose = _finite_pose(selected, "pose_world")
    rebuilt = transform_se2(
        prediction["observation_pose_world"], relative_pose
    )
    position_error = float(np.linalg.norm(rebuilt[:2] - world_pose[:2]))
    yaw_error = abs(angular_error(rebuilt[2], world_pose[2]))
    if position_error > 1e-5 or yaw_error > 1e-5:
        raise ValueError(
            "selected relative/world poses fail the SE(2) round trip: "
            f"position={position_error:.6g}m yaw={math.degrees(yaw_error):.6g}deg"
        )

    cube_pose = [float(value) for value in observation["cube_pose_world"]]
    observation_id = str(prediction["observation_id"])
    selection_id = f"{selection_kind}{selection_index:04d}"
    candidate_id = f"{observation_id}__n2m_{selection_id}"
    candidate = {
        "candidate_id": candidate_id,
        "scene_id": observation["scene_id"],
        "category": "n2m_prediction",
        "approach_edge": observation["approach_edge"],
        "cube_pose": cube_pose,
        "base_pose": world_pose,
        "joint1_target": float(joint1_towards_cube(world_pose, cube_pose)),
        "source": "n2m_prediction",
        "observation_id": observation_id,
        "selection_kind": selection_kind,
        "selection_index": selection_index,
    }
    selection = {
        "kind": selection_kind,
        "index": selection_index,
        "pose_observation": relative_pose,
        "pose_world": world_pose,
        "weight": selected.get("weight"),
        "covariance": selected.get("covariance"),
    }
    return candidate, selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", type=Path, required=True,
        help="JSON produced by learning/infer.py",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="dataset containing the prediction observation and cube pose",
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--mode-index", type=int,
        help="Gaussian mean to replay; default is the highest-weight mode",
    )
    selection.add_argument(
        "--sample-index", type=int,
        help="saved random sample in the prediction JSON to replay",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="default: <prediction directory>/prediction_replays/<candidate id>",
    )
    parser.add_argument(
        "--stop-after", choices=("arm_init", "lift"), default="arm_init",
        help="ARM_INIT is the safe visual preflight; lift runs the full pipeline",
    )
    parser.add_argument("--grasp-checkpoint")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--required-successes", type=int, default=1)
    parser.add_argument("--wrist-min-object-pixels", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", dest="render", action="store_true")
    display.add_argument("--headless", dest="render", action="store_false")
    parser.set_defaults(render=False)
    parser.add_argument("--pause-on-stage", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stop_after == "lift" and not args.grasp_checkpoint:
        raise SystemExit("--grasp-checkpoint is required with --stop-after lift")
    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")
    if not 1 <= args.required_successes <= args.attempts:
        raise SystemExit("--required-successes must be between 1 and --attempts")
    if args.pause_on_stage and not args.render:
        print("[N2M] NOTE: --pause-on-stage has no effect without --render", flush=True)

    prediction_path = args.prediction.resolve()
    dataset_dir = args.dataset_dir.resolve()
    prediction = read_json(prediction_path)
    observation = _observation_for_prediction(dataset_dir, prediction)
    candidate, selected = build_prediction_candidate(
        prediction,
        observation,
        mode_index=args.mode_index,
        sample_index=args.sample_index,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else prediction_path.parent / "prediction_replays" / candidate["candidate_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EpisodeConfig(
        wrist_min_object_pixels=args.wrist_min_object_pixels,
        max_episode_steps=args.max_episode_steps,
        reject_initial_table_contact=True,
    )
    run_config = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "prediction": str(prediction_path),
        "checkpoint": prediction.get("checkpoint"),
        "dataset_dir": str(dataset_dir),
        "scene": str(args.scene.resolve()),
        "observation_id": prediction["observation_id"],
        "candidate": candidate,
        "selection": selected,
        "stop_after": args.stop_after,
        "attempts": args.attempts,
        "required_successes": args.required_successes,
        "episode_config": config.to_dict(),
    }
    write_json(output_dir / "run_config.json", run_config)
    print(
        f"[N2M][PREDICTION] candidate={candidate['candidate_id']} "
        f"base={np.round(candidate['base_pose'], 6)} "
        f"joint1={candidate['joint1_target']:+.6f}rad",
        flush=True,
    )

    controller = InteractiveController()
    runner = ManipulationEpisodeRunner(
        args.scene,
        config,
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        pause_on_stage=args.pause_on_stage,
        interactive=controller,
        debug_dir=output_dir / "debug",
    )
    try:
        # Use the exact robot roll/pitch, arm state and collision model that the
        # episode will use.  Reject any non-floor environment contact before
        # starting ARM_INIT.
        place_episode_state(
            runner.env,
            candidate["cube_pose"],
            candidate["base_pose"],
            config.robot_root_z,
            config.robot_root_roll,
            config.robot_root_pitch,
        )
        environment_hits = robot_environment_contacts(runner.env)
        table_hits = robot_table_contacts(runner.env)
        preflight = {
            "created_at": _utc_now(),
            "candidate_id": candidate["candidate_id"],
            "environment_contacts": environment_hits,
            "table_contacts": table_hits,
            "passed": not environment_hits,
        }
        write_json(output_dir / "collision_preflight.json", preflight)
        print(
            f"[N2M][COLLISION_PREFLIGHT] passed={not environment_hits} "
            f"contacts={environment_hits}",
            flush=True,
        )
        if environment_hits:
            write_json(
                output_dir / "summary.json",
                {
                    **run_config,
                    "completed_at": _utc_now(),
                    "attempt_count": 0,
                    "success_count": 0,
                    "stable_success": False,
                    "decision": "collision_preflight_failed",
                    "collision_preflight": preflight,
                },
            )
            return 2

        results = []
        for attempt in range(args.attempts):
            if controller.quit:
                break
            result = runner.run_episode(
                candidate, attempt=attempt, stop_after=args.stop_after
            )
            row = result.to_dict()
            row.update(
                {
                    "created_at": _utc_now(),
                    "observation_id": prediction["observation_id"],
                    "selection": selected,
                }
            )
            append_jsonl(output_dir / "rollouts.jsonl", row)
            results.append(result)
            passed = (
                result.failure_reason is None
                if args.stop_after == "arm_init"
                else result.success is True
            )
            print(
                f"[N2M][PREDICTION_REPLAY] attempt={attempt} passed={passed} "
                f"completed={result.completed_stage} "
                f"failure={result.failure_stage}:{result.failure_reason}",
                flush=True,
            )
            success_count = sum(
                int(
                    item.failure_reason is None
                    if args.stop_after == "arm_init"
                    else item.success is True
                )
                for item in results
            )
            if success_count >= args.required_successes:
                break
            if result.failure_stage in {"BASE_PLACED", "ARM_INIT", "OPERATOR", "INTERNAL"}:
                break

        success_count = sum(
            int(
                item.failure_reason is None
                if args.stop_after == "arm_init"
                else item.success is True
            )
            for item in results
        )
        stable = success_count >= args.required_successes
        summary = {
            **run_config,
            "completed_at": _utc_now(),
            "attempt_count": len(results),
            "success_count": success_count,
            "stable_success": stable,
            "decision": (
                "success_threshold_reached"
                if stable
                else "attempt_limit_or_deterministic_failure"
            ),
            "collision_preflight": preflight,
        }
        write_json(output_dir / "summary.json", summary)
        print(
            f"[N2M][PREDICTION_SUMMARY] stable={stable} "
            f"success={success_count}/{len(results)} output={output_dir}",
            flush=True,
        )
        if args.render and not controller.quit:
            controller.wait_for_advance(runner.env, "prediction replay finished")
        return 0 if stable else 1
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
