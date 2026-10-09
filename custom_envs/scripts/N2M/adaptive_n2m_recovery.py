#!/usr/bin/env python3
"""Try a ranked queue of manipulation poses from one initial N2M inference.

The prediction JSON is loaded exactly once.  Its Gaussian means and saved
samples are ranked by the density of that same initial prediction, then
de-duplicated in SE(2).  After a grasp failure the robot may move to the next
predicted M, but the cube is never reset and N2M is never run again.  AnyGrasp
is intentionally run afresh for every grasp attempt.
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
    rpy_to_quat_wxyz,
)
from custom_envs.scripts.N2M.interactive import InteractiveController
from custom_envs.scripts.N2M.manifests import append_jsonl, read_json, write_json
from custom_envs.scripts.N2M.manipulation_episode import ARM_HOME, EpisodeFailure
from custom_envs.scripts.N2M.mujoco_state import (
    robot_environment_contacts,
    robot_table_contacts,
)
from custom_envs.scripts.N2M.replay_n2m_prediction import (
    _finite_pose,
    _observation_for_prediction,
    build_prediction_candidate,
)
from custom_envs.scripts.N2M.retry_n2m_grasp_until_success import (
    PersistentGraspRunner,
    pose_motion,
)


DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def gmm_log_density(relative_pose, modes: list[dict]) -> float:
    """Evaluate one SE(2) pose under the initial N2M Gaussian mixture."""
    pose = np.asarray(relative_pose, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid relative pose: {relative_pose}")
    if not modes:
        raise ValueError("prediction contains no Gaussian modes")

    terms = []
    dimension = 3
    for index, mode in enumerate(modes):
        weight = float(mode.get("weight", 0.0))
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        mean = np.asarray(_finite_pose(mode, "pose_observation"), dtype=np.float64)
        covariance = np.asarray(mode.get("covariance"), dtype=np.float64)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            raise ValueError(f"mode {index} has no valid 3x3 covariance")
        covariance = 0.5 * (covariance + covariance.T)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            raise ValueError(f"mode {index} covariance is not positive definite")
        difference = pose - mean
        difference[2] = angular_error(pose[2], mean[2])
        mahalanobis = float(difference @ np.linalg.solve(covariance, difference))
        terms.append(
            math.log(weight)
            - 0.5 * (dimension * math.log(2.0 * math.pi) + logdet + mahalanobis)
        )
    if not terms:
        raise ValueError("prediction contains no positive-weight Gaussian modes")
    return _logsumexp(np.asarray(terms, dtype=np.float64))


def build_candidate_queue(
    prediction: dict,
    observation: dict,
    *,
    min_xy_m: float = 0.08,
    min_yaw_rad: float = math.radians(10.0),
    max_candidates: int = 0,
) -> tuple[list[dict], dict]:
    """Rank means/samples by initial density and greedily suppress near copies.

    A pose is a near copy only when *both* its XY and yaw differences are below
    the thresholds.  Consequently a candidate survives when it provides a
    meaningful translation or a meaningful change in facing direction.
    """
    if min_xy_m < 0.0 or min_yaw_rad < 0.0:
        raise ValueError("candidate separation thresholds must be non-negative")
    if max_candidates < 0:
        raise ValueError("max_candidates must be >= 0")

    raw = []
    for index, row in enumerate(prediction.get("modes", [])):
        relative = _finite_pose(row, "pose_observation")
        raw.append(
            {
                "kind": "mode",
                "index": index,
                "pose_observation": relative,
                "log_density": gmm_log_density(relative, prediction["modes"]),
            }
        )
    for index, row in enumerate(prediction.get("samples", [])):
        relative = _finite_pose(row, "pose_observation")
        raw.append(
            {
                "kind": "sample",
                "index": index,
                "pose_observation": relative,
                "log_density": gmm_log_density(relative, prediction["modes"]),
            }
        )
    # A Gaussian mean wins exact ties.  For K=1 it is also the unique density
    # maximum and therefore becomes the first M.
    raw.sort(
        key=lambda row: (
            -row["log_density"],
            0 if row["kind"] == "mode" else 1,
            row["index"],
        )
    )

    accepted = []
    suppressed = []
    accepted_poses: list[np.ndarray] = []
    for row in raw:
        pose = np.asarray(row["pose_observation"], dtype=np.float64)
        near = None
        for accepted_index, other in enumerate(accepted_poses):
            xy = float(np.linalg.norm(pose[:2] - other[:2]))
            yaw = abs(angular_error(pose[2], other[2]))
            if xy < min_xy_m and yaw < min_yaw_rad:
                near = {
                    "accepted_queue_index": accepted_index,
                    "xy_m": xy,
                    "yaw_rad": yaw,
                }
                break
        if near is not None:
            suppressed.append({**row, "near": near})
            continue

        if row["kind"] == "mode":
            candidate, selection = build_prediction_candidate(
                prediction, observation, mode_index=row["index"]
            )
        else:
            candidate, selection = build_prediction_candidate(
                prediction, observation, sample_index=row["index"]
            )
        accepted.append(
            {
                "queue_index": len(accepted),
                "candidate": candidate,
                "selection": selection,
                "initial_log_density": row["log_density"],
            }
        )
        accepted_poses.append(pose)
        if max_candidates and len(accepted) >= max_candidates:
            break

    if not accepted:
        raise ValueError("no N2M candidates survived queue construction")
    diagnostics = {
        "raw_candidate_count": len(raw),
        "accepted_candidate_count": len(accepted),
        "suppressed_near_duplicate_count": len(suppressed),
        "truncated": bool(max_candidates and len(accepted) >= max_candidates),
        "min_xy_m": min_xy_m,
        "min_yaw_rad": min_yaw_rad,
        "suppressed": suppressed,
    }
    return accepted, diagnostics


class AdaptiveRecoveryRunner(PersistentGraspRunner):
    """Persistent runner that can reposition only the robot base."""

    def retract_for_reposition(self) -> None:
        self._detach_safely()
        self.env.disable_gripper_force_hold()
        current = self._arm_q()
        retract = ARM_HOME.copy()
        # First retract without sweeping joint1 across the table.  run_arm_init
        # at the next M will rotate joint1 toward the cube after the base moves.
        retract[0] = current[0]
        self._run_arm_target(retract, "RECOVERY_ARM_RETRACTED")

    def reposition_base_preserving_cube(self, candidate: dict) -> dict:
        before = self.current_object_pose()
        self.weld.disable()
        self.env.set_root_pose(
            np.asarray(
                [candidate["base_pose"][0], candidate["base_pose"][1], self.cfg.robot_root_z],
                dtype=np.float64,
            ),
            rpy_to_quat_wxyz(
                self.cfg.robot_root_roll,
                self.cfg.robot_root_pitch,
                candidate["base_pose"][2],
            ),
        )
        after = self.current_object_pose()
        translation, rotation = pose_motion(before, after)
        if translation > 1e-8 or rotation > 1e-8:
            raise EpisodeFailure(
                "BASE_REPOSITIONED",
                "cube_pose_changed_while_repositioning_base",
                cube_motion_m=translation,
                cube_rotation_rad=rotation,
            )
        live_candidate = dict(candidate)
        live_candidate["cube_pose"] = after
        self.current_candidate = live_candidate
        self.joint1_target = float(
            joint1_towards_cube(candidate["base_pose"], after)
        )
        self._update_overlay()
        environment_hits = robot_environment_contacts(self.env)
        table_hits = robot_table_contacts(self.env)
        self._checkpoint(
            "BASE_REPOSITIONED",
            f"candidate={candidate['candidate_id']} "
            f"base={np.round(candidate['base_pose'], 4)} "
            f"cube_motion={translation * 1000.0:.6f}mm "
            f"contacts={environment_hits}",
        )
        return {
            "cube_pose_before": before,
            "cube_pose_after": after,
            "cube_motion_m": translation,
            "cube_rotation_rad": rotation,
            "environment_contacts": environment_hits,
            "table_contacts": table_hits,
            "passed": not environment_hits and not table_hits,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--grasp-checkpoint", required=True)
    parser.add_argument(
        "--attempts-per-m",
        type=int,
        default=3,
        help="fresh AnyGrasp attempts before advancing to the next M (default: 3)",
    )
    parser.add_argument(
        "--max-m-candidates",
        type=int,
        default=0,
        help="0 uses every de-duplicated candidate in the saved prediction",
    )
    parser.add_argument("--min-m-xy", type=float, default=0.08)
    parser.add_argument("--min-m-yaw-deg", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <prediction directory>/adaptive_n2m_recovery/<observation>",
    )
    parser.add_argument("--wrist-min-object-pixels", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", dest="render", action="store_true")
    display.add_argument("--headless", dest="render", action="store_false")
    parser.set_defaults(render=True)
    parser.add_argument("--pause-on-stage", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.attempts_per_m < 1:
        raise SystemExit("--attempts-per-m must be >= 1")
    if args.max_m_candidates < 0:
        raise SystemExit("--max-m-candidates must be >= 0")
    if args.min_m_xy < 0.0 or args.min_m_yaw_deg < 0.0:
        raise SystemExit("M separation thresholds must be non-negative")
    if args.pause_on_stage and not args.render:
        print("[N2M] NOTE: --pause-on-stage has no effect without --render", flush=True)

    prediction_path = args.prediction.resolve()
    dataset_dir = args.dataset_dir.resolve()
    prediction = read_json(prediction_path)
    observation = _observation_for_prediction(dataset_dir, prediction)
    queue, queue_diagnostics = build_candidate_queue(
        prediction,
        observation,
        min_xy_m=args.min_m_xy,
        min_yaw_rad=math.radians(args.min_m_yaw_deg),
        max_candidates=args.max_m_candidates,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else prediction_path.parent
        / "adaptive_n2m_recovery"
        / str(prediction["observation_id"])
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
        "n2m_inference_count": 1,
        "reuse_initial_n2m_prediction_after_cube_motion": True,
        "reset_cube_between_attempts": False,
        "reset_cube_between_m_candidates": False,
        "rerun_anygrasp_every_attempt": True,
        "attempts_per_m": args.attempts_per_m,
        "max_m_candidates": args.max_m_candidates,
        "queue_diagnostics": {
            key: value
            for key, value in queue_diagnostics.items()
            if key != "suppressed"
        },
        "episode_config": config.to_dict(),
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(
        output_dir / "candidate_queue.json",
        {"queue": queue, "diagnostics": queue_diagnostics},
    )
    print(
        f"[N2M][ADAPTIVE_START] observation={prediction['observation_id']} "
        f"initial_inference=1 queue={len(queue)} attempts_per_m={args.attempts_per_m}",
        flush=True,
    )

    controller = InteractiveController()
    runner = AdaptiveRecoveryRunner(
        args.scene,
        config,
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        pause_on_stage=args.pause_on_stage,
        interactive=controller,
        debug_dir=output_dir / "debug" / "initial",
    )
    outcome = "not_started"
    return_code = 1
    attempt_count = 0
    attempted_m_count = 0
    collision_rejected_count = 0
    successful_candidate = None
    initial_cube_pose = None
    needs_retraction = False
    fatal = False

    try:
        for queue_index, queue_row in enumerate(queue):
            if controller.quit:
                outcome = "operator_quit"
                return_code = 130
                break
            candidate = queue_row["candidate"]
            selection = queue_row["selection"]
            runner.debug_dir = output_dir / "debug" / f"m_{queue_index:04d}" / "transition"

            if queue_index == 0:
                # This is the only full MuJoCo reset and the only cube placement.
                table_hits = runner.reset_candidate(candidate)
                initial_cube_pose = runner.current_object_pose()
                environment_hits = robot_environment_contacts(runner.env)
                preflight = {
                    "cube_pose_before": initial_cube_pose,
                    "cube_pose_after": initial_cube_pose,
                    "cube_motion_m": 0.0,
                    "cube_rotation_rad": 0.0,
                    "environment_contacts": environment_hits,
                    "table_contacts": table_hits,
                    "passed": not environment_hits and not table_hits,
                }
            else:
                if needs_retraction:
                    # X means "leave this M", not "abort the recovery
                    # retraction".  Q remains latched and was handled above.
                    controller.clear_candidate_abort()
                    try:
                        runner.retract_for_reposition()
                    except EpisodeFailure as failure:
                        outcome = "retraction_failed"
                        return_code = 2
                        fatal = True
                        append_jsonl(
                            output_dir / "m_summaries.jsonl",
                            {
                                "created_at": _utc_now(),
                                "queue_index": queue_index,
                                "candidate_id": candidate["candidate_id"],
                                "status": "transition_failed",
                                "failure_stage": failure.stage,
                                "failure_reason": failure.reason,
                                "metrics": failure.metrics,
                            },
                        )
                        print(
                            f"[N2M][ADAPTIVE_STOP] retraction failed: {failure}",
                            flush=True,
                        )
                        break
                    needs_retraction = False
                preflight = runner.reposition_base_preserving_cube(candidate)

            preflight_row = {
                "created_at": _utc_now(),
                "queue_index": queue_index,
                "candidate_id": candidate["candidate_id"],
                "base_pose": candidate["base_pose"],
                "selection": selection,
                "initial_log_density": queue_row["initial_log_density"],
                **preflight,
            }
            append_jsonl(output_dir / "collision_preflights.jsonl", preflight_row)
            if not preflight["passed"]:
                collision_rejected_count += 1
                append_jsonl(
                    output_dir / "m_summaries.jsonl",
                    {**preflight_row, "status": "collision_rejected", "attempts": 0},
                )
                print(
                    f"[N2M][M_REJECTED] rank={queue_index} "
                    f"candidate={candidate['candidate_id']} contacts="
                    f"{preflight['environment_contacts']}",
                    flush=True,
                )
                continue

            runner.weld.enable()
            attempted_m_count += 1
            m_success = False
            m_attempts = 0
            m_last_failure = None
            m_status = "attempts_exhausted"
            print(
                f"[N2M][M_START] rank={queue_index} candidate={candidate['candidate_id']} "
                f"base={np.round(candidate['base_pose'], 6)} "
                f"log_density={queue_row['initial_log_density']:.4f}",
                flush=True,
            )

            for local_attempt in range(args.attempts_per_m):
                if controller.quit:
                    outcome = "operator_quit"
                    return_code = 130
                    fatal = True
                    m_status = "operator_quit"
                    break
                runner.debug_dir = (
                    output_dir
                    / "debug"
                    / f"m_{queue_index:04d}"
                    / f"attempt_{local_attempt:04d}"
                )
                print(
                    f"[N2M][ADAPTIVE_ATTEMPT] global={attempt_count} "
                    f"m_rank={queue_index} local={local_attempt} "
                    f"cube={np.round(runner.current_object_pose()[:3], 6)}",
                    flush=True,
                )
                result = runner.run_preserved_attempt(candidate, attempt_count)
                row = result.to_dict()
                row.update(
                    {
                        "created_at": _utc_now(),
                        "observation_id": prediction["observation_id"],
                        "n2m_inference_count": 1,
                        "queue_index": queue_index,
                        "attempt_within_m": local_attempt,
                        "selection": selection,
                        "initial_log_density": queue_row["initial_log_density"],
                    }
                )
                append_jsonl(output_dir / "attempts.jsonl", row)
                attempt_count += 1
                m_attempts += 1
                needs_retraction = True
                m_last_failure = (
                    None
                    if result.success
                    else f"{result.failure_stage}:{result.failure_reason}"
                )
                print(
                    f"[N2M][ADAPTIVE_DONE] global={attempt_count - 1} "
                    f"m_rank={queue_index} success={result.success} "
                    f"completed={result.completed_stage} failure={m_last_failure} "
                    f"cube_motion={result.metrics['cube_motion_m'] * 1000.0:.2f}mm",
                    flush=True,
                )
                if result.success:
                    m_success = True
                    m_status = "success"
                    successful_candidate = candidate["candidate_id"]
                    outcome = "lift_success"
                    return_code = 0
                    break
                if result.failure_stage == "OPERATOR":
                    if controller.quit:
                        outcome = "operator_quit"
                        return_code = 130
                        fatal = True
                        m_status = "operator_quit"
                    else:
                        m_status = "operator_skipped"
                    # X skips the rest of this M and advances to the next one.
                    break
                if result.failure_stage in {"SAFETY", "INTERNAL", "EPISODE"}:
                    outcome = "fatal_failure"
                    return_code = 2
                    fatal = True
                    m_status = "fatal_failure"
                    break

            append_jsonl(
                output_dir / "m_summaries.jsonl",
                {
                    "created_at": _utc_now(),
                    "queue_index": queue_index,
                    "candidate_id": candidate["candidate_id"],
                    "base_pose": candidate["base_pose"],
                    "selection": selection,
                    "initial_log_density": queue_row["initial_log_density"],
                    "status": m_status,
                    "attempts": m_attempts,
                    "last_failure": m_last_failure,
                    "cube_pose_after": runner.current_object_pose(),
                },
            )
            if m_success or fatal:
                break
        else:
            outcome = "candidate_queue_exhausted"
            return_code = 1
    except KeyboardInterrupt:
        outcome = "keyboard_interrupt"
        return_code = 130
    except EpisodeFailure as failure:
        outcome = "transition_failed"
        return_code = 2
        print(f"[N2M][ADAPTIVE_STOP] {failure}", flush=True)
    finally:
        final_cube_pose = None
        try:
            final_cube_pose = runner.current_object_pose()
        except Exception:
            pass
        summary = {
            **run_config,
            "completed_at": _utc_now(),
            "outcome": outcome,
            "success": outcome == "lift_success",
            "successful_candidate": successful_candidate,
            "attempt_count": attempt_count,
            "attempted_m_count": attempted_m_count,
            "collision_rejected_m_count": collision_rejected_count,
            "initial_cube_pose": initial_cube_pose,
            "final_cube_pose": final_cube_pose,
        }
        write_json(output_dir / "summary.json", summary)
        print(
            f"[N2M][ADAPTIVE_SUMMARY] outcome={outcome} attempts={attempt_count} "
            f"attempted_m={attempted_m_count} collision_rejected="
            f"{collision_rejected_count} output={output_dir}",
            flush=True,
        )
        runner.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
