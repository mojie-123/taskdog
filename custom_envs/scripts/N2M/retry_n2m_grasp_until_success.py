#!/usr/bin/env python3
"""Retry grasping one N2M-predicted M without resetting the disturbed cube.

The scene is initialized exactly once.  Each retry retracts and redeploys the
arm, captures a fresh wrist RGB-D scan, runs AnyGrasp again, and executes the
full pipeline through LIFT.  A failed attempt therefore leaves the cube at its
current MuJoCo pose for the next attempt.  The loop stops after a successful
LIFT, when the operator presses Q, on a safety/internal failure, or at an
optional finite attempt limit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.config import EpisodeConfig, PROJECT_ROOT
from custom_envs.scripts.N2M.interactive import InteractiveController
from custom_envs.scripts.N2M.manifests import append_jsonl, read_json, write_json
from custom_envs.scripts.N2M.manipulation_episode import (
    EpisodeFailure,
    ManipulationEpisodeRunner,
)
from custom_envs.scripts.N2M.mujoco_state import (
    robot_environment_contacts,
    robot_table_contacts,
)
from custom_envs.scripts.N2M.replay_n2m_prediction import (
    _observation_for_prediction,
    build_prediction_candidate,
)


DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quaternion_angle_rad(first, second) -> float:
    """Return the sign-invariant angular distance between wxyz quaternions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def pose_motion(before, after) -> tuple[float, float]:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    return (
        float(np.linalg.norm(after[:3] - before[:3])),
        quaternion_angle_rad(before[3:7], after[3:7]),
    )


class PersistentGraspRunner(ManipulationEpisodeRunner):
    """Manipulation runner whose retry method deliberately skips env.reset()."""

    def current_object_pose(self) -> list[float]:
        import mujoco

        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.object_name
        )
        if body_id < 0:
            raise RuntimeError(f"body not found: {self.cfg.object_name}")
        position = self.env.data.xpos[body_id].astype(np.float64).copy()
        quaternion = self.env.data.xquat[body_id].astype(np.float64).copy()
        return np.concatenate([position, quaternion]).tolist()

    def run_preserved_attempt(self, candidate: dict, attempt: int):
        """Run ARM_INIT -> LIFT while preserving the current MuJoCo scene."""
        started = time.monotonic()
        live_candidate = dict(candidate)
        live_candidate["cube_pose"] = self.current_object_pose()
        self.current_candidate = live_candidate
        self.step_count = 0
        self.joint1_target = None
        self.interactive.clear_candidate_abort()

        before = list(live_candidate["cube_pose"])
        arm_info: dict = {}
        metrics: dict = {"cube_pose_before": before}
        completed = "PRESERVED_SCENE"
        success = False
        failure_stage = None
        failure_reason = None
        selected = None

        try:
            # run_arm_init opens the gripper and first retracts from whatever
            # pose the preceding failed attempt left behind.  No MuJoCo reset
            # or object placement occurs in this path.
            arm_info = self.run_arm_init(live_candidate)
            metrics.update(arm_info)
            completed = "ARM_INIT"

            self._ensure_moveit()
            grasp = self._scan_and_propose()
            completed = "GRASP_PLAN"
            selected = self._select_and_execute_pregrasp(grasp)
            metrics["selected_anygrasp_index"] = selected["candidate_index"]
            completed = "MOVEIT_PRE_GRASP"
            self._execute_approach(selected)
            completed = "MOVEIT_APPROACH"
            self._run_local_reach(selected)
            completed = "LOCAL_REACH"
            close_info = self._run_close()
            completed = "CLOSE"
            lift_info = self._run_lift(close_info)
            metrics.update(lift_info)
            completed = "LIFT"
            success = True
            self._update_overlay("success")
        except EpisodeFailure as failure:
            failure_stage = failure.stage
            failure_reason = failure.reason
            metrics.update(failure.metrics)
            self._update_overlay("failure")
        except Exception as exc:
            failure_stage = "INTERNAL"
            failure_reason = repr(exc)
            self._update_overlay("failure")
        finally:
            self._detach_safely()
            try:
                self.env.disable_gripper_force_hold()
            except Exception:
                pass

        after = self.current_object_pose()
        translation, rotation = pose_motion(before, after)
        metrics.update(
            {
                "cube_pose_after": after,
                "cube_motion_m": translation,
                "cube_rotation_rad": rotation,
                "fresh_anygrasp_inference": True,
            }
        )
        result = self._result(
            live_candidate,
            attempt,
            success,
            completed,
            failure_stage,
            failure_reason,
            started,
            arm_info,
            metrics,
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mode-index", type=int)
    selection.add_argument("--sample-index", type=int)
    parser.add_argument("--grasp-checkpoint", required=True)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 means retry until success or operator quit (default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <prediction directory>/persistent_grasp_retries/<candidate>",
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
    if args.max_attempts < 0:
        raise SystemExit("--max-attempts must be >= 0")
    if not args.render and args.max_attempts == 0:
        raise SystemExit(
            "unlimited retries require --render so Q can stop the loop; "
            "use --max-attempts N with --headless"
        )
    if args.pause_on_stage and not args.render:
        print("[N2M] NOTE: --pause-on-stage has no effect without --render", flush=True)

    prediction_path = args.prediction.resolve()
    dataset_dir = args.dataset_dir.resolve()
    prediction = read_json(prediction_path)
    observation = _observation_for_prediction(dataset_dir, prediction)
    candidate, selection = build_prediction_candidate(
        prediction,
        observation,
        mode_index=args.mode_index,
        sample_index=args.sample_index,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else prediction_path.parent
        / "persistent_grasp_retries"
        / candidate["candidate_id"]
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
        "selection": selection,
        "max_attempts": args.max_attempts,
        "preserve_cube_between_attempts": True,
        "rerun_anygrasp_every_attempt": True,
        "episode_config": config.to_dict(),
    }
    write_json(output_dir / "run_config.json", run_config)
    print(
        f"[N2M][PERSISTENT_RETRY] candidate={candidate['candidate_id']} "
        f"base={np.round(candidate['base_pose'], 6)} "
        f"max_attempts={'unlimited' if args.max_attempts == 0 else args.max_attempts}",
        flush=True,
    )

    controller = InteractiveController()
    runner = PersistentGraspRunner(
        args.scene,
        config,
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        pause_on_stage=args.pause_on_stage,
        interactive=controller,
        debug_dir=output_dir / "debug" / "initial",
    )
    results = []
    outcome = "not_started"
    initial_cube_pose = None
    try:
        # The only scene reset in this program.
        initial_table_hits = runner.reset_candidate(candidate)
        initial_cube_pose = runner.current_object_pose()
        environment_hits = robot_environment_contacts(runner.env)
        preflight = {
            "created_at": _utc_now(),
            "environment_contacts": environment_hits,
            "table_contacts": initial_table_hits,
            "passed": not environment_hits and not initial_table_hits,
        }
        write_json(output_dir / "collision_preflight.json", preflight)
        if environment_hits or initial_table_hits:
            print(
                "[N2M][PERSISTENT_RETRY] initial collision; refusing to start: "
                f"environment={environment_hits} table={initial_table_hits}",
                flush=True,
            )
            outcome = "collision_preflight_failed"
            return_code = 2
        else:
            runner.weld.enable()
            return_code = 1
            attempt = 0
            while args.max_attempts == 0 or attempt < args.max_attempts:
                if controller.quit:
                    outcome = "operator_quit"
                    return_code = 130
                    break
                runner.debug_dir = output_dir / "debug" / f"attempt_{attempt:04d}"
                print(
                    f"[N2M][RETRY_START] attempt={attempt} "
                    f"cube={np.round(runner.current_object_pose()[:3], 6)}",
                    flush=True,
                )
                result = runner.run_preserved_attempt(candidate, attempt)
                row = result.to_dict()
                row.update(
                    {
                        "created_at": _utc_now(),
                        "observation_id": prediction["observation_id"],
                        "selection": selection,
                    }
                )
                append_jsonl(output_dir / "attempts.jsonl", row)
                results.append(result)
                print(
                    f"[N2M][RETRY_DONE] attempt={attempt} success={result.success} "
                    f"completed={result.completed_stage} "
                    f"failure={result.failure_stage}:{result.failure_reason} "
                    f"cube_motion={result.metrics['cube_motion_m'] * 1000:.2f}mm",
                    flush=True,
                )
                if result.success:
                    outcome = "lift_success"
                    return_code = 0
                    break
                if controller.quit:
                    outcome = "operator_quit"
                    return_code = 130
                    break
                if result.failure_stage in {"SAFETY", "INTERNAL", "EPISODE"}:
                    outcome = "fatal_failure"
                    return_code = 2
                    print(
                        "[N2M][PERSISTENT_RETRY] stopping after non-retryable "
                        f"failure {result.failure_stage}:{result.failure_reason}",
                        flush=True,
                    )
                    break
                attempt += 1
            else:
                outcome = "attempt_limit_reached"

            if outcome == "not_started":
                outcome = "attempt_limit_reached"
    except KeyboardInterrupt:
        outcome = "keyboard_interrupt"
        return_code = 130
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
            "attempt_count": len(results),
            "success": any(result.success is True for result in results),
            "initial_cube_pose": initial_cube_pose,
            "final_cube_pose": final_cube_pose,
        }
        write_json(output_dir / "summary.json", summary)
        print(
            f"[N2M][PERSISTENT_SUMMARY] outcome={outcome} "
            f"attempts={len(results)} output={output_dir}",
            flush=True,
        )
        runner.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
