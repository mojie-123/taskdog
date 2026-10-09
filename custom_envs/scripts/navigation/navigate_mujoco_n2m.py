#!/usr/bin/env python3
"""Navigate to a cube, infer N2M, grasp it, deliver it, and put it down.

This is an integration entry point; the validated
``navigate_mujoco_moveit.py`` remains unchanged.  The deprecated ``--goal``
argument is accepted for command-line compatibility but deliberately ignored.
The logical target comes from a replaceable TargetProvider.  For this first
MuJoCo-only version it is the named body's simulator pose.  N2M runs once;
the grasp recovery policy tries three fresh AnyGrasp plans at each of three
collision-free M poses before declaring failure.  A successful LIFT continues
through the original backoff, Nav2 destination, arm rotation and release flow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _path in (_ROOT, _ROOT / "custom_envs" / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from custom_envs.scripts.N2M.adaptive_n2m_recovery import (
    AdaptiveRecoveryRunner,
    build_candidate_queue,
)
from custom_envs.scripts.N2M.config import EpisodeConfig
from custom_envs.scripts.N2M.data_generation.live_observation import (
    LiveObservationConfig,
    capture_live_observation,
)
from custom_envs.scripts.N2M.interactive import InteractiveController, ViewerOverlay
from custom_envs.scripts.N2M.learning.runtime import N2MRuntime
from custom_envs.scripts.N2M.manifests import append_jsonl, write_json
from custom_envs.scripts.N2M.manipulation_episode import (
    ARM_HOME,
    GRIPPER_OPEN,
    EpisodeFailure,
)
from custom_envs.scripts.N2M.mujoco_state import (
    BaseWeld,
    robot_environment_contacts,
)
from custom_envs.scripts.N2M.target_provider import (
    MuJoCoTruthTargetProvider,
    navigation_target_metrics,
    observation_gate,
    standoff_goal,
)


DEFAULT_SCENE = _ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"
DEFAULT_NAV2_PARAMS = _ROOT / "custom_envs" / "config" / "nav2_params.yaml"
DEFAULT_CHECKPOINT = (
    _ROOT
    / "custom_envs"
    / "scripts"
    / "N2M"
    / "data"
    / "training"
    / "d30_o100_k1_v1"
    / "best.pt"
)
DEFAULT_ROBOT_X = 1.6
DEFAULT_ROBOT_Y = 1.6
DEFAULT_ROBOT_YAW = 0.0
DELIVERY_ARM_SIDE = np.array(
    [-math.pi / 2.0, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float64
)
RETRACT_COLLAPSE_RATIO = 0.55
RETRACT_COLLAPSE_ABSOLUTE = 0.012
RETRACT_RAPID_CLOSE_WINDOW = 5
RETRACT_RAPID_CLOSE_DISTANCE = 0.006


def load_nav2_runtime_settings(path: Path) -> dict:
    """Read the values that affect the MuJoCo/Nav2 bridge from Nav2 YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    controller = document["controller_server"]["ros__parameters"]
    smoother = document["velocity_smoother"]["ros__parameters"]
    progress = controller["progress_checker"]
    follow_path = controller["FollowPath"]
    max_velocity = [float(value) for value in smoother["max_velocity"]]
    return {
        "source": str(path.resolve()),
        "controller_frequency_hz": float(controller["controller_frequency"]),
        "smoothing_frequency_hz": float(smoother["smoothing_frequency"]),
        "velocity_timeout_s": float(smoother["velocity_timeout"]),
        "max_linear_velocity": abs(max_velocity[0]),
        "max_angular_velocity": abs(max_velocity[2]),
        "desired_linear_velocity": float(follow_path["desired_linear_vel"]),
        "required_movement_radius_m": float(progress["required_movement_radius"]),
        "movement_time_allowance_s": float(progress["movement_time_allowance"]),
    }


class RealtimePacer:
    """Keep real-time ROS 2 timers aligned with MuJoCo policy-time steps."""

    def __init__(self, period_s: float):
        self.period_s = float(period_s)
        self.deadline = time.monotonic()

    def wait(self) -> None:
        self.deadline += self.period_s
        now = time.monotonic()
        delay = self.deadline - now
        if delay > 0.0:
            time.sleep(delay)
        elif -delay > 5.0 * self.period_s:
            self.deadline = now


def _clip_nav_command(vx: float, wz: float, settings: dict) -> tuple[float, float]:
    return (
        float(np.clip(vx, -settings["max_linear_velocity"], settings["max_linear_velocity"])),
        float(np.clip(wz, -settings["max_angular_velocity"], settings["max_angular_velocity"])),
    )


def _update_progress_watchdog(state: dict, position_xy, settings: dict) -> None:
    """Mirror SimpleProgressChecker so a stalled Nav2 task cannot wait forever."""
    position = np.asarray(position_xy, dtype=np.float64)
    if float(np.linalg.norm(position - state["anchor"])) >= settings[
        "required_movement_radius_m"
    ]:
        state["anchor"] = position.copy()
        state["last_progress_time"] = time.monotonic()
        return
    stalled_s = time.monotonic() - state["last_progress_time"]
    if stalled_s > settings["movement_time_allowance_s"]:
        raise RuntimeError(
            "Nav2 progress timeout: moved less than "
            f"{settings['required_movement_radius_m']:.3f}m for "
            f"{stalled_s:.1f}s"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class AttachedPersistentGraspRunner(AdaptiveRecoveryRunner):
    """Run the validated manipulation episode on an already navigated env."""

    def __init__(
        self,
        env,
        config: EpisodeConfig,
        *,
        grasp_checkpoint: str,
        render: bool,
        interactive: InteractiveController,
        debug_dir: Path,
    ):
        self.cfg = config
        self.grasp_checkpoint = grasp_checkpoint
        self.render = bool(render)
        self.pause_on_stage = False
        self.interactive = interactive
        self.debug_dir = Path(debug_dir)
        self.env = env
        self.overlay = ViewerOverlay(env)
        self.weld = BaseWeld(env)
        self.zeros_action = np.zeros(16, dtype=np.float32)
        self.step_count = 0
        self.current_candidate = None
        self.joint1_target = None
        self.moveit = None
        self.attached = False
        self.attached_id = "n2m_carried_cube"
        self._imports_ready = False
        self._prepare_indices()

    def _delivery_grasp_guard(self, state: dict, initial_object_tcp) -> float:
        """Detect a dropped cube by finger collapse, not benign in-hand slip."""
        contacts = self._finger_contacts()
        if contacts["link7"] or contacts["link8"]:
            state["contact_miss_steps"] = 0
        else:
            state["contact_miss_steps"] += 1
        if state["contact_miss_steps"] >= int(
            self.cfg.moveit_lift_contact_miss_steps
        ):
            raise EpisodeFailure(
                "LIFT_RETRACT",
                "both_finger_contacts_lost",
                miss_steps=state["contact_miss_steps"],
            )

        grip = self._grip_q()
        aperture = max(0.0, float(grip[0] - grip[1]))
        closure = max(0.0, float(state["previous_aperture"] - aperture))
        state["previous_aperture"] = aperture
        state["minimum_aperture"] = min(state["minimum_aperture"], aperture)
        state["recent_closures"].append(closure)
        state["recent_closures"] = state["recent_closures"][-RETRACT_RAPID_CLOSE_WINDOW:]

        collapse_threshold = max(
            RETRACT_COLLAPSE_ABSOLUTE,
            state["initial_aperture"] * RETRACT_COLLAPSE_RATIO,
        )
        recent_closure = float(sum(state["recent_closures"]))
        if (
            aperture <= collapse_threshold
            and recent_closure >= RETRACT_RAPID_CLOSE_DISTANCE
        ):
            raise EpisodeFailure(
                "LIFT_RETRACT",
                "gripper_rapid_close",
                initial_aperture=state["initial_aperture"],
                current_aperture=aperture,
                collapse_threshold=collapse_threshold,
                recent_closure=recent_closure,
                contact_link7=bool(contacts["link7"]),
                contact_link8=bool(contacts["link8"]),
            )

        current_object_tcp = self._object_pose_in_tcp().position
        slip = float(np.linalg.norm(current_object_tcp - initial_object_tcp))
        state["max_object_tcp_slip"] = max(
            state["max_object_tcp_slip"], slip
        )
        return slip

    def retract_lifted_object_for_delivery(self) -> dict:
        """Reproduce the original post-LIFT MoveIt ARM_SIDE retraction."""
        if self.moveit is None:
            raise EpisodeFailure("LIFT_RETRACT", "moveit_bridge_not_started")

        # The standalone attempt runner disables force mode in its finally
        # block.  No physics step has happened since the successful LIFT, so
        # restoring the same hold force here is continuous in simulation time.
        self.env.set_gripper_force_hold(2.0)
        grip_target = self._grip_q().copy()
        initial_aperture = max(0.0, float(grip_target[0] - grip_target[1]))
        guard_state = {
            "initial_aperture": initial_aperture,
            "previous_aperture": initial_aperture,
            "minimum_aperture": initial_aperture,
            "recent_closures": [],
            "contact_miss_steps": 0,
            "max_object_tcp_slip": 0.0,
        }
        object_tcp = self._object_pose_in_tcp()
        initial_object_tcp = object_tcp.position.copy()
        default_sizes = {
            "cube": (0.050, 0.050, 0.050),
            "apple": (0.085, 0.085, 0.085),
            "bowl": (0.120, 0.120, 0.070),
            "banana": (0.180, 0.060, 0.055),
        }
        size = self.cfg.moveit_attached_object_size or default_sizes.get(
            self.cfg.object_name, (0.060, 0.060, 0.060)
        )
        attach = self.moveit.attach_box(
            self.attached_id,
            self.pose_to_ros_dict(object_tcp),
            size,
            link_name="grasp_tcp",
            touch_links=["gripper_base", "grasp_tcp", "link6", "link7", "link8"],
            timeout=6.0,
        )
        if not attach.get("ok", False):
            raise EpisodeFailure(
                "LIFT_RETRACT", "moveit_attach_failed", reply=attach
            )
        self.attached = True

        try:
            names, positions = self._moveit_state()
            plan = self.moveit.plan_to_joint(
                self.arm_joint_names,
                DELIVERY_ARM_SIDE,
                names,
                positions,
                joint_tolerance=0.02,
                allowed_planning_time=max(3.0, float(self.cfg.moveit_plan_time)),
                planning_attempts=4,
                velocity_scaling=0.18,
                acceleration_scaling=0.18,
                timeout=9.0,
            )
            if not plan.get("ok", False):
                raise EpisodeFailure(
                    "LIFT_RETRACT", "arm_side_plan_failed", reply=plan
                )
            commands = self._dense_commands(plan)
            if len(commands) == 0:
                raise EpisodeFailure("LIFT_RETRACT", "empty_arm_side_trajectory")

            for command in commands:
                self._step(command, grip_target)
                self._delivery_grasp_guard(guard_state, initial_object_tcp)

            final = np.asarray(commands[-1], dtype=np.float64)
            for settle in range(301):
                qerr = float(np.max(np.abs(self._arm_q() - final)))
                if qerr <= float(self.cfg.moveit_lift_joint_tolerance):
                    break
                if settle < 300:
                    self._step(final, grip_target)
                    self._delivery_grasp_guard(
                        guard_state, initial_object_tcp
                    )
            else:
                raise EpisodeFailure(
                    "LIFT_RETRACT",
                    "arm_side_execution_not_settled",
                    qerr=qerr,
                )

            self._checkpoint(
                "LIFT_RETRACT_DONE",
                f"commands={len(commands)} qerr={qerr:.4f} "
                f"aperture={initial_aperture * 1000.0:.1f}->"
                f"{guard_state['minimum_aperture'] * 1000.0:.1f}mm "
                f"max_slip={guard_state['max_object_tcp_slip'] * 1000.0:.1f}mm",
            )
            return {
                "target_joint_positions": DELIVERY_ARM_SIDE.tolist(),
                "trajectory_commands": int(len(commands)),
                "final_joint_error": qerr,
                "initial_gripper_aperture": initial_aperture,
                "minimum_gripper_aperture": guard_state["minimum_aperture"],
                "max_object_tcp_slip_diagnostic": guard_state[
                    "max_object_tcp_slip"
                ],
            }
        finally:
            self._detach_safely()

    def close_attached(self) -> None:
        """Release manipulation resources without closing the shared env."""
        try:
            self._detach_safely()
        except Exception:
            pass
        try:
            self.weld.disable()
        except Exception:
            pass
        try:
            self.env.disable_gripper_force_hold()
        except Exception:
            pass
        if self.moveit is not None:
            self.moveit.stop()


def _policy_step(env, policy, device, last_action, commands):
    from custom_envs.mujoco.obs_builder import build_policy_obs

    rs = env.get_robot_state()
    cmd_vx, cmd_vy, cmd_wz = commands
    env.set_arm_target(ARM_HOME)
    env.set_gripper_target(GRIPPER_OPEN)
    obs = build_policy_obs(
        rs["joint_pos"],
        rs["joint_vel"],
        rs["ang_vel"],
        rs["quat"],
        float(cmd_vx),
        float(cmd_vy),
        float(cmd_wz),
        last_action,
    )
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        action = policy(tensor).detach().cpu().numpy().reshape(-1)[:16]
    env.step(action)
    return action.astype(np.float32, copy=True)


# These values intentionally match the validated post-LIFT states in
# navigate_mujoco_moveit.py.
DELIVERY_BACKOFF_MAX_STEPS = 450
DELIVERY_ALIGN_MAX_STEPS = 600
DELIVERY_PAN_MAX_STEPS = 500
DELIVERY_ROTATE_STEPS = 600
DELIVERY_PUT_DOWN_STEPS = 100
DELIVERY_MAX_PLACEMENT_DISTANCE_M = 1.5


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _slew(current: float, target: float, max_delta: float) -> float:
    return float(current + np.clip(target - current, -max_delta, max_delta))


def _yaw_velocity_target(
    error: float,
    *,
    deadband: float,
    kp: float = 4.0,
    min_abs: float = 0.4,
    max_abs: float = 0.9,
) -> float:
    if abs(error) <= deadband:
        return 0.0
    magnitude = min(max_abs, max(min_abs, kp * abs(error)))
    return math.copysign(magnitude, error)


def _carry_policy_step(
    env,
    policy,
    device,
    last_action,
    commands,
    arm_target,
    gripper_target,
):
    """Advance locomotion while preserving the post-LIFT arm/gripper targets."""
    from custom_envs.mujoco.obs_builder import build_policy_obs

    rs = env.get_robot_state()
    cmd_vx, cmd_vy, cmd_wz = commands
    env.set_arm_target(np.asarray(arm_target, dtype=np.float64))
    env.set_gripper_target(np.asarray(gripper_target, dtype=np.float64))
    obs = build_policy_obs(
        rs["joint_pos"],
        rs["joint_vel"],
        rs["ang_vel"],
        rs["quat"],
        float(cmd_vx),
        float(cmd_vy),
        float(cmd_wz),
        last_action,
    )
    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        action = policy(tensor).detach().cpu().numpy().reshape(-1)[:16]
    env.step(action)
    return action.astype(np.float32, copy=True)


def _delivery_world_x_pan(
    *,
    name: str,
    env,
    policy,
    bridge,
    device,
    interactive,
    last_action,
    arm_target,
    gripper_target,
    target_x: float,
    target_yaw: float,
    max_steps: int,
    position_tolerance: float,
    resume_tolerance: float,
    pacer: RealtimePacer,
):
    """Run the original smooth world-X PAN controller while carrying the cube."""
    cmd_vx = cmd_vy = cmd_wz = 0.0
    settling = False
    settle_count = 0
    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        rs = env.get_robot_state()
        bridge.update_robot_pose(
            rs["pos"], rs["quat"], rs["lin_vel"], rs["ang_vel"]
        )
        world_error = float(target_x - rs["pos"][0])
        abs_error = abs(world_error)
        if settling:
            if abs_error > resume_tolerance:
                settling = False
                settle_count = 0
        elif abs_error <= position_tolerance:
            settling = True
            settle_count = 0

        world_vx_target = (
            0.0
            if settling
            else math.copysign(min(0.25, max(0.25, abs_error)), world_error)
        )
        yaw = float(rs["yaw"])
        cmd_vx_target = math.cos(yaw) * world_vx_target
        cmd_vy_target = -math.sin(yaw) * world_vx_target
        cmd_vx = _slew(cmd_vx, cmd_vx_target, 0.030)
        cmd_vy = _slew(cmd_vy, cmd_vy_target, 0.030)
        yaw_error = _wrap_angle(target_yaw - yaw)
        cmd_wz = _slew(
            cmd_wz,
            _yaw_velocity_target(
                yaw_error,
                deadband=math.radians(1.5),
                kp=3.0,
                min_abs=0.3,
                max_abs=0.6,
            ),
            0.060,
        )

        axis_speed = abs(float(np.asarray(rs["lin_vel"])[0]))
        if settling and axis_speed <= 0.06 and abs(yaw_error) <= math.radians(3.0):
            settle_count += 1
        else:
            settle_count = 0
        if step == 0 or step % 25 == 0:
            print(
                f"[N2M][{name}] step={step} x_error={world_error:+.3f}m "
                f"yaw_error={math.degrees(yaw_error):+.2f}deg "
                f"cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                f"settle={settle_count}/8",
                flush=True,
            )
        if settle_count >= 8:
            return last_action, {
                "steps": step + 1,
                "target_x": target_x,
                "final_x": float(rs["pos"][0]),
                "final_yaw": yaw,
            }
        last_action = _carry_policy_step(
            env,
            policy,
            device,
            last_action,
            (cmd_vx, cmd_vy, cmd_wz),
            arm_target,
            gripper_target,
        )
        pacer.wait()
    rs = env.get_robot_state()
    print(
        f"[N2M][{name}_TIMEOUT] proceeding like the validated state machine; "
        f"final_x_error={target_x - float(rs['pos'][0]):+.3f}m",
        flush=True,
    )
    return last_action, {
        "steps": int(max_steps),
        "target_x": target_x,
        "final_x": float(rs["pos"][0]),
        "final_yaw": float(rs["yaw"]),
        "timed_out": True,
    }


def _delivery_align_yaw(
    *,
    env,
    policy,
    bridge,
    device,
    interactive,
    last_action,
    arm_target,
    gripper_target,
    target_yaw: float,
    pacer: RealtimePacer,
):
    cmd_wz = 0.0
    settling = False
    settle_count = 0
    for step in range(DELIVERY_ALIGN_MAX_STEPS):
        if interactive.quit:
            raise KeyboardInterrupt
        rs = env.get_robot_state()
        bridge.update_robot_pose(
            rs["pos"], rs["quat"], rs["lin_vel"], rs["ang_vel"]
        )
        error = _wrap_angle(target_yaw - float(rs["yaw"]))
        if settling:
            if abs(error) > math.radians(3.0):
                settling = False
                settle_count = 0
        elif abs(error) <= math.radians(2.0):
            settling = True
            settle_count = 0
        target_wz = (
            0.0
            if settling
            else _yaw_velocity_target(error, deadband=math.radians(2.0))
        )
        cmd_wz = _slew(cmd_wz, target_wz, 0.08)
        if (
            settling
            and abs(error) <= math.radians(3.0)
            and abs(float(rs["ang_vel"][2])) <= 0.15
            and abs(cmd_wz) <= 0.08
        ):
            settle_count += 1
        else:
            settle_count = 0
        if step == 0 or step % 20 == 0:
            print(
                f"[N2M][DELIVERY_ALIGN] step={step} "
                f"yaw_error={math.degrees(error):+.2f}deg "
                f"cmd_wz={cmd_wz:+.3f} settle={settle_count}/6",
                flush=True,
            )
        if settle_count >= 6:
            return last_action, {
                "steps": step + 1,
                "target_yaw": target_yaw,
                "final_yaw": float(rs["yaw"]),
            }
        last_action = _carry_policy_step(
            env,
            policy,
            device,
            last_action,
            (0.0, 0.0, cmd_wz),
            arm_target,
            gripper_target,
        )
        pacer.wait()
    rs = env.get_robot_state()
    print(
        "[N2M][DELIVERY_ALIGN_TIMEOUT] proceeding like the validated "
        f"state machine; final_yaw={float(rs['yaw']):+.3f}rad",
        flush=True,
    )
    return last_action, {
        "steps": DELIVERY_ALIGN_MAX_STEPS,
        "target_yaw": target_yaw,
        "final_yaw": float(rs["yaw"]),
        "timed_out": True,
    }


def deliver_lifted_object(
    *,
    env,
    runner,
    policy,
    bridge,
    device,
    interactive,
    destination,
    destination_max_steps: int,
    render: bool,
    nav2_settings: dict,
) -> dict:
    """Continue the validated state machine from successful LIFT through release."""
    metrics: dict = {
        "destination": [float(destination[0]), float(destination[1])],
        "cube_pose_after_lift": runner.current_object_pose(),
    }
    carry_arm = runner._arm_q().copy()
    carry_gripper = runner._grip_q().copy()
    # run_preserved_attempt deliberately returns its standalone test runner to
    # position mode in finally.  No physics step occurs between LIFT success
    # and this call, so restore the same bilateral hold force before locomotion.
    env.set_gripper_force_hold(2.0)
    last_action = np.zeros(16, dtype=np.float32)
    pacer = RealtimePacer(env.dt)
    runner.weld.disable()

    start = env.get_robot_state()
    backoff_x = float(start["pos"][0]) - 0.5
    print(
        f"[N2M][DELIVERY_BACKOFF_START] x={start['pos'][0]:.3f} "
        f"target_x={backoff_x:.3f}",
        flush=True,
    )
    last_action, metrics["backoff"] = _delivery_world_x_pan(
        name="DELIVERY_BACKOFF",
        env=env,
        policy=policy,
        bridge=bridge,
        device=device,
        interactive=interactive,
        last_action=last_action,
        arm_target=carry_arm,
        gripper_target=carry_gripper,
        target_x=backoff_x,
        target_yaw=math.pi / 2.0,
        max_steps=DELIVERY_BACKOFF_MAX_STEPS,
        position_tolerance=0.03,
        resume_tolerance=0.08,
        pacer=pacer,
    )

    rs = env.get_robot_state()
    bridge.update_robot_pose(rs["pos"], rs["quat"], rs["lin_vel"], rs["ang_vel"])
    bridge.send_goal(float(destination[0]), float(destination[1]))
    progress = {
        "anchor": np.asarray(rs["pos"][:2], dtype=np.float64).copy(),
        "last_progress_time": time.monotonic(),
    }
    print(f"[N2M][DELIVERY_NAV_START] destination={destination}", flush=True)
    nav_started = time.monotonic()
    for step in range(int(destination_max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        rs = env.get_robot_state()
        bridge.update_robot_pose(
            rs["pos"], rs["quat"], rs["lin_vel"], rs["ang_vel"]
        )
        nav_done, nav_failed = bridge.get_nav_status()
        if nav_failed:
            bridge.cancel_goal()
            raise RuntimeError("Nav2 failed while carrying cube to destination")
        if nav_done:
            bridge.cancel_goal()
            metrics["navigation"] = {
                "steps": step,
                "elapsed_s": time.monotonic() - nav_started,
                "final_base_pose": [
                    float(rs["pos"][0]),
                    float(rs["pos"][1]),
                    float(rs["yaw"]),
                ],
            }
            print(
                f"[N2M][DELIVERY_NAV_DONE] step={step} "
                f"base={np.round(metrics['navigation']['final_base_pose'], 4)}",
                flush=True,
            )
            break
        cmd_vx, cmd_wz = bridge.get_cmd_vel()
        cmd_vx, cmd_wz = _clip_nav_command(cmd_vx, cmd_wz, nav2_settings)
        _update_progress_watchdog(progress, rs["pos"][:2], nav2_settings)
        last_action = _carry_policy_step(
            env,
            policy,
            device,
            last_action,
            (cmd_vx, 0.0, cmd_wz),
            carry_arm,
            carry_gripper,
        )
        if step % 100 == 0:
            print(
                f"[N2M][DELIVERY_NAV] step={step} "
                f"base={np.round(rs['pos'][:2], 3)} "
                f"cmd=({cmd_vx:+.3f},{cmd_wz:+.3f})",
                flush=True,
            )
        pacer.wait()
    else:
        bridge.cancel_goal()
        raise RuntimeError(
            f"DELIVERY_NAV timed out after {destination_max_steps} steps"
        )

    last_action, metrics["align"] = _delivery_align_yaw(
        env=env,
        policy=policy,
        bridge=bridge,
        device=device,
        interactive=interactive,
        last_action=last_action,
        arm_target=carry_arm,
        gripper_target=carry_gripper,
        target_yaw=-math.pi / 2.0,
        pacer=pacer,
    )
    last_action, metrics["destination_pan"] = _delivery_world_x_pan(
        name="DELIVERY_PAN_X",
        env=env,
        policy=policy,
        bridge=bridge,
        device=device,
        interactive=interactive,
        last_action=last_action,
        arm_target=carry_arm,
        gripper_target=carry_gripper,
        target_x=float(destination[0]),
        target_yaw=-math.pi / 2.0,
        max_steps=DELIVERY_PAN_MAX_STEPS,
        position_tolerance=0.08,
        resume_tolerance=0.16,
        pacer=pacer,
    )

    runner.weld.enable()
    rotate_start = runner._arm_q().copy()
    rotate_target = rotate_start.copy()
    rotate_target[:3] = [math.pi / 2.0, 1.8, -1.8]
    for step in range(DELIVERY_ROTATE_STEPS):
        if interactive.quit:
            raise KeyboardInterrupt
        alpha = 0.5 * (
            1.0 - math.cos(math.pi * min(1.0, step / DELIVERY_ROTATE_STEPS))
        )
        command = rotate_start + alpha * (rotate_target - rotate_start)
        last_action = _carry_policy_step(
            env,
            policy,
            device,
            last_action,
            (0.0, 0.0, 0.0),
            command,
            carry_gripper,
        )
        if step == 0 or step % 100 == 0:
            print(
                f"[N2M][DELIVERY_ROTATE] step={step}/{DELIVERY_ROTATE_STEPS}",
                flush=True,
            )

    env.disable_gripper_force_hold()
    for step in range(DELIVERY_PUT_DOWN_STEPS):
        if interactive.quit:
            raise KeyboardInterrupt
        last_action = _carry_policy_step(
            env,
            policy,
            device,
            last_action,
            (0.0, 0.0, 0.0),
            rotate_target,
            GRIPPER_OPEN,
        )
        if (step + 1) % 20 == 0:
            print(
                f"[N2M][DELIVERY_PUT_DOWN] "
                f"step={step + 1}/{DELIVERY_PUT_DOWN_STEPS}",
                flush=True,
            )
    metrics["cube_pose_after_release"] = runner.current_object_pose()
    final_rs = env.get_robot_state()
    metrics["final_base_pose"] = [
        float(final_rs["pos"][0]),
        float(final_rs["pos"][1]),
        float(final_rs["yaw"]),
    ]
    metrics["rotate_steps"] = DELIVERY_ROTATE_STEPS
    metrics["put_down_steps"] = DELIVERY_PUT_DOWN_STEPS
    metrics["cube_destination_xy_error_m"] = float(
        np.linalg.norm(
            np.asarray(metrics["cube_pose_after_release"][:2], dtype=np.float64)
            - np.asarray(destination[:2], dtype=np.float64)
        )
    )
    runner.weld.disable()
    if metrics["cube_destination_xy_error_m"] > DELIVERY_MAX_PLACEMENT_DISTANCE_M:
        raise RuntimeError(
            "cube was released too far from the destination: "
            f"distance={metrics['cube_destination_xy_error_m']:.3f}m > "
            f"{DELIVERY_MAX_PLACEMENT_DISTANCE_M:.3f}m"
        )
    print(
        f"[N2M][DELIVERY_DONE] cube="
        f"{np.round(metrics['cube_pose_after_release'][:3], 4)} "
        f"destination_error={metrics['cube_destination_xy_error_m']:.3f}m",
        flush=True,
    )
    return metrics


def _effective_yaw_command(error: float) -> float:
    """Turn with the minimum command known to move the locomotion policy."""
    magnitude = abs(float(error))
    if magnitude <= math.radians(3.0):
        return 0.0
    target = min(0.9, max(0.4, 3.0 * magnitude))
    return math.copysign(target, error)


def navigate_to_observation(env, policy, bridge, provider, args, device):
    """Run Nav2 to a standoff point, then truth-servo into the O gate."""
    pacer = RealtimePacer(env.dt)
    nav2_settings = args.nav2_settings
    initial = env.get_robot_state()
    target = provider.estimate(env, args.object)
    waypoint = standoff_goal(
        initial["pos"][:2], target.position_world[:2], args.nav_standoff
    )
    bridge.send_goal(float(waypoint[0]), float(waypoint[1]))
    progress = {
        "anchor": np.asarray(initial["pos"][:2], dtype=np.float64).copy(),
        "last_progress_time": time.monotonic(),
    }
    print(
        f"[N2M][NAV_GOAL] source={target.source} cube="
        f"{np.round(target.position_world, 4)} waypoint={np.round(waypoint, 4)} "
        f"standoff={args.nav_standoff:.2f}m",
        flush=True,
    )

    state = "NAV_GLOBAL"
    stable = 0
    last_action = np.zeros(16, dtype=np.float32)
    max_bearing = math.radians(float(args.nav_max_bearing_deg))
    for step in range(int(args.nav_max_steps)):
        rs = env.get_robot_state()
        bridge.update_robot_pose(
            rs["pos"], rs["quat"], rs["lin_vel"], rs["ang_vel"]
        )
        target = provider.estimate(env, args.object)
        metrics = navigation_target_metrics(
            [rs["pos"][0], rs["pos"][1], rs["yaw"]], target.position_world
        )
        nav_done, nav_failed = bridge.get_nav_status()

        if state == "NAV_GLOBAL":
            cmd_vx, cmd_wz = bridge.get_cmd_vel()
            cmd_vx, cmd_wz = _clip_nav_command(
                cmd_vx, cmd_wz, nav2_settings
            )
            cmd_vy = 0.0
            reached_band = (
                args.nav_min_distance
                <= metrics.horizontal_distance_m
                <= args.nav_max_distance
            )
            if reached_band or nav_done:
                bridge.cancel_goal()
                state = "TARGET_ALIGN"
                cmd_vx = cmd_vy = cmd_wz = 0.0
                print(
                    f"[N2M][NAV_ALIGN] trigger="
                    f"{'distance_band' if reached_band else 'nav_done'} "
                    f"distance={metrics.horizontal_distance_m:.3f}m "
                    f"bearing={math.degrees(metrics.bearing_error_rad):+.2f}deg",
                    flush=True,
                )
            elif nav_failed:
                raise RuntimeError(
                    "Nav2 failed before reaching the observation distance band: "
                    f"distance={metrics.horizontal_distance_m:.3f}m"
                )
            else:
                _update_progress_watchdog(
                    progress, rs["pos"][:2], nav2_settings
                )
        else:
            cmd_wz = _effective_yaw_command(metrics.bearing_error_rad)
            cmd_vy = 0.0
            if abs(metrics.bearing_error_rad) > math.radians(15.0):
                cmd_vx = 0.0
            elif metrics.horizontal_distance_m > args.nav_max_distance:
                cmd_vx = min(0.22, metrics.horizontal_distance_m - args.nav_standoff)
            elif metrics.horizontal_distance_m < args.nav_min_distance:
                cmd_vx = max(-0.15, metrics.horizontal_distance_m - args.nav_standoff)
            else:
                cmd_vx = 0.0

            in_gate = observation_gate(
                metrics,
                min_distance_m=args.nav_min_distance,
                max_distance_m=args.nav_max_distance,
                max_bearing_rad=max_bearing,
            )
            stopped = (
                float(np.linalg.norm(np.asarray(rs["lin_vel"])[:2]))
                <= args.nav_stop_speed
                and abs(float(rs["ang_vel"][2])) <= args.nav_stop_yaw_rate
                and abs(cmd_vx) < 1e-9
                and abs(cmd_wz) < 1e-9
            )
            stable = stable + 1 if in_gate and stopped else 0
            if stable >= int(args.nav_settle_frames):
                print(
                    f"[N2M][NAV_OBSERVATION_READY] step={step} "
                    f"distance={metrics.horizontal_distance_m:.3f}m "
                    f"bearing={math.degrees(metrics.bearing_error_rad):+.2f}deg "
                    f"stable={stable}",
                    flush=True,
                )
                return rs, target, {
                    "waypoint_world": waypoint.tolist(),
                    "horizontal_distance_m": metrics.horizontal_distance_m,
                    "bearing_error_rad": metrics.bearing_error_rad,
                    "navigation_steps": step + 1,
                }

        if step % 100 == 0:
            print(
                f"[N2M][{state}] step={step} pos={np.round(rs['pos'][:2], 3)} "
                f"distance={metrics.horizontal_distance_m:.3f}m "
                f"bearing={math.degrees(metrics.bearing_error_rad):+.1f}deg "
                f"cmd=({cmd_vx:+.2f},{cmd_wz:+.2f}) stable={stable}",
                flush=True,
            )
        last_action = _policy_step(
            env, policy, device, last_action, (cmd_vx, cmd_vy, cmd_wz)
        )
        pacer.wait()

    bridge.cancel_goal()
    raise RuntimeError(f"NAV observation gate timed out after {args.nav_max_steps} steps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        default=None,
        help="deprecated compatibility option; ignored in favor of cube truth",
    )
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--n2m-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grasp-checkpoint", required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--nav2-params",
        type=Path,
        default=DEFAULT_NAV2_PARAMS,
        help=(
            "Nav2 YAML used by terminal 1; its controller/smoother limits and "
            "progress timeout are also applied to the MuJoCo bridge"
        ),
    )
    parser.add_argument("--object", default="cube")
    parser.add_argument(
        "--destination",
        nargs=2,
        type=float,
        required=True,
        metavar=("X", "Y"),
        help="final Nav2 target used after a successful LIFT",
    )
    parser.add_argument(
        "--destination-max-steps",
        type=int,
        default=6000,
        help="maximum locomotion-policy steps for delivery Nav2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--robot-x",
        "--robot_x",
        dest="robot_x",
        type=float,
        default=DEFAULT_ROBOT_X,
        help=f"initial robot world X (default: {DEFAULT_ROBOT_X})",
    )
    parser.add_argument(
        "--robot-y",
        "--robot_y",
        dest="robot_y",
        type=float,
        default=DEFAULT_ROBOT_Y,
        help=f"initial robot world Y (default: {DEFAULT_ROBOT_Y})",
    )
    parser.add_argument(
        "--robot-yaw",
        "--robot_yaw",
        dest="robot_yaw",
        type=float,
        default=DEFAULT_ROBOT_YAW,
        help=f"initial robot world yaw in radians (default: {DEFAULT_ROBOT_YAW})",
    )
    parser.add_argument("--nav-standoff", type=float, default=1.4)
    parser.add_argument("--nav-min-distance", type=float, default=1.2)
    parser.add_argument("--nav-max-distance", type=float, default=1.7)
    parser.add_argument("--nav-max-bearing-deg", type=float, default=15.0)
    parser.add_argument("--nav-stop-speed", type=float, default=0.05)
    parser.add_argument("--nav-stop-yaw-rate", type=float, default=0.10)
    parser.add_argument("--nav-settle-frames", type=int, default=15)
    parser.add_argument("--nav-max-steps", type=int, default=8000)
    parser.add_argument("--n2m-num-samples", type=int, default=100)
    parser.add_argument(
        "--n2m-max-candidates",
        "--max-m-candidates",
        dest="n2m_max_candidates",
        type=int,
        default=3,
        help="maximum collision-free M poses to attempt (default: 3)",
    )
    parser.add_argument("--n2m-seed", type=int, default=42)
    parser.add_argument(
        "--max-grasp-attempts",
        "--attempts-per-m",
        dest="max_grasp_attempts",
        type=int,
        default=3,
        help="fresh AnyGrasp attempts at each M (default: 3)",
    )
    parser.add_argument("--output-dir", type=Path)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--render", dest="render", action="store_true")
    display.add_argument("--headless", dest="render", action="store_false")
    parser.set_defaults(render=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.goal is not None:
        print(
            f"[N2M][GOAL_IGNORED] --goal={args.goal}; using live cube truth instead",
            flush=True,
        )
    if not (0.0 < args.nav_min_distance < args.nav_max_distance):
        raise SystemExit("require 0 < --nav-min-distance < --nav-max-distance")
    if (
        args.nav_settle_frames <= 0
        or args.max_grasp_attempts <= 0
        or args.n2m_max_candidates <= 0
        or args.destination_max_steps <= 0
    ):
        raise SystemExit("settle frames, M/attempt limits and delivery steps must be positive")
    if not args.scene.is_file():
        raise SystemExit(f"scene does not exist: {args.scene}")
    if not args.n2m_checkpoint.is_file():
        raise SystemExit(f"N2M checkpoint does not exist: {args.n2m_checkpoint}")
    if not args.nav2_params.is_file():
        raise SystemExit(f"Nav2 params do not exist: {args.nav2_params}")
    try:
        nav2_settings = load_nav2_runtime_settings(args.nav2_params)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(
            f"cannot read required navigation settings from "
            f"{args.nav2_params}: {exc}"
        ) from exc
    args.nav2_settings = nav2_settings
    bridge_feedback_timeout = max(
        0.35, 2.5 * nav2_settings["velocity_timeout_s"]
    )
    print(
        f"[N2M][NAV2_SETTINGS] source={nav2_settings['source']} "
        f"controller={nav2_settings['controller_frequency_hz']:.1f}Hz "
        f"smoother={nav2_settings['smoothing_frequency_hz']:.1f}Hz "
        f"desired_vx={nav2_settings['desired_linear_velocity']:.2f}m/s "
        f"limits=({nav2_settings['max_linear_velocity']:.2f}m/s," 
        f"{nav2_settings['max_angular_velocity']:.2f}rad/s) "
        f"cmd_timeout={nav2_settings['velocity_timeout_s']:.2f}s "
        f"progress={nav2_settings['required_movement_radius_m']:.2f}m/"
        f"{nav2_settings['movement_time_allowance_s']:.1f}s",
        flush=True,
    )

    os.environ.setdefault("MUJOCO_GL", "glx" if args.render else "egl")
    device_name = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            _ROOT
            / "custom_envs"
            / "scripts"
            / "N2M"
            / "data"
            / "integrated_runs"
            / _timestamp_token()
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from custom_envs.mujoco.env import MuJoCoEnv
    from custom_envs.mujoco.policy_loader import load_locomotion_policy
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge

    controller = InteractiveController()
    env = MuJoCoEnv(
        robot_init_pos=(args.robot_x, args.robot_y, 0.58),
        robot_init_quat=(
            math.cos(args.robot_yaw / 2.0),
            0.0,
            0.0,
            math.sin(args.robot_yaw / 2.0),
        ),
        render=args.render,
        key_callback=controller.key_callback if args.render else None,
        scene_path=str(args.scene.resolve()),
    )
    initial_contacts = robot_environment_contacts(env)
    if initial_contacts:
        env.close()
        raise SystemExit(
            "initial robot pose collides with the environment: "
            f"pose=({args.robot_x:.3f}, {args.robot_y:.3f}, "
            f"{args.robot_yaw:.3f}), contacts={initial_contacts}. "
            "Choose a free pose with --robot-x/--robot-y/--robot-yaw."
        )
    print(
        f"[N2M][ROBOT_INITIALIZED] pose="
        f"({args.robot_x:.3f}, {args.robot_y:.3f}, {args.robot_yaw:.3f}) "
        "contacts=[]",
        flush=True,
    )
    policy = load_locomotion_policy(args.policy_path, device=device_name)
    provider = MuJoCoTruthTargetProvider()
    bridge = IsaacROS2Bridge(
        cmd_vel_timeout=nav2_settings["velocity_timeout_s"],
        feedback_timeout=bridge_feedback_timeout,
    )
    bridge_started = False
    runner = None
    result_rows = []
    outcome = "not_started"
    selected_candidate = None
    attempted_m_count = 0
    collision_rejected_m_count = 0
    delivery_metrics = None
    delivery_retraction_metrics = None
    lift_succeeded = False
    run_config = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "scene": str(args.scene.resolve()),
        "nav2_runtime_settings": nav2_settings,
        "mujoco_policy_frequency_hz": 1.0 / float(env.dt),
        "realtime_navigation_pacing": True,
        "bridge_feedback_timeout_s": bridge_feedback_timeout,
        "policy_path": str(Path(args.policy_path).expanduser().resolve()),
        "checkpoint": str(args.n2m_checkpoint.expanduser().resolve()),
        "grasp_checkpoint": str(Path(args.grasp_checkpoint).expanduser().resolve()),
        "object": args.object,
        "destination": [float(args.destination[0]), float(args.destination[1])],
        "destination_max_steps": args.destination_max_steps,
        "robot_initial_pose": [args.robot_x, args.robot_y, args.robot_yaw],
        "goal_argument_ignored": args.goal,
        "target_provider": "MuJoCoTruthTargetProvider",
        "nav_standoff_m": args.nav_standoff,
        "nav_distance_band_m": [args.nav_min_distance, args.nav_max_distance],
        "nav_max_bearing_deg": args.nav_max_bearing_deg,
        "n2m_num_samples": args.n2m_num_samples,
        "n2m_max_attempted_candidates": args.n2m_max_candidates,
        "grasp_attempts_per_m": args.max_grasp_attempts,
        "maximum_total_grasp_attempts": (
            args.n2m_max_candidates * args.max_grasp_attempts
        ),
        "rerun_anygrasp_every_attempt": True,
        "rerun_n2m_after_cube_motion": False,
        "reset_cube_between_attempts": False,
        "reset_cube_between_m_candidates": False,
        "restart_nav_bridge_after_lift": True,
    }
    write_json(output_dir / "run_config.json", run_config)

    try:
        bridge.start(timeout=30.0)
        bridge_started = True
        _, target, nav_metrics = navigate_to_observation(
            env, policy, bridge, provider, args, device
        )
        bridge.cancel_goal()
        write_json(output_dir / "navigation_result.json", nav_metrics)

        observation_lock = BaseWeld(env)
        observation_lock.enable()
        live = capture_live_observation(
            env,
            LiveObservationConfig(
                target_body=args.object,
                max_camera_distance_m=1.7,
            ),
        )
        np.savez_compressed(
            output_dir / "online_observation.npz",
            xyz=live.xyz_base,
            rgb=live.rgb,
        )
        np.save(output_dir / "front_rgb.npy", live.image_rgb)
        np.save(output_dir / "front_depth.npy", live.image_depth)
        np.save(output_dir / "front_target_mask.npy", live.target_mask)
        write_json(
            output_dir / "observation.json",
            {
                "observation_pose_world": live.observation_pose_world,
                "target_pose_world": live.target_pose_world,
                "target_centroid_base": live.target_centroid_base.tolist(),
                "metrics": live.metrics,
            },
        )
        print(
            f"[N2M][OBS_CAPTURED] points={len(live.xyz_base)} "
            f"target_points={live.metrics['target_points_after_voxel']} "
            f"pixels={live.metrics['target_visible_pixels']} "
            f"camera_distance={live.metrics['target_camera_distance_m']:.3f}m",
            flush=True,
        )

        runtime = N2MRuntime(args.n2m_checkpoint, device=device_name)
        prediction = runtime.predict(
            live.xyz_base,
            live.rgb,
            live.observation_pose_world,
            observation_id="online__" + _timestamp_token(),
            num_samples=args.n2m_num_samples,
            sample_seed=args.n2m_seed,
        )
        write_json(output_dir / "prediction.json", prediction)
        runtime.close()
        del runtime

        target = provider.estimate(env, args.object)
        observation = {
            "observation_id": prediction["observation_id"],
            "scene_id": "online",
            "approach_edge": "online",
            "observation_pose_world": live.observation_pose_world,
            "cube_pose_world": target.pose_wxyz(),
        }
        queue, queue_diagnostics = build_candidate_queue(
            prediction,
            observation,
            min_xy_m=0.08,
            min_yaw_rad=math.radians(10.0),
            # Keep the full density-ordered queue so collision-rejected poses
            # do not consume one of the three M attempts.
            max_candidates=0,
        )
        write_json(
            output_dir / "candidate_queue.json",
            {"queue": queue, "diagnostics": queue_diagnostics},
        )

        observation_lock.disable()
        # The ROS child emits cmd_vel continuously.  The manipulation runner
        # does not drain that stream, so close this client before potentially
        # long AnyGrasp retries and create a fresh client only after LIFT.
        bridge.stop()
        bridge_started = False
        if device.type == "cuda":
            torch.cuda.empty_cache()

        episode_config = EpisodeConfig(object_name=args.object)
        runner = AttachedPersistentGraspRunner(
            env,
            episode_config,
            grasp_checkpoint=args.grasp_checkpoint,
            render=args.render,
            interactive=controller,
            debug_dir=output_dir / "debug",
        )
        outcome = "grasp_candidate_queue_exhausted"
        global_attempt = 0
        needs_retraction = False
        fatal = False

        for queue_row in queue:
            if attempted_m_count >= int(args.n2m_max_candidates):
                outcome = "grasp_attempts_exhausted"
                break
            if controller.quit:
                raise KeyboardInterrupt

            candidate = queue_row["candidate"]
            queue_index = int(queue_row["queue_index"])
            if needs_retraction:
                runner.debug_dir = output_dir / "debug" / f"m_{queue_index:04d}" / "transition"
                runner.retract_for_reposition()
                needs_retraction = False

            preflight = runner.reposition_base_preserving_cube(candidate)
            preflight_row = {
                "created_at": _utc_now(),
                "queue_index": queue_index,
                "candidate_id": candidate["candidate_id"],
                "base_pose": candidate["base_pose"],
                "selection": queue_row["selection"],
                "initial_log_density": queue_row["initial_log_density"],
                **preflight,
            }
            append_jsonl(output_dir / "collision_preflights.jsonl", preflight_row)
            if not preflight["passed"]:
                collision_rejected_m_count += 1
                append_jsonl(
                    output_dir / "m_summaries.jsonl",
                    {**preflight_row, "status": "collision_rejected", "attempts": 0},
                )
                print(
                    f"[N2M][M_REJECTED] queue={queue_index} "
                    f"candidate={candidate['candidate_id']} contacts="
                    f"{preflight['environment_contacts']}",
                    flush=True,
                )
                continue

            attempted_m_count += 1
            selected_candidate = candidate
            runner.weld.enable()
            m_success = False
            m_last_failure = None
            m_attempts = 0
            print(
                f"[N2M][M_START] attempted_m={attempted_m_count}/"
                f"{args.n2m_max_candidates} queue={queue_index} "
                f"base={np.round(candidate['base_pose'], 4)}",
                flush=True,
            )

            for local_attempt in range(int(args.max_grasp_attempts)):
                runner.debug_dir = (
                    output_dir
                    / "debug"
                    / f"m_{queue_index:04d}"
                    / f"attempt_{local_attempt:04d}"
                )
                result = runner.run_preserved_attempt(candidate, global_attempt)
                row = result.to_dict()
                row.update(
                    {
                        "created_at": _utc_now(),
                        "observation_id": prediction["observation_id"],
                        "queue_index": queue_index,
                        "attempted_m_index": attempted_m_count - 1,
                        "attempt_within_m": local_attempt,
                        "selection": queue_row["selection"],
                        "initial_log_density": queue_row["initial_log_density"],
                    }
                )
                append_jsonl(output_dir / "manipulation_attempts.jsonl", row)
                result_rows.append(row)
                global_attempt += 1
                m_attempts += 1
                needs_retraction = True
                m_last_failure = (
                    None
                    if result.success
                    else f"{result.failure_stage}:{result.failure_reason}"
                )
                print(
                    f"[N2M][GRASP_ATTEMPT] global={global_attempt}/"
                    f"{args.n2m_max_candidates * args.max_grasp_attempts} "
                    f"m={attempted_m_count}/{args.n2m_max_candidates} "
                    f"local={local_attempt + 1}/{args.max_grasp_attempts} "
                    f"success={result.success} completed={result.completed_stage} "
                    f"failure={m_last_failure}",
                    flush=True,
                )
                if result.success:
                    m_success = True
                    lift_succeeded = True
                    outcome = "lift_success"
                    selected_candidate = dict(candidate)
                    selected_candidate["cube_pose_after_lift"] = runner.current_object_pose()
                    break
                if result.failure_stage == "OPERATOR":
                    if controller.quit:
                        raise KeyboardInterrupt
                    break
                if result.failure_stage in {"SAFETY", "INTERNAL", "EPISODE"}:
                    outcome = "fatal_manipulation_failure"
                    fatal = True
                    break

            append_jsonl(
                output_dir / "m_summaries.jsonl",
                {
                    "created_at": _utc_now(),
                    "queue_index": queue_index,
                    "attempted_m_index": attempted_m_count - 1,
                    "candidate_id": candidate["candidate_id"],
                    "base_pose": candidate["base_pose"],
                    "status": (
                        "success"
                        if m_success
                        else "fatal_failure" if fatal else "attempts_exhausted"
                    ),
                    "attempts": m_attempts,
                    "last_failure": m_last_failure,
                    "cube_pose_after": runner.current_object_pose(),
                },
            )
            if m_success or fatal:
                break

        if not lift_succeeded:
            if not fatal and attempted_m_count >= int(args.n2m_max_candidates):
                outcome = "grasp_attempts_exhausted"
            print(
                f"[N2M][GRASP_EXHAUSTED] outcome={outcome} "
                f"attempted_m={attempted_m_count}/"
                f"{args.n2m_max_candidates} attempts={len(result_rows)}/"
                f"{args.n2m_max_candidates * args.max_grasp_attempts}",
                flush=True,
            )
            return_code = 2
        else:
            write_json(
                output_dir / "selected_candidate.json",
                {
                    "candidate": selected_candidate,
                    "attempted_m_count": attempted_m_count,
                    "grasp_attempt_count": len(result_rows),
                },
            )
            delivery_retraction_metrics = (
                runner.retract_lifted_object_for_delivery()
            )
            write_json(
                output_dir / "lift_retraction_result.json",
                delivery_retraction_metrics,
            )
            bridge = IsaacROS2Bridge(
                cmd_vel_timeout=nav2_settings["velocity_timeout_s"],
                feedback_timeout=bridge_feedback_timeout,
            )
            bridge.start(timeout=30.0)
            bridge_started = True
            delivery_metrics = deliver_lifted_object(
                env=env,
                runner=runner,
                policy=policy,
                bridge=bridge,
                device=device,
                interactive=controller,
                destination=args.destination,
                destination_max_steps=args.destination_max_steps,
                render=args.render,
                nav2_settings=nav2_settings,
            )
            write_json(output_dir / "delivery_result.json", delivery_metrics)
            outcome = "task_success"
            return_code = 0
    except KeyboardInterrupt:
        outcome = "keyboard_interrupt"
        return_code = 130
    except EpisodeFailure as exc:
        outcome = (
            "lift_retraction_failure"
            if exc.stage == "LIFT_RETRACT"
            else "manipulation_transition_failure"
        )
        print(f"[N2M][INTEGRATED_ERROR] {exc.stage}: {exc.reason}", flush=True)
        return_code = 2
    except Exception as exc:
        if outcome == "lift_success":
            outcome = "delivery_failure"
        elif outcome == "not_started":
            outcome = "integration_failure"
        print(f"[N2M][INTEGRATED_ERROR] {type(exc).__name__}: {exc}", flush=True)
        return_code = 2
    finally:
        if runner is not None:
            runner.close_attached()
        if bridge_started:
            try:
                bridge.cancel_goal()
                bridge.stop()
            except Exception:
                pass
        final_target = None
        try:
            final_target = provider.estimate(env, args.object).pose_wxyz()
        except Exception:
            pass
        write_json(
            output_dir / "summary.json",
            {
                **run_config,
                "completed_at": _utc_now(),
                "outcome": outcome,
                "success": outcome == "task_success",
                "lift_succeeded": lift_succeeded,
                "selected_candidate": selected_candidate,
                "attempt_count": len(result_rows),
                "attempted_m_count": attempted_m_count,
                "collision_rejected_m_count": collision_rejected_m_count,
                "lift_retraction": delivery_retraction_metrics,
                "delivery": delivery_metrics,
                "final_target_pose": final_target,
            },
        )
        print(
            f"[N2M][INTEGRATED_SUMMARY] outcome={outcome} "
            f"attempts={len(result_rows)} output={output_dir}",
            flush=True,
        )
        env.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
