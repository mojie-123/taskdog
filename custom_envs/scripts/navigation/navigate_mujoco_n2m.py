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
from custom_envs.scripts.N2M.geometry import relative_se2, rpy_to_quat_wxyz
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
    robot_table_contacts,
)
from custom_envs.scripts.N2M.target_provider import (
    MuJoCoTruthTargetProvider,
    navigation_target_metrics,
    observation_gate,
    standoff_goal,
)
from custom_envs.scripts.N2M.retry_n2m_grasp_until_success import pose_motion


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
PREDOCK_ROTATION_SAMPLES = 24
PREDOCK_DOCK_SAMPLE_SPACING_M = 0.04
PREDOCK_DOCK_YAW_MARGIN_RAD = math.radians(3.0)
PREDOCK_ROTATION_DRIFT_MARGIN_M = 0.27

# Keep these values in lock-step with the PAN_XY controller that has already
# been validated in navigate_mujoco_moveit.py.  The locomotion policy tends to
# oscillate in the 0.1--0.2 m/s range, so a moving axis uses an effective
# minimum command and crosses the low-speed region with a bounded slew.
PAN_MIN_MOVE_SPEED = 0.25
PAN_DV_PER_STEP = 0.030
PAN_DW_PER_STEP = 0.060
PAN_YAW_KP = 3.0
PAN_YAW_MIN = 0.30
PAN_YAW_MAX = 0.60
PAN_YAW_ENTER_TOL_RAD = math.radians(2.0)
PAN_POSITION_RESUME_MARGIN_M = 0.03
# The final P-to-M controller uses a set-valued terminal pose instead of
# chasing one exact SE(2) state.  It first drives into a deliberately tighter
# inner capture region, then keeps every command at zero while the base settles.
# Small gait drift is tolerated inside the outer region, which prevents the
# yaw -> lateral drift -> XY correction limit cycle observed near M.
PAN_TERMINAL_LONGITUDINAL_INNER_MARGIN_M = 0.01
PAN_TERMINAL_LATERAL_INNER_MARGIN_M = 0.02
PAN_TERMINAL_POSITION_OUTER_MARGIN_M = 0.01
PAN_TERMINAL_YAW_CAPTURE_MARGIN_RAD = math.radians(2.0)
PAN_TERMINAL_YAW_HARD_MARGIN_RAD = math.radians(4.0)
PAN_TERMINAL_YAW_RECHECK_STEPS = 15
# N2M predicts a distribution of usable manipulation poses, not one exact
# setpoint.  For the final docking stage the selected M remains the collision-
# checked reference, while its Gaussian component supplies the shape and scale
# of a local terminal region.  Physical caps below deliberately remain much
# tighter than the raw learned covariance.
N2M_TERMINAL_ENTER_RADIUS = 1.0
N2M_TERMINAL_HOLD_RADIUS = 1.5
N2M_TERMINAL_SLOW_RADIUS = 2.0
N2M_TERMINAL_HOLD_POSITION_MARGIN_M = 0.02
N2M_TERMINAL_EXIT_FRAMES = 15
N2M_TERMINAL_MIN_SPEED_SCALE = 0.35
N2M_TERMINAL_SLOW_OUTSIDE_M = 0.18
N2M_TERMINAL_SLOW_LATERAL_M = 0.15
N2M_TERMINAL_SLOW_YAW_RAD = math.radians(20.0)
PAN_SETTLE_LINEAR_RATE = 0.03
PAN_SETTLE_YAW_RATE = 0.10
PAN_SETTLE_WINDOW_FRAMES = 15
PAN_SETTLE_STEPS = 8
PAN_DEFAULT_VMAX_X = 0.25
PAN_DEFAULT_VMAX_Y = 0.35

ALIGN_YAW_KP = 4.0
ALIGN_YAW_MIN = 0.40
ALIGN_YAW_MAX = 0.90
ALIGN_YAW_DW_PER_STEP = 0.08
ALIGN_YAW_ENTER_TOL_RAD = math.radians(2.0)
ALIGN_YAW_RESUME_TOL_RAD = math.radians(3.0)
ALIGN_YAW_RATE_TOL = 0.15
ALIGN_YAW_SETTLE_STEPS = 6
NAV_OBSERVATION_YAW_ENTER_RAD = math.radians(3.0)
NAV_OBSERVATION_YAW_RESUME_RAD = math.radians(6.0)


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
        moveit = self.moveit
        self.moveit = None
        if moveit is not None:
            moveit.stop()
        # Drop the overlay's viewer/env references before the MuJoCo viewer is
        # closed by main().  close_attached() is intentionally idempotent.
        self.overlay = None


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


def _base_pose_from_state(state: dict) -> np.ndarray:
    return np.array(
        [state["pos"][0], state["pos"][1], state["yaw"]],
        dtype=np.float64,
    )


def _body_frame_xy_error(current_pose, target_xy) -> np.ndarray:
    """Return a world-frame XY target error expressed in the robot frame."""
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    delta = target[:2] - current[:2]
    c, s = math.cos(float(current[2])), math.sin(float(current[2]))
    return np.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]],
        dtype=np.float64,
    )


def _approach_frame_position_errors(
    current_pose, target_pose, approach_direction
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return target-current errors along/across the P-to-M direction.

    Positive longitudinal error means the base is still outside M and farther
    from the object.  Negative longitudinal error means it has moved inward,
    past M and toward the object.
    """
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    inward = np.asarray(approach_direction, dtype=np.float64)[:2]
    norm = float(np.linalg.norm(inward))
    if norm < 1e-9:
        raise ValueError("approach_direction must be non-zero")
    inward = inward / norm
    lateral = np.array([-inward[1], inward[0]], dtype=np.float64)
    delta = target[:2] - current[:2]
    return (
        float(np.dot(delta, inward)),
        float(np.dot(delta, lateral)),
        inward,
        lateral,
    )


def _n2m_candidate_terminal_region(
    prediction: dict,
    queue_row: dict,
    *,
    enter_radius: float = N2M_TERMINAL_ENTER_RADIUS,
    hold_radius: float = N2M_TERMINAL_HOLD_RADIUS,
    slow_radius: float = N2M_TERMINAL_SLOW_RADIUS,
) -> dict:
    """Build a covariance-shaped terminal region for one selected M.

    The reference pose is the collision-checked candidate (a mode or sample).
    Its covariance comes from the Gaussian component that most likely produced
    it.  Centering the local region on the selected candidate is important:
    later queue entries may be used precisely because the global mode mean had
    no collision-free approach.
    """
    radii = np.asarray(
        [enter_radius, hold_radius, slow_radius], dtype=np.float64
    )
    if (
        not np.isfinite(radii).all()
        or np.any(radii <= 0.0)
        or not (radii[0] < radii[1] < radii[2])
    ):
        raise ValueError(
            "N2M terminal radii must satisfy 0 < enter < hold < slow"
        )

    selection = queue_row["selection"]
    reference = np.asarray(
        selection["pose_observation"], dtype=np.float64
    ).reshape(3)
    modes = prediction.get("modes", [])
    if not modes:
        raise ValueError("N2M prediction contains no Gaussian modes")

    if selection.get("kind") == "mode":
        component_index = int(selection["index"])
    else:
        component_scores = []
        for index, mode in enumerate(modes):
            mean = np.asarray(
                mode["pose_observation"], dtype=np.float64
            ).reshape(3)
            covariance = np.asarray(
                mode["covariance"], dtype=np.float64
            ).reshape(3, 3)
            covariance = 0.5 * (covariance + covariance.T)
            difference = reference - mean
            difference[2] = _wrap_angle(float(difference[2]))
            sign, logdet = np.linalg.slogdet(covariance)
            weight = float(mode.get("weight", 0.0))
            if sign <= 0.0 or weight <= 0.0:
                component_scores.append(float("-inf"))
                continue
            mahalanobis = float(
                difference @ np.linalg.solve(covariance, difference)
            )
            component_scores.append(
                math.log(weight) - 0.5 * (logdet + mahalanobis)
            )
        component_index = int(np.argmax(component_scores))
        if not math.isfinite(component_scores[component_index]):
            raise ValueError("N2M sample has no valid Gaussian component")

    if not 0 <= component_index < len(modes):
        raise ValueError(f"invalid N2M component index {component_index}")
    covariance = np.asarray(
        modes[component_index]["covariance"], dtype=np.float64
    ).reshape(3, 3)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or float(eigenvalues[0]) <= 0.0:
        raise ValueError("N2M terminal covariance is not positive definite")

    return {
        "observation_pose_world": np.asarray(
            prediction["observation_pose_world"], dtype=np.float64
        ).reshape(3),
        "reference_pose_observation": reference,
        "component_index": component_index,
        "covariance": covariance,
        "precision": np.linalg.inv(covariance),
        "enter_radius": float(enter_radius),
        "hold_radius": float(hold_radius),
        "slow_radius": float(slow_radius),
    }


def _n2m_terminal_pose_metrics(current_pose, terminal_region: dict) -> dict:
    """Evaluate a world pose in a candidate-local N2M Gaussian region."""
    current_observation = relative_se2(
        terminal_region["observation_pose_world"], current_pose
    )
    error = np.asarray(current_observation, dtype=np.float64) - np.asarray(
        terminal_region["reference_pose_observation"], dtype=np.float64
    )
    error[2] = _wrap_angle(float(error[2]))
    mahalanobis_sq = float(
        error @ np.asarray(terminal_region["precision"]) @ error
    )
    # Numerical roundoff can produce a tiny negative number around the center.
    mahalanobis_sq = max(0.0, mahalanobis_sq)
    return {
        "current_pose_observation": current_observation,
        "error_observation": error,
        "mahalanobis_sq": mahalanobis_sq,
        "mahalanobis_radius": math.sqrt(mahalanobis_sq),
    }


def _new_pan_xy_control_state() -> dict:
    """Return the persistent command/settling state used by PAN_XY."""
    return {
        "cmd_vx": 0.0,
        "cmd_vy": 0.0,
        "cmd_wz": 0.0,
        "settling": False,
        "yaw_settling": False,
        "terminal_capture": False,
        "terminal_exit_count": 0,
        "settle_count": 0,
        "pose_window": [],
    }


def _pan_world_xy_control_step(
    control: dict,
    *,
    current_pose,
    target_pose,
    base_lin_vel,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
    step_dt: float = 0.02,
    settle_window_frames: int = PAN_SETTLE_WINDOW_FRAMES,
    resume_tolerance_m: float | None = None,
    approach_direction=None,
    inward_tolerance_m: float = 0.0,
    terminal_region: dict | None = None,
    control_yaw: bool = True,
    kp_x: float = 1.0,
    kp_y: float = 0.8,
    vmax_x: float = PAN_DEFAULT_VMAX_X,
    vmax_y: float = PAN_DEFAULT_VMAX_Y,
) -> tuple[tuple[float, float, float], bool, dict]:
    """One exact PAN_XY-style smooth-control update.

    X/Y are closed independently in the world frame and then transformed to
    body-frame commands.  Command slew, the minimum effective walking speed,
    hysteretic stopping match the controller in
    ``navigate_mujoco_moveit.py``.  Directional P-to-M control adds a tighter
    inner capture region and a wider zero-command hold region.  When an N2M
    terminal region is supplied, PAN_XY and yaw remain jointly active outside
    that region, commands are reduced in an outer slowdown contour, and entry
    is latched until the pose has remained outside the hold contour for several
    frames.  The terminal check uses a short pose trend rather than
    instantaneous base velocity so periodic gait sway does not prevent an
    otherwise stationary manipulation pose from being accepted.
    """
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    ex = float(target[0] - current[0])
    ey = float(target[1] - current[1])
    ax = abs(ex)
    ay = abs(ey)
    yaw_error = _wrap_angle(float(target[2] - current[2]))
    tolerance = float(position_tolerance_m)
    resume = (
        tolerance + PAN_POSITION_RESUME_MARGIN_M
        if resume_tolerance_m is None
        else float(resume_tolerance_m)
    )
    directional = approach_direction is not None
    n2m_terminal_enabled = directional and terminal_region is not None
    yaw_control_enabled = bool(control_yaw) and (
        not directional or n2m_terminal_enabled
    )
    terminal_metrics = (
        _n2m_terminal_pose_metrics(current, terminal_region)
        if n2m_terminal_enabled
        else None
    )
    mahalanobis_radius = (
        float(terminal_metrics["mahalanobis_radius"])
        if terminal_metrics is not None
        else None
    )
    terminal_speed_scale = 1.0
    inside_slow_region = False
    if directional:
        if float(inward_tolerance_m) <= 0.0:
            raise ValueError("inward_tolerance_m must be positive")
        longitudinal_error, lateral_error, inward_axis, lateral_axis = (
            _approach_frame_position_errors(
                current, target, approach_direction
            )
        )
        if n2m_terminal_enabled:
            # N2M supplies the statistical shape of the terminal set.  The
            # explicit approach/lateral caps retain the known manipulation
            # geometry: finishing slightly inward is allowed, but finishing
            # outside (farther from the table than M) is not.
            longitudinal_inner_bounds = (
                -float(inward_tolerance_m),
                -min(
                    PAN_TERMINAL_LONGITUDINAL_INNER_MARGIN_M,
                    0.25 * float(inward_tolerance_m),
                ),
            )
            lateral_inner_tolerance = tolerance
            longitudinal_outer_bounds = (
                -float(inward_tolerance_m)
                - N2M_TERMINAL_HOLD_POSITION_MARGIN_M,
                0.0,
            )
            lateral_outer_tolerance = (
                tolerance + N2M_TERMINAL_HOLD_POSITION_MARGIN_M
            )
        else:
            longitudinal_inner_margin = min(
                PAN_TERMINAL_LONGITUDINAL_INNER_MARGIN_M,
                0.25 * float(inward_tolerance_m),
            )
            lateral_inner_margin = min(
                PAN_TERMINAL_LATERAL_INNER_MARGIN_M,
                0.4 * tolerance,
            )
            longitudinal_inner_bounds = (
                -float(inward_tolerance_m) + longitudinal_inner_margin,
                -longitudinal_inner_margin,
            )
            lateral_inner_tolerance = tolerance - lateral_inner_margin
            longitudinal_outer_bounds = (
                -float(inward_tolerance_m)
                - PAN_TERMINAL_POSITION_OUTER_MARGIN_M,
                # Preserve the manipulation constraint: the base may settle
                # inward of M, but it must never finish outside M.
                0.0,
            )
            lateral_outer_tolerance = (
                tolerance + PAN_TERMINAL_POSITION_OUTER_MARGIN_M
            )
        yaw_capture_tolerance = (
            float(yaw_tolerance_rad)
            + PAN_TERMINAL_YAW_CAPTURE_MARGIN_RAD
        )
        yaw_hard_tolerance = (
            float(yaw_tolerance_rad)
            + PAN_TERMINAL_YAW_HARD_MARGIN_RAD
        )
        in_x = (
            longitudinal_inner_bounds[0]
            <= longitudinal_error
            <= longitudinal_inner_bounds[1]
        )
        in_y = abs(lateral_error) <= lateral_inner_tolerance
        inside_inner_position = in_x and in_y and (
            not n2m_terminal_enabled
            or mahalanobis_radius <= float(terminal_region["enter_radius"])
        )
        inside_outer_position = (
            longitudinal_outer_bounds[0]
            <= longitudinal_error
            <= longitudinal_outer_bounds[1]
            and abs(lateral_error) <= lateral_outer_tolerance
            and (
                not n2m_terminal_enabled
                or mahalanobis_radius
                <= float(terminal_region["hold_radius"])
            )
        )
        if n2m_terminal_enabled:
            inside_slow_region = (
                mahalanobis_radius
                <= float(terminal_region["slow_radius"])
                and (
                    -float(inward_tolerance_m)
                    - N2M_TERMINAL_SLOW_OUTSIDE_M
                    <= longitudinal_error
                    <= N2M_TERMINAL_SLOW_OUTSIDE_M
                )
                and abs(lateral_error) <= N2M_TERMINAL_SLOW_LATERAL_M
                and abs(yaw_error) <= N2M_TERMINAL_SLOW_YAW_RAD
            )
            if inside_slow_region:
                enter_radius = float(terminal_region["enter_radius"])
                slow_radius = float(terminal_region["slow_radius"])
                radial_fraction = np.clip(
                    (mahalanobis_radius - enter_radius)
                    / max(slow_radius - enter_radius, 1e-9),
                    0.0,
                    1.0,
                )
                terminal_speed_scale = float(
                    N2M_TERMINAL_MIN_SPEED_SCALE
                    + (1.0 - N2M_TERMINAL_MIN_SPEED_SCALE)
                    * radial_fraction
                )
    else:
        longitudinal_error = None
        lateral_error = None
        inward_axis = None
        lateral_axis = None
        longitudinal_inner_bounds = None
        lateral_inner_tolerance = None
        longitudinal_outer_bounds = None
        lateral_outer_tolerance = None
        yaw_capture_tolerance = None
        yaw_hard_tolerance = None
        in_x = ax <= tolerance
        in_y = ay <= tolerance
        inside_inner_position = in_x and in_y
        inside_outer_position = inside_inner_position

    if directional:
        if control["terminal_capture"]:
            left_capture_region = (
                not inside_outer_position
                or abs(yaw_error) > yaw_hard_tolerance
            )
            if n2m_terminal_enabled and left_capture_region:
                control["terminal_exit_count"] += 1
            else:
                control["terminal_exit_count"] = 0
            if left_capture_region and (
                not n2m_terminal_enabled
                or control["terminal_exit_count"]
                >= N2M_TERMINAL_EXIT_FRAMES
            ):
                control["terminal_capture"] = False
                control["settling"] = False
                control["yaw_settling"] = False
                control["terminal_exit_count"] = 0
                control["settle_count"] = 0
                control["pose_window"].clear()
        # Once the full terminal set is entered, position capture has priority
        # over residual XY/yaw commands.  The controller then ramps all three
        # commands to zero and does not restart for one noisy boundary sample.
        if (
            not control["terminal_capture"]
            and inside_inner_position
            and abs(yaw_error) <= yaw_capture_tolerance
        ):
            control["terminal_capture"] = True
            control["settling"] = True
            control["yaw_settling"] = True
            control["terminal_exit_count"] = 0
            control["settle_count"] = 0
            control["pose_window"].clear()
        control["settling"] = bool(control["terminal_capture"])
    else:
        if control["settling"]:
            if ax > resume or ay > resume:
                control["settling"] = False
                control["settle_count"] = 0
        elif in_x and in_y:
            control["settling"] = True
            control["settle_count"] = 0

    if control["settling"]:
        world_vx_target = 0.0
        world_vy_target = 0.0
    else:
        if directional:
            effective_vmax_x = float(vmax_x) * terminal_speed_scale
            effective_vmax_y = float(vmax_y) * terminal_speed_scale
            effective_min_move_speed = (
                PAN_MIN_MOVE_SPEED * terminal_speed_scale
            )
            if n2m_terminal_enabled and not inside_inner_position:
                longitudinal_distance = longitudinal_error - 0.5 * (
                    longitudinal_inner_bounds[0]
                    + longitudinal_inner_bounds[1]
                )
                lateral_distance = lateral_error
                longitudinal_velocity = (
                    0.0
                    if abs(longitudinal_distance) <= 0.005
                    else math.copysign(
                        min(
                            effective_vmax_x,
                            max(
                                effective_min_move_speed,
                                float(kp_x) * abs(longitudinal_distance),
                            ),
                        ),
                        longitudinal_distance,
                    )
                )
                lateral_velocity = (
                    0.0
                    if abs(lateral_distance) <= 0.005
                    else math.copysign(
                        min(
                            effective_vmax_y,
                            max(
                                effective_min_move_speed,
                                float(kp_y) * abs(lateral_distance),
                            ),
                        ),
                        lateral_distance,
                    )
                )
            elif in_x:
                longitudinal_velocity = 0.0
            else:
                longitudinal_distance = (
                    longitudinal_error - longitudinal_inner_bounds[1]
                    if longitudinal_error > longitudinal_inner_bounds[1]
                    else longitudinal_error - longitudinal_inner_bounds[0]
                )
                longitudinal_velocity = math.copysign(
                    min(
                        effective_vmax_x,
                        max(
                            effective_min_move_speed,
                            float(kp_x) * abs(longitudinal_distance),
                        ),
                    ),
                    longitudinal_distance,
                )
            if not (n2m_terminal_enabled and not inside_inner_position):
                if in_y:
                    lateral_velocity = 0.0
                else:
                    lateral_distance = math.copysign(
                        abs(lateral_error) - lateral_inner_tolerance,
                        lateral_error,
                    )
                    lateral_velocity = math.copysign(
                        min(
                            effective_vmax_y,
                            max(
                                effective_min_move_speed,
                                float(kp_y) * abs(lateral_distance),
                            ),
                        ),
                        lateral_distance,
                    )
            world_velocity = (
                longitudinal_velocity * inward_axis
                + lateral_velocity * lateral_axis
            )
            world_vx_target = float(world_velocity[0])
            world_vy_target = float(world_velocity[1])
        else:
            world_vx_target = 0.0
            if not in_x:
                world_vx_target = math.copysign(
                    min(
                        float(vmax_x),
                        max(PAN_MIN_MOVE_SPEED, float(kp_x) * ax),
                    ),
                    ex,
                )
            world_vy_target = 0.0
            if not in_y:
                world_vy_target = math.copysign(
                    min(
                        float(vmax_y),
                        max(PAN_MIN_MOVE_SPEED, float(kp_y) * ay),
                    ),
                    ey,
                )

    yaw = float(current[2])
    c, s = math.cos(yaw), math.sin(yaw)
    vx_target = c * world_vx_target + s * world_vy_target
    vy_target = -s * world_vx_target + c * world_vy_target
    control["cmd_vx"] = _slew(
        control["cmd_vx"], vx_target, PAN_DV_PER_STEP
    )
    control["cmd_vy"] = _slew(
        control["cmd_vy"], vy_target, PAN_DV_PER_STEP
    )

    if control["terminal_capture"]:
        control["yaw_settling"] = True
        wz_target = 0.0
    elif not yaw_control_enabled:
        # P-to-M is translation-only.  If its terminal yaw is unacceptable,
        # the caller retreats to P and runs the safe yaw-alignment phase
        # again; P-centering likewise leaves yaw to the following dedicated
        # alignment phase.  In both cases PAN_XY must never rotate the base.
        control["yaw_settling"] = True
        wz_target = 0.0
    else:
        yaw_enter_tolerance = min(
            float(yaw_tolerance_rad), PAN_YAW_ENTER_TOL_RAD
        )
        yaw_resume_tolerance = float(yaw_tolerance_rad)
        if control["yaw_settling"]:
            if abs(yaw_error) > yaw_resume_tolerance:
                control["yaw_settling"] = False
                control["settle_count"] = 0
        elif abs(yaw_error) <= yaw_enter_tolerance:
            control["yaw_settling"] = True
            control["settle_count"] = 0
        wz_target = (
            0.0
            if control["yaw_settling"]
            else _yaw_velocity_target(
                yaw_error,
                deadband=yaw_enter_tolerance,
                kp=PAN_YAW_KP,
                min_abs=PAN_YAW_MIN * terminal_speed_scale,
                max_abs=PAN_YAW_MAX * terminal_speed_scale,
            )
        )
    control["cmd_wz"] = _slew(
        control["cmd_wz"], wz_target, PAN_DW_PER_STEP
    )

    velocity_xy = np.asarray(base_lin_vel, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(velocity_xy))
    if directional:
        within_terminal_pose = (
            control["terminal_capture"]
            and inside_outer_position
            and abs(yaw_error)
            <= (
                yaw_hard_tolerance
                if n2m_terminal_enabled
                else yaw_capture_tolerance
            )
        )
    else:
        within_terminal_pose = (
            control["settling"]
            and control["yaw_settling"]
            and in_x
            and in_y
            and (
                not yaw_control_enabled
                or abs(yaw_error) <= float(yaw_tolerance_rad)
            )
        )
    if within_terminal_pose:
        control["pose_window"].append(current.copy())
        control["pose_window"] = control["pose_window"][-int(settle_window_frames):]
    else:
        control["pose_window"].clear()

    window_ready = len(control["pose_window"]) >= int(settle_window_frames)
    if window_ready:
        trend_speed, trend_yaw_rate = _pose_window_motion_rates(
            control["pose_window"], step_dt
        )
    else:
        trend_speed = float("inf")
        trend_yaw_rate = float("inf")
    commands_stopped = (
        math.hypot(control["cmd_vx"], control["cmd_vy"])
        <= PAN_DV_PER_STEP
        and abs(control["cmd_wz"]) <= PAN_DW_PER_STEP
    )
    if (
        within_terminal_pose
        and window_ready
        and commands_stopped
        and trend_speed <= PAN_SETTLE_LINEAR_RATE
        and (
            not yaw_control_enabled
            or trend_yaw_rate <= PAN_SETTLE_YAW_RATE
        )
    ):
        control["settle_count"] += 1
    else:
        control["settle_count"] = 0
    done = control["settle_count"] >= PAN_SETTLE_STEPS
    command = (
        float(control["cmd_vx"]),
        float(control["cmd_vy"]),
        float(control["cmd_wz"]),
    )
    return command, done, {
        "err_x": ex,
        "err_y": ey,
        "abs_err_x": ax,
        "abs_err_y": ay,
        "xy_error_m": float(math.hypot(ex, ey)),
        "yaw_error_rad": yaw_error,
        "speed_mps": speed,
        "trend_speed_mps": trend_speed,
        "trend_yaw_rate_radps": trend_yaw_rate,
        "settle_window_count": len(control["pose_window"]),
        "settle_window_frames": int(settle_window_frames),
        "commands_stopped": commands_stopped,
        "in_x": in_x,
        "in_y": in_y,
        "settling": bool(control["settling"]),
        "yaw_settling": bool(control["yaw_settling"]),
        "terminal_capture": bool(control["terminal_capture"]),
        "terminal_exit_count": int(control["terminal_exit_count"]),
        "settle_count": int(control["settle_count"]),
        "world_vx_target": world_vx_target,
        "world_vy_target": world_vy_target,
        "resume_tolerance_m": resume,
        "directional_position_control": directional,
        "yaw_control_enabled": yaw_control_enabled,
        "n2m_terminal_enabled": bool(n2m_terminal_enabled),
        "n2m_component_index": (
            int(terminal_region["component_index"])
            if n2m_terminal_enabled
            else None
        ),
        "mahalanobis_sq": (
            float(terminal_metrics["mahalanobis_sq"])
            if terminal_metrics is not None
            else None
        ),
        "mahalanobis_radius": mahalanobis_radius,
        "inside_slow_region": bool(inside_slow_region),
        "terminal_speed_scale": float(terminal_speed_scale),
        "longitudinal_error_m": longitudinal_error,
        "lateral_error_m": lateral_error,
        "inward_tolerance_m": (
            float(inward_tolerance_m) if directional else None
        ),
        "longitudinal_inner_bounds_m": longitudinal_inner_bounds,
        "lateral_inner_tolerance_m": lateral_inner_tolerance,
        "longitudinal_outer_bounds_m": longitudinal_outer_bounds,
        "lateral_outer_tolerance_m": lateral_outer_tolerance,
        "yaw_capture_tolerance_rad": yaw_capture_tolerance,
        "yaw_hard_tolerance_rad": yaw_hard_tolerance,
        "terminal_yaw_realign_required": bool(
            directional
            and not yaw_control_enabled
            and (inside_inner_position or control["terminal_capture"])
            and abs(yaw_error) > yaw_capture_tolerance
        ),
        "inside_inner_position": bool(inside_inner_position),
        "inside_outer_position": bool(inside_outer_position),
    }


def build_predock_pose_options(
    manipulation_pose,
    cube_pose,
    *,
    distance_m: float,
    distance_step_m: float,
    distance_count: int,
    tangent_offset_m: float,
) -> list[np.ndarray]:
    """Generate the original cube-radial P poses outside the table.

    P lies on the ray from the cube through M, so every P-to-M transition
    advances toward the object instead of first driving to the body-front side
    of M and then reversing.  For each radial distance the centerline option is
    followed by the two tangent-offset fallbacks used by the initial physical
    transition implementation.
    """
    manipulation = np.asarray(manipulation_pose, dtype=np.float64)
    cube = np.asarray(cube_pose, dtype=np.float64)
    away = manipulation[:2] - cube[:2]
    distance_to_cube = float(np.linalg.norm(away))
    if distance_to_cube < 1e-9:
        raise ValueError("manipulation pose and cube XY must not coincide")
    away /= distance_to_cube
    tangent = np.array([-away[1], away[0]], dtype=np.float64)
    tangent_offset = float(tangent_offset_m)

    options = []
    for index in range(int(distance_count)):
        distance = distance_m + index * distance_step_m
        center = manipulation[:2] + distance * away
        offsets = [0.0]
        if tangent_offset > 0.0:
            offsets.extend((tangent_offset, -tangent_offset))
        for offset in offsets:
            xy = center + offset * tangent
            options.append(
                np.array([xy[0], xy[1], manipulation[2]], dtype=np.float64)
            )
    return options


def _sample_pose_contacts(env, pose, runner) -> tuple[list[str], list[str]]:
    env.set_root_pose(
        np.array(
            [pose[0], pose[1], runner.cfg.robot_root_z], dtype=np.float64
        ),
        rpy_to_quat_wxyz(
            runner.cfg.robot_root_roll,
            runner.cfg.robot_root_pitch,
            pose[2],
        ),
    )
    return robot_environment_contacts(env), robot_table_contacts(env)


def preflight_predock_motion(
    env,
    runner,
    manipulation_pose,
    predock_options,
    *,
    rotation_clearance_radius_m: float,
    dock_inward_tolerance_m: float = 0.0,
    dock_lateral_tolerance_m: float = 0.0,
    dock_yaw_margin_rad: float = PREDOCK_DOCK_YAW_MARGIN_RAD,
) -> dict:
    """Collision-check P rotation clearance and the complete P-to-M corridor.

    The simulator state is restored exactly after probing; this is a read-only
    geometric preflight, not another base teleport in the execution path.
    """
    import mujoco

    manipulation = np.asarray(manipulation_pose, dtype=np.float64)
    data = env.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    time_value = float(data.time)
    eq_active = data.eq_active.copy()
    mocap_pos = data.mocap_pos.copy()
    mocap_quat = data.mocap_quat.copy()
    rejected = []
    selected = None
    selected_samples = 0
    try:
        for option_index, predock in enumerate(predock_options):
            collision = None
            sample_count = 0
            margin = float(rotation_clearance_radius_m)
            rotation_offsets = [(0.0, 0.0)]
            rotation_offsets.extend(
                (
                    margin * math.cos(angle),
                    margin * math.sin(angle),
                )
                for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
            )
            for dx, dy in rotation_offsets:
                for yaw in np.linspace(
                    -math.pi,
                    math.pi,
                    PREDOCK_ROTATION_SAMPLES,
                    endpoint=False,
                ):
                    pose = np.array(
                        [predock[0] + dx, predock[1] + dy, yaw],
                        dtype=np.float64,
                    )
                    environment, table = _sample_pose_contacts(
                        env, pose, runner
                    )
                    sample_count += 1
                    if environment or table:
                        collision = {
                            "stage": "predock_rotation",
                            "pose": pose.tolist(),
                            "environment_contacts": environment,
                            "table_contacts": table,
                        }
                        break
                if collision is not None:
                    break

            if collision is None:
                corridor = manipulation[:2] - predock[:2]
                corridor_length = float(np.linalg.norm(corridor))
                if corridor_length < 1e-9:
                    raise ValueError("predock and manipulation pose coincide")
                inward = corridor / corridor_length
                lateral = np.array([-inward[1], inward[0]], dtype=np.float64)
                checked_endpoint = (
                    manipulation[:2]
                    + float(dock_inward_tolerance_m) * inward
                )
                distance = float(
                    np.linalg.norm(checked_endpoint - predock[:2])
                )
                count = max(
                    2,
                    int(
                        math.ceil(
                            distance / PREDOCK_DOCK_SAMPLE_SPACING_M
                        )
                    )
                    + 1,
                )
                lateral_margin = float(dock_lateral_tolerance_m)
                lateral_offsets = (
                    (0.0,)
                    if lateral_margin <= 0.0
                    else (-lateral_margin, 0.0, lateral_margin)
                )
                for alpha in np.linspace(0.0, 1.0, count):
                    xy = (
                        (1.0 - alpha) * predock[:2]
                        + alpha * checked_endpoint
                    )
                    for lateral_offset in lateral_offsets:
                        sample_xy = xy + lateral_offset * lateral
                        for yaw_delta in (
                            -float(dock_yaw_margin_rad),
                            0.0,
                            float(dock_yaw_margin_rad),
                        ):
                            pose = np.array(
                                [
                                    sample_xy[0],
                                    sample_xy[1],
                                    manipulation[2] + yaw_delta,
                                ],
                                dtype=np.float64,
                            )
                            environment, table = _sample_pose_contacts(
                                env, pose, runner
                            )
                            sample_count += 1
                            if environment or table:
                                collision = {
                                    "stage": "dock_corridor",
                                    "pose": pose.tolist(),
                                    "environment_contacts": environment,
                                    "table_contacts": table,
                                }
                                break
                        if collision is not None:
                            break
                    if collision is not None:
                        break

            if collision is None:
                selected = predock.copy()
                selected_samples = sample_count
                break
            rejected.append(
                {
                    "option_index": option_index,
                    "predock_pose": predock.tolist(),
                    "checked_samples": sample_count,
                    "first_collision": collision,
                }
            )
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.time = time_value
        data.eq_active[:] = eq_active
        data.mocap_pos[:] = mocap_pos
        data.mocap_quat[:] = mocap_quat
        mujoco.mj_forward(env.model, data)

    if selected is None:
        selected_approach_mode = None
        selected_axis_alignment = None
    else:
        corridor = manipulation[:2] - selected[:2]
        corridor /= max(float(np.linalg.norm(corridor)), 1e-9)
        heading = np.array(
            [math.cos(manipulation[2]), math.sin(manipulation[2])],
            dtype=np.float64,
        )
        selected_axis_alignment = float(np.dot(corridor, heading))
        selected_approach_mode = "cube_radial"

    return {
        "passed": selected is not None,
        "predock_pose": None if selected is None else selected.tolist(),
        "approach_mode": selected_approach_mode,
        "approach_axis_alignment": selected_axis_alignment,
        "selected_checked_samples": selected_samples,
        "rejected_options": rejected,
        "option_count": len(predock_options),
        "rotation_samples": PREDOCK_ROTATION_SAMPLES,
        "rotation_clearance_radius_m": float(rotation_clearance_radius_m),
        "dock_sample_spacing_m": PREDOCK_DOCK_SAMPLE_SPACING_M,
        "dock_yaw_margin_deg": math.degrees(float(dock_yaw_margin_rad)),
        "dock_inward_tolerance_m": float(dock_inward_tolerance_m),
        "dock_lateral_tolerance_m": float(dock_lateral_tolerance_m),
    }


def _checked_base_motion_step(
    *,
    stage: str,
    env,
    policy,
    device,
    last_action,
    commands,
    arm_target,
    gripper_target,
    pacer: RealtimePacer | None = None,
):
    last_action = _carry_policy_step(
        env,
        policy,
        device,
        last_action,
        commands,
        arm_target,
        gripper_target,
    )
    contacts = robot_environment_contacts(env)
    if contacts:
        raise EpisodeFailure(
            stage,
            "environment_contact_during_base_motion",
            contacts=contacts,
        )
    if pacer is not None:
        pacer.wait()
    return last_action


def navigate_to_predock(
    *,
    env,
    runner,
    policy,
    bridge,
    device,
    interactive,
    predock_pose,
    nav2_settings,
    max_steps: int,
    position_tolerance_m: float,
) -> tuple[np.ndarray, dict]:
    """Use Nav2 for the coarse part of O/current-M to P motion."""
    target = np.asarray(predock_pose, dtype=np.float64)
    initial = env.get_robot_state()
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    last_action = np.zeros(16, dtype=np.float32)
    pacer = RealtimePacer(env.dt)
    bridge.update_robot_pose(
        initial["pos"], initial["quat"], initial["lin_vel"], initial["ang_vel"]
    )
    bridge.send_goal(float(target[0]), float(target[1]))
    progress = {
        "anchor": np.asarray(initial["pos"][:2], dtype=np.float64).copy(),
        "last_progress_time": time.monotonic(),
    }
    started = time.monotonic()
    print(
        f"[N2M][PREDOCK_NAV_START] current="
        f"{np.round(_base_pose_from_state(initial), 3)} "
        f"target={np.round(target, 3)}",
        flush=True,
    )
    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        bridge.update_robot_pose(
            state["pos"], state["quat"], state["lin_vel"], state["ang_vel"]
        )
        distance = float(np.linalg.norm(state["pos"][:2] - target[:2]))
        nav_done, nav_failed = bridge.get_nav_status()
        if distance <= position_tolerance_m or nav_done:
            bridge.cancel_goal()
            trigger = "local_tolerance" if distance <= position_tolerance_m else "nav_done"
            print(
                f"[N2M][PREDOCK_NAV_DONE] trigger={trigger} step={step} "
                f"distance={distance:.3f}m",
                flush=True,
            )
            return last_action, {
                "steps": step,
                "elapsed_s": time.monotonic() - started,
                "trigger": trigger,
                "final_distance_m": distance,
                "final_base_pose": _base_pose_from_state(state).tolist(),
            }
        if nav_failed:
            bridge.cancel_goal()
            raise EpisodeFailure(
                "PREDOCK_NAV",
                "nav2_failed",
                distance_m=distance,
            )
        cmd_vx, cmd_wz = _clip_nav_command(
            *bridge.get_cmd_vel(), nav2_settings
        )
        _update_progress_watchdog(progress, state["pos"][:2], nav2_settings)
        last_action = _checked_base_motion_step(
            stage="PREDOCK_NAV",
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=(cmd_vx, 0.0, cmd_wz),
            arm_target=arm_target,
            gripper_target=gripper_target,
            pacer=pacer,
        )
        if step % 100 == 0:
            print(
                f"[N2M][PREDOCK_NAV] step={step} distance={distance:.3f}m "
                f"cmd=({cmd_vx:+.3f},{cmd_wz:+.3f})",
                flush=True,
            )
    bridge.cancel_goal()
    raise EpisodeFailure(
        "PREDOCK_NAV",
        "timeout",
        max_steps=int(max_steps),
    )


def pulse_servo_to_pose(
    *,
    name: str,
    env,
    runner,
    policy,
    device,
    interactive,
    target_pose,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
    translation_speed: float,
    yaw_speed: float,
    pulse_steps: int,
    settle_steps: int,
    max_cycles: int,
    enforce_yaw: bool,
    last_action=None,
) -> tuple[np.ndarray, dict]:
    """Closed-loop fixed-amplitude pulses separated by zero-command settling."""
    target = np.asarray(target_pose, dtype=np.float64)
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    if last_action is None:
        last_action = np.zeros(16, dtype=np.float32)
    started = time.monotonic()

    # Nav2 and the preceding pulse may leave non-zero body velocity.  Always
    # brake before measuring the next closed-loop error.
    for _ in range(int(settle_steps)):
        last_action = _checked_base_motion_step(
            stage=name,
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=(0.0, 0.0, 0.0),
            arm_target=arm_target,
            gripper_target=gripper_target,
        )

    for cycle in range(int(max_cycles)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        pose = _base_pose_from_state(state)
        xy_error = target[:2] - pose[:2]
        distance = float(np.linalg.norm(xy_error))
        yaw_error = _wrap_angle(float(target[2] - pose[2]))
        position_ok = distance <= position_tolerance_m
        yaw_ok = (not enforce_yaw) or abs(yaw_error) <= yaw_tolerance_rad

        if position_ok and yaw_ok:
            stable = 0
            for _ in range(int(settle_steps)):
                last_action = _checked_base_motion_step(
                    stage=name,
                    env=env,
                    policy=policy,
                    device=device,
                    last_action=last_action,
                    commands=(0.0, 0.0, 0.0),
                    arm_target=arm_target,
                    gripper_target=gripper_target,
                )
                check = env.get_robot_state()
                check_pose = _base_pose_from_state(check)
                check_distance = float(
                    np.linalg.norm(target[:2] - check_pose[:2])
                )
                check_yaw = abs(_wrap_angle(target[2] - check_pose[2]))
                stopped = (
                    float(np.linalg.norm(np.asarray(check["lin_vel"])[:2])) <= 0.08
                    and abs(float(check["ang_vel"][2])) <= 0.15
                )
                if (
                    check_distance <= position_tolerance_m
                    and ((not enforce_yaw) or check_yaw <= yaw_tolerance_rad)
                    and stopped
                ):
                    stable += 1
                else:
                    stable = 0
                if stable >= int(settle_steps):
                    break
            if stable >= int(settle_steps):
                final = env.get_robot_state()
                final_pose = _base_pose_from_state(final)
                final_distance = float(
                    np.linalg.norm(target[:2] - final_pose[:2])
                )
                final_yaw_error = abs(
                    _wrap_angle(target[2] - final_pose[2])
                )
                print(
                    f"[N2M][{name}_DONE] cycles={cycle} "
                    f"xy_error={final_distance:.3f}m "
                    f"yaw_error={math.degrees(final_yaw_error):.2f}deg",
                    flush=True,
                )
                return last_action, {
                    "cycles": cycle,
                    "elapsed_s": time.monotonic() - started,
                    "target_pose": target.tolist(),
                    "final_base_pose": final_pose.tolist(),
                    "xy_error_m": final_distance,
                    "position_tolerance_m": position_tolerance_m,
                    "yaw_error_rad": final_yaw_error,
                    "enforce_yaw": enforce_yaw,
                }
            continue

        needs_yaw = (
            enforce_yaw
            and position_ok
            and abs(yaw_error) > yaw_tolerance_rad
        )
        if needs_yaw:
            body_error = _body_frame_xy_error(pose, target[:2])
            body_norm = max(float(np.linalg.norm(body_error)), 1e-9)
            hold_speed = 0.0 if distance < 0.05 else min(0.25, 1.5 * distance)
            command = (
                hold_speed * float(body_error[0] / body_norm),
                hold_speed * float(body_error[1] / body_norm),
                math.copysign(yaw_speed, yaw_error),
            )
            phase = "yaw"
        else:
            body_error = _body_frame_xy_error(pose, target[:2])
            body_norm = max(float(np.linalg.norm(body_error)), 1e-9)
            gentle_wz = (
                float(np.clip(4.0 * yaw_error, -0.40, 0.40))
                if enforce_yaw
                else 0.0
            )
            command = (
                translation_speed * float(body_error[0] / body_norm),
                translation_speed * float(body_error[1] / body_norm),
                gentle_wz,
            )
            phase = "translate"

        active_steps = int(pulse_steps)
        if phase == "yaw" and abs(yaw_error) < 2.0 * yaw_tolerance_rad:
            active_steps = max(2, active_steps // 4)
        elif phase == "translate":
            if enforce_yaw:
                # Re-measure yaw at least twice as often while following the
                # narrow, collision-checked P-to-M corridor.
                active_steps = min(active_steps, max(2, pulse_steps // 2))
            # Keep the command amplitude high enough for the learned policy to
            # respond, but shorten the pulse close to the capture boundary so
            # the base does not repeatedly jump across it.
            remaining = max(0.0, distance - position_tolerance_m)
            if remaining < 0.03:
                active_steps = min(active_steps, max(2, pulse_steps // 4))
            elif remaining < 0.08:
                active_steps = min(active_steps, max(2, pulse_steps // 2))
        for _ in range(active_steps):
            last_action = _checked_base_motion_step(
                stage=name,
                env=env,
                policy=policy,
                device=device,
                last_action=last_action,
                commands=command,
                arm_target=arm_target,
                gripper_target=gripper_target,
            )
        for _ in range(int(settle_steps)):
            last_action = _checked_base_motion_step(
                stage=name,
                env=env,
                policy=policy,
                device=device,
                last_action=last_action,
                commands=(0.0, 0.0, 0.0),
                arm_target=arm_target,
                gripper_target=gripper_target,
            )
        if cycle == 0 or cycle % 5 == 0:
            print(
                f"[N2M][{name}] cycle={cycle}/{max_cycles} phase={phase} "
                f"xy_error={distance:.3f}m "
                f"yaw_error={math.degrees(yaw_error):+.2f}deg "
                f"cmd=({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f})",
                flush=True,
            )

    final = env.get_robot_state()
    final_pose = _base_pose_from_state(final)
    raise EpisodeFailure(
        name,
        "pulse_servo_timeout",
        target_pose=target.tolist(),
        final_base_pose=final_pose.tolist(),
        xy_error_m=float(np.linalg.norm(target[:2] - final_pose[:2])),
        yaw_error_rad=abs(_wrap_angle(target[2] - final_pose[2])),
        max_cycles=int(max_cycles),
    )


def continuous_servo_to_pose(
    *,
    name: str,
    env,
    runner,
    policy,
    device,
    interactive,
    target_pose,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
    translation_speed: float,
    yaw_speed: float,
    stable_steps: int,
    max_steps: int,
    last_action=None,
) -> tuple[np.ndarray, dict]:
    """Continuously servo a pose without zero-command gaps between updates.

    The learned locomotion policy drifts during long zero-command intervals.
    Recomputing the body-frame translation and yaw correction every policy
    step avoids the pulse/settle limit cycle observed during lateral docking.
    """
    target = np.asarray(target_pose, dtype=np.float64)
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    if last_action is None:
        last_action = np.zeros(16, dtype=np.float32)
    stable = 0
    started = time.monotonic()
    max_yaw_speed = 2.0 * float(yaw_speed)

    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        pose = _base_pose_from_state(state)
        distance = float(np.linalg.norm(target[:2] - pose[:2]))
        yaw_error = _wrap_angle(float(target[2] - pose[2]))
        position_ok = distance <= position_tolerance_m
        yaw_ok = abs(yaw_error) <= yaw_tolerance_rad

        if position_ok and yaw_ok:
            command = (0.0, 0.0, 0.0)
            phase = "verify"
            stable += 1
        else:
            stable = 0
            body_error = _body_frame_xy_error(pose, target[:2])
            body_norm = max(float(np.linalg.norm(body_error)), 1e-9)
            speed = 0.0 if position_ok else float(translation_speed)
            command = (
                speed * float(body_error[0] / body_norm),
                speed * float(body_error[1] / body_norm),
                float(
                    np.clip(
                        8.0 * yaw_error,
                        -max_yaw_speed,
                        max_yaw_speed,
                    )
                ),
            )
            phase = "yaw" if position_ok else "translate"

        last_action = _checked_base_motion_step(
            stage=name,
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=command,
            arm_target=arm_target,
            gripper_target=gripper_target,
        )
        if step == 0 or step % 50 == 0:
            print(
                f"[N2M][{name}] step={step}/{max_steps} phase={phase} "
                f"xy_error={distance:.3f}m "
                f"yaw_error={math.degrees(yaw_error):+.2f}deg "
                f"cmd=({command[0]:+.3f},{command[1]:+.3f},"
                f"{command[2]:+.3f}) stable={stable}/{stable_steps}",
                flush=True,
            )
        if stable >= int(stable_steps):
            final = env.get_robot_state()
            final_pose = _base_pose_from_state(final)
            final_distance = float(
                np.linalg.norm(target[:2] - final_pose[:2])
            )
            final_yaw_error = abs(_wrap_angle(target[2] - final_pose[2]))
            if (
                final_distance > position_tolerance_m
                or final_yaw_error > yaw_tolerance_rad
            ):
                stable = 0
                continue
            print(
                f"[N2M][{name}_DONE] steps={step + 1} "
                f"xy_error={final_distance:.3f}m "
                f"yaw_error={math.degrees(final_yaw_error):.2f}deg",
                flush=True,
            )
            return last_action, {
                "steps": step + 1,
                "elapsed_s": time.monotonic() - started,
                "target_pose": target.tolist(),
                "final_base_pose": final_pose.tolist(),
                "xy_error_m": final_distance,
                "position_tolerance_m": position_tolerance_m,
                "yaw_error_rad": final_yaw_error,
                "yaw_tolerance_rad": yaw_tolerance_rad,
                "stable_steps": int(stable_steps),
                "controller": "continuous_body_frame_servo",
            }

    final = env.get_robot_state()
    final_pose = _base_pose_from_state(final)
    raise EpisodeFailure(
        name,
        "continuous_servo_timeout",
        target_pose=target.tolist(),
        final_base_pose=final_pose.tolist(),
        xy_error_m=float(np.linalg.norm(target[:2] - final_pose[:2])),
        yaw_error_rad=abs(_wrap_angle(target[2] - final_pose[2])),
        max_steps=int(max_steps),
    )


def pan_xy_to_pose(
    *,
    name: str,
    env,
    runner,
    policy,
    device,
    interactive,
    target_pose,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
    max_steps: int,
    approach_direction=None,
    inward_tolerance_m: float = 0.0,
    terminal_region: dict | None = None,
    control_yaw: bool = True,
    last_action=None,
) -> tuple[np.ndarray, dict]:
    """Drive to a world-frame pose with the validated PAN_XY controller.

    ``control_yaw=False`` makes this a pure world-XY centering phase.  Its
    completion test deliberately ignores yaw and yaw-rate because a separate
    collision-checked in-place alignment phase follows at P.
    """
    target = np.asarray(target_pose, dtype=np.float64)
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    if last_action is None:
        last_action = np.zeros(16, dtype=np.float32)
    control = _new_pan_xy_control_state()
    started = time.monotonic()
    terminal_yaw_recheck_count = 0

    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        pose = _base_pose_from_state(state)
        command, done, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=pose,
            target_pose=target,
            base_lin_vel=state["lin_vel"],
            position_tolerance_m=position_tolerance_m,
            yaw_tolerance_rad=yaw_tolerance_rad,
            step_dt=env.dt,
            approach_direction=approach_direction,
            inward_tolerance_m=inward_tolerance_m,
            terminal_region=terminal_region,
            control_yaw=control_yaw,
        )
        if diagnostics["terminal_yaw_realign_required"]:
            terminal_yaw_recheck_count += 1
        else:
            terminal_yaw_recheck_count = 0
        if terminal_yaw_recheck_count >= PAN_TERMINAL_YAW_RECHECK_STEPS:
            raise EpisodeFailure(
                name,
                "terminal_yaw_realign_required",
                target_pose=target.tolist(),
                current_base_pose=pose.tolist(),
                longitudinal_error_m=diagnostics[
                    "longitudinal_error_m"
                ],
                lateral_error_m=diagnostics["lateral_error_m"],
                yaw_error_rad=abs(diagnostics["yaw_error_rad"]),
                yaw_capture_tolerance_rad=diagnostics[
                    "yaw_capture_tolerance_rad"
                ],
                yaw_hard_tolerance_rad=diagnostics[
                    "yaw_hard_tolerance_rad"
                ],
                recheck_steps=terminal_yaw_recheck_count,
            )
        last_action = _checked_base_motion_step(
            stage=name,
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=command,
            arm_target=arm_target,
            gripper_target=gripper_target,
        )
        if step == 0 or step % 25 == 0:
            directional_text = ""
            if diagnostics["directional_position_control"]:
                gaussian_text = ""
                if diagnostics["n2m_terminal_enabled"]:
                    gaussian_text = (
                        f"n2m_r={diagnostics['mahalanobis_radius']:.2f} "
                        f"zone={'slow' if diagnostics['inside_slow_region'] else 'track'} "
                        f"scale={diagnostics['terminal_speed_scale']:.2f} "
                    )
                directional_text = (
                    f"approach_error="
                    f"{diagnostics['longitudinal_error_m']:+.3f}m "
                    f"lateral_error="
                    f"{diagnostics['lateral_error_m']:+.3f}m "
                    f"capture={diagnostics['terminal_capture']} "
                    f"exit={diagnostics['terminal_exit_count']}/"
                    f"{N2M_TERMINAL_EXIT_FRAMES} "
                    f"{gaussian_text}"
                    f"yaw_mode="
                    f"{'joint' if diagnostics['yaw_control_enabled'] else 'hold'} "
                    f"yaw_recheck={terminal_yaw_recheck_count}/"
                    f"{PAN_TERMINAL_YAW_RECHECK_STEPS} "
                )
            elif not diagnostics["yaw_control_enabled"]:
                directional_text = "yaw_mode=free "
            print(
                f"[N2M][{name}] step={step}/{max_steps} "
                f"err=({diagnostics['err_x']:+.3f},"
                f"{diagnostics['err_y']:+.3f})m "
                f"{directional_text}"
                f"yaw_error={math.degrees(diagnostics['yaw_error_rad']):+.2f}deg "
                f"instant_speed={diagnostics['speed_mps']:.3f}m/s "
                f"trend=({diagnostics['trend_speed_mps']:.3f}m/s,"
                f"{diagnostics['trend_yaw_rate_radps']:.3f}rad/s) "
                f"cmd=({command[0]:+.3f},{command[1]:+.3f},"
                f"{command[2]:+.3f}) "
                f"window={diagnostics['settle_window_count']}/"
                f"{PAN_SETTLE_WINDOW_FRAMES} "
                f"settle={diagnostics['settle_count']}/{PAN_SETTLE_STEPS}",
                flush=True,
            )
        if done:
            final = env.get_robot_state()
            final_pose = _base_pose_from_state(final)
            final_error = target[:2] - final_pose[:2]
            final_yaw_error = abs(_wrap_angle(target[2] - final_pose[2]))
            if approach_direction is not None:
                final_longitudinal, final_lateral, _, _ = (
                    _approach_frame_position_errors(
                        final_pose, target, approach_direction
                    )
                )
                if terminal_region is not None:
                    final_terminal_metrics = _n2m_terminal_pose_metrics(
                        final_pose, terminal_region
                    )
                    hold_margin = N2M_TERMINAL_HOLD_POSITION_MARGIN_M
                    final_mahalanobis_radius = float(
                        final_terminal_metrics["mahalanobis_radius"]
                    )
                    final_gaussian_ok = (
                        final_mahalanobis_radius
                        <= float(terminal_region["hold_radius"])
                    )
                else:
                    hold_margin = PAN_TERMINAL_POSITION_OUTER_MARGIN_M
                    final_mahalanobis_radius = None
                    final_gaussian_ok = True
                final_position_ok = (
                    -float(inward_tolerance_m)
                    - hold_margin
                    <= final_longitudinal
                    <= 0.0
                    and abs(final_lateral)
                    <= float(position_tolerance_m)
                    + hold_margin
                    and final_gaussian_ok
                )
                final_yaw_tolerance = float(yaw_tolerance_rad) + (
                    PAN_TERMINAL_YAW_HARD_MARGIN_RAD
                    if terminal_region is not None
                    else PAN_TERMINAL_YAW_CAPTURE_MARGIN_RAD
                )
            else:
                final_longitudinal = None
                final_lateral = None
                final_mahalanobis_radius = None
                final_position_ok = (
                    abs(float(final_error[0])) <= position_tolerance_m
                    and abs(float(final_error[1])) <= position_tolerance_m
                )
                final_yaw_tolerance = (
                    float(yaw_tolerance_rad) if control_yaw else None
                )
            if (
                not final_position_ok
                or (
                    final_yaw_tolerance is not None
                    and final_yaw_error > final_yaw_tolerance
                )
            ):
                control["settle_count"] = 0
                control["pose_window"].clear()
                continue
            directional_text = ""
            if final_longitudinal is not None:
                directional_text = (
                    f"approach_error={final_longitudinal:+.3f}m "
                    f"lateral_error={final_lateral:+.3f}m "
                )
            print(
                f"[N2M][{name}_DONE] steps={step + 1} "
                f"xy_error={np.linalg.norm(final_error):.3f}m "
                f"axis_error=({final_error[0]:+.3f},{final_error[1]:+.3f})m "
                f"{directional_text}"
                f"yaw_error={math.degrees(final_yaw_error):.2f}deg",
                flush=True,
            )
            return last_action, {
                "steps": step + 1,
                "elapsed_s": time.monotonic() - started,
                "target_pose": target.tolist(),
                "final_base_pose": final_pose.tolist(),
                "xy_error_m": float(np.linalg.norm(final_error)),
                "axis_error_m": final_error.tolist(),
                "longitudinal_error_m": final_longitudinal,
                "lateral_error_m": final_lateral,
                "n2m_terminal_enabled": terminal_region is not None,
                "n2m_component_index": (
                    int(terminal_region["component_index"])
                    if terminal_region is not None
                    else None
                ),
                "n2m_mahalanobis_radius": final_mahalanobis_radius,
                "n2m_enter_radius": (
                    float(terminal_region["enter_radius"])
                    if terminal_region is not None
                    else None
                ),
                "n2m_hold_radius": (
                    float(terminal_region["hold_radius"])
                    if terminal_region is not None
                    else None
                ),
                "inward_tolerance_m": (
                    float(inward_tolerance_m)
                    if approach_direction is not None
                    else None
                ),
                "position_tolerance_m": float(position_tolerance_m),
                "yaw_error_rad": final_yaw_error,
                "yaw_tolerance_rad": float(yaw_tolerance_rad),
                "yaw_control_enabled": bool(
                    diagnostics["yaw_control_enabled"]
                ),
                "terminal_yaw_capture_tolerance_rad": (
                    float(yaw_tolerance_rad)
                    + PAN_TERMINAL_YAW_CAPTURE_MARGIN_RAD
                    if approach_direction is not None
                    else None
                ),
                "terminal_yaw_hard_tolerance_rad": (
                    float(yaw_tolerance_rad)
                    + PAN_TERMINAL_YAW_HARD_MARGIN_RAD
                    if approach_direction is not None
                    else None
                ),
                "terminal_capture": bool(
                    diagnostics["terminal_capture"]
                ),
                "longitudinal_inner_bounds_m": diagnostics[
                    "longitudinal_inner_bounds_m"
                ],
                "lateral_inner_tolerance_m": diagnostics[
                    "lateral_inner_tolerance_m"
                ],
                "longitudinal_outer_bounds_m": diagnostics[
                    "longitudinal_outer_bounds_m"
                ],
                "lateral_outer_tolerance_m": diagnostics[
                    "lateral_outer_tolerance_m"
                ],
                "settle_window_frames": PAN_SETTLE_WINDOW_FRAMES,
                "settle_linear_rate_mps": diagnostics["trend_speed_mps"],
                "settle_yaw_rate_radps": diagnostics[
                    "trend_yaw_rate_radps"
                ],
                "settle_steps": PAN_SETTLE_STEPS,
                "controller": "validated_pan_world_xy",
            }

    final = env.get_robot_state()
    final_pose = _base_pose_from_state(final)
    if approach_direction is not None:
        final_longitudinal, final_lateral, _, _ = (
            _approach_frame_position_errors(
                final_pose, target, approach_direction
            )
        )
    else:
        final_longitudinal = None
        final_lateral = None
    raise EpisodeFailure(
        name,
        "pan_xy_timeout",
        target_pose=target.tolist(),
        final_base_pose=final_pose.tolist(),
        xy_error_m=float(np.linalg.norm(target[:2] - final_pose[:2])),
        longitudinal_error_m=final_longitudinal,
        lateral_error_m=final_lateral,
        yaw_error_rad=abs(_wrap_angle(target[2] - final_pose[2])),
        max_steps=int(max_steps),
    )


def align_yaw_in_predock_clearance(
    *,
    env,
    runner,
    policy,
    device,
    interactive,
    predock_pose,
    yaw_tolerance_rad: float,
    clearance_radius_m: float,
    max_steps: int,
    last_action=None,
) -> tuple[np.ndarray, dict]:
    """Smoothly align yaw only inside the collision-checked P clearance disk."""
    target = np.asarray(predock_pose, dtype=np.float64)
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    if last_action is None:
        last_action = np.zeros(16, dtype=np.float32)
    cmd_wz = 0.0
    settling = False
    settle_count = 0
    started = time.monotonic()
    enter_tolerance = min(float(yaw_tolerance_rad), ALIGN_YAW_ENTER_TOL_RAD)
    resume_tolerance = max(float(yaw_tolerance_rad), ALIGN_YAW_RESUME_TOL_RAD)
    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        pose = _base_pose_from_state(state)
        distance = float(np.linalg.norm(pose[:2] - target[:2]))
        if distance > clearance_radius_m:
            raise EpisodeFailure(
                "PREDOCK_ALIGN",
                "left_collision_checked_rotation_clearance",
                predock_pose=target.tolist(),
                current_base_pose=pose.tolist(),
                distance_m=distance,
                clearance_radius_m=clearance_radius_m,
            )
        yaw_error = _wrap_angle(target[2] - pose[2])
        if settling:
            if abs(yaw_error) > resume_tolerance:
                settling = False
                settle_count = 0
        elif abs(yaw_error) <= enter_tolerance:
            settling = True
            settle_count = 0
        target_wz = 0.0 if settling else _yaw_velocity_target(
            yaw_error,
            deadband=enter_tolerance,
            kp=ALIGN_YAW_KP,
            min_abs=ALIGN_YAW_MIN,
            max_abs=ALIGN_YAW_MAX,
        )
        cmd_wz = _slew(cmd_wz, target_wz, ALIGN_YAW_DW_PER_STEP)
        if (
            settling
            and abs(yaw_error) <= resume_tolerance
            and abs(float(state["ang_vel"][2])) <= ALIGN_YAW_RATE_TOL
            and abs(cmd_wz) <= ALIGN_YAW_DW_PER_STEP
        ):
            settle_count += 1
        else:
            settle_count = 0
        if step == 0 or step % 25 == 0:
            print(
                f"[N2M][PREDOCK_ALIGN] step={step}/{max_steps} "
                f"yaw_error={math.degrees(yaw_error):+.2f}deg "
                f"P_distance={distance:.3f}m "
                f"cmd=(+0.000,+0.000,{cmd_wz:+.3f}) "
                f"settle={settle_count}/{ALIGN_YAW_SETTLE_STEPS}",
                flush=True,
            )
        if settle_count >= ALIGN_YAW_SETTLE_STEPS:
            final = env.get_robot_state()
            final_pose = _base_pose_from_state(final)
            final_yaw_error = abs(_wrap_angle(target[2] - final_pose[2]))
            print(
                f"[N2M][PREDOCK_ALIGN_DONE] step={step} "
                f"yaw_error={math.degrees(final_yaw_error):.2f}deg "
                f"P_distance={np.linalg.norm(final_pose[:2] - target[:2]):.3f}m",
                flush=True,
            )
            return last_action, {
                "steps": step + 1,
                "elapsed_s": time.monotonic() - started,
                "target_yaw": float(target[2]),
                "final_base_pose": final_pose.tolist(),
                "yaw_error_rad": final_yaw_error,
                "predock_distance_m": float(
                    np.linalg.norm(final_pose[:2] - target[:2])
                ),
                "clearance_radius_m": clearance_radius_m,
                "controller": "validated_align_yaw",
            }
        last_action = _checked_base_motion_step(
            stage="PREDOCK_ALIGN",
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=(0.0, 0.0, cmd_wz),
            arm_target=arm_target,
            gripper_target=gripper_target,
        )
    final = env.get_robot_state()
    final_pose = _base_pose_from_state(final)
    raise EpisodeFailure(
        "PREDOCK_ALIGN",
        "yaw_alignment_timeout",
        target_yaw=float(target[2]),
        final_base_pose=final_pose.tolist(),
        yaw_error_rad=abs(_wrap_angle(target[2] - final_pose[2])),
        max_steps=int(max_steps),
    )


def retreat_into_predock_clearance(
    *,
    env,
    runner,
    policy,
    device,
    interactive,
    predock_pose,
    translation_speed: float,
    clearance_radius_m: float,
    trigger_margin_m: float = 0.05,
    brake_steps: int = 30,
    max_steps: int = 2000,
    correct_yaw: bool = True,
) -> dict:
    """Leave M once, brake inside P's checked disk, and never re-approach M."""
    target = np.asarray(predock_pose, dtype=np.float64)
    trigger_radius = clearance_radius_m - trigger_margin_m
    if trigger_radius <= 0.0:
        raise ValueError("retreat trigger radius must be positive")
    arm_target = runner._arm_q().copy()
    gripper_target = GRIPPER_OPEN.copy()
    last_action = np.zeros(16, dtype=np.float32)
    started = time.monotonic()
    initial_pose = _base_pose_from_state(env.get_robot_state())
    target_yaw = float(target[2])
    print(
        f"[N2M][M_RETREAT_START] current={np.round(initial_pose, 3)} "
        f"P={np.round(target, 3)} trigger={trigger_radius:.3f}m",
        flush=True,
    )
    for step in range(int(max_steps)):
        if interactive.quit:
            raise KeyboardInterrupt
        state = env.get_robot_state()
        pose = _base_pose_from_state(state)
        distance = float(np.linalg.norm(target[:2] - pose[:2]))
        if distance <= trigger_radius:
            for _ in range(int(brake_steps)):
                last_action = _checked_base_motion_step(
                    stage="M_RETREAT",
                    env=env,
                    policy=policy,
                    device=device,
                    last_action=last_action,
                    commands=(0.0, 0.0, 0.0),
                    arm_target=arm_target,
                    gripper_target=gripper_target,
                )
            final_state = env.get_robot_state()
            final_pose = _base_pose_from_state(final_state)
            final_distance = float(
                np.linalg.norm(target[:2] - final_pose[:2])
            )
            if final_distance > clearance_radius_m:
                raise EpisodeFailure(
                    "M_RETREAT",
                    "braking_left_predock_clearance",
                    predock_pose=target.tolist(),
                    final_base_pose=final_pose.tolist(),
                    final_distance_m=final_distance,
                    clearance_radius_m=clearance_radius_m,
                )
            print(
                f"[N2M][M_RETREAT_DONE] step={step} "
                f"P_distance={final_distance:.3f}m "
                f"yaw_error={math.degrees(abs(_wrap_angle(target_yaw - final_pose[2]))):.2f}deg",
                flush=True,
            )
            return {
                "steps": step,
                "elapsed_s": time.monotonic() - started,
                "initial_base_pose": initial_pose.tolist(),
                "final_base_pose": final_pose.tolist(),
                "predock_pose": target.tolist(),
                "trigger_radius_m": trigger_radius,
                "clearance_radius_m": clearance_radius_m,
                "final_distance_m": final_distance,
                "brake_steps": int(brake_steps),
                "correct_yaw": bool(correct_yaw),
            }

        body_error = _body_frame_xy_error(pose, target[:2])
        body_norm = max(float(np.linalg.norm(body_error)), 1e-9)
        yaw_error = _wrap_angle(target_yaw - pose[2])
        command = (
            translation_speed * float(body_error[0] / body_norm),
            translation_speed * float(body_error[1] / body_norm),
            (
                float(np.clip(4.0 * yaw_error, -0.40, 0.40))
                if correct_yaw
                else 0.0
            ),
        )
        last_action = _checked_base_motion_step(
            stage="M_RETREAT",
            env=env,
            policy=policy,
            device=device,
            last_action=last_action,
            commands=command,
            arm_target=arm_target,
            gripper_target=gripper_target,
        )
        if step == 0 or step % 25 == 0:
            print(
                f"[N2M][M_RETREAT] step={step}/{max_steps} "
                f"P_distance={distance:.3f}m "
                f"yaw_error={math.degrees(yaw_error):+.2f}deg "
                f"cmd=({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f})",
                flush=True,
            )
    raise EpisodeFailure(
        "M_RETREAT",
        "clearance_entry_timeout",
        predock_pose=target.tolist(),
        max_steps=int(max_steps),
    )


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


def _observation_yaw_capture_step(
    captured: bool,
    bearing_error_rad: float,
    max_bearing_rad: float,
) -> bool:
    """Apply enter/resume hysteresis to the NAV observation yaw gate."""
    maximum = float(max_bearing_rad)
    if maximum <= 0.0:
        raise ValueError("max_bearing_rad must be positive")
    enter = min(NAV_OBSERVATION_YAW_ENTER_RAD, maximum)
    resume = min(
        NAV_OBSERVATION_YAW_RESUME_RAD,
        maximum,
    )
    magnitude = abs(float(bearing_error_rad))
    if captured:
        return magnitude <= resume
    return magnitude <= enter


def _pose_window_motion_rates(poses, step_dt: float) -> tuple[float, float]:
    """Estimate planar and yaw drift from a short pose window.

    A least-squares trend rejects the locomotion policy's periodic body sway,
    unlike a per-frame velocity threshold.  It still exposes sustained drift.
    """
    samples = np.asarray(poses, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise ValueError("pose window must contain at least two [x, y, yaw] rows")
    if float(step_dt) <= 0.0:
        raise ValueError("step_dt must be positive")

    times = np.arange(len(samples), dtype=np.float64) * float(step_dt)
    times -= float(np.mean(times))
    denominator = float(np.dot(times, times))
    xy = samples[:, :2] - np.mean(samples[:, :2], axis=0, keepdims=True)
    yaw = np.unwrap(samples[:, 2])
    yaw -= float(np.mean(yaw))
    xy_velocity = times @ xy / denominator
    yaw_rate = float(np.dot(times, yaw) / denominator)
    return float(np.linalg.norm(xy_velocity)), abs(yaw_rate)


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
    settle_window = []
    settle_linear_rate = float("inf")
    settle_yaw_rate = float("inf")
    yaw_captured = False
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
            yaw_captured = _observation_yaw_capture_step(
                yaw_captured,
                metrics.bearing_error_rad,
                max_bearing,
            )
            cmd_wz = (
                0.0
                if yaw_captured
                else _effective_yaw_command(metrics.bearing_error_rad)
            )
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
            zero_command = abs(cmd_vx) < 1e-9 and abs(cmd_wz) < 1e-9
            if in_gate and zero_command:
                settle_window.append(
                    np.array([rs["pos"][0], rs["pos"][1], rs["yaw"]])
                )
                settle_window = settle_window[-int(args.nav_settle_frames):]
            else:
                settle_window.clear()
                settle_linear_rate = float("inf")
                settle_yaw_rate = float("inf")

            window_ready = len(settle_window) >= int(args.nav_settle_frames)
            if window_ready:
                settle_linear_rate, settle_yaw_rate = _pose_window_motion_rates(
                    settle_window, env.dt
                )
            stopped = (
                window_ready
                and settle_linear_rate <= args.nav_stop_speed
                and settle_yaw_rate <= args.nav_stop_yaw_rate
            )
            if stopped:
                print(
                    f"[N2M][NAV_OBSERVATION_READY] step={step} "
                    f"distance={metrics.horizontal_distance_m:.3f}m "
                    f"bearing={math.degrees(metrics.bearing_error_rad):+.2f}deg "
                    f"window={len(settle_window)}/"
                    f"{args.nav_settle_frames} "
                    f"avg_speed={settle_linear_rate:.4f}m/s "
                    f"avg_yaw_rate={settle_yaw_rate:.4f}rad/s",
                    flush=True,
                )
                return rs, target, {
                    "waypoint_world": waypoint.tolist(),
                    "horizontal_distance_m": metrics.horizontal_distance_m,
                    "bearing_error_rad": metrics.bearing_error_rad,
                    "navigation_steps": step + 1,
                    "settle_window_frames": len(settle_window),
                    "settle_linear_rate_mps": settle_linear_rate,
                    "settle_yaw_rate_radps": settle_yaw_rate,
                }

        if step % 100 == 0:
            print(
                f"[N2M][{state}] step={step} pos={np.round(rs['pos'][:2], 3)} "
                f"distance={metrics.horizontal_distance_m:.3f}m "
                f"bearing={math.degrees(metrics.bearing_error_rad):+.1f}deg "
                f"cmd=({cmd_vx:+.2f},{cmd_wz:+.2f}) "
                f"yaw_capture={yaw_captured} "
                f"window={len(settle_window)}/{args.nav_settle_frames} "
                f"avg=({settle_linear_rate:.3f}m/s,"
                f"{settle_yaw_rate:.3f}rad/s)",
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
    parser.add_argument(
        "--predock-distance",
        type=float,
        default=0.55,
        help="first cube-radial distance from P to the N2M manipulation pose",
    )
    parser.add_argument(
        "--predock-distance-step",
        type=float,
        default=0.12,
        help="extra cube-radial distance for each fallback P",
    )
    parser.add_argument("--predock-distance-options", type=int, default=3)
    parser.add_argument(
        "--predock-tangent-offset",
        type=float,
        default=0.12,
        help="left/right tangent offset for each cube-radial P distance",
    )
    parser.add_argument("--predock-nav-max-steps", type=int, default=4000)
    parser.add_argument("--predock-nav-tolerance", type=float, default=0.30)
    parser.add_argument(
        "--predock-rotation-radius",
        type=float,
        default=0.18,
        help=(
            "nominal P radius; together with the drift margin it defines the "
            "collision-checked yaw-alignment clearance"
        ),
    )
    parser.add_argument("--predock-align-max-steps", type=int, default=800)
    parser.add_argument("--predock-position-tolerance", type=float, default=0.07)
    parser.add_argument("--dock-position-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--dock-inward-tolerance",
        type=float,
        default=0.05,
        help=(
            "allowed signed P-to-M overshoot toward the object; the final "
            "longitudinal error must be in [-value, 0] (default: 0.05m)"
        ),
    )
    parser.add_argument("--dock-yaw-tolerance-deg", type=float, default=4.0)
    parser.add_argument(
        "--dock-n2m-enter-radius",
        type=float,
        default=N2M_TERMINAL_ENTER_RADIUS,
        help="Mahalanobis radius of the N2M terminal capture contour",
    )
    parser.add_argument(
        "--dock-n2m-hold-radius",
        type=float,
        default=N2M_TERMINAL_HOLD_RADIUS,
        help="larger Mahalanobis radius retained after terminal capture",
    )
    parser.add_argument(
        "--dock-n2m-slow-radius",
        type=float,
        default=N2M_TERMINAL_SLOW_RADIUS,
        help="outer Mahalanobis radius where final docking starts slowing",
    )
    parser.add_argument("--dock-translation-speed", type=float, default=0.25)
    parser.add_argument("--dock-yaw-speed", type=float, default=0.40)
    parser.add_argument(
        "--dock-pulse-steps",
        type=int,
        default=8,
        help=(
            "legacy step-budget factor; P-to-M yaw pulses are disabled"
        ),
    )
    parser.add_argument(
        "--dock-settle-steps",
        type=int,
        default=8,
        help=(
            "legacy step-budget factor; retained for CLI compatibility"
        ),
    )
    parser.add_argument("--dock-max-cycles", type=int, default=120)
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
        args.nav_settle_frames < 2
        or args.max_grasp_attempts <= 0
        or args.n2m_max_candidates <= 0
        or args.destination_max_steps <= 0
        or args.predock_distance <= 0.0
        or args.predock_distance_step < 0.0
        or args.predock_distance_options <= 0
        or args.predock_tangent_offset < 0.0
        or args.predock_nav_max_steps <= 0
        or args.predock_nav_tolerance <= 0.0
        or args.predock_rotation_radius <= 0.0
        or args.predock_align_max_steps <= 0
        or args.predock_position_tolerance <= 0.0
        or args.dock_position_tolerance <= 0.0
        or args.dock_inward_tolerance <= 0.0
        or args.dock_yaw_tolerance_deg <= 0.0
        or args.dock_n2m_enter_radius <= 0.0
        or args.dock_n2m_hold_radius <= args.dock_n2m_enter_radius
        or args.dock_n2m_slow_radius <= args.dock_n2m_hold_radius
        or args.dock_translation_speed <= 0.0
        or args.dock_yaw_speed <= 0.0
        or args.dock_pulse_steps <= 0
        or args.dock_settle_steps <= 0
        or args.dock_max_cycles <= 0
    ):
        raise SystemExit(
            "navigation/docking tolerances, ordered N2M radii, speeds, step budgets and "
            "M/attempt limits must be positive; --nav-settle-frames must "
            "be at least 2"
        )
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
        "nav_observation_yaw_enter_deg": math.degrees(
            NAV_OBSERVATION_YAW_ENTER_RAD
        ),
        "nav_observation_yaw_resume_deg": math.degrees(
            min(
                NAV_OBSERVATION_YAW_RESUME_RAD,
                math.radians(args.nav_max_bearing_deg),
            )
        ),
        "manipulation_pose_transition": (
            "cube_radial_predock_then_joint_pan_xy_yaw_to_n2m_terminal_region"
        ),
        "predock": {
            "distance_m": args.predock_distance,
            "distance_step_m": args.predock_distance_step,
            "distance_options": args.predock_distance_options,
            "tangent_offset_m": args.predock_tangent_offset,
            "nav_max_steps": args.predock_nav_max_steps,
            "nav_tolerance_m": args.predock_nav_tolerance,
            "rotation_radius_m": args.predock_rotation_radius,
            "rotation_drift_margin_m": PREDOCK_ROTATION_DRIFT_MARGIN_M,
            "rotation_clearance_radius_m": (
                args.predock_rotation_radius
                + PREDOCK_ROTATION_DRIFT_MARGIN_M
            ),
            "align_max_steps": args.predock_align_max_steps,
            "position_tolerance_m": args.predock_position_tolerance,
            "dock_position_tolerance_m": args.dock_position_tolerance,
            "dock_inward_tolerance_m": args.dock_inward_tolerance,
            "dock_yaw_tolerance_deg": args.dock_yaw_tolerance_deg,
            "pan_xy": {
                "min_move_speed_mps": PAN_MIN_MOVE_SPEED,
                "dv_per_step_mps": PAN_DV_PER_STEP,
                "dw_per_step_radps": PAN_DW_PER_STEP,
                "vmax_world_x_mps": PAN_DEFAULT_VMAX_X,
                "vmax_world_y_mps": PAN_DEFAULT_VMAX_Y,
                "yaw_min_radps": PAN_YAW_MIN,
                "yaw_max_radps": PAN_YAW_MAX,
                "yaw_enter_tolerance_deg": math.degrees(
                    PAN_YAW_ENTER_TOL_RAD
                ),
                "yaw_resume_tolerance_deg": args.dock_yaw_tolerance_deg,
                "position_resume_margin_m": PAN_POSITION_RESUME_MARGIN_M,
                "terminal_capture": {
                    "region": "candidate_local_n2m_covariance",
                    "enter_mahalanobis_radius": (
                        args.dock_n2m_enter_radius
                    ),
                    "hold_mahalanobis_radius": (
                        args.dock_n2m_hold_radius
                    ),
                    "slow_mahalanobis_radius": (
                        args.dock_n2m_slow_radius
                    ),
                    "exit_hysteresis_frames": N2M_TERMINAL_EXIT_FRAMES,
                    "slow_min_speed_scale": (
                        N2M_TERMINAL_MIN_SPEED_SCALE
                    ),
                    "longitudinal_inner_margin_m": (
                        PAN_TERMINAL_LONGITUDINAL_INNER_MARGIN_M
                    ),
                    "lateral_inner_margin_m": (
                        PAN_TERMINAL_LATERAL_INNER_MARGIN_M
                    ),
                    "position_outer_margin_m": (
                        PAN_TERMINAL_POSITION_OUTER_MARGIN_M
                    ),
                    "yaw_capture_tolerance_deg": (
                        args.dock_yaw_tolerance_deg
                        + math.degrees(
                            PAN_TERMINAL_YAW_CAPTURE_MARGIN_RAD
                        )
                    ),
                    "yaw_hard_tolerance_deg": (
                        args.dock_yaw_tolerance_deg
                        + math.degrees(
                            PAN_TERMINAL_YAW_HARD_MARGIN_RAD
                        )
                    ),
                    "yaw_control_near_m": "joint_pan_xy_yaw_until_capture",
                    "yaw_recheck_steps": PAN_TERMINAL_YAW_RECHECK_STEPS,
                    "safe_predock_realign_retries": 1,
                },
                "settle_window_frames": PAN_SETTLE_WINDOW_FRAMES,
                "settle_linear_rate_mps": PAN_SETTLE_LINEAR_RATE,
                "settle_yaw_rate_radps": PAN_SETTLE_YAW_RATE,
                "settle_steps": PAN_SETTLE_STEPS,
            },
            "max_control_steps": (
                args.dock_max_cycles
                * (args.dock_pulse_steps + args.dock_settle_steps)
            ),
        },
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
        active_predock_pose = None
        active_execution_candidate = None
        fatal = False

        for queue_row in queue:
            if attempted_m_count >= int(args.n2m_max_candidates):
                outcome = "grasp_attempts_exhausted"
                break
            if controller.quit:
                raise KeyboardInterrupt

            candidate = queue_row["candidate"]
            queue_index = int(queue_row["queue_index"])
            terminal_region = _n2m_candidate_terminal_region(
                prediction,
                queue_row,
                enter_radius=args.dock_n2m_enter_radius,
                hold_radius=args.dock_n2m_hold_radius,
                slow_radius=args.dock_n2m_slow_radius,
            )
            terminal_region_summary = {
                "reference_pose_observation": terminal_region[
                    "reference_pose_observation"
                ].tolist(),
                "component_index": terminal_region["component_index"],
                "covariance": terminal_region["covariance"].tolist(),
                "enter_radius": terminal_region["enter_radius"],
                "hold_radius": terminal_region["hold_radius"],
                "slow_radius": terminal_region["slow_radius"],
            }
            if needs_retraction:
                runner.debug_dir = output_dir / "debug" / f"m_{queue_index:04d}" / "transition"
                runner.retract_for_reposition()
                runner.weld.disable()
                if active_predock_pose is None:
                    raise EpisodeFailure(
                        "M_RETREAT", "missing_active_predock_pose"
                    )
                retreat_cube_before = runner.current_object_pose()
                retreat_clearance = (
                    args.predock_rotation_radius
                    + PREDOCK_ROTATION_DRIFT_MARGIN_M
                )
                retreat_metrics = retreat_into_predock_clearance(
                    env=env,
                    runner=runner,
                    policy=policy,
                    device=device,
                    interactive=controller,
                    predock_pose=active_predock_pose,
                    translation_speed=args.dock_translation_speed,
                    clearance_radius_m=retreat_clearance,
                )
                retreat_cube_after = runner.current_object_pose()
                retreat_translation, retreat_rotation = pose_motion(
                    retreat_cube_before, retreat_cube_after
                )
                append_jsonl(
                    output_dir / "motion_transitions.jsonl",
                    {
                        "created_at": _utc_now(),
                        "kind": "retreat_from_failed_m",
                        "from_candidate": active_execution_candidate,
                        "target_predock_pose": active_predock_pose,
                        "controller": retreat_metrics,
                        "cube_motion_m": retreat_translation,
                        "cube_rotation_rad": retreat_rotation,
                    },
                )
                active_predock_pose = None
                active_execution_candidate = None
                needs_retraction = False

            predock_options = build_predock_pose_options(
                candidate["base_pose"],
                candidate["cube_pose"],
                distance_m=args.predock_distance,
                distance_step_m=args.predock_distance_step,
                distance_count=args.predock_distance_options,
                tangent_offset_m=args.predock_tangent_offset,
            )
            preflight = preflight_predock_motion(
                env,
                runner,
                candidate["base_pose"],
                predock_options,
                rotation_clearance_radius_m=(
                    args.predock_rotation_radius
                    + PREDOCK_ROTATION_DRIFT_MARGIN_M
                ),
                dock_inward_tolerance_m=(
                    args.dock_inward_tolerance
                    + N2M_TERMINAL_HOLD_POSITION_MARGIN_M
                ),
                dock_lateral_tolerance_m=(
                    args.dock_position_tolerance
                    + N2M_TERMINAL_HOLD_POSITION_MARGIN_M
                ),
                dock_yaw_margin_rad=math.radians(
                    args.dock_yaw_tolerance_deg
                )
                + PAN_TERMINAL_YAW_HARD_MARGIN_RAD,
            )
            preflight_row = {
                "created_at": _utc_now(),
                "queue_index": queue_index,
                "candidate_id": candidate["candidate_id"],
                "base_pose": candidate["base_pose"],
                "selection": queue_row["selection"],
                "terminal_region": terminal_region_summary,
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
                    f"candidate={candidate['candidate_id']} "
                    f"reason=no_collision_free_predock options="
                    f"{preflight['option_count']}",
                    flush=True,
                )
                continue

            attempted_m_count += 1
            predock_pose = np.asarray(preflight["predock_pose"], dtype=np.float64)
            print(
                f"[N2M][PREDOCK_SELECTED] attempted_m={attempted_m_count}/"
                f"{args.n2m_max_candidates} queue={queue_index} "
                f"approach={preflight['approach_mode']} "
                f"P={np.round(predock_pose, 4)} "
                f"M={np.round(candidate['base_pose'], 4)}",
                flush=True,
            )
            transition_cube_before = runner.current_object_pose()
            bridge = IsaacROS2Bridge(
                cmd_vel_timeout=nav2_settings["velocity_timeout_s"],
                feedback_timeout=bridge_feedback_timeout,
            )
            try:
                bridge.start(timeout=30.0)
                bridge_started = True
                last_action, coarse_metrics = navigate_to_predock(
                    env=env,
                    runner=runner,
                    policy=policy,
                    bridge=bridge,
                    device=device,
                    interactive=controller,
                    predock_pose=predock_pose,
                    nav2_settings=nav2_settings,
                    max_steps=args.predock_nav_max_steps,
                    position_tolerance_m=args.predock_nav_tolerance,
                )
            finally:
                # stop() is idempotent and also reaps a child from a partially
                # failed start.  Do not guard cleanup only on start() success.
                try:
                    bridge.cancel_goal()
                except Exception:
                    pass
                try:
                    bridge.stop()
                finally:
                    bridge_started = False

            rotation_clearance_radius = (
                args.predock_rotation_radius
                + PREDOCK_ROTATION_DRIFT_MARGIN_M
            )
            print(
                f"[N2M][PREDOCK_CLEARANCE] checked_radius="
                f"{rotation_clearance_radius:.3f}m "
                f"P_center_tolerance={args.predock_position_tolerance:.3f}m",
                flush=True,
            )
            pan_xy_max_steps = (
                args.dock_max_cycles
                * (args.dock_pulse_steps + args.dock_settle_steps)
            )

            # Nav2 may declare success at its comparatively loose goal
            # tolerance.  First use the validated PAN_XY law to translate to
            # P without any yaw correction.  Once centered, the following
            # dedicated stage rotates in place to M's yaw; only then do we
            # translate along the checked P-to-M corridor.  Decoupling these
            # phases prevents yaw correction from kicking the base back out
            # of the P capture region.
            predock_pan_target = predock_pose.copy()
            last_action, center_metrics = pan_xy_to_pose(
                name="PREDOCK_CENTER_PAN_XY",
                env=env,
                runner=runner,
                policy=policy,
                device=device,
                interactive=controller,
                target_pose=predock_pan_target,
                position_tolerance_m=args.predock_position_tolerance,
                yaw_tolerance_rad=math.radians(args.dock_yaw_tolerance_deg),
                max_steps=pan_xy_max_steps,
                control_yaw=False,
                last_action=last_action,
            )
            last_action, align_metrics = align_yaw_in_predock_clearance(
                env=env,
                runner=runner,
                policy=policy,
                device=device,
                interactive=controller,
                predock_pose=predock_pose,
                yaw_tolerance_rad=math.radians(args.dock_yaw_tolerance_deg),
                clearance_radius_m=rotation_clearance_radius,
                max_steps=args.predock_align_max_steps,
                last_action=last_action,
            )
            dock_realignments = []
            dock_metrics = None
            for dock_pass in range(2):
                try:
                    last_action, dock_metrics = pan_xy_to_pose(
                        name="M_DOCK_PAN_XY",
                        env=env,
                        runner=runner,
                        policy=policy,
                        device=device,
                        interactive=controller,
                        target_pose=candidate["base_pose"],
                        position_tolerance_m=args.dock_position_tolerance,
                        yaw_tolerance_rad=math.radians(
                            args.dock_yaw_tolerance_deg
                        ),
                        max_steps=pan_xy_max_steps,
                        approach_direction=(
                            np.asarray(
                                candidate["base_pose"], dtype=np.float64
                            )[:2]
                            - predock_pose[:2]
                        ),
                        inward_tolerance_m=args.dock_inward_tolerance,
                        terminal_region=terminal_region,
                        last_action=last_action,
                    )
                    break
                except EpisodeFailure as exc:
                    if (
                        exc.reason != "terminal_yaw_realign_required"
                        or dock_pass >= 1
                    ):
                        raise
                    print(
                        f"[N2M][M_DOCK_REALIGN] pass={dock_pass + 1} "
                        f"yaw_error="
                        f"{math.degrees(float(exc.metrics['yaw_error_rad'])):.2f}deg "
                        "retreating_to=P",
                        flush=True,
                    )
                    retreat_metrics = retreat_into_predock_clearance(
                        env=env,
                        runner=runner,
                        policy=policy,
                        device=device,
                        interactive=controller,
                        predock_pose=predock_pose,
                        translation_speed=args.dock_translation_speed,
                        clearance_radius_m=rotation_clearance_radius,
                        correct_yaw=False,
                    )
                    retry_center_target = predock_pose.copy()
                    last_action, retry_center_metrics = pan_xy_to_pose(
                        name="PREDOCK_RECENTER_PAN_XY",
                        env=env,
                        runner=runner,
                        policy=policy,
                        device=device,
                        interactive=controller,
                        target_pose=retry_center_target,
                        position_tolerance_m=(
                            args.predock_position_tolerance
                        ),
                        yaw_tolerance_rad=math.radians(
                            args.dock_yaw_tolerance_deg
                        ),
                        max_steps=pan_xy_max_steps,
                        control_yaw=False,
                        last_action=None,
                    )
                    last_action, retry_align_metrics = (
                        align_yaw_in_predock_clearance(
                            env=env,
                            runner=runner,
                            policy=policy,
                            device=device,
                            interactive=controller,
                            predock_pose=predock_pose,
                            yaw_tolerance_rad=math.radians(
                                args.dock_yaw_tolerance_deg
                            ),
                            clearance_radius_m=rotation_clearance_radius,
                            max_steps=args.predock_align_max_steps,
                            last_action=last_action,
                        )
                    )
                    dock_realignments.append(
                        {
                            "pass": dock_pass + 1,
                            "trigger": {
                                "stage": exc.stage,
                                "reason": exc.reason,
                                **exc.metrics,
                            },
                            "retreat": retreat_metrics,
                            "recenter": retry_center_metrics,
                            "realign": retry_align_metrics,
                        }
                    )
            if dock_metrics is None:
                raise EpisodeFailure(
                    "M_DOCK_PAN_XY", "missing_dock_result"
                )
            del last_action
            transition_cube_after = runner.current_object_pose()
            transition_translation, transition_rotation = pose_motion(
                transition_cube_before, transition_cube_after
            )
            actual_base_pose = _base_pose_from_state(env.get_robot_state())
            execution_candidate = dict(candidate)
            execution_candidate["n2m_target_base_pose"] = list(
                candidate["base_pose"]
            )
            execution_candidate["base_pose"] = actual_base_pose.tolist()
            execution_candidate["predock_pose"] = predock_pose.tolist()
            execution_candidate["terminal_region"] = terminal_region_summary
            transition_metrics = {
                "created_at": _utc_now(),
                "kind": "observation_or_previous_predock_to_m",
                "queue_index": queue_index,
                "candidate_id": candidate["candidate_id"],
                "n2m_target_base_pose": candidate["base_pose"],
                "actual_base_pose": actual_base_pose.tolist(),
                "predock_pose": predock_pose.tolist(),
                "terminal_region": terminal_region_summary,
                "coarse_nav": coarse_metrics,
                "predock_approach": center_metrics,
                "predock_align": align_metrics,
                "predock_center": center_metrics,
                "dock": dock_metrics,
                "dock_realignments": dock_realignments,
                "cube_pose_before": transition_cube_before,
                "cube_pose_after": transition_cube_after,
                "cube_motion_m": transition_translation,
                "cube_rotation_rad": transition_rotation,
            }
            append_jsonl(
                output_dir / "motion_transitions.jsonl", transition_metrics
            )
            print(
                f"[N2M][M_REACHED] target="
                f"{np.round(candidate['base_pose'], 4)} actual="
                f"{np.round(actual_base_pose, 4)} "
                f"xy_error={dock_metrics['xy_error_m']:.3f}m "
                f"approach_error="
                f"{dock_metrics['longitudinal_error_m']:+.3f}m "
                f"lateral_error={dock_metrics['lateral_error_m']:+.3f}m "
                f"yaw_error={math.degrees(dock_metrics['yaw_error_rad']):.2f}deg",
                flush=True,
            )

            selected_candidate = execution_candidate
            active_predock_pose = predock_pose.tolist()
            active_execution_candidate = execution_candidate
            runner.weld.enable()
            m_success = False
            m_last_failure = None
            m_attempts = 0
            print(
                f"[N2M][M_START] attempted_m={attempted_m_count}/"
                f"{args.n2m_max_candidates} queue={queue_index} "
                f"target={np.round(candidate['base_pose'], 4)} "
                f"actual={np.round(execution_candidate['base_pose'], 4)}",
                flush=True,
            )

            for local_attempt in range(int(args.max_grasp_attempts)):
                runner.debug_dir = (
                    output_dir
                    / "debug"
                    / f"m_{queue_index:04d}"
                    / f"attempt_{local_attempt:04d}"
                )
                result = runner.run_preserved_attempt(
                    execution_candidate, global_attempt
                )
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
                    selected_candidate = dict(execution_candidate)
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
                    "n2m_target_base_pose": candidate["base_pose"],
                    "actual_base_pose": execution_candidate["base_pose"],
                    "predock_pose": predock_pose.tolist(),
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
        # Read final MuJoCo state before releasing wrappers that retain native
        # resources.  The summary is written even if one cleanup step reports
        # a Python-level error.
        final_target = None
        try:
            final_target = provider.estimate(env, args.object).pose_wxyz()
        except Exception:
            pass

        cleanup_errors = []
        if runner is not None:
            try:
                runner.close_attached()
            except Exception as exc:
                cleanup_errors.append(f"runner:{type(exc).__name__}:{exc}")
            runner = None
        # Always stop the most recently constructed bridge.  start() may have
        # launched a child before raising, in which case bridge_started remains
        # false but the process and its reader thread still need reaping.
        try:
            bridge.cancel_goal()
        except Exception:
            pass
        try:
            bridge.stop()
        except Exception as exc:
            cleanup_errors.append(f"bridge:{type(exc).__name__}:{exc}")
        bridge_started = False

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
                "cleanup_errors": cleanup_errors,
            },
        )
        print(
            f"[N2M][INTEGRATED_SUMMARY] outcome={outcome} "
            f"attempts={len(result_rows)} output={output_dir}",
            flush=True,
        )
        try:
            env.close()
        except Exception as exc:
            # Python exceptions are reportable; a native viewer fault is
            # avoided after main() by the standalone os._exit boundary below.
            cleanup_errors.append(f"env:{type(exc).__name__}:{exc}")
            print(f"[N2M][CLEANUP_WARNING] {cleanup_errors[-1]}", flush=True)
        print(
            f"[N2M][CLEANUP_DONE] errors={len(cleanup_errors)}",
            flush=True,
        )
    return return_code


if __name__ == "__main__":
    # MuJoCo/GLFW, CUDA extensions and the system-Python ROS children load
    # several independent native runtimes.  All owned resources have already
    # been closed and reaped above; bypassing CPython's second native-library
    # finalization pass prevents the shutdown-only SIGSEGV seen after a valid
    # INTEGRATED_SUMMARY.  Flush explicitly because os._exit skips atexit.
    _return_code = int(main())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_return_code)
