"""Independent, finite ``ARM_INIT -> LIFT`` manipulation episode runner.

The implementation reuses the project's validated AnyGrasp conversion, MoveIt
bridge and MuJoCo local servo, but deliberately does not start Nav2 or the
locomotion policy.  Every call begins from a full MuJoCo reset and ends as soon
as Cartesian LIFT is verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import numpy as np

from custom_envs.scripts.N2M.config import EpisodeConfig
from custom_envs.scripts.N2M.geometry import angular_error, joint1_towards_cube
from custom_envs.scripts.N2M.interactive import InteractiveController, ViewerOverlay
from custom_envs.scripts.N2M.mujoco_state import (
    BaseWeld,
    place_episode_state,
    robot_table_contacts,
)


ARM_HOME = np.array([0.0, 0.5, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
ARM_DEPLOY_TEMPLATE = np.array([0.0, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float64)
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float64)


class EpisodeFailure(RuntimeError):
    def __init__(self, stage: str, reason: str, **metrics):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.metrics = metrics


class CandidateAborted(EpisodeFailure):
    def __init__(self, reason="operator_abort"):
        super().__init__("OPERATOR", reason)


@dataclass
class EpisodeResult:
    candidate_id: str
    scene_id: str
    attempt: int
    success: Optional[bool]
    completed_stage: str
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    cube_pose: list
    base_pose: list
    joint1_target: Optional[float]
    joint1_actual: Optional[float]
    joint1_error: Optional[float]
    elapsed_s: float
    episode_steps: int
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ManipulationEpisodeRunner:
    def __init__(
        self,
        scene_path: Path,
        config: EpisodeConfig,
        grasp_checkpoint: Optional[str] = None,
        render: bool = False,
        pause_on_stage: bool = False,
        interactive: Optional[InteractiveController] = None,
        debug_dir: Optional[Path] = None,
    ):
        self.cfg = config
        self.grasp_checkpoint = grasp_checkpoint
        self.render = bool(render)
        self.pause_on_stage = bool(pause_on_stage)
        self.interactive = interactive or InteractiveController()
        self.debug_dir = Path(debug_dir) if debug_dir else None

        # RGB-D rendering is required even in batch mode.  Use EGL without a
        # viewer and GLX only for the interactive X11 viewer.  An explicit user
        # MUJOCO_GL setting still wins.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "glx" if self.render else "egl"
        if self.render:
            os.environ.setdefault("DISPLAY", ":0")

        from custom_envs.mujoco.env import MuJoCoEnv

        self.env = MuJoCoEnv(
            robot_init_pos=(0.0, 0.0, config.robot_root_z),
            render=self.render,
            key_callback=self.interactive.key_callback if self.render else None,
            scene_path=str(scene_path),
        )
        self.overlay = ViewerOverlay(self.env)
        self.weld = BaseWeld(self.env)
        self.zeros_action = np.zeros(16, dtype=np.float32)
        self.step_count = 0
        self.current_candidate = None
        self.joint1_target = None
        self.moveit = None
        self.attached = False
        self.attached_id = "n2m_carried_cube"
        self._imports_ready = False
        self._prepare_indices()

    def _prepare_indices(self):
        import mujoco

        self.arm_idx = np.asarray(self.env._arm_all_idx, dtype=np.int32)
        self.grip_idx = np.asarray(self.env._grip_all_idx, dtype=np.int32)
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.moveit_state_names = self.arm_joint_names + ["joint7", "joint8"]
        joint_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
        ]
        if min(joint_ids) < 0:
            raise RuntimeError("Missing one or more Piper arm joints")
        self.arm_lo = np.array([self.env.model.jnt_range[j, 0] for j in joint_ids])
        self.arm_hi = np.array([self.env.model.jnt_range[j, 1] for j in joint_ids])

    def _prepare_pipeline_imports(self):
        if self._imports_ready:
            return
        root = Path(__file__).resolve().parents[3]
        navigation = root / "custom_envs" / "scripts" / "navigation"
        utils = root / "custom_envs" / "utils"
        for path in (str(navigation), str(utils)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from custom_envs.utils.moveit_bridge import MoveItROS2Bridge
        from mujoco_moveit_frames import (
            Pose,
            T_gripper_base_grasp_tcp,
            anygrasp_candidate_to_tcp,
            capture_scan_transforms,
            inv_T,
            mujoco_body_pose_world,
            pose_error,
            pose_to_ros_dict,
            tcp_pose_to_gripper_base_pose,
            validate_camera_optical_transform,
        )
        from mujoco_local_servo import MuJoCoGripperBaseServo, finger_object_contact
        from navigate_mujoco_moveit import PipelineState, _run_grasp_plan

        self.MoveItROS2Bridge = MoveItROS2Bridge
        self.Pose = Pose
        self.T_gripper_base_grasp_tcp = T_gripper_base_grasp_tcp
        self.anygrasp_candidate_to_tcp = anygrasp_candidate_to_tcp
        self.capture_scan_transforms = capture_scan_transforms
        self.inv_T = inv_T
        self.mujoco_body_pose_world = mujoco_body_pose_world
        self.pose_error = pose_error
        self.pose_to_ros_dict = pose_to_ros_dict
        self.tcp_pose_to_gripper_base_pose = tcp_pose_to_gripper_base_pose
        self.validate_camera_optical_transform = validate_camera_optical_transform
        self.MuJoCoGripperBaseServo = MuJoCoGripperBaseServo
        self.finger_object_contact = finger_object_contact
        self.PipelineState = PipelineState
        self.run_grasp_plan = _run_grasp_plan
        self._imports_ready = True

    def _ensure_moveit(self):
        self._prepare_pipeline_imports()
        if not self.grasp_checkpoint:
            raise EpisodeFailure("SCAN", "grasp_checkpoint_not_provided")
        if self.moveit is None:
            self.moveit = self.MoveItROS2Bridge(request_timeout=10.0)
            self.moveit.start(timeout=35.0)
            diag = self.validate_camera_optical_transform(self.env)
            print(
                "[N2M] wrist camera transform verified: "
                f"{diag['position_error_m'] * 1e6:.2f}um, "
                f"{math.degrees(diag['rotation_error_rad']):.6f}deg",
                flush=True,
            )

    def close(self):
        try:
            self._detach_safely()
            if self.moveit is not None:
                self.moveit.stop()
        finally:
            self.env.close()

    def _arm_q(self) -> np.ndarray:
        return self.env.get_robot_state()["joint_pos"][self.arm_idx].copy()

    def _grip_q(self) -> np.ndarray:
        return self.env.get_robot_state()["joint_pos"][self.grip_idx].copy()

    def _moveit_state(self):
        state = self.env.get_robot_state()["joint_pos"]
        positions = np.concatenate([state[self.arm_idx], state[self.grip_idx]])
        return self.moveit_state_names, positions.astype(np.float64)

    def _update_moveit_state(self):
        if self.moveit is not None:
            names, positions = self._moveit_state()
            self.moveit.update_joint_state(names, positions)

    def _step(self, arm_target=None, grip_target=None):
        if arm_target is not None:
            self.env.set_arm_target(np.asarray(arm_target, dtype=np.float64))
        if grip_target is not None:
            self.env.set_gripper_target(np.asarray(grip_target, dtype=np.float64))
        if not self.interactive.honor_pause(self.env):
            raise CandidateAborted()
        self.env.step(self.zeros_action)
        self.step_count += 1
        arm_table_hits = robot_table_contacts(self.env, robot_root_body="link1")
        if arm_table_hits:
            raise EpisodeFailure(
                "SAFETY", "arm_table_contact", contacts=arm_table_hits,
                step_count=self.step_count,
            )
        self._update_moveit_state()
        self._update_overlay()
        if self.step_count > int(self.cfg.max_episode_steps):
            raise EpisodeFailure("EPISODE", "step_budget_exceeded", step_count=self.step_count)
        if self.render:
            time.sleep(0.005)

    def _update_overlay(self, status="candidate"):
        if not self.current_candidate:
            return
        actual = None
        try:
            actual = float(self._arm_q()[0])
        except Exception:
            pass
        self.overlay.update(
            self.current_candidate["cube_pose"][:3],
            self.current_candidate["base_pose"],
            self.joint1_target,
            actual,
            status=status,
        )

    def _checkpoint(self, stage: str, details: str = ""):
        print(f"[N2M][{stage}] {details}", flush=True)
        self._update_overlay()
        if self.pause_on_stage and self.render:
            if not self.interactive.wait_for_advance(self.env, stage):
                raise CandidateAborted()

    def reset_candidate(self, candidate: dict) -> list[str]:
        self.current_candidate = candidate
        self.step_count = 0
        self.joint1_target = None
        self.interactive.clear_candidate_abort()
        self.weld.active = False
        self.attached = False
        place_episode_state(
            self.env,
            candidate["cube_pose"],
            candidate["base_pose"],
            self.cfg.robot_root_z,
            self.cfg.robot_root_roll,
            self.cfg.robot_root_pitch,
        )
        hits = robot_table_contacts(self.env)
        self._update_overlay()
        self._checkpoint(
            "BASE_PLACED",
            f"candidate={candidate['candidate_id']} base={np.round(candidate['base_pose'], 4)} "
            f"cube={np.round(candidate['cube_pose'][:3], 4)} contacts={hits}",
        )
        return hits

    def preview_candidate(self, candidate: dict) -> dict:
        hits = self.reset_candidate(candidate)
        self.joint1_target = joint1_towards_cube(
            candidate["base_pose"], candidate["cube_pose"]
        )
        self._update_overlay("failure" if hits else "candidate")
        if self.render:
            self.interactive.wait_for_advance(
                self.env,
                f"preview {candidate['candidate_id']} (contacts={hits})",
            )
        return {
            "candidate_id": candidate["candidate_id"],
            "scene_id": candidate["scene_id"],
            "base_pose": candidate["base_pose"],
            "cube_pose": candidate["cube_pose"],
            "joint1_target": self.joint1_target,
            "initial_table_contacts": hits,
        }

    def _run_arm_target(self, target, stage: str, budget: Optional[int] = None):
        target = np.clip(np.asarray(target, dtype=np.float64), self.arm_lo, self.arm_hi)
        budget = int(budget or self.cfg.arm_stage_budget)
        start = self._arm_q()
        ramp_steps = max(1, int(budget * 0.65))
        for index in range(budget):
            alpha = min(1.0, float(index + 1) / ramp_steps)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            command = start + alpha * (target - start)
            self._step(command, GRIPPER_OPEN)
            error = float(np.max(np.abs(self._arm_q() - target)))
            if alpha >= 1.0 and error <= float(self.cfg.arm_joint_tolerance):
                self._checkpoint(stage, f"converged steps={index + 1} max_error={error:.5f}rad")
                return
        error = float(np.max(np.abs(self._arm_q() - target)))
        raise EpisodeFailure(stage, "arm_target_timeout", max_joint_error=error)

    def run_arm_init(self, candidate: dict) -> dict:
        base_pose = candidate["base_pose"]
        cube_pose = candidate["cube_pose"]
        target = joint1_towards_cube(base_pose, cube_pose)
        margin = math.radians(5.0)
        if not (self.arm_lo[0] + margin <= target <= self.arm_hi[0] - margin):
            raise EpisodeFailure(
                "ARM_INIT", "joint1_out_of_range", target=target,
                lower=float(self.arm_lo[0]), upper=float(self.arm_hi[0]),
            )
        self.joint1_target = float(target)
        self.env.disable_gripper_force_hold()

        current = self._arm_q()
        retract = ARM_HOME.copy()
        retract[0] = current[0]
        self._run_arm_target(retract, "ARM_RETRACTED")

        rotate = ARM_HOME.copy()
        rotate[0] = target
        self._run_arm_target(rotate, "JOINT1_ROTATED")

        deploy = ARM_DEPLOY_TEMPLATE.copy()
        deploy[0] = target
        self._run_arm_target(deploy, "ARM_DEPLOYED")

        actual = float(self._arm_q()[0])
        error = abs(angular_error(actual, target))
        rgb, depth = self.env.render_camera("wrist_cam")
        object_mask = self._object_segmentation_mask(
            "wrist_cam", int(rgb.shape[0]), int(rgb.shape[1])
        )
        visible_pixels = int(object_mask.sum())
        if self.debug_dir is not None:
            out = self.debug_dir / candidate["candidate_id"]
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "wrist_rgb.npy", rgb)
            np.save(out / "wrist_depth.npy", depth)
            np.save(out / "wrist_object_mask.npy", object_mask)

        details = (
            f"joint1 target={target:+.4f} actual={actual:+.4f} "
            f"error={error:.5f}rad object_pixels={visible_pixels}"
        )
        self._checkpoint("ARM_INIT_DONE", details)
        if error > float(self.cfg.arm_joint_tolerance):
            raise EpisodeFailure("ARM_INIT", "joint1_not_converged", error=error)
        if visible_pixels < int(self.cfg.wrist_min_object_pixels):
            raise EpisodeFailure(
                "ARM_INIT", "wrist_cube_not_visible",
                object_pixels=visible_pixels,
            )
        return {
            "joint1_target": target,
            "joint1_actual": actual,
            "joint1_error": error,
            "wrist_object_pixels": visible_pixels,
            "deploy_arm_target": deploy.tolist(),
        }

    @staticmethod
    def _ransac_fit_plane(points, n_iter=150, dist_thresh=0.008):
        points = np.asarray(points, dtype=np.float64)
        count = len(points)
        if count < 10:
            return None, None, np.zeros(count, dtype=bool)
        rng = np.random.default_rng(42)
        best_mask = np.zeros(count, dtype=bool)
        best_normal = best_d = None
        for _ in range(n_iter):
            a, b, c = points[rng.choice(count, 3, replace=False)]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:
                continue
            normal /= norm
            d = float(normal @ a)
            mask = np.abs(points @ normal - d) < dist_thresh
            if int(mask.sum()) > int(best_mask.sum()):
                best_mask, best_normal, best_d = mask, normal, d
        return best_normal, best_d, best_mask

    @classmethod
    def _ransac_remove_plane(cls, points, n_iter=150, dist_thresh=0.008,
                             min_inlier_ratio=0.15):
        normal, _, mask = cls._ransac_fit_plane(points, n_iter, dist_thresh)
        if normal is None or int(mask.sum()) < int(len(points) * min_inlier_ratio):
            return np.ones(len(points), dtype=bool)
        return ~mask

    def _current_tcp_pose_world(self):
        T_w_gb = self.mujoco_body_pose_world(self.env, "gripper_base").matrix()
        T_w_tcp = T_w_gb @ self.T_gripper_base_grasp_tcp()
        return self.Pose(T_w_tcp[:3, 3].copy(), T_w_tcp[:3, :3].copy())

    def _current_tcp_pose_base(self):
        T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
        T_b_tcp = self.inv_T(T_w_b) @ self._current_tcp_pose_world().matrix()
        return self.Pose(T_b_tcp[:3, 3].copy(), T_b_tcp[:3, :3].copy())

    def _tcp_error_base(self, target):
        pos, rot = self.pose_error(self._current_tcp_pose_base(), target)
        return float(np.linalg.norm(pos)), float(np.linalg.norm(rot))

    def _object_pose_in_tcp(self):
        T_w_tcp = self._current_tcp_pose_world().matrix()
        T_w_obj = self.mujoco_body_pose_world(self.env, self.cfg.object_name).matrix()
        T_tcp_obj = self.inv_T(T_w_tcp) @ T_w_obj
        return self.Pose(T_tcp_obj[:3, 3].copy(), T_tcp_obj[:3, :3].copy())

    def _finger_contacts(self):
        return self.finger_object_contact(self.env, self.cfg.object_name)

    def _object_segmentation_mask(self, camera_name: str, height: int, width: int):
        """Render a lighting-independent mask for the configured target body."""
        import mujoco

        object_bid = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.object_name
        )
        if object_bid < 0:
            raise EpisodeFailure("ARM_INIT", "target_body_not_found")

        def belongs_to_object(body_id):
            current = int(body_id)
            while current > 0:
                if current == object_bid:
                    return True
                current = int(self.env.model.body_parentid[current])
            return current == object_bid

        geom_ids = np.asarray([
            geom_id for geom_id in range(int(self.env.model.ngeom))
            if belongs_to_object(self.env.model.geom_bodyid[geom_id])
        ], dtype=np.int32)
        renderer = mujoco.Renderer(self.env.model, height=height, width=width)
        try:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.env.data, camera=camera_name)
            segmentation = renderer.render().copy()
        finally:
            renderer.close()
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            raise EpisodeFailure(
                "ARM_INIT", "unexpected_segmentation_shape",
                shape=list(segmentation.shape),
            )
        return (
            np.isin(segmentation[..., 0], geom_ids)
            & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        )

    def _finger_normal_forces(self):
        import mujoco

        out = {"link7": 0.0, "link8": 0.0}
        model, data = self.env.model, self.env.data
        object_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.object_name
        )
        finger_bids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in out
        }
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            b1 = int(model.geom_bodyid[int(contact.geom1)])
            b2 = int(model.geom_bodyid[int(contact.geom2)])
            if object_bid not in (b1, b2):
                continue
            for name, bid in finger_bids.items():
                if bid in (b1, b2):
                    force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, index, force)
                    out[name] += max(0.0, float(force[0]))
        return out

    def _dense_commands(self, reply):
        path = self.MoveItROS2Bridge.extract_arm_path(
            reply.get("trajectory", {}), self.arm_joint_names
        )
        if len(path) == 0:
            return path
        return self.MoveItROS2Bridge.densify_joint_path(
            path, np.array([0.012, 0.018, 0.018, 0.015, 0.015, 0.018])
        )

    def _execute_path(self, commands, target_pose, stage, joint_tolerance,
                      position_tolerance=0.015, rotation_tolerance=math.radians(10),
                      grip_target=GRIPPER_OPEN, extra_settle=220):
        commands = np.asarray(commands, dtype=np.float64)
        if len(commands) == 0:
            raise EpisodeFailure(stage, "empty_trajectory")
        for command in commands:
            self._step(command, grip_target)
        final = commands[-1]
        for settle in range(int(extra_settle) + 1):
            qerr = float(np.max(np.abs(self._arm_q() - final)))
            pos_err, rot_err = self._tcp_error_base(target_pose)
            if (qerr <= float(joint_tolerance)
                    and pos_err <= float(position_tolerance)
                    and rot_err <= float(rotation_tolerance)):
                self._checkpoint(
                    stage,
                    f"qerr={qerr:.4f} pos={pos_err*1000:.1f}mm "
                    f"rot={math.degrees(rot_err):.2f}deg",
                )
                return {"qerr": qerr, "tcp_pos_error": pos_err, "tcp_rot_error": rot_err}
            if settle < int(extra_settle):
                self._step(final, grip_target)
        raise EpisodeFailure(
            stage, "trajectory_not_settled", qerr=qerr,
            tcp_pos_error=pos_err, tcp_rot_error=rot_err,
        )

    def _scan_and_propose(self):
        depth_accum = []
        scan_rgb = None
        deploy = ARM_DEPLOY_TEMPLATE.copy()
        deploy[0] = self.joint1_target
        for index in range(int(self.cfg.scan_warmup + self.cfg.scan_frames)):
            rgb, depth = self.env.render_camera("wrist_cam")
            if index >= int(self.cfg.scan_warmup):
                if depth.ndim == 3:
                    depth = depth[..., 0]
                depth_accum.append(depth.astype(np.float32))
                if scan_rgb is None:
                    scan_rgb = rgb
            self._step(deploy, GRIPPER_OPEN)

        state = self.env.get_robot_state()
        scan_transforms = self.capture_scan_transforms(self.env)
        args = SimpleNamespace(**self.cfg.to_dict())
        args.grasp_checkpoint = self.grasp_checkpoint
        args.object = self.cfg.object_name
        # The reused worker communicates through this legacy temporary file.
        # Removing it first prevents a crashed worker from leaving a stale
        # result that could be mistaken for the current observation.
        Path("/tmp/grasp_result.npz").unlink(missing_ok=True)
        next_state, result = self.run_grasp_plan(
            self.env,
            args,
            depth_accum,
            scan_rgb,
            self._arm_q(),
            state["pos"],
            state["quat"],
            state["pos"],
            state["quat"],
            self._ransac_fit_plane,
            self._ransac_remove_plane,
            scan_transforms,
        )
        if result is None or next_state != self.PipelineState.MOVEIT_PRE_GRASP:
            raise EpisodeFailure("GRASP_PLAN", "anygrasp_no_usable_candidate")
        self._checkpoint(
            "SCAN_DONE",
            f"frames={len(depth_accum)} candidates={len(result.get('candidate_idxs', []))}",
        )
        return result

    def _select_and_execute_pregrasp(self, grasp_result):
        names, positions = self._moveit_state()
        T_b_c = grasp_result["scan_transforms"]["T_base_camera_optical"]
        R_w_b = grasp_result["scan_transforms"]["T_world_base"][:3, :3]
        selected = None
        last_reason = "no_candidate"
        for candidate_index in grasp_result.get("candidate_idxs", []):
            try:
                geometry = self.anygrasp_candidate_to_tcp(
                    T_b_c,
                    grasp_result["gr_translations"][candidate_index],
                    grasp_result["gr_rotations"][candidate_index],
                    float(grasp_result["gr_depths"][candidate_index]),
                    insertion=float(self.cfg.moveit_insertion),
                    pregrasp_distance=float(self.cfg.moveit_pregrasp_distance),
                    local_reach_distance=float(self.cfg.moveit_local_reach_distance),
                )
                approach_world = R_w_b @ np.asarray(geometry["approach_base"])
                if float(approach_world[2]) > float(self.cfg.moveit_max_approach_world_z):
                    last_reason = "upward_approach"
                    continue
                target = self.pose_to_ros_dict(geometry["pregrasp_tcp_pose_base"])
                ik = self.moveit.compute_ik(target, names, positions, avoid_collisions=True)
                if not ik.get("ok", False):
                    last_reason = "ik_or_goal_collision"
                    continue
                plan = self.moveit.plan_to_pose(
                    target, names, positions,
                    position_tolerance=float(self.cfg.moveit_position_tolerance),
                    orientation_tolerance=float(self.cfg.moveit_orientation_tolerance),
                    allowed_planning_time=float(self.cfg.moveit_plan_time),
                )
                commands = self._dense_commands(plan) if plan.get("ok", False) else []
                if len(commands) == 0:
                    last_reason = "path_planning_failed"
                    continue
                selected = {"geometry": geometry, "commands": commands,
                            "candidate_index": int(candidate_index)}
                break
            except Exception as exc:
                last_reason = f"candidate_exception:{exc}"
        if selected is None:
            raise EpisodeFailure("MOVEIT_PRE_GRASP", last_reason)

        self._execute_path(
            selected["commands"],
            selected["geometry"]["pregrasp_tcp_pose_base"],
            "MOVEIT_PRE_GRASP",
            self.cfg.moveit_exec_joint_tolerance,
            self.cfg.moveit_exec_tcp_position_tolerance,
            self.cfg.moveit_exec_tcp_orientation_tolerance,
        )
        return selected

    def _execute_approach(self, selected):
        names, positions = self._moveit_state()
        target_pose = selected["geometry"]["near_tcp_pose_base"]
        reply = self.moveit.compute_cartesian_path(
            [self.pose_to_ros_dict(target_pose)],
            names, positions,
            max_step=float(self.cfg.moveit_cart_step),
            jump_threshold=0.0,
            avoid_collisions=True,
        )
        fraction = float(reply.get("fraction", 0.0))
        if not reply.get("ok", False) or fraction < float(self.cfg.moveit_cart_fraction):
            raise EpisodeFailure(
                "MOVEIT_APPROACH", "cartesian_path_failed", fraction=fraction,
            )
        commands = self._dense_commands(reply)
        self._execute_path(
            commands, target_pose, "MOVEIT_APPROACH",
            self.cfg.moveit_approach_joint_tolerance,
            self.cfg.moveit_exec_tcp_position_tolerance,
            self.cfg.moveit_exec_tcp_orientation_tolerance,
            extra_settle=180,
        )

    def _run_local_reach(self, selected):
        servo = self.MuJoCoGripperBaseServo(self.env)
        target_tcp_b = selected["geometry"]["contact_tcp_pose_base"]
        target_gb_b = self.tcp_pose_to_gripper_base_pose(target_tcp_b)
        first_contact_position = None
        first_contact_side = None
        overforce_steps = 0

        for index in range(int(self.cfg.local_reach_budget)):
            T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
            T_w_gb = T_w_b @ target_gb_b.matrix()
            target_gb_w = self.Pose(T_w_gb[:3, 3], T_w_gb[:3, :3])
            cur_tcp = self._current_tcp_pose_world()
            T_w_tcp_target = T_w_b @ target_tcp_b.matrix()
            target_tcp = self.Pose(T_w_tcp_target[:3, 3], T_w_tcp_target[:3, :3])
            approach = T_w_b[:3, :3] @ target_tcp_b.rotation[:, 2]
            approach /= max(float(np.linalg.norm(approach)), 1e-12)
            contacts = self._finger_contacts()
            count = int(contacts["link7"]) + int(contacts["link8"])
            if count == 2:
                self._checkpoint("LOCAL_REACH", "bilateral finger contact")
                return

            speed_scale = 1.0
            if count == 1:
                side = "link7" if contacts["link7"] else "link8"
                if first_contact_position is None or first_contact_side != side:
                    first_contact_position = cur_tcp.position.copy()
                    first_contact_side = side
                    overforce_steps = 0
                advance = max(
                    0.0,
                    float(np.dot(cur_tcp.position - first_contact_position, approach)),
                )
                force = float(self._finger_normal_forces()[side])
                overforce_steps = overforce_steps + 1 if (
                    force >= float(self.cfg.local_single_contact_force_limit)
                ) else 0
                if overforce_steps >= int(self.cfg.local_single_contact_force_confirm_steps):
                    raise EpisodeFailure(
                        "LOCAL_REACH", "unilateral_overforce",
                        side=side, force=force, advance=advance,
                    )
                if advance >= float(self.cfg.local_single_contact_max_advance):
                    self._checkpoint(
                        "LOCAL_REACH",
                        f"unilateral bounded advance complete side={side} advance={advance:.4f}m",
                    )
                    return
                speed_scale = float(self.cfg.local_single_contact_speed_scale)
            else:
                first_contact_position = None
                first_contact_side = None
                overforce_steps = 0

            raw_command, diagnostics = servo.step_world(target_gb_w)
            current = self._arm_q()
            command = current + speed_scale * (raw_command - current)
            self._step(command, GRIPPER_OPEN)
            if diagnostics.pos_error_m < 0.004 and diagnostics.rot_error_rad < 0.10:
                self._checkpoint(
                    "LOCAL_REACH",
                    f"geometric target pos={diagnostics.pos_error_m:.4f}m",
                )
                return
        raise EpisodeFailure("LOCAL_REACH", "timeout")

    def _run_close(self):
        arm_hold = self._arm_q()
        finger_targets = [None, None]
        contact_history = []
        object_history = []
        normal_delta = float(self.env._grip_max_delta)
        force_hold_started = False
        for _ in range(int(self.cfg.close_budget)):
            contacts = self._finger_contacts()
            grip = self._grip_q()
            contact_history.append((bool(contacts["link7"]), bool(contacts["link8"])))
            contact_history = contact_history[-20:]
            object_history.append(self.env.get_object_pos(self.cfg.object_name).copy())
            object_history = object_history[-20:]

            if finger_targets[0] is None and contacts["link7"]:
                finger_targets[0] = float(grip[0] - 0.003125)
            if finger_targets[1] is None and contacts["link8"]:
                finger_targets[1] = float(grip[1] + 0.003125)

            touched = int(finger_targets[0] is not None) + int(finger_targets[1] is not None)
            self.env._grip_max_delta = 0.0004 if (touched == 1 and not force_hold_started) else normal_delta
            if touched == 2 and not force_hold_started:
                self.env.set_gripper_force_hold(2.0)
                force_hold_started = True

            target = np.array([
                finger_targets[0] if finger_targets[0] is not None else 0.0,
                finger_targets[1] if finger_targets[1] is not None else 0.0,
            ])
            self._step(arm_hold, target)

        self.env._grip_max_delta = normal_delta
        rate7 = sum(int(x[0]) for x in contact_history) / max(len(contact_history), 1)
        rate8 = sum(int(x[1]) for x in contact_history) / max(len(contact_history), 1)
        motion = (
            float(np.linalg.norm(object_history[-1] - object_history[0]))
            if len(object_history) >= 2 else float("inf")
        )
        self._checkpoint(
            "CLOSE_DONE",
            f"contact_rate={rate7:.2f}/{rate8:.2f} object_motion={motion*1000:.2f}mm",
        )
        if not force_hold_started:
            raise EpisodeFailure(
                "CLOSE", "bilateral_contact_never_established",
                contact_rate=[rate7, rate8], object_motion=motion,
            )
        return {
            "arm_hold": arm_hold,
            "finger_targets": target,
            "contact_rate": [rate7, rate8],
            "object_motion": motion,
        }

    def _detach_safely(self):
        if not self.attached or self.moveit is None:
            return
        try:
            self.moveit.detach_object(self.attached_id, link_name="grasp_tcp", timeout=6.0)
        except Exception as exc:
            print(f"[N2M] detach warning: {exc}", flush=True)
        self.attached = False

    def _lift_guard(self, initial_object_tcp, miss_steps):
        contacts = self._finger_contacts()
        miss_steps = miss_steps + 1 if not (contacts["link7"] or contacts["link8"]) else 0
        if miss_steps >= int(self.cfg.moveit_lift_contact_miss_steps):
            raise EpisodeFailure("LIFT", "both_finger_contacts_lost", miss_steps=miss_steps)
        current = self._object_pose_in_tcp().position
        slip = float(np.linalg.norm(current - initial_object_tcp))
        if slip > float(self.cfg.moveit_lift_max_object_slip):
            raise EpisodeFailure("LIFT", "object_tcp_slip", slip=slip)
        return miss_steps, slip

    def _run_lift(self, close_info):
        object_start = self.env.get_object_pos(self.cfg.object_name).copy()
        seen = np.zeros(2, dtype=np.int32)
        hold_start = object_start.copy()
        for _ in range(int(self.cfg.lift_hold_steps)):
            self._step(close_info["arm_hold"], close_info["finger_targets"])
            contacts = self._finger_contacts()
            seen += [int(contacts["link7"]), int(contacts["link8"])]
        rates = seen / max(int(self.cfg.lift_hold_steps), 1)
        hold_move = float(np.linalg.norm(
            self.env.get_object_pos(self.cfg.object_name) - hold_start
        ))
        if rates.min() < 0.40 or hold_move > 0.002:
            raise EpisodeFailure(
                "LIFT", "hold_unstable",
                contact_rate=rates.tolist(), object_motion=hold_move,
            )

        object_tcp = self._object_pose_in_tcp()
        initial_object_tcp = object_tcp.position.copy()
        size = self.cfg.moveit_attached_object_size or (0.05, 0.05, 0.05)
        attach = self.moveit.attach_box(
            self.attached_id,
            self.pose_to_ros_dict(object_tcp),
            size,
            link_name="grasp_tcp",
            touch_links=["gripper_base", "grasp_tcp", "link6", "link7", "link8"],
            timeout=6.0,
        )
        if not attach.get("ok", False):
            raise EpisodeFailure("LIFT", "moveit_attach_failed", reply=attach)
        self.attached = True

        T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
        T_w_target = self._current_tcp_pose_world().matrix()
        T_w_target[:3, 3] += np.array([0.0, 0.0, float(self.cfg.moveit_lift_distance)])
        T_b_target = self.inv_T(T_w_b) @ T_w_target
        target = self.Pose(T_b_target[:3, 3].copy(), T_b_target[:3, :3].copy())
        names, positions = self._moveit_state()
        reply = self.moveit.compute_cartesian_path(
            [self.pose_to_ros_dict(target)], names, positions,
            max_step=float(self.cfg.moveit_cart_step),
            jump_threshold=0.0,
            avoid_collisions=True,
            velocity_scaling=0.15,
            acceleration_scaling=0.15,
            timeout=8.0,
        )
        fraction = float(reply.get("fraction", 0.0))
        if not reply.get("ok", False) or fraction < float(self.cfg.moveit_lift_cart_fraction):
            raise EpisodeFailure("LIFT", "cartesian_lift_plan_failed", fraction=fraction)
        commands = self._dense_commands(reply)
        if len(commands) == 0:
            raise EpisodeFailure("LIFT", "empty_lift_trajectory")

        miss_steps = 0
        max_slip = 0.0
        for command in commands:
            self._step(command, close_info["finger_targets"])
            miss_steps, slip = self._lift_guard(initial_object_tcp, miss_steps)
            max_slip = max(max_slip, slip)
        final = commands[-1]
        for _ in range(220):
            qerr = float(np.max(np.abs(self._arm_q() - final)))
            pos_err, rot_err = self._tcp_error_base(target)
            if (qerr <= float(self.cfg.moveit_lift_joint_tolerance)
                    and pos_err <= 0.020 and rot_err <= math.radians(12.0)):
                break
            self._step(final, close_info["finger_targets"])
            miss_steps, slip = self._lift_guard(initial_object_tcp, miss_steps)
            max_slip = max(max_slip, slip)
        else:
            raise EpisodeFailure(
                "LIFT", "lift_execution_not_settled",
                qerr=qerr, tcp_pos_error=pos_err, tcp_rot_error=rot_err,
            )

        object_final = self.env.get_object_pos(self.cfg.object_name).copy()
        rise = float(object_final[2] - object_start[2])
        self._checkpoint(
            "LIFT_DONE",
            f"rise={rise*1000:.1f}mm qerr={qerr:.4f} max_slip={max_slip*1000:.1f}mm",
        )
        if rise < float(self.cfg.lift_success_rise):
            raise EpisodeFailure("LIFT", "insufficient_object_rise", rise=rise)
        self._detach_safely()
        return {
            "object_start": object_start.tolist(),
            "object_final": object_final.tolist(),
            "object_rise": rise,
            "max_object_tcp_slip": max_slip,
            "lift_fraction": fraction,
            "hold_contact_rate": rates.tolist(),
            "hold_object_motion": hold_move,
        }

    def run_episode(self, candidate: dict, attempt: int = 0,
                    stop_after: str = "lift") -> EpisodeResult:
        started = time.monotonic()
        arm_info = {}
        completed = "RESET"
        try:
            initial_hits = self.reset_candidate(candidate)
            if initial_hits and self.cfg.reject_initial_table_contact:
                raise EpisodeFailure(
                    "BASE_PLACED", "initial_robot_table_collision", contacts=initial_hits
                )
            if initial_hits:
                print(
                    f"[N2M][BASE_PLACED] warning: retaining candidate with "
                    f"initial table contacts {initial_hits}",
                    flush=True,
                )
            self.weld.enable()
            arm_info = self.run_arm_init(candidate)
            arm_info["initial_table_contacts"] = initial_hits
            completed = "ARM_INIT"
            if stop_after.lower() == "arm_init":
                self._update_overlay("success")
                return self._result(
                    candidate, attempt, None, completed, None, None,
                    started, arm_info, arm_info,
                )

            self._ensure_moveit()
            grasp = self._scan_and_propose()
            completed = "GRASP_PLAN"
            selected = self._select_and_execute_pregrasp(grasp)
            completed = "MOVEIT_PRE_GRASP"
            self._execute_approach(selected)
            completed = "MOVEIT_APPROACH"
            self._run_local_reach(selected)
            completed = "LOCAL_REACH"
            close_info = self._run_close()
            completed = "CLOSE"
            lift_info = self._run_lift(close_info)
            completed = "LIFT"
            metrics = dict(arm_info)
            metrics.update(lift_info)
            metrics["selected_anygrasp_index"] = selected["candidate_index"]
            self._update_overlay("success")
            return self._result(
                candidate, attempt, True, completed, None, None,
                started, arm_info, metrics,
            )
        except EpisodeFailure as failure:
            metrics = dict(arm_info)
            metrics.update(failure.metrics)
            self._update_overlay("failure")
            return self._result(
                candidate, attempt, False, completed,
                failure.stage, failure.reason, started, arm_info, metrics,
            )
        except Exception as exc:
            self._update_overlay("failure")
            return self._result(
                candidate, attempt, False, completed,
                "INTERNAL", repr(exc), started, arm_info, arm_info,
            )
        finally:
            self._detach_safely()
            try:
                self.env.disable_gripper_force_hold()
            except Exception:
                pass

    def _result(self, candidate, attempt, success, completed, failure_stage,
                failure_reason, started, arm_info, metrics):
        actual = arm_info.get("joint1_actual")
        target = arm_info.get("joint1_target", self.joint1_target)
        error = arm_info.get("joint1_error")
        drift_position, drift_rotation = self.weld.drift()
        full_metrics = dict(metrics)
        full_metrics.update({
            "base_weld_drift_m": drift_position,
            "base_weld_drift_rad": drift_rotation,
        })
        return EpisodeResult(
            candidate_id=candidate["candidate_id"],
            scene_id=candidate["scene_id"],
            attempt=int(attempt),
            success=success,
            completed_stage=completed,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            cube_pose=list(candidate["cube_pose"]),
            base_pose=list(candidate["base_pose"]),
            joint1_target=None if target is None else float(target),
            joint1_actual=None if actual is None else float(actual),
            joint1_error=None if error is None else float(error),
            elapsed_s=float(time.monotonic() - started),
            episode_steps=int(self.step_count),
            metrics=full_metrics,
        )
