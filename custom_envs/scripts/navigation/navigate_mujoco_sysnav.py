#!/usr/bin/env python3
"""Search for an unknown cube with SysNav, then run the existing N2M task.

The executable keeps the validated ``navigate_mujoco_n2m.py`` unchanged.  A
SysNav search runs first in the same MuJoCo environment.  Its visual 3-D target
estimate and a detector-derived target mask are then handed to the existing
observation, N2M, grasp and delivery stages without resetting the simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.data_generation.camera_geometry import (
    camera_points_to_frame,
    crop_point_mask,
    depth_to_optical_points,
    voxel_downsample_indices,
)
from custom_envs.scripts.N2M.data_generation.live_observation import (
    LiveObservation,
    LiveObservationConfig,
    LiveObservationError,
)
from custom_envs.scripts.N2M.manifests import write_json
from custom_envs.scripts.N2M.manipulation_episode import ARM_HOME, GRIPPER_OPEN
from custom_envs.scripts.N2M.target_provider import TargetEstimate
from custom_envs.scripts.navigation import navigate_mujoco_n2m as n2m
from custom_envs.scripts.sysnav.config import DEFAULT_CONFIG
from custom_envs.scripts.sysnav.obstacle_edge_view import (
    plan_obstacle_edge_views,
    save_obstacle_edge_view_debug,
)
from custom_envs.scripts.sysnav.rgbd import capture_rgbd_frame
from custom_envs.scripts.sysnav.runtime import SysNavComponents, build_sysnav_components
from custom_envs.scripts.sysnav.search_runner import SysNavSearchRunner
from custom_envs.scripts.sysnav.transit_detection import (
    ContinuousTransitDetector,
    TransitCandidate,
)
from custom_envs.scripts.sysnav.types import SearchInstruction


class _PreparsedParser:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def parse_args(self):
        return self.args


class _VisionTargetProvider:
    def __init__(self, position_world: np.ndarray):
        self.position_world = np.asarray(position_world, dtype=np.float64).reshape(3)

    def estimate(self, env, object_name: str = "cube") -> TargetEstimate:
        del env, object_name
        return TargetEstimate(
            position_world=self.position_world.copy(),
            quaternion_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            source="sysnav_yoloe_sam2_depth",
        )

    def update(self, position_world: np.ndarray) -> None:
        """Refresh the visual target after a later observation round."""

        self.position_world = np.asarray(
            position_world, dtype=np.float64
        ).reshape(3)


class _TransitNavigationInterrupt(RuntimeError):
    """Internal control-flow signal raised after a moving-frame detection."""

    def __init__(self, candidate: TransitCandidate):
        super().__init__("navigation interrupted for transit target confirmation")
        self.candidate = candidate


class _ObservedEnvProxy:
    """Observe N2M's unmodified teleport loop after each MuJoCo step."""

    def __init__(self, env, detector: ContinuousTransitDetector, name: str):
        self._env = env
        self._detector = detector
        self._name = str(name)
        self._origin = n2m._base_pose_from_state(env.get_robot_state())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        result = self._env.step(action)
        pose = n2m._base_pose_from_state(self._env.get_robot_state())
        moved = (
            float(np.linalg.norm(pose[:2] - self._origin[:2])) > 1e-5
            or abs(n2m._wrap_angle(float(pose[2] - self._origin[2]))) > 1e-5
        )
        if moved:
            candidate = self._detector.sample(
                pose,
                navigation_name=self._name,
            )
            if candidate is not None:
                raise _TransitNavigationInterrupt(candidate)
        return result


def _path_pose_at_fraction(
    path_xy,
    fraction: float,
) -> tuple[np.ndarray, float]:
    """Interpolate a pose at an arc-length fraction of a planned XY path."""

    points = np.asarray(path_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("planned path must contain at least two XY points")
    points = points[:, :2]
    if not np.all(np.isfinite(points)):
        raise ValueError("planned path contains non-finite coordinates")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("path fraction must be strictly between zero and one")

    # Nav2 paths can contain repeated poses. Remove only consecutive repeats
    # so the cumulative length and tangent remain well-defined.
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9)
    )
    points = points[keep]
    if len(points) < 2:
        raise ValueError("planned path has zero length")
    vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    distance = float(fraction) * total_length
    segment_index = min(
        int(np.searchsorted(cumulative, distance, side="right") - 1),
        len(segment_lengths) - 1,
    )
    alpha = (
        distance - float(cumulative[segment_index])
    ) / float(segment_lengths[segment_index])
    xy = points[segment_index] + alpha * vectors[segment_index]
    yaw = math.atan2(
        float(vectors[segment_index, 1]),
        float(vectors[segment_index, 0]),
    )
    return np.array([xy[0], xy[1], yaw], dtype=np.float64), total_length


class _LegacySearchNavigator:
    """Execute SysNav metric viewpoints with the already validated controllers."""

    def __init__(
        self,
        env,
        bridge,
        args,
        controller,
        *,
        policy=None,
        device=None,
        transit_detector: ContinuousTransitDetector | None = None,
    ):
        self.env = env
        self.bridge = bridge
        self.args = args
        self.controller = controller
        self.policy = policy
        self.device = device
        self.transit_detector = transit_detector
        self._pending_transit_candidate: TransitCandidate | None = None

    def take_transit_candidate(self) -> TransitCandidate | None:
        candidate = self._pending_transit_candidate
        self._pending_transit_candidate = None
        return candidate

    def current_pose(self) -> np.ndarray:
        return n2m._base_pose_from_state(self.env.get_robot_state())

    def move_to(self, pose: tuple[float, float, float], *, name: str) -> dict:
        target = np.asarray(pose, dtype=np.float64).reshape(3)
        current = self.current_pose()
        direct_distance = float(np.linalg.norm(target[:2] - current[:2]))
        split_threshold = float(self.args.sysnav_max_direct_goal_distance)
        if direct_distance > split_threshold:
            path = self.bridge.plan_path(
                current[:2],
                target[:2],
                timeout=float(self.args.teleport_plan_timeout),
            )
            if len(path) < 2:
                raise RuntimeError(
                    f"{name}: Nav2 found no complete path to the final target; "
                    "cannot choose a safe path midpoint"
                )
            midpoint, path_length = _path_pose_at_fraction(path, 0.5)
            print(
                f"[SysNav][NAV_SPLIT] name={name} "
                f"distance={direct_distance:.3f}m "
                f"threshold={split_threshold:.3f}m "
                f"strategy=nav2_path_arclength "
                f"path_points={len(path)} path_length={path_length:.3f}m "
                f"midpoint={np.round(midpoint, 3)} "
                f"target={np.round(target, 3)}",
                flush=True,
            )
            midpoint_metrics = self._move_once(
                midpoint,
                name=f"{name}_MIDPOINT",
            )
            if midpoint_metrics.get("interrupted_for_transit_candidate"):
                midpoint_metrics["direct_distance_m"] = direct_distance
                midpoint_metrics["split_threshold_m"] = split_threshold
                midpoint_metrics["split_strategy"] = "nav2_path_arclength"
                midpoint_metrics["split_path_points"] = len(path)
                midpoint_metrics["split_path_length_m"] = path_length
                midpoint_metrics["midpoint_pose"] = midpoint.astype(float).tolist()
                return midpoint_metrics
            target_metrics = self._move_once(target, name=name)
            result = dict(target_metrics)
            result["mode"] = f"{target_metrics['mode']}_via_midpoint"
            result["direct_distance_m"] = direct_distance
            result["split_threshold_m"] = split_threshold
            result["split_strategy"] = "nav2_path_arclength"
            result["split_path_points"] = len(path)
            result["split_path_length_m"] = path_length
            result["midpoint_pose"] = midpoint.astype(float).tolist()
            result["legs"] = [midpoint_metrics, target_metrics]
            result["elapsed_s"] = sum(
                float(metrics.get("elapsed_s", 0.0))
                for metrics in result["legs"]
            )
            for key in ("steps", "teleport_count", "path_waypoints", "align_steps"):
                values = [metrics.get(key) for metrics in result["legs"]]
                if all(value is not None for value in values):
                    result[key] = sum(int(value) for value in values)
            return result
        return self._move_once(target, name=name)

    def _move_once(self, target: np.ndarray, *, name: str) -> dict:
        if self.args.teleport_navigation:
            observed_env = (
                self.env
                if self.transit_detector is None
                else _ObservedEnvProxy(self.env, self.transit_detector, name)
            )
            started = time.monotonic()
            try:
                return n2m._follow_nav2_path_by_teleport(
                    name=name,
                    env=observed_env,
                    bridge=self.bridge,
                    interactive=self.controller,
                    goal_xy=target[:2],
                    final_yaw=float(target[2]),
                    max_steps=self.args.sysnav_nav_max_steps,
                    teleport_every_steps=self.args.teleport_every_steps,
                    teleport_spacing_m=self.args.teleport_step_distance,
                    plan_timeout_s=self.args.teleport_plan_timeout,
                    snapshot_writer=getattr(self.args, "nav_snapshot_writer", None),
                    arm_target=ARM_HOME,
                    gripper_target=GRIPPER_OPEN,
                )
            except _TransitNavigationInterrupt as interrupt:
                self._pending_transit_candidate = interrupt.candidate
                self.bridge.cancel_goal()
                pose = self.current_pose()
                print(
                    f"[SYSNAV][NAV_INTERRUPTED_FOR_TARGET] name={name} "
                    f"pose={np.round(pose, 3)} "
                    f"sequence={interrupt.candidate.observation.sequence}",
                    flush=True,
                )
                return {
                    "mode": "nav2_path_teleport",
                    "interrupted_for_transit_candidate": True,
                    "elapsed_s": time.monotonic() - started,
                    "final_base_pose": pose.astype(float).tolist(),
                }
        return self._move_physically(target, name=name)

    def _move_physically(self, target: np.ndarray, *, name: str) -> dict:
        if self.policy is None or self.device is None:
            raise RuntimeError("physical SysNav search requires a locomotion policy")
        state = self.env.get_robot_state()
        self.bridge.update_robot_pose(
            state["pos"], state["quat"], state["lin_vel"], state["ang_vel"]
        )
        self.bridge.send_goal(float(target[0]), float(target[1]))
        pacer = n2m.RealtimePacer(self.env.dt)
        conditioner = n2m.NavCommandConditioner(self.env.dt)
        last_action = np.zeros(16, dtype=np.float32)
        started = time.monotonic()
        for step in range(int(self.args.sysnav_nav_max_steps)):
            if self.controller.quit:
                raise KeyboardInterrupt
            state = self.env.get_robot_state()
            self.bridge.update_robot_pose(
                state["pos"], state["quat"], state["lin_vel"], state["ang_vel"]
            )
            distance = float(np.linalg.norm(state["pos"][:2] - target[:2]))
            done, failed = self.bridge.get_nav_status()
            if distance <= self.args.sysnav_position_tolerance or done:
                self.bridge.cancel_goal()
                break
            if failed:
                raise RuntimeError(f"{name}: Nav2 failed at distance {distance:.3f}m")
            vx, wz = n2m._clip_nav_command(
                *self.bridge.get_cmd_vel(), self.args.nav2_settings
            )
            vx, wz, _ = conditioner.step(vx, wz)
            last_action = n2m._policy_step(
                self.env, self.policy, self.device, last_action, (vx, 0.0, wz)
            )
            contacts = n2m.robot_environment_contacts(self.env)
            if contacts:
                raise RuntimeError(f"{name}: environment contact: {contacts}")
            if self.transit_detector is not None:
                candidate = self.transit_detector.sample(
                    self.current_pose(), navigation_name=name
                )
                if candidate is not None:
                    self._pending_transit_candidate = candidate
                    self.bridge.cancel_goal()
                    return {
                        "mode": "physical_locomotion_policy",
                        "interrupted_for_transit_candidate": True,
                        "elapsed_s": time.monotonic() - started,
                        "final_base_pose": self.current_pose().tolist(),
                    }
            pacer.wait()
        else:
            self.bridge.cancel_goal()
            raise RuntimeError(f"{name}: position navigation timed out")

        settled = 0
        align_steps = 0
        for align_steps in range(int(self.args.sysnav_align_max_steps)):
            state = self.env.get_robot_state()
            error = n2m._wrap_angle(float(target[2] - state["yaw"]))
            if abs(error) <= math.radians(self.args.sysnav_yaw_tolerance_deg):
                settled += 1
                command = 0.0
            else:
                settled = 0
                command = n2m._effective_yaw_command(error)
            last_action = n2m._policy_step(
                self.env,
                self.policy,
                self.device,
                last_action,
                (0.0, 0.0, command),
            )
            if self.transit_detector is not None:
                candidate = self.transit_detector.sample(
                    self.current_pose(), navigation_name=f"{name}_ALIGN"
                )
                if candidate is not None:
                    self._pending_transit_candidate = candidate
                    self.bridge.cancel_goal()
                    return {
                        "mode": "physical_locomotion_policy",
                        "interrupted_for_transit_candidate": True,
                        "elapsed_s": time.monotonic() - started,
                        "final_base_pose": self.current_pose().tolist(),
                    }
            if settled >= 8:
                break
            pacer.wait()
        else:
            raise RuntimeError(f"{name}: final yaw alignment timed out")
        return {
            "mode": "physical_locomotion_policy",
            "elapsed_s": time.monotonic() - started,
            "align_steps": align_steps + 1,
            "final_base_pose": self.current_pose().tolist(),
        }


class _DirectTeleportTestNavigator:
    """Explicit test-only controller that bypasses Nav2 but keeps collision checks."""

    def __init__(self, env):
        self.env = env

    def current_pose(self) -> np.ndarray:
        return n2m._base_pose_from_state(self.env.get_robot_state())

    def move_to(self, pose: tuple[float, float, float], *, name: str) -> dict:
        before = self.current_pose()
        n2m._teleport_base_pose(self.env, pose)
        contacts = n2m.robot_environment_contacts(self.env)
        if contacts:
            raise RuntimeError(f"{name}: direct test pose collides: {contacts}")
        for _ in range(2):
            self.env.set_arm_target(ARM_HOME)
            self.env.set_gripper_target(GRIPPER_OPEN)
            self.env.step(np.zeros(16, dtype=np.float32))
        return {
            "mode": "direct_teleport_test_only",
            "from_pose": before.tolist(),
            "final_base_pose": self.current_pose().tolist(),
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = n2m.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--instruction",
        required=True,
        help='natural-language task, e.g. "去客厅茶几上拿 cube"',
    )
    parser.add_argument("--sysnav-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sysnav-max-viewpoints", type=int, default=120)
    parser.add_argument("--sysnav-nav-max-steps", type=int, default=8000)
    parser.add_argument("--sysnav-position-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--sysnav-max-direct-goal-distance",
        type=float,
        default=6.0,
        help=(
            "split a longer SysNav navigation into current-to-midpoint and "
            "midpoint-to-target legs (default: 6.0 m)"
        ),
    )
    parser.add_argument("--sysnav-align-max-steps", type=int, default=800)
    parser.add_argument("--sysnav-yaw-tolerance-deg", type=float, default=4.0)
    parser.add_argument(
        "--sysnav-max-n2m-rounds",
        type=int,
        default=2,
        help=(
            "maximum observe/predict/grasp rounds; after a complete 3x3 "
            "grasp exhaustion, move to a different observation view and "
            "rerun N2M (default: 2 total rounds)"
        ),
    )
    parser.add_argument(
        "--sysnav-retry-min-view-angle-deg",
        type=float,
        default=60.0,
        help=(
            "preferred minimum angular separation around the target between "
            "N2M observation rounds (default: 60 degrees)"
        ),
    )
    parser.add_argument(
        "--sysnav-disable-obstacle-edge-view",
        action="store_true",
        help=(
            "disable obstacle-edge N2M observation planning and use the "
            "legacy radial standoff/retry selection"
        ),
    )
    parser.add_argument(
        "--sysnav-skip-vlm",
        action="store_true",
        help=(
            "test mode: treat --object as a class-only parsed instruction; "
            "disables room semantics and early-stop VLM calls"
        ),
    )
    parser.add_argument("--sysnav-skip-room-semantics", action="store_true")
    parser.add_argument("--sysnav-disable-early-stop", action="store_true")
    parser.add_argument(
        "--sysnav-disable-transit-detection",
        action="store_true",
        help="disable RGB-D target detection while the robot is navigating",
    )
    parser.add_argument(
        "--sysnav-transit-detection-hz",
        type=float,
        default=None,
        help="override the config sampling-rate upper bound (default: 2 Hz)",
    )
    parser.add_argument(
        "--sysnav-search-only",
        action="store_true",
        help="stop successfully after visually locating the target",
    )
    parser.add_argument(
        "--sysnav-direct-teleport-test",
        action="store_true",
        help="test only: bypass Nav2 during the search stage",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.sysnav_max_viewpoints,
        args.sysnav_nav_max_steps,
        args.sysnav_position_tolerance,
        args.sysnav_max_direct_goal_distance,
        args.sysnav_align_max_steps,
        args.sysnav_yaw_tolerance_deg,
        args.sysnav_max_n2m_rounds,
        args.sysnav_retry_min_view_angle_deg,
    )
    if any(float(value) <= 0.0 for value in positive):
        raise SystemExit("SysNav viewpoint/navigation limits must be positive")
    if args.sysnav_retry_min_view_angle_deg > 180.0:
        raise SystemExit("--sysnav-retry-min-view-angle-deg must be <= 180")
    if (
        getattr(args, "sysnav_transit_detection_hz", None) is not None
        and args.sysnav_transit_detection_hz <= 0.0
    ):
        raise SystemExit("--sysnav-transit-detection-hz must be positive")
    if args.sysnav_direct_teleport_test and not args.sysnav_search_only:
        raise SystemExit(
            "--sysnav-direct-teleport-test requires --sysnav-search-only"
        )
    if not args.scene.is_file():
        raise SystemExit(f"scene does not exist: {args.scene}")
    if not args.nav2_params.is_file():
        raise SystemExit(f"Nav2 params do not exist: {args.nav2_params}")
    if not args.sysnav_search_only:
        for path, label in (
            (args.n2m_checkpoint, "N2M checkpoint"),
            (Path(args.policy_path).expanduser(), "locomotion policy"),
            (Path(args.grasp_checkpoint).expanduser(), "grasp checkpoint"),
        ):
            if not path.is_file():
                raise SystemExit(f"{label} does not exist: {path}")


def _prepare_output_dir(args: argparse.Namespace) -> Path:
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            _ROOT
            / "custom_envs"
            / "scripts"
            / "N2M"
            / "data"
            / "integrated_runs"
            / ("sysnav_" + n2m._timestamp_token())
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    return output


def _build_visual_live_capture(
    components: SysNavComponents,
    target_provider: _VisionTargetProvider,
    target_label: str,
):
    used = False

    def capture(env, config: LiveObservationConfig | None = None) -> LiveObservation:
        nonlocal used
        if used:
            raise RuntimeError("SysNav N2M observation capture may only run once")
        used = True
        cfg = config or LiveObservationConfig()
        frame = capture_rgbd_frame(env, cfg.camera_name)
        detections = components.perception.detect(frame.rgb)
        localized = components.localizer.localize_all(frame, detections)
        candidates = [
            item for item in localized if item.detection.label == target_label
        ]
        if not candidates:
            raise LiveObservationError(
                "target_not_redetected_by_sysnav",
                target_label=target_label,
                detections=[item.label for item in detections],
            )
        target = min(
            candidates,
            key=lambda item: float(
                np.linalg.norm(
                    item.position_world - target_provider.position_world
                )
            ),
        )
        # The object may have shifted during the previous failed manipulation
        # round.  Navigation uses the last visual estimate; all prediction and
        # candidate transforms after this capture use the refreshed estimate.
        target_provider.update(target.position_world)
        target_mask = target.detection.mask
        valid = (
            np.isfinite(frame.depth)
            & (frame.depth > 0.0)
            & (frame.depth <= float(cfg.max_depth_m))
        )
        optical, flat_indices = depth_to_optical_points(
            frame.depth, frame.intrinsics, valid
        )
        point_rgb = frame.rgb.reshape(-1, 3)[flat_indices]
        target_flags = target_mask.reshape(-1)[flat_indices]
        import mujoco

        base_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if base_id < 0:
            raise LiveObservationError("missing_base_link")
        xyz_base = camera_points_to_frame(
            optical,
            frame.camera_position_world,
            frame.camera_rotation_world,
            env.data.xpos[base_id],
            env.data.xmat[base_id],
        )
        keep = crop_point_mask(xyz_base, cfg.crop_min, cfg.crop_max)
        xyz_base = xyz_base[keep]
        point_rgb = point_rgb[keep]
        target_flags = target_flags[keep]
        keep = voxel_downsample_indices(xyz_base, float(cfg.voxel_size_m))
        xyz_base = xyz_base[keep]
        point_rgb = point_rgb[keep]
        target_flags = target_flags[keep]
        metrics = {
            "target_visible_pixels": int(target_mask.sum()),
            "target_camera_distance_m": float(
                np.linalg.norm(
                    target.position_world - frame.camera_position_world
                )
            ),
            "point_count": int(len(xyz_base)),
            "target_points_after_voxel": int(target_flags.sum()),
            "detector_confidence": float(target.detection.confidence),
            "segmentation_source": "yoloe_box_plus_sam2",
        }
        if len(xyz_base) < int(cfg.minimum_point_count):
            raise LiveObservationError("too_few_cloud_points", **metrics)
        if int(target_flags.sum()) < int(cfg.min_target_points):
            raise LiveObservationError(
                "too_few_target_points_after_voxel", **metrics
            )
        target_centroid_base = np.median(xyz_base[target_flags], axis=0)
        robot = env.get_robot_state()
        observation = LiveObservation(
            xyz_base=xyz_base.astype(np.float32),
            rgb=point_rgb.astype(np.uint8),
            observation_pose_world=[
                float(robot["pos"][0]),
                float(robot["pos"][1]),
                float(robot["yaw"]),
            ],
            target_pose_world=np.concatenate(
                (target.position_world, [1.0, 0.0, 0.0, 0.0])
            ).astype(float).tolist(),
            target_centroid_base=target_centroid_base.astype(np.float64),
            metrics=metrics,
            image_rgb=frame.rgb,
            image_depth=frame.depth,
            target_mask=target_mask,
        )
        # N2M and AnyGrasp need the same GPU next.  Release YOLOE/SAM2 after
        # every successful capture; a later retry round reloads them lazily.
        components.perception.close()
        return observation

    return capture


def _angular_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _select_retry_observation_pose(
    components: SysNavComponents,
    target_position_world: np.ndarray,
    previous_observation_poses: list[np.ndarray],
    current_robot_pose: np.ndarray,
    *,
    preferred_distance_m: float,
    min_distance_m: float,
    max_distance_m: float,
    preferred_angle_separation_deg: float,
) -> tuple[tuple[float, float, float], dict]:
    """Choose a free in-room target view distinct from earlier N2M views."""

    target = np.asarray(target_position_world, dtype=np.float64).reshape(3)
    current = np.asarray(current_robot_pose, dtype=np.float64).reshape(3)
    room_id = components.room_graph.room_at(float(target[0]), float(target[1]))
    if room_id is None:
        raise RuntimeError("cannot select a retry view: target is outside room map")

    candidate_mask = (
        components.room_map.searchable_mask(room_id)
        & components.nav_map.free_mask
    )
    rows, cols = np.nonzero(candidate_mask)
    if len(rows) == 0:
        raise RuntimeError(f"cannot select a retry view: {room_id} has no free cells")
    points = np.asarray(
        [
            components.nav_map.grid_to_world(int(row), int(col))
            for row, col in zip(rows, cols, strict=True)
        ],
        dtype=np.float64,
    )
    offsets = points - target[:2]
    distances = np.linalg.norm(offsets, axis=1)
    in_observation_band = (
        (distances >= float(min_distance_m))
        & (distances <= float(max_distance_m))
    )
    points = points[in_observation_band]
    offsets = offsets[in_observation_band]
    distances = distances[in_observation_band]
    if len(points) == 0:
        raise RuntimeError(
            f"cannot select a retry view: no free {room_id} cell lies in "
            f"the {min_distance_m:.2f}-{max_distance_m:.2f}m observation band"
        )

    if previous_observation_poses:
        prior_xy = np.asarray(
            [pose[:2] for pose in previous_observation_poses],
            dtype=np.float64,
        )
        nearest_prior_distance = np.linalg.norm(
            points[:, None, :] - prior_xy[None, :, :], axis=2
        ).min(axis=1)
        distinct = nearest_prior_distance >= 0.30
        points = points[distinct]
        offsets = offsets[distinct]
        distances = distances[distinct]
        nearest_prior_distance = nearest_prior_distance[distinct]
        if len(points) == 0:
            raise RuntimeError(
                "cannot select a retry view: no observation-band cell is "
                "at least 0.30m from every previous observation pose"
            )
    else:
        nearest_prior_distance = np.full(len(points), math.inf)

    candidate_angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    prior_angles = np.asarray(
        [
            math.atan2(
                float(pose[1] - target[1]),
                float(pose[0] - target[0]),
            )
            for pose in previous_observation_poses
        ],
        dtype=np.float64,
    )
    if len(prior_angles):
        separations = np.column_stack(
            [
                _angular_separation(candidate_angles, prior)
                for prior in prior_angles
            ]
        )
        minimum_separation = separations.min(axis=1)
    else:
        minimum_separation = np.full(len(points), math.pi, dtype=np.float64)

    requested_separation = math.radians(preferred_angle_separation_deg)
    preferred = minimum_separation >= requested_separation
    pool = np.flatnonzero(preferred)
    used_fallback = False
    if len(pool) == 0:
        pool = np.arange(len(points))
        used_fallback = True

    travel = np.linalg.norm(points - current[:2], axis=1)
    radial_error = np.abs(distances - float(preferred_distance_m))
    # Angular diversity is the primary objective.  Radial accuracy keeps the
    # camera in the validated capture geometry, while travel is only a small
    # tie-breaker among otherwise comparable views.
    score = 8.0 * minimum_separation - 2.0 * radial_error - 0.05 * travel
    selected = int(pool[np.argmax(score[pool])])
    xy = points[selected]
    yaw = math.atan2(float(target[1] - xy[1]), float(target[0] - xy[0]))
    pose = (float(xy[0]), float(xy[1]), float(yaw))
    diagnostics = {
        "room_id": room_id,
        "pose": list(pose),
        "target_position_world": target.astype(float).tolist(),
        "distance_to_target_m": float(distances[selected]),
        "minimum_prior_view_angle_deg": math.degrees(
            float(minimum_separation[selected])
        ),
        "preferred_minimum_angle_deg": float(preferred_angle_separation_deg),
        "used_angle_fallback": used_fallback,
        "travel_distance_m": float(travel[selected]),
        "nearest_prior_view_distance_m": (
            float(nearest_prior_distance[selected])
            if np.isfinite(nearest_prior_distance[selected])
            else None
        ),
        "candidate_count_in_observation_band": int(len(points)),
    }
    return pose, diagnostics


def _retry_joint_positions(env) -> tuple[np.ndarray, np.ndarray]:
    """Read the six arm and two gripper joints from the shared environment."""

    state = env.get_robot_state()
    if "joint_pos" not in state:
        raise RuntimeError("retry recovery requires joint_pos in robot state")
    arm_indices = np.asarray(env._arm_all_idx, dtype=np.int32)
    grip_indices = np.asarray(env._grip_all_idx, dtype=np.int32)
    joints = np.asarray(state["joint_pos"], dtype=np.float64)
    return joints[arm_indices].copy(), joints[grip_indices].copy()


def _move_retry_arm_stage(
    env,
    target: np.ndarray,
    *,
    stage: str,
    interactive=None,
    max_steps: int = 260,
    tolerance_rad: float = 0.02,
) -> dict:
    """Smoothly move one recovery arm stage while the base weld is active."""

    target = np.asarray(target, dtype=np.float64).reshape(6)
    start, _ = _retry_joint_positions(env)
    ramp_steps = max(1, int(max_steps * 0.65))
    last_arm_error = math.inf
    last_gripper_error = math.inf
    for index in range(int(max_steps)):
        if interactive is not None and getattr(interactive, "quit", False):
            raise KeyboardInterrupt
        alpha = min(1.0, float(index + 1) / float(ramp_steps))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        command = start + alpha * (target - start)
        env.set_arm_target(command)
        env.set_gripper_target(GRIPPER_OPEN)
        env.step(np.zeros(16, dtype=np.float32))

        contacts = n2m.robot_environment_contacts(
            env,
            ignored_environment_geoms=("floor",),
        )
        if contacts:
            raise RuntimeError(f"{stage} collided: {contacts}")
        arm_q, grip_q = _retry_joint_positions(env)
        last_arm_error = float(np.max(np.abs(arm_q - target)))
        last_gripper_error = float(np.max(np.abs(grip_q - GRIPPER_OPEN)))
        if (
            alpha >= 1.0
            and last_arm_error <= float(tolerance_rad)
            and last_gripper_error <= float(tolerance_rad)
        ):
            metrics = {
                "stage": stage,
                "steps": index + 1,
                "max_arm_error_rad": last_arm_error,
                "max_gripper_error_rad": last_gripper_error,
            }
            print(
                f"[SYSNAV][{stage}] steps={index + 1} "
                f"arm_error={last_arm_error:.5f}rad "
                f"gripper_error={last_gripper_error:.5f}rad",
                flush=True,
            )
            return metrics
    raise RuntimeError(
        f"{stage} timed out after {max_steps} steps: "
        f"arm_error={last_arm_error:.5f}rad "
        f"gripper_error={last_gripper_error:.5f}rad"
    )


def _prepare_retry_navigation_state(
    env,
    target_position_world: np.ndarray,
    *,
    safe_distance_m: float,
    retreat_step_distance_m: float = 0.04,
    interactive=None,
    arm_stage_max_steps: int = 260,
    arm_tolerance_rad: float = 0.02,
) -> dict:
    """Restore a collision-safe navigation state before another N2M round.

    A failed manipulation round leaves the base close to the table with the arm
    deployed.  NavFn plans only XY translation and its first path tangent may
    rotate the robot in place, sweeping the hind legs into the table.  Fold the
    arm first, then preserve the current yaw while moving radially away from the
    target in short, collision-checked pose resets.
    """

    safe_distance = float(safe_distance_m)
    step_distance = float(retreat_step_distance_m)
    if safe_distance <= 0.0:
        raise ValueError("safe_distance_m must be positive")
    if step_distance <= 0.0:
        raise ValueError("retreat_step_distance_m must be positive")

    target = np.asarray(target_position_world, dtype=np.float64).reshape(3)
    initial_pose = n2m._base_pose_from_state(env.get_robot_state())
    print(
        "[SYSNAV][N2M_RETRY_RECOVERY_START] "
        f"base={np.round(initial_pose, 4).tolist()} "
        f"target={np.round(target, 4).tolist()} "
        f"safe_distance={safe_distance:.3f}m",
        flush=True,
    )

    lock = n2m.BaseWeld(env)
    lock.enable()
    try:
        if hasattr(env, "disable_gripper_force_hold"):
            env.disable_gripper_force_hold()

        current_arm, _ = _retry_joint_positions(env)
        folded_at_current_azimuth = ARM_HOME.copy()
        folded_at_current_azimuth[0] = current_arm[0]
        arm_stages = [
            _move_retry_arm_stage(
                env,
                folded_at_current_azimuth,
                stage="N2M_RETRY_ARM_RETRACTED",
                interactive=interactive,
                max_steps=arm_stage_max_steps,
                tolerance_rad=arm_tolerance_rad,
            ),
            _move_retry_arm_stage(
                env,
                ARM_HOME,
                stage="N2M_RETRY_ARM_HOME",
                interactive=interactive,
                max_steps=arm_stage_max_steps,
                tolerance_rad=arm_tolerance_rad,
            ),
        ]

        retreat_start = n2m._base_pose_from_state(env.get_robot_state())
        radial = retreat_start[:2] - target[:2]
        initial_distance = float(np.linalg.norm(radial))
        if initial_distance > 1e-6:
            direction = radial / initial_distance
        else:
            # A robot observing the target normally faces it, so negative
            # heading is the safest deterministic radial fallback.
            direction = -np.array(
                [math.cos(retreat_start[2]), math.sin(retreat_start[2])],
                dtype=np.float64,
            )
        retreat_distance = max(0.0, safe_distance - initial_distance)
        retreat_steps = (
            int(math.ceil(retreat_distance / step_distance))
            if retreat_distance > 0.0
            else 0
        )
        final_xy = retreat_start[:2] + direction * retreat_distance
        previous_pose = retreat_start.copy()
        for index in range(retreat_steps):
            if interactive is not None and getattr(interactive, "quit", False):
                raise KeyboardInterrupt
            fraction = float(index + 1) / float(retreat_steps)
            candidate = np.array(
                [
                    retreat_start[0]
                    + fraction * (final_xy[0] - retreat_start[0]),
                    retreat_start[1]
                    + fraction * (final_xy[1] - retreat_start[1]),
                    retreat_start[2],
                ],
                dtype=np.float64,
            )
            n2m._teleport_base_pose(env, candidate, weld=lock)
            contacts = n2m.robot_environment_contacts(
                env,
                ignored_environment_geoms=("floor",),
            )
            if contacts:
                n2m._teleport_base_pose(env, previous_pose, weld=lock)
                raise RuntimeError(
                    "retry radial retreat pose collides at "
                    f"step {index + 1}/{retreat_steps}: {contacts}"
                )

            env.set_arm_target(ARM_HOME)
            env.set_gripper_target(GRIPPER_OPEN)
            env.step(np.zeros(16, dtype=np.float32))
            contacts = n2m.robot_environment_contacts(
                env,
                ignored_environment_geoms=("floor",),
            )
            if contacts:
                n2m._teleport_base_pose(env, previous_pose, weld=lock)
                raise RuntimeError(
                    "retry radial retreat settled into a collision at "
                    f"step {index + 1}/{retreat_steps}: {contacts}"
                )
            previous_pose = n2m._base_pose_from_state(env.get_robot_state())
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == retreat_steps:
                print(
                    "[SYSNAV][N2M_RETRY_RETREAT] "
                    f"step={index + 1}/{retreat_steps} "
                    f"pose={np.round(previous_pose, 4).tolist()}",
                    flush=True,
                )

        final_pose = n2m._base_pose_from_state(env.get_robot_state())
        final_distance = float(np.linalg.norm(final_pose[:2] - target[:2]))
        if final_distance + 1e-3 < safe_distance:
            raise RuntimeError(
                "retry radial retreat did not reach its safe distance: "
                f"{final_distance:.3f}m < {safe_distance:.3f}m"
            )
        metrics = {
            "mode": "collision_checked_radial_pose_reset",
            "initial_base_pose": initial_pose.astype(float).tolist(),
            "final_base_pose": final_pose.astype(float).tolist(),
            "target_position_world": target.astype(float).tolist(),
            "initial_target_distance_m": initial_distance,
            "safe_distance_m": safe_distance,
            "final_target_distance_m": final_distance,
            "retreat_distance_m": retreat_distance,
            "retreat_step_distance_m": step_distance,
            "retreat_step_count": retreat_steps,
            "yaw_preserved": True,
            "arm_stages": arm_stages,
        }
        print(
            "[SYSNAV][N2M_RETRY_RECOVERY_DONE] "
            f"base={np.round(final_pose, 4).tolist()} "
            f"target_distance={final_distance:.3f}m "
            f"retreat_steps={retreat_steps}",
            flush=True,
        )
        return metrics
    finally:
        if lock.active:
            lock.disable()


def _plan_n2m_observation_view(
    components: SysNavComponents,
    target_position_world: np.ndarray,
    previous_observation_poses: list[np.ndarray],
    current_robot_pose: np.ndarray,
    *,
    args: argparse.Namespace,
    root_output_dir: Path,
    round_index: int,
) -> dict | None:
    """Plan and persist one obstacle-edge view, or request legacy fallback."""

    if bool(getattr(args, "sysnav_disable_obstacle_edge_view", False)):
        return None
    output_dir = root_output_dir / "viewpoint_planning"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"round_{round_index:04d}.json"
    image_path = output_dir / f"round_{round_index:04d}_debug.png"
    try:
        plan = plan_obstacle_edge_views(
            components.nav_map,
            components.room_map,
            target_position_world,
            current_robot_pose,
            previous_observation_poses=previous_observation_poses,
            preferred_distance_m=float(args.nav_standoff),
            min_distance_m=float(args.nav_min_distance),
            max_distance_m=float(args.nav_max_distance),
            preferred_angle_separation_deg=float(
                args.sysnav_retry_min_view_angle_deg
            ),
        )
        save_obstacle_edge_view_debug(
            image_path,
            components.nav_map,
            plan,
            previous_observation_poses=previous_observation_poses,
        )
        document = plan.to_dict()
        document["round"] = int(round_index)
        document["debug_image"] = str(image_path.resolve())
        write_json(json_path, document)
        selected = plan.selected
        result = {
            "source": "obstacle_edge_view",
            "pose": list(selected.pose),
            "room_id": plan.room_id,
            "target_position_world": list(plan.target_position_world),
            "target_grid_cell": list(plan.target_grid_cell),
            "edge_point_world": list(selected.edge_point_world),
            "edge_distance_m": float(selected.edge_distance_m),
            "distance_to_target_m": float(selected.target_distance_m),
            "obstacle_clearance_m": float(selected.obstacle_clearance_m),
            "view_side_angle_deg": math.degrees(selected.view_side_angle_rad),
            "minimum_prior_view_angle_deg": math.degrees(
                selected.minimum_prior_view_angle_rad
            ),
            "preferred_minimum_angle_deg": float(
                args.sysnav_retry_min_view_angle_deg
            ),
            "used_angle_fallback": bool(plan.used_angle_fallback),
            "travel_distance_m": float(selected.travel_distance_m),
            "candidate_count": len(plan.candidates),
            "component_bounds_world": [
                list(plan.component_bounds_world[0]),
                list(plan.component_bounds_world[1]),
            ],
            "planning_json": str(json_path.resolve()),
            "debug_image": str(image_path.resolve()),
            "map_mutation": "none_derived_masks_only",
        }
        print(
            f"[SYSNAV][N2M_EDGE_VIEW] round={round_index} "
            f"pose={np.round(selected.pose, 4).tolist()} "
            f"edge={np.round(selected.edge_point_world, 4).tolist()} "
            f"edge_distance={selected.edge_distance_m:.3f}m "
            f"clearance={selected.obstacle_clearance_m:.3f}m "
            f"candidates={len(plan.candidates)}",
            flush=True,
        )
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(
            json_path,
            {
                "schema_version": 1,
                "planner": "obstacle_edge_view",
                "round": int(round_index),
                "outcome": "fallback",
                "error": error,
                "map_mutation": "none_derived_masks_only",
            },
        )
        print(
            f"[SYSNAV][N2M_EDGE_VIEW_FALLBACK] round={round_index} "
            f"error={error}",
            flush=True,
        )
        return None


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_n2m_round(
    round_dir: Path,
    *,
    round_index: int,
    root_output_dir: Path,
    observation_view: dict | None,
) -> None:
    for filename in ("run_config.json", "summary.json"):
        path = round_dir / filename
        if not path.is_file():
            continue
        document = _load_json(path)
        document["target_provider"] = "SysNavVisionTargetProvider"
        document["target_observation_mask"] = "YOLOE_box_plus_SAM2"
        document["sysnav_search_result"] = str(
            (root_output_dir / "sysnav_search" / "result.json").resolve()
        )
        document["sysnav_n2m_round"] = int(round_index)
        document["sysnav_observation_view"] = observation_view
        document["sysnav_retry_observation_view"] = (
            observation_view if round_index > 1 else None
        )
        write_json(path, document)


def _handoff_to_n2m(
    args: argparse.Namespace,
    env,
    controller,
    components: SysNavComponents,
    target_position_world: np.ndarray,
) -> int:
    """Run one or more legacy N2M rounds against the same live environment."""

    import custom_envs.mujoco.env as env_module

    provider = _VisionTargetProvider(target_position_world)
    root_output_dir = Path(args.output_dir).resolve()
    original_parser = n2m.build_parser
    original_provider = n2m.MuJoCoTruthTargetProvider
    original_capture = n2m.capture_live_observation
    original_controller = n2m.InteractiveController
    original_standoff_goal = n2m.standoff_goal
    original_env_class = env_module.MuJoCoEnv
    original_env_close = env.close
    original_output_dir = args.output_dir
    original_robot_pose = [args.robot_x, args.robot_y, args.robot_yaw]
    original_snapshot_writer = getattr(args, "nav_snapshot_writer", None)
    env_requests = 0

    def reuse_env(*unused_args, **unused_kwargs):
        nonlocal env_requests
        env_requests += 1
        if env_requests > 1:
            raise RuntimeError(
                "the live MuJoCo environment was requested twice in one N2M round"
            )
        return env

    args.sysnav_original_robot_pose = list(original_robot_pose)
    rounds: list[dict] = []
    previous_observation_poses: list[np.ndarray] = []
    active_observation_view = _plan_n2m_observation_view(
        components,
        provider.position_world,
        previous_observation_poses,
        n2m._base_pose_from_state(env.get_robot_state()),
        args=args,
        root_output_dir=root_output_dir,
        round_index=1,
    )
    return_code = 2
    orchestration_error: str | None = None

    # The unchanged legacy main owns and closes an environment.  Suppress that
    # one close while it runs so a failed round can reuse the same simulation;
    # this wrapper restores and performs the real close exactly once below.
    env.close = lambda: None
    try:
        for round_index in range(1, int(args.sysnav_max_n2m_rounds) + 1):
            env_requests = 0
            round_dir = (
                root_output_dir
                / "n2m_rounds"
                / f"round_{round_index:04d}"
            )
            round_dir.mkdir(parents=True, exist_ok=True)
            pose = n2m._base_pose_from_state(env.get_robot_state())
            args.robot_x, args.robot_y, args.robot_yaw = map(float, pose)
            args.output_dir = round_dir

            # A previous successful live capture released the visual models
            # to leave GPU memory for N2M/AnyGrasp.  Restore the task prompts
            # (and lazy detector state) before a possible retry capture.
            components.perception.set_prompts(
                components.search_policy.instruction.detector_prompts
            )

            n2m.build_parser = lambda: _PreparsedParser(args)
            n2m.MuJoCoTruthTargetProvider = lambda: provider
            n2m.capture_live_observation = _build_visual_live_capture(
                components,
                provider,
                components.search_policy.instruction.target_object,
            )
            n2m.InteractiveController = lambda: controller
            env_module.MuJoCoEnv = reuse_env
            if active_observation_view is None:
                n2m.standoff_goal = original_standoff_goal
            else:
                retry_xy = np.asarray(
                    active_observation_view["pose"][:2], dtype=np.float64
                )

                def retry_standoff_goal(
                    unused_robot_xy,
                    unused_target_xy,
                    unused_distance_m,
                    *,
                    _retry_xy=retry_xy,
                ):
                    return _retry_xy.copy()

                n2m.standoff_goal = retry_standoff_goal

            print(
                f"[SYSNAV][N2M_ROUND_START] round={round_index}/"
                f"{args.sysnav_max_n2m_rounds} observation_view="
                f"{None if active_observation_view is None else active_observation_view['pose']}",
                flush=True,
            )
            return_code = int(n2m.main())
            _annotate_n2m_round(
                round_dir,
                round_index=round_index,
                root_output_dir=root_output_dir,
                observation_view=active_observation_view,
            )
            summary = _load_json(round_dir / "summary.json")
            observation = _load_json(round_dir / "observation.json")
            observation_pose = observation.get("observation_pose_world")
            if observation_pose is not None:
                previous_observation_poses.append(
                    np.asarray(observation_pose, dtype=np.float64).reshape(3)
                )
            rounds.append(
                {
                    "round": round_index,
                    "output_dir": str(round_dir),
                    "return_code": return_code,
                    "outcome": summary.get("outcome", "missing_summary"),
                    "success": bool(summary.get("success", False)),
                    "lift_succeeded": bool(
                        summary.get("lift_succeeded", False)
                    ),
                    "attempt_count": int(summary.get("attempt_count", 0)),
                    "attempted_m_count": int(
                        summary.get("attempted_m_count", 0)
                    ),
                    "collision_rejected_m_count": int(
                        summary.get("collision_rejected_m_count", 0)
                    ),
                    "observation_pose_world": observation_pose,
                    "planned_observation_view": active_observation_view,
                    "retry_observation_view": (
                        active_observation_view if round_index > 1 else None
                    ),
                }
            )
            print(
                f"[SYSNAV][N2M_ROUND_DONE] round={round_index} "
                f"outcome={summary.get('outcome', 'missing_summary')} "
                f"attempts={summary.get('attempt_count', 0)}",
                flush=True,
            )
            if return_code == 0:
                break
            if summary.get("outcome") != "grasp_attempts_exhausted":
                break
            if round_index >= int(args.sysnav_max_n2m_rounds):
                break

            active_observation_view = _plan_n2m_observation_view(
                components,
                provider.position_world,
                previous_observation_poses,
                n2m._base_pose_from_state(env.get_robot_state()),
                args=args,
                root_output_dir=root_output_dir,
                round_index=round_index + 1,
            )
            if active_observation_view is None:
                try:
                    retry_pose, active_observation_view = (
                        _select_retry_observation_pose(
                            components,
                            provider.position_world,
                            previous_observation_poses,
                            n2m._base_pose_from_state(env.get_robot_state()),
                            preferred_distance_m=float(args.nav_standoff),
                            min_distance_m=float(args.nav_min_distance),
                            max_distance_m=float(args.nav_max_distance),
                            preferred_angle_separation_deg=float(
                                args.sysnav_retry_min_view_angle_deg
                            ),
                        )
                    )
                    active_observation_view["source"] = "angular_grid_fallback"
                    active_observation_view["pose"] = list(retry_pose)
                except Exception as exc:
                    orchestration_error = (
                        f"retry_view_selection_failed:{type(exc).__name__}:{exc}"
                    )
                    print(
                        f"[SYSNAV][N2M_RETRY_ABORTED] {orchestration_error}",
                        flush=True,
                    )
                    break
            print(
                f"[SYSNAV][N2M_RETRY_VIEW] next_round={round_index + 1} "
                f"source={active_observation_view.get('source')} "
                f"pose={np.round(active_observation_view['pose'], 4).tolist()} "
                f"view_change="
                f"{active_observation_view['minimum_prior_view_angle_deg']:.1f}deg",
                flush=True,
            )
            try:
                recovery = _prepare_retry_navigation_state(
                    env,
                    provider.position_world,
                    safe_distance_m=max(
                        float(args.nav_standoff),
                        float(args.nav_min_distance),
                    ),
                    interactive=controller,
                )
            except Exception as exc:
                orchestration_error = (
                    "retry_navigation_recovery_failed:"
                    f"{type(exc).__name__}:{exc}"
                )
                print(
                    f"[SYSNAV][N2M_RETRY_ABORTED] {orchestration_error}",
                    flush=True,
                )
                break
            active_observation_view["pre_navigation_recovery"] = recovery
    finally:
        n2m.build_parser = original_parser
        n2m.MuJoCoTruthTargetProvider = original_provider
        n2m.capture_live_observation = original_capture
        n2m.InteractiveController = original_controller
        n2m.standoff_goal = original_standoff_goal
        env_module.MuJoCoEnv = original_env_class
        args.output_dir = original_output_dir
        args.robot_x, args.robot_y, args.robot_yaw = original_robot_pose
        args.nav_snapshot_writer = original_snapshot_writer
        env.close = original_env_close
        original_env_close()

    final_round = rounds[-1] if rounds else {}
    final_outcome = str(final_round.get("outcome", "n2m_not_run"))
    if (
        final_outcome == "grasp_attempts_exhausted"
        and len(rounds) >= int(args.sysnav_max_n2m_rounds)
    ):
        final_outcome = "grasp_attempts_exhausted_after_observation_retries"
    if orchestration_error is not None:
        final_outcome = orchestration_error.split(":", 1)[0]
    aggregate = {
        "schema_version": 1,
        "outcome": final_outcome,
        "success": bool(final_round.get("success", False)),
        "lift_succeeded": bool(final_round.get("lift_succeeded", False)),
        "n2m_round_count": len(rounds),
        "n2m_round_limit": int(args.sysnav_max_n2m_rounds),
        "attempt_count": sum(int(item["attempt_count"]) for item in rounds),
        "attempted_m_count": sum(
            int(item["attempted_m_count"]) for item in rounds
        ),
        "collision_rejected_m_count": sum(
            int(item["collision_rejected_m_count"]) for item in rounds
        ),
        "target_provider": "SysNavVisionTargetProvider",
        "target_observation_mask": "YOLOE_box_plus_SAM2",
        "final_target_position_world": provider.position_world.astype(float).tolist(),
        "sysnav_search_result": str(
            (root_output_dir / "sysnav_search" / "result.json").resolve()
        ),
        "orchestration_error": orchestration_error,
        "rounds": rounds,
    }
    write_json(root_output_dir / "n2m_rounds.json", aggregate)
    write_json(root_output_dir / "summary.json", aggregate)
    return return_code


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    output_dir = _prepare_output_dir(args)
    os.environ.setdefault("MUJOCO_GL", "glx" if args.render else "egl")
    config = build_sysnav_components(
        args.sysnav_config,
        enable_vlm=not args.sysnav_skip_vlm,
        refine_with_sam2=True,
    )
    if args.sysnav_skip_vlm:
        instruction = SearchInstruction(
            raw_text=args.instruction,
            target_object=args.object,
        )
        print(
            "[SYSNAV][VLM_SKIPPED] class-only test instruction; room semantics "
            "and VLM early stopping disabled",
            flush=True,
        )
    else:
        instruction = config.vlm.parse_instruction(args.instruction)
    if instruction.target_object != str(args.object).strip().lower():
        raise SystemExit(
            f"instruction target {instruction.target_object!r} does not match "
            f"the manipulable MuJoCo body --object={args.object!r}"
        )

    args.nav2_settings = n2m.load_nav2_runtime_settings(args.nav2_params)
    args.nav_snapshot_writer = n2m.NavigationSnapshotWriter(
        args.nav_snapshot_map,
        args.nav_snapshot_dir,
        args.nav_snapshot_every,
    )
    transit_config = config.config.transit_detection
    if args.sysnav_transit_detection_hz is not None:
        transit_config = replace(
            transit_config,
            frequency_hz=float(args.sysnav_transit_detection_hz),
        )
    if args.sysnav_disable_transit_detection or args.sysnav_direct_teleport_test:
        transit_config = replace(transit_config, enabled=False)

    write_json(
        output_dir / "sysnav_config.json",
        {
            "schema_version": 2,
            "instruction": instruction.to_dict(),
            "scene": str(args.scene.resolve()),
            "initial_robot_pose": [args.robot_x, args.robot_y, args.robot_yaw],
            "destination": [float(value) for value in args.destination],
            "room_map": str(config.config.search.room_map_manifest),
            "nav2_map": str(config.config.search.nav2_map_yaml),
            "skip_vlm": bool(args.sysnav_skip_vlm),
            "search_only": bool(args.sysnav_search_only),
            "direct_teleport_test": bool(args.sysnav_direct_teleport_test),
            "max_direct_goal_distance_m": float(
                args.sysnav_max_direct_goal_distance
            ),
            "max_n2m_rounds": int(args.sysnav_max_n2m_rounds),
            "retry_min_view_angle_deg": float(
                args.sysnav_retry_min_view_angle_deg
            ),
            "obstacle_edge_view_planning": {
                "enabled": not bool(args.sysnav_disable_obstacle_edge_view),
                "map_mutation": "none_derived_masks_only",
                "debug_output": str(
                    (output_dir / "viewpoint_planning").resolve()
                ),
            },
            "room_semantics": {
                "max_images": int(
                    config.config.search.room_semantic_max_images
                ),
                "refresh_min_new_images": int(
                    config.config.search.room_semantic_refresh_min_new_images
                ),
                "stable_min_ordinary_views": int(
                    config.config.search.room_semantic_stable_min_ordinary_views
                ),
                "stable_confidence": float(
                    config.config.search.room_semantic_stable_confidence
                ),
                "room_condition_source": "stable_multi_view_room_node",
                "local_target_prompt_includes_room_condition": False,
            },
            "transit_detection": {
                "enabled": bool(transit_config.enabled),
                "frequency_hz": float(transit_config.frequency_hz),
                "min_translation_m": float(transit_config.min_translation_m),
                "min_yaw_deg": float(transit_config.min_yaw_deg),
                "confirmation_distance_m": float(
                    transit_config.confirmation_distance_m
                ),
                "candidate_dedup_radius_m": float(
                    transit_config.candidate_dedup_radius_m
                ),
                "reobserve_min_translation_m": float(
                    transit_config.reobserve_min_translation_m
                ),
                "reobserve_min_yaw_deg": float(
                    transit_config.reobserve_min_yaw_deg
                ),
                "save_images": bool(transit_config.save_images),
                "output": str(
                    (output_dir / "sysnav_search" / "transit").resolve()
                ),
            },
        },
    )

    from custom_envs.mujoco.env import MuJoCoEnv
    from custom_envs.mujoco.policy_loader import load_locomotion_policy
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge

    controller = n2m.InteractiveController()
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
    contacts = n2m.robot_environment_contacts(env)
    if contacts:
        env.close()
        raise SystemExit(f"initial robot pose collides: {contacts}")

    bridge = None
    handed_off = False
    try:
        def capture_frame():
            return capture_rgbd_frame(env, config.config.search.camera_name)

        transit_detector = (
            ContinuousTransitDetector(
                config,
                capture_frame,
                output_dir / "sysnav_search" / "transit",
                instruction.target_object,
                transit_config,
            )
            if transit_config.enabled
            else None
        )
        if transit_detector is not None:
            print(
                f"[SYSNAV][TRANSIT_DETECTION] enabled=true "
                f"frequency={transit_config.frequency_hz:.2f}Hz "
                f"save_images={transit_config.save_images}",
                flush=True,
            )
        print(
            "[SYSNAV][ROOM_SEMANTICS] mode=revisable_multi_view "
            f"max_images={config.config.search.room_semantic_max_images} "
            "stable_gate="
            f"{config.config.search.room_semantic_stable_min_ordinary_views}ordinary/"
            f"{config.config.search.room_semantic_stable_confidence:.2f}confidence "
            "target_room_source=stable_room_node",
            flush=True,
        )
        policy = None
        device = None
        if args.sysnav_direct_teleport_test:
            navigator = _DirectTeleportTestNavigator(env)
        else:
            bridge_feedback_timeout = max(
                0.35, 2.5 * args.nav2_settings["velocity_timeout_s"]
            )
            bridge = IsaacROS2Bridge(
                cmd_vel_timeout=args.nav2_settings["velocity_timeout_s"],
                feedback_timeout=bridge_feedback_timeout,
            )
            bridge.start(timeout=30.0)
            if not args.teleport_navigation:
                device_name = args.device if torch.cuda.is_available() else "cpu"
                device = torch.device(device_name)
                policy = load_locomotion_policy(args.policy_path, device=device_name)
            navigator = _LegacySearchNavigator(
                env,
                bridge,
                args,
                controller,
                policy=policy,
                device=device,
                transit_detector=transit_detector,
            )
        runner = SysNavSearchRunner(
            config,
            navigator,
            capture_frame,
            output_dir / "sysnav_search",
            enable_room_semantics=(
                not args.sysnav_skip_vlm
                and not args.sysnav_skip_room_semantics
            ),
            enable_early_stop=(
                not args.sysnav_skip_vlm
                and not args.sysnav_disable_early_stop
            ),
            max_viewpoints=args.sysnav_max_viewpoints,
        )
        result = runner.run(instruction)
        print(
            f"[SYSNAV][SEARCH_DONE] found={result.found} "
            f"reason={result.reason} viewpoints={result.searched_viewpoints} "
            f"rooms={list(result.searched_rooms)}",
            flush=True,
        )
        if bridge is not None:
            bridge.cancel_goal()
            bridge.stop()
            bridge = None
        if not result.found:
            return 3
        if args.sysnav_search_only:
            return 0
        handed_off = True
        return _handoff_to_n2m(
            args,
            env,
            controller,
            config,
            result.target_position_world,
        )
    finally:
        if bridge is not None:
            try:
                bridge.cancel_goal()
            except Exception:
                pass
            bridge.stop()
        config.close()
        if not handed_off:
            env.close()


if __name__ == "__main__":
    # Match the validated legacy entry point: MuJoCo/GLFW, CUDA extensions and
    # the system-Python ROS child load independent native runtimes.  All owned
    # resources are closed by main(), so skip CPython's second native-library
    # finalization pass, which can otherwise crash after a successful run.
    _return_code = 1
    try:
        _return_code = int(main())
    except SystemExit as exc:
        _return_code = exc.code if isinstance(exc.code, int) else 1
    except KeyboardInterrupt:
        _return_code = 130
    except BaseException:
        traceback.print_exc()
        _return_code = 1
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(_return_code)
