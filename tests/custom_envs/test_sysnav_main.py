from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from custom_envs.scripts.navigation import navigate_mujoco_n2m as n2m
from custom_envs.scripts.N2M.geometry import joint1_towards_cube
from custom_envs.scripts.N2M.manipulation_episode import ARM_DEPLOY_TEMPLATE
from custom_envs.scripts.navigation.navigate_mujoco_sysnav import (
    _LegacySearchNavigator,
    _PreparsedParser,
    _SysNavVisionGraspRunner,
    _VisionTargetProvider,
    _build_parser,
    _handoff_to_n2m,
    _path_pose_at_fraction,
    _prepare_retry_navigation_state,
    _select_retry_observation_pose,
    _validate_args,
)
from custom_envs.scripts.map.nav2_map_io import load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap
from custom_envs.scripts.sysnav.config import load_sysnav_config
from custom_envs.scripts.sysnav.room_graph import AnonymousRoomGraph


def test_parser_adds_sysnav_options_without_changing_legacy_defaults():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--instruction",
            "find the cube",
            "--policy_path",
            "policy.pt",
            "--grasp-checkpoint",
            "grasp.pt",
            "--destination",
            "12",
            "12",
            "--sysnav-skip-vlm",
            "--sysnav-search-only",
        ]
    )
    assert args.instruction == "find the cube"
    assert args.sysnav_skip_vlm
    assert args.sysnav_search_only
    assert args.object == "cube"
    assert args.teleport_navigation
    assert args.destination == [12.0, 12.0]
    assert args.sysnav_max_n2m_rounds == 2
    assert args.sysnav_retry_min_view_angle_deg == 60.0
    assert args.sysnav_max_direct_goal_distance == 6.0
    assert not args.sysnav_disable_obstacle_edge_view
    assert not args.sysnav_disable_transit_detection
    assert args.sysnav_transit_detection_hz is None


def test_sysnav_arm_init_uses_vision_and_caches_complete_target_per_m(
    monkeypatch,
):
    calls = []

    def fake_run_arm_init(unused_self, candidate):
        target = joint1_towards_cube(
            candidate["base_pose"], candidate["cube_pose"]
        )
        deploy = ARM_DEPLOY_TEMPLATE.copy()
        deploy[0] = target
        calls.append((str(candidate["candidate_id"]), target, deploy))
        return {
            "joint1_target": target,
            "deploy_arm_target": deploy.tolist(),
        }

    monkeypatch.setattr(
        n2m.AttachedPersistentGraspRunner,
        "run_arm_init",
        fake_run_arm_init,
    )
    provider = _VisionTargetProvider(np.array([5.0, 4.0, 0.45]))
    runner = object.__new__(_SysNavVisionGraspRunner)
    runner._sysnav_target_provider = provider
    runner._sysnav_arm_init_targets = {}
    runner.env = object()
    runner.cfg = SimpleNamespace(object_name="cube")

    first_m = {
        "candidate_id": "m_0001",
        "base_pose": [4.0, 3.5, 0.2],
        # Deliberately disagree with vision: ARM_INIT must ignore this truth.
        "cube_pose": [20.0, 20.0, 0.45, 1.0, 0.0, 0.0, 0.0],
    }
    first = runner.run_arm_init(first_m)
    provider.update(np.array([4.1, 5.2, 0.46]))
    second = runner.run_arm_init(first_m)

    second_m = {
        "candidate_id": "m_0002",
        "base_pose": [4.4, 3.2, -0.1],
        "cube_pose": [-10.0, -10.0, 0.45, 1.0, 0.0, 0.0, 0.0],
    }
    third = runner.run_arm_init(second_m)

    expected_first = joint1_towards_cube(first_m["base_pose"], [5.0, 4.0])
    expected_third = joint1_towards_cube(
        second_m["base_pose"], [4.1, 5.2]
    )
    assert calls[0][1] == pytest.approx(expected_first)
    assert calls[1][1] == pytest.approx(expected_first)
    assert calls[2][1] == pytest.approx(expected_third)
    np.testing.assert_allclose(
        first["deploy_arm_target"], second["deploy_arm_target"]
    )
    assert first["arm_init_target_position_world"] == [5.0, 4.0, 0.45]
    assert second["arm_init_target_position_world"] == [5.0, 4.0, 0.45]
    assert third["arm_init_target_position_world"] == [4.1, 5.2, 0.46]
    assert first["arm_init_target_reused"] is False
    assert second["arm_init_target_reused"] is True
    assert third["arm_init_target_reused"] is False


def test_navfn_uses_dijkstra_for_stable_long_path_extraction():
    params_path = (
        Path(__file__).resolve().parents[2]
        / "custom_envs"
        / "config"
        / "nav2_params.yaml"
    )
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    planner = params["planner_server"]["ros__parameters"]["GridBased"]
    assert planner["plugin"] == "nav2_navfn_planner/NavfnPlanner"
    assert planner["use_astar"] is False


def test_long_sysnav_navigation_is_split_at_nav2_path_midpoint(monkeypatch):
    class FakeEnv:
        def get_robot_state(self):
            return {
                "pos": np.array([3.975, 5.175, 0.58]),
                "yaw": 1.54,
            }

    class FakeBridge:
        def __init__(self):
            self.plan_calls = []

        def plan_path(self, start_xy, goal_xy, *, timeout):
            self.plan_calls.append((list(start_xy), list(goal_xy), timeout))
            # This route goes around the top/right of the coffee table. Its
            # arc midpoint is intentionally different from the line midpoint,
            # which lies inside the table at (7.675, 3.5).
            return [
                [3.975, 5.175],
                [10.5, 5.175],
                [10.5, 1.825],
                [11.375, 1.825],
            ]

    args = Namespace(
        teleport_navigation=True,
        sysnav_max_direct_goal_distance=6.0,
        sysnav_nav_max_steps=8000,
        teleport_every_steps=5,
        teleport_step_distance=0.08,
        teleport_plan_timeout=15.0,
        nav_snapshot_writer=None,
    )
    calls = []

    def fake_follow(**kwargs):
        calls.append(kwargs)
        goal = np.asarray(kwargs["goal_xy"], dtype=float)
        return {
            "mode": "nav2_path_teleport",
            "steps": 10,
            "teleport_count": 2,
            "path_waypoints": 2,
            "elapsed_s": 0.5,
            "final_base_pose": [*goal.tolist(), kwargs["final_yaw"]],
        }

    monkeypatch.setattr(n2m, "_follow_nav2_path_by_teleport", fake_follow)
    bridge = FakeBridge()
    navigator = _LegacySearchNavigator(
        FakeEnv(), bridge, args, object()
    )
    result = navigator.move_to((11.375, 1.825, 0.25), name="SEARCH_room_0_0304")

    assert [call["name"] for call in calls] == [
        "SEARCH_room_0_0304_MIDPOINT",
        "SEARCH_room_0_0304",
    ]
    np.testing.assert_allclose(calls[0]["goal_xy"], [9.35, 5.175])
    np.testing.assert_allclose(calls[1]["goal_xy"], [11.375, 1.825])
    assert bridge.plan_calls == [
        ([3.975, 5.175], [11.375, 1.825], 15.0)
    ]
    assert result["mode"] == "nav2_path_teleport_via_midpoint"
    assert result["split_strategy"] == "nav2_path_arclength"
    assert result["split_path_points"] == 4
    assert result["split_path_length_m"] == pytest.approx(10.75)
    assert result["steps"] == 20
    assert result["teleport_count"] == 4
    assert result["path_waypoints"] == 4
    assert result["elapsed_s"] == 1.0
    assert len(result["legs"]) == 2


def test_path_fraction_pose_ignores_duplicate_nav2_poses():
    pose, total_length = _path_pose_at_fraction(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
        0.5,
    )
    # Exactly at a corner, face along the following segment so the second
    # navigation leg starts in the continuation direction.
    np.testing.assert_allclose(pose, [2.0, 0.0, np.pi / 2.0])
    assert total_length == pytest.approx(4.0)


def test_long_navigation_does_not_start_second_leg_after_transit_interrupt(
    monkeypatch,
):
    class FakeEnv:
        def get_robot_state(self):
            return {"pos": np.array([0.0, 0.0, 0.58]), "yaw": 0.0}

    class FakeBridge:
        def plan_path(self, start_xy, goal_xy, *, timeout):
            return [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]

    args = Namespace(
        teleport_navigation=True,
        sysnav_max_direct_goal_distance=6.0,
        teleport_plan_timeout=15.0,
    )
    navigator = _LegacySearchNavigator(FakeEnv(), FakeBridge(), args, object())
    calls = []

    def interrupted(target, *, name):
        calls.append((np.asarray(target).copy(), name))
        return {
            "mode": "nav2_path_teleport",
            "interrupted_for_transit_candidate": True,
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(navigator, "_move_once", interrupted)
    result = navigator.move_to((8.0, 0.0, 0.0), name="SEARCH")

    assert len(calls) == 1
    assert calls[0][1] == "SEARCH_MIDPOINT"
    assert result["interrupted_for_transit_candidate"]
    assert result["split_strategy"] == "nav2_path_arclength"


def test_long_sysnav_navigation_stops_when_full_path_is_unavailable():
    class FakeEnv:
        def get_robot_state(self):
            return {"pos": np.array([0.0, 0.0, 0.58]), "yaw": 0.0}

    class FakeBridge:
        def plan_path(self, start_xy, goal_xy, *, timeout):
            return []

    args = Namespace(
        teleport_navigation=True,
        sysnav_max_direct_goal_distance=6.0,
        sysnav_nav_max_steps=8000,
        teleport_every_steps=5,
        teleport_step_distance=0.08,
        teleport_plan_timeout=15.0,
        nav_snapshot_writer=None,
    )
    navigator = _LegacySearchNavigator(
        FakeEnv(), FakeBridge(), args, object()
    )
    with pytest.raises(RuntimeError, match="no complete path"):
        navigator.move_to((8.0, 0.0, 0.0), name="SEARCH_room_0_0001")


def test_near_sysnav_navigation_is_not_split(monkeypatch):
    class FakeEnv:
        def get_robot_state(self):
            return {
                "pos": np.array([1.0, 2.0, 0.58]),
                "yaw": 0.0,
            }

    args = Namespace(
        teleport_navigation=True,
        sysnav_max_direct_goal_distance=6.0,
        sysnav_nav_max_steps=8000,
        teleport_every_steps=5,
        teleport_step_distance=0.08,
        teleport_plan_timeout=15.0,
        nav_snapshot_writer=None,
    )
    calls = []

    def fake_follow(**kwargs):
        calls.append(kwargs)
        return {"mode": "nav2_path_teleport", "final_base_pose": [2.0, 3.0, 0.5]}

    monkeypatch.setattr(n2m, "_follow_nav2_path_by_teleport", fake_follow)
    navigator = _LegacySearchNavigator(
        FakeEnv(), object(), args, object()
    )
    result = navigator.move_to((2.0, 3.0, 0.5), name="SEARCH_room_0_0008")

    assert len(calls) == 1
    assert calls[0]["name"] == "SEARCH_room_0_0008"
    assert result["mode"] == "nav2_path_teleport"
    assert "legs" not in result


def test_direct_test_navigation_is_search_only():
    args = Namespace(
        sysnav_max_viewpoints=1,
        sysnav_nav_max_steps=1,
        sysnav_position_tolerance=0.1,
        sysnav_max_direct_goal_distance=6.0,
        sysnav_align_max_steps=1,
        sysnav_yaw_tolerance_deg=1.0,
        sysnav_max_n2m_rounds=2,
        sysnav_retry_min_view_angle_deg=60.0,
        sysnav_direct_teleport_test=True,
        sysnav_search_only=False,
    )
    with pytest.raises(SystemExit, match="requires --sysnav-search-only"):
        _validate_args(args)


def test_visual_target_provider_returns_cached_search_position():
    provider = _VisionTargetProvider(np.array([1.2, 3.4, 0.5]))
    estimate = provider.estimate(object(), "cube")
    np.testing.assert_allclose(estimate.position_world, [1.2, 3.4, 0.5])
    np.testing.assert_allclose(estimate.quaternion_world_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert estimate.source == "sysnav_yoloe_sam2_depth"
    provider.update(np.array([2.0, 4.0, 0.6]))
    np.testing.assert_allclose(
        provider.estimate(object(), "cube").position_world,
        [2.0, 4.0, 0.6],
    )


def test_preparsed_parser_returns_the_same_namespace():
    args = Namespace(output_dir=Path("run"))
    assert _PreparsedParser(args).parse_args() is args


def test_retry_observation_pose_changes_target_view_angle():
    config = load_sysnav_config()
    room_map = RoomMap.load(config.search.room_map_manifest)
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    components = Namespace(
        room_map=room_map,
        nav_map=nav_map,
        room_graph=AnonymousRoomGraph(room_map),
    )
    target = np.array([5.3, 3.65, 0.48])
    previous = [np.array([5.3, 5.05, -np.pi / 2.0])]
    pose, diagnostics = _select_retry_observation_pose(
        components,
        target,
        previous,
        np.array([5.1, 4.25, 0.2]),
        preferred_distance_m=1.4,
        min_distance_m=1.2,
        max_distance_m=1.7,
        preferred_angle_separation_deg=60.0,
    )
    distance = float(np.linalg.norm(np.asarray(pose[:2]) - target[:2]))
    assert 1.2 <= distance <= 1.7
    assert diagnostics["minimum_prior_view_angle_deg"] >= 60.0
    assert room_map.room_id_at_world(*pose[:2]) == "room_0"
    yaw_to_target = np.arctan2(target[1] - pose[1], target[0] - pose[0])
    assert np.isclose(pose[2], yaw_to_target)


def test_retry_recovery_folds_arm_then_retreats_without_changing_yaw(
    monkeypatch,
):
    class FakeEnv:
        def __init__(self):
            self.pose = np.array([5.324, 4.364, 0.155], dtype=np.float64)
            self.joints = np.array(
                [-1.54, 1.2, -1.1, 0.4, 0.6, -0.2, 0.3, -0.3],
                dtype=np.float64,
            )
            self._arm_all_idx = np.arange(6, dtype=np.int32)
            self._grip_all_idx = np.arange(6, 8, dtype=np.int32)
            self.arm_target = self.joints[:6].copy()
            self.grip_target = self.joints[6:].copy()
            self.force_hold_disabled = False

        def get_robot_state(self):
            return {
                "pos": np.array([self.pose[0], self.pose[1], 0.58]),
                "yaw": float(self.pose[2]),
                "joint_pos": self.joints.copy(),
            }

        def set_arm_target(self, target):
            self.arm_target = np.asarray(target, dtype=np.float64).copy()

        def set_gripper_target(self, target):
            self.grip_target = np.asarray(target, dtype=np.float64).copy()

        def disable_gripper_force_hold(self):
            self.force_hold_disabled = True

        def step(self, unused_action):
            self.joints[:6] = self.arm_target
            self.joints[6:] = self.grip_target

    class FakeWeld:
        def __init__(self, unused_env):
            self.active = False

        def enable(self):
            self.active = True

        def disable(self):
            self.active = False

    env = FakeEnv()
    teleported_poses = []

    def fake_teleport(live_env, pose, **unused_kwargs):
        live_env.pose = np.asarray(pose, dtype=np.float64).copy()
        teleported_poses.append(live_env.pose.copy())
        return {}

    monkeypatch.setattr(n2m, "BaseWeld", FakeWeld)
    monkeypatch.setattr(n2m, "_teleport_base_pose", fake_teleport)
    monkeypatch.setattr(n2m, "robot_environment_contacts", lambda *a, **k: [])

    target = np.array([5.2928, 3.675, 0.478])
    metrics = _prepare_retry_navigation_state(
        env,
        target,
        safe_distance_m=1.4,
        retreat_step_distance_m=0.10,
        arm_stage_max_steps=10,
    )

    assert env.force_hold_disabled
    np.testing.assert_allclose(env.joints[:6], n2m.ARM_HOME)
    np.testing.assert_allclose(env.joints[6:], n2m.GRIPPER_OPEN)
    assert metrics["retreat_step_count"] > 0
    assert metrics["final_target_distance_m"] >= 1.4 - 1e-3
    assert all(np.isclose(pose[2], 0.155) for pose in teleported_poses)
    radial_distances = [
        np.linalg.norm(pose[:2] - target[:2]) for pose in teleported_poses
    ]
    assert np.all(np.diff(radial_distances) > 0.0)


def test_n2m_exhaustion_runs_a_second_round_in_the_same_env(
    tmp_path, monkeypatch
):
    import custom_envs.mujoco.env as env_module
    from custom_envs.scripts.navigation import navigate_mujoco_n2m as n2m

    config = load_sysnav_config()
    room_map = RoomMap.load(config.search.room_map_manifest)
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    components = SimpleNamespace(
        room_map=room_map,
        nav_map=nav_map,
        room_graph=AnonymousRoomGraph(room_map),
        perception=SimpleNamespace(
            set_prompts=lambda unused: None,
        ),
        search_policy=SimpleNamespace(
            instruction=SimpleNamespace(
                target_object="cube",
                detector_prompts=("cube",),
            )
        ),
    )

    class FakeEnv:
        def __init__(self):
            self.close_count = 0

        def get_robot_state(self):
            return {
                "pos": np.array([5.1, 4.25, 0.58]),
                "yaw": 0.2,
            }

        def close(self):
            self.close_count += 1

    env = FakeEnv()
    args = Namespace(
        output_dir=tmp_path,
        robot_x=1.6,
        robot_y=1.6,
        robot_yaw=0.0,
        nav_snapshot_writer=object(),
        sysnav_max_n2m_rounds=2,
        sysnav_retry_min_view_angle_deg=60.0,
        nav_standoff=1.4,
        nav_min_distance=1.2,
        nav_max_distance=1.7,
    )
    calls = []
    planned_waypoints = []
    retry_waypoints = []
    recoveries = []
    original_grasp_runner = n2m.AttachedPersistentGraspRunner

    def fake_retry_recovery(unused_env, target, **kwargs):
        recoveries.append((np.asarray(target).copy(), kwargs))
        return {
            "mode": "test_recovery",
            "final_base_pose": [5.1, 4.25, 0.2],
        }

    def fake_n2m_main():
        assert n2m.AttachedPersistentGraspRunner is not original_grasp_runner
        parsed = n2m.build_parser().parse_args()
        live_env = env_module.MuJoCoEnv()
        round_index = len(calls) + 1
        calls.append(round_index)
        round_dir = Path(parsed.output_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        planned_waypoints.append(
            n2m.standoff_goal([5.1, 4.25], [5.3, 3.65], 1.4)
        )
        if round_index == 2:
            retry_waypoints.append(
                planned_waypoints[-1]
            )
        outcome = "grasp_attempts_exhausted" if round_index == 1 else "task_success"
        (round_dir / "summary.json").write_text(
            json.dumps(
                {
                    "outcome": outcome,
                    "success": round_index == 2,
                    "lift_succeeded": round_index == 2,
                    "attempt_count": 9 if round_index == 1 else 1,
                    "attempted_m_count": 3 if round_index == 1 else 1,
                    "collision_rejected_m_count": 2,
                }
            ),
            encoding="utf-8",
        )
        observation_pose = (
            [5.3, 5.05, -np.pi / 2.0]
            if round_index == 1
            else [4.2, 3.1, 0.4]
        )
        (round_dir / "observation.json").write_text(
            json.dumps({"observation_pose_world": observation_pose}),
            encoding="utf-8",
        )
        live_env.close()
        return 2 if round_index == 1 else 0

    monkeypatch.setattr(n2m, "main", fake_n2m_main)
    monkeypatch.setattr(
        "custom_envs.scripts.navigation.navigate_mujoco_sysnav."
        "_prepare_retry_navigation_state",
        fake_retry_recovery,
    )
    return_code = _handoff_to_n2m(
        args,
        env,
        object(),
        components,
        np.array([5.3, 3.65, 0.48]),
    )

    assert return_code == 0
    assert n2m.AttachedPersistentGraspRunner is original_grasp_runner
    assert calls == [1, 2]
    assert env.close_count == 1
    assert len(planned_waypoints) == 2
    assert len(retry_waypoints) == 1
    assert len(recoveries) == 1
    assert recoveries[0][1]["safe_distance_m"] == 1.4
    aggregate = json.loads((tmp_path / "summary.json").read_text())
    assert aggregate["success"]
    assert aggregate["n2m_round_count"] == 2
    assert aggregate["attempt_count"] == 10
    assert aggregate["rounds"][0]["outcome"] == "grasp_attempts_exhausted"
    assert aggregate["rounds"][1]["outcome"] == "task_success"
    assert (
        aggregate["rounds"][0]["planned_observation_view"]["source"]
        == "obstacle_edge_view"
    )
    np.testing.assert_allclose(
        planned_waypoints[0],
        aggregate["rounds"][0]["planned_observation_view"]["pose"][:2],
    )
    np.testing.assert_allclose(
        retry_waypoints[0],
        aggregate["rounds"][1]["retry_observation_view"]["pose"][:2],
    )
    assert (
        aggregate["rounds"][1]["retry_observation_view"]
        ["pre_navigation_recovery"]["mode"]
        == "test_recovery"
    )
