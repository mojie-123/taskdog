from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import numpy as np

from custom_envs.scripts.N2M.learning.runtime import _pose_rows
from custom_envs.scripts.N2M.target_provider import (
    NavigationTargetMetrics,
    navigation_target_metrics,
    observation_gate,
    standoff_goal,
)
from custom_envs.scripts.N2M.manipulation_episode import EpisodeFailure
from custom_envs.scripts.navigation.navigate_mujoco_n2m import (
    AttachedPersistentGraspRunner,
    _clip_nav_command,
    _update_progress_watchdog,
    build_parser,
    load_nav2_runtime_settings,
)
from custom_envs.utils.ros2_bridge import IsaacROS2Bridge


class NavigationTargetTests(unittest.TestCase):
    def test_standoff_goal_stays_on_robot_side_of_target(self):
        goal = standoff_goal([0.0, 0.0], [5.0, 0.0], 1.4)
        np.testing.assert_allclose(goal, [3.6, 0.0], atol=1e-12)

    def test_metrics_report_bearing_in_base_frame(self):
        metrics = navigation_target_metrics(
            [1.0, 2.0, math.pi / 2], [2.0, 2.0, 0.5]
        )
        self.assertAlmostEqual(metrics.horizontal_distance_m, 1.0)
        self.assertAlmostEqual(metrics.bearing_error_rad, -math.pi / 2)

    def test_observation_gate_checks_distance_and_angle(self):
        self.assertTrue(
            observation_gate(
                NavigationTargetMetrics(1.4, math.radians(10.0)),
                min_distance_m=1.2,
                max_distance_m=1.7,
                max_bearing_rad=math.radians(15.0),
            )
        )
        self.assertFalse(
            observation_gate(
                NavigationTargetMetrics(1.4, math.radians(16.0)),
                min_distance_m=1.2,
                max_distance_m=1.7,
                max_bearing_rad=math.radians(15.0),
            )
        )


class RuntimeGeometryTests(unittest.TestCase):
    def test_live_relative_pose_is_exported_in_world_frame(self):
        rows = _pose_rows(np.array([[1.0, 0.0, 0.2]]), [2.0, 3.0, math.pi / 2])
        np.testing.assert_allclose(rows[0]["pose_world"][:2], [2.0, 4.0])
        self.assertAlmostEqual(rows[0]["pose_world"][2], math.pi / 2 + 0.2)


class BridgeProtocolTests(unittest.TestCase):
    def test_cancel_goal_is_additive_protocol_message(self):
        bridge = IsaacROS2Bridge()
        sent = []
        bridge._send = sent.append
        bridge._vx = 0.4
        bridge._omega_z = -0.2
        bridge.cancel_goal()
        self.assertEqual(sent, [{"type": "cancel_goal"}])
        self.assertEqual((bridge._vx, bridge._omega_z), (0.0, 0.0))


class IntegratedPipelineArgumentTests(unittest.TestCase):
    def test_default_recovery_budget_is_three_m_times_three_attempts(self):
        args = build_parser().parse_args(
            [
                "--policy_path",
                "policy.pt",
                "--grasp-checkpoint",
                "grasp.tar",
                "--destination",
                "12.0",
                "12.0",
            ]
        )
        self.assertEqual(args.n2m_max_candidates, 3)
        self.assertEqual(args.max_grasp_attempts, 3)
        self.assertEqual(args.destination, [12.0, 12.0])

    def test_test_script_argument_aliases_are_supported(self):
        args = build_parser().parse_args(
            [
                "--policy_path",
                "policy.pt",
                "--grasp-checkpoint",
                "grasp.tar",
                "--destination",
                "8.0",
                "6.0",
                "--max-m-candidates",
                "2",
                "--attempts-per-m",
                "4",
            ]
        )
        self.assertEqual(args.n2m_max_candidates, 2)
        self.assertEqual(args.max_grasp_attempts, 4)


class Nav2RuntimeSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_nav2_runtime_settings(
            Path("custom_envs/config/nav2_params.yaml")
        )

    def test_project_yaml_drives_bridge_limits_and_timing(self):
        self.assertEqual(self.settings["controller_frequency_hz"], 20.0)
        self.assertEqual(self.settings["smoothing_frequency_hz"], 20.0)
        self.assertEqual(self.settings["velocity_timeout_s"], 0.2)
        self.assertEqual(self.settings["max_linear_velocity"], 0.8)
        self.assertEqual(self.settings["max_angular_velocity"], 1.0)
        self.assertEqual(self.settings["required_movement_radius_m"], 0.1)
        self.assertEqual(self.settings["movement_time_allowance_s"], 60.0)

    def test_nav_command_is_clipped_to_yaml_limits(self):
        self.assertEqual(
            _clip_nav_command(1.7, -2.2, self.settings),
            (0.8, -1.0),
        )

    def test_progress_watchdog_uses_yaml_allowance(self):
        state = {
            "anchor": np.array([1.0, 2.0]),
            "last_progress_time": (
                time.monotonic()
                - self.settings["movement_time_allowance_s"]
                - 1.0
            ),
        }
        with self.assertRaisesRegex(RuntimeError, "Nav2 progress timeout"):
            _update_progress_watchdog(
                state, np.array([1.01, 2.0]), self.settings
            )


class DeliveryGraspGuardTests(unittest.TestCase):
    @staticmethod
    def _state(aperture=0.05):
        return {
            "initial_aperture": aperture,
            "previous_aperture": aperture,
            "minimum_aperture": aperture,
            "recent_closures": [],
            "contact_miss_steps": 0,
            "max_object_tcp_slip": 0.0,
        }

    def test_large_tcp_slip_alone_is_diagnostic(self):
        runner = object.__new__(AttachedPersistentGraspRunner)
        runner.cfg = SimpleNamespace(moveit_lift_contact_miss_steps=8)
        runner._finger_contacts = lambda: {"link7": True, "link8": True}
        runner._grip_q = lambda: np.array([0.025, -0.025])
        runner._object_pose_in_tcp = lambda: SimpleNamespace(
            position=np.array([0.030, 0.0, 0.0])
        )
        state = self._state()
        slip = runner._delivery_grasp_guard(state, np.zeros(3))
        self.assertAlmostEqual(slip, 0.030)
        self.assertAlmostEqual(state["max_object_tcp_slip"], 0.030)

    def test_fast_aperture_collapse_reports_drop(self):
        runner = object.__new__(AttachedPersistentGraspRunner)
        runner.cfg = SimpleNamespace(moveit_lift_contact_miss_steps=8)
        runner._finger_contacts = lambda: {"link7": True, "link8": True}
        apertures = iter((0.040, 0.020))
        runner._grip_q = lambda: np.array(
            [(value := next(apertures)) / 2.0, -value / 2.0]
        )
        runner._object_pose_in_tcp = lambda: SimpleNamespace(
            position=np.zeros(3)
        )
        state = self._state()
        runner._delivery_grasp_guard(state, np.zeros(3))
        with self.assertRaises(EpisodeFailure) as caught:
            runner._delivery_grasp_guard(state, np.zeros(3))
        self.assertEqual(caught.exception.stage, "LIFT_RETRACT")
        self.assertEqual(caught.exception.reason, "gripper_rapid_close")


if __name__ == "__main__":
    unittest.main()
