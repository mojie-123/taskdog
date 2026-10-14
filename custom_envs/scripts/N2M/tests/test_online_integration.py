from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

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
    PAN_DV_PER_STEP,
    N2M_TERMINAL_EXIT_FRAMES,
    PAN_SETTLE_STEPS,
    PAN_SETTLE_WINDOW_FRAMES,
    PAN_SETTLE_YAW_RATE,
    AttachedPersistentGraspRunner,
    NavCommandConditioner,
    _approach_frame_position_errors,
    _body_frame_xy_error,
    _clip_nav_command,
    _door_cross_control_command,
    _door_point_approach_command,
    _kitchen_living_door_plan,
    _new_pan_xy_control_state,
    _n2m_candidate_terminal_region,
    _observation_yaw_capture_step,
    _pan_world_xy_control_step,
    _pose_window_motion_rates,
    _resample_navigation_path,
    _update_progress_watchdog,
    _wxyz_multiply,
    build_predock_pose_options,
    build_parser,
    load_nav2_runtime_settings,
    pan_xy_to_pose,
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

    def test_observation_yaw_capture_has_three_six_degree_hysteresis(self):
        maximum = math.radians(15.0)
        captured = _observation_yaw_capture_step(
            False, math.radians(2.9), maximum
        )
        self.assertTrue(captured)
        self.assertTrue(
            _observation_yaw_capture_step(
                captured, math.radians(3.1), maximum
            )
        )
        self.assertFalse(
            _observation_yaw_capture_step(
                captured, math.radians(6.1), maximum
            )
        )

    def test_nav_path_resampling_uses_fixed_spacing_and_exact_goal(self):
        samples = _resample_navigation_path(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            [0.0, 0.0],
            [1.0, 1.0],
            0.4,
        )
        np.testing.assert_allclose(samples[-1], [1.0, 1.0], atol=1e-12)
        self.assertEqual(len(samples), 5)
        steps = np.linalg.norm(
            np.diff(np.vstack(([0.0, 0.0], samples)), axis=0), axis=1
        )
        self.assertTrue(np.all(steps <= 0.4 + 1e-12))

    def test_wxyz_multiply_rotates_yaw_without_losing_unit_norm(self):
        yaw_90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        product = _wxyz_multiply(yaw_90, yaw_90)
        np.testing.assert_allclose(product, [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_kitchen_living_door_plan_only_crosses_horizontal_wall(self):
        southbound = _kitchen_living_door_plan([12.0, 12.0], [6.2, 4.7])
        self.assertIsNotNone(southbound)
        self.assertEqual(southbound["direction_name"], "kitchen_to_living")
        np.testing.assert_allclose(
            southbound["approach_pose"], [4.0, 6.85, -math.pi / 2]
        )
        np.testing.assert_allclose(
            southbound["exit_pose"], [4.0, 5.15, -math.pi / 2]
        )

        northbound = _kitchen_living_door_plan([2.0, 3.0], [4.0, 8.0])
        self.assertEqual(northbound["direction_name"], "living_to_kitchen")
        np.testing.assert_allclose(
            northbound["approach_pose"], [4.0, 5.15, math.pi / 2]
        )
        np.testing.assert_allclose(
            northbound["exit_pose"], [4.0, 6.85, math.pi / 2]
        )

        self.assertIsNone(
            _kitchen_living_door_plan([1.6, 1.6], [4.1, 4.8])
        )

    def test_door_cross_command_is_forward_only_and_centers_heading(self):
        plan = _kitchen_living_door_plan([12.0, 12.0], [6.2, 4.7])
        command, diagnostics = _door_cross_control_command(
            [4.08, 6.50, -math.pi / 2], plan
        )
        self.assertGreater(command[0], 0.0)
        self.assertEqual(command[1], 0.0)
        self.assertLess(command[2], 0.0)
        self.assertFalse(diagnostics["tracking_warning"])

        command, diagnostics = _door_cross_control_command(
            [4.0, 5.10, -math.pi / 2], plan
        )
        self.assertEqual(command, (0.0, 0.0, 0.0))
        self.assertTrue(diagnostics["done"])

        command, diagnostics = _door_cross_control_command(
            [4.30, 6.0, -math.pi / 2], plan
        )
        self.assertGreater(command[0], 0.0)
        self.assertEqual(command[1], 0.0)
        self.assertTrue(diagnostics["tracking_warning"])

    def test_door_point_approach_never_strafes_or_reverses(self):
        command, captured, diagnostics = _door_point_approach_command(
            [4.466, 6.846, math.radians(177.6)],
            [4.0, 6.85],
            captured=False,
        )
        self.assertFalse(captured)
        self.assertEqual(diagnostics["mode"], "forward")
        self.assertAlmostEqual(command[0], 0.45)
        self.assertEqual(command[1], 0.0)

        command, captured, diagnostics = _door_point_approach_command(
            [4.466, 6.846, 0.0], [4.0, 6.85], captured=False
        )
        self.assertFalse(captured)
        self.assertEqual(diagnostics["mode"], "turn")
        self.assertEqual(command[0], 0.0)
        self.assertEqual(command[1], 0.0)

        command, captured, diagnostics = _door_point_approach_command(
            [4.09, 6.85, 0.0], [4.0, 6.85], captured=False
        )
        self.assertTrue(captured)
        self.assertEqual(command, (0.0, 0.0, 0.0))


class RuntimeGeometryTests(unittest.TestCase):
    def test_live_relative_pose_is_exported_in_world_frame(self):
        rows = _pose_rows(np.array([[1.0, 0.0, 0.2]]), [2.0, 3.0, math.pi / 2])
        np.testing.assert_allclose(rows[0]["pose_world"][:2], [2.0, 4.0])
        self.assertAlmostEqual(rows[0]["pose_world"][2], math.pi / 2 + 0.2)

    def test_predock_is_cube_radial_with_tangent_fallbacks(self):
        manipulation = np.array([4.8, 3.55, math.pi / 2])
        cube = np.array([5.3, 3.65, 0.475])
        options = build_predock_pose_options(
            manipulation,
            cube,
            distance_m=0.55,
            distance_step_m=0.12,
            distance_count=3,
            tangent_offset_m=0.12,
        )
        self.assertEqual(len(options), 9)
        away = manipulation[:2] - cube[:2]
        away /= np.linalg.norm(away)
        tangent = np.array([-away[1], away[0]])
        center = manipulation[:2] + 0.55 * away
        np.testing.assert_allclose(
            options[0][:2], center
        )
        np.testing.assert_allclose(options[1][:2], center + 0.12 * tangent)
        np.testing.assert_allclose(options[2][:2], center - 0.12 * tangent)
        for option in options:
            self.assertAlmostEqual(option[2], manipulation[2])
            self.assertGreater(
                np.linalg.norm(option[:2] - cube[:2]),
                np.linalg.norm(manipulation[:2] - cube[:2]),
            )

    def test_predock_does_not_generate_body_front_back_pairs(self):
        manipulation = np.array([0.0, 0.0, 0.0])
        cube = np.array([-1.0, 0.0, 0.5])
        options = build_predock_pose_options(
            manipulation,
            cube,
            distance_m=0.5,
            distance_step_m=0.1,
            distance_count=1,
            tangent_offset_m=0.12,
        )
        self.assertEqual(len(options), 3)
        # All options are outside M relative to the cube; there is no mirrored
        # P on the cube side of M merely to enable a backward approach.
        np.testing.assert_allclose(options[0], [0.5, 0.0, 0.0])
        self.assertTrue(all(option[0] > 0.0 for option in options))

    def test_world_target_error_is_rotated_into_body_frame(self):
        error = _body_frame_xy_error(
            [1.0, 2.0, math.pi / 2], [2.0, 2.0]
        )
        np.testing.assert_allclose(error, [0.0, -1.0], atol=1e-12)

    def test_approach_error_is_positive_outside_and_negative_inward(self):
        outside, lateral, _, _ = _approach_frame_position_errors(
            [0.97, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0]
        )
        inward, _, _, _ = _approach_frame_position_errors(
            [1.03, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0]
        )
        self.assertAlmostEqual(outside, 0.03)
        self.assertAlmostEqual(inward, -0.03)
        self.assertAlmostEqual(lateral, 0.0)

    def test_settle_window_rejects_periodic_sway_but_reports_drift(self):
        steps = np.arange(15, dtype=np.float64)
        sway = np.column_stack(
            (
                0.01 * np.sin(steps * math.pi / 2.0),
                0.006 * np.cos(steps * math.pi / 2.0),
                0.02 * np.sin(steps * math.pi / 2.0),
            )
        )
        sway_speed, sway_yaw_rate = _pose_window_motion_rates(sway, 0.02)
        self.assertLess(sway_speed, 0.01)
        self.assertLess(sway_yaw_rate, 0.02)

        times = steps * 0.02
        drifting = sway + np.column_stack(
            (0.08 * times, np.zeros_like(times), 0.2 * times)
        )
        drift_speed, drift_yaw_rate = _pose_window_motion_rates(
            drifting, 0.02
        )
        self.assertGreater(drift_speed, 0.07)
        self.assertGreater(drift_yaw_rate, 0.18)

    def test_pan_xy_uses_world_error_and_slews_body_command(self):
        control = _new_pan_xy_control_state()
        command, done, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[0.0, 0.0, math.pi / 2.0],
            target_pose=[1.0, 0.0, math.pi / 2.0],
            base_lin_vel=[0.0, 0.0, 0.0],
            position_tolerance_m=0.05,
            yaw_tolerance_rad=math.radians(3.0),
        )
        self.assertFalse(done)
        self.assertAlmostEqual(command[0], 0.0, places=12)
        self.assertAlmostEqual(command[1], -PAN_DV_PER_STEP, places=12)
        self.assertAlmostEqual(command[2], 0.0, places=12)
        self.assertAlmostEqual(diagnostics["world_vx_target"], 0.25)
        self.assertAlmostEqual(diagnostics["world_vy_target"], 0.0)

    def test_pan_xy_requires_consecutive_stopped_samples(self):
        control = _new_pan_xy_control_state()
        done = False
        for _ in range(PAN_SETTLE_WINDOW_FRAMES + PAN_SETTLE_STEPS - 1):
            command, done, diagnostics = _pan_world_xy_control_step(
                control,
                current_pose=[1.01, 1.02, 0.01],
                target_pose=[1.0, 1.0, 0.0],
                # Instantaneous root velocity contains periodic gait sway and
                # must not veto an otherwise stationary pose window.
                base_lin_vel=[0.15, -0.08, 0.0],
                position_tolerance_m=0.05,
                yaw_tolerance_rad=math.radians(3.0),
            )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0, 0.0))
        self.assertEqual(diagnostics["settle_count"], PAN_SETTLE_STEPS)
        self.assertLess(diagnostics["trend_speed_mps"], 1e-12)

    def test_pan_xy_position_hysteresis_prevents_threshold_chatter(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [0.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(3.0),
        }
        _pan_world_xy_control_step(
            control, current_pose=[0.04, 0.0, 0.0], **common
        )
        command, _, diagnostics = _pan_world_xy_control_step(
            control, current_pose=[0.08, 0.0, 0.0], **common
        )
        self.assertTrue(diagnostics["settling"])
        self.assertEqual(command[:2], (0.0, 0.0))

        command, _, diagnostics = _pan_world_xy_control_step(
            control, current_pose=[0.11, 0.0, 0.0], **common
        )
        self.assertFalse(diagnostics["settling"])
        self.assertLess(command[0], 0.0)

    def test_pan_xy_yaw_hysteresis_does_not_correct_inside_three_degrees(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 1.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(3.0),
        }
        _pan_world_xy_control_step(
            control,
            current_pose=[1.0, 1.0, math.radians(1.0)],
            **common,
        )
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.0, 1.0, math.radians(2.5)],
            **common,
        )
        self.assertTrue(diagnostics["yaw_settling"])
        self.assertAlmostEqual(command[2], 0.0)

    def test_translation_only_pan_never_corrects_large_yaw_error(self):
        control = _new_pan_xy_control_state()
        command, done, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[0.0, 0.0, math.radians(25.0)],
            target_pose=[0.5, 0.0, 0.0],
            base_lin_vel=[0.0, 0.0, 0.0],
            position_tolerance_m=0.05,
            yaw_tolerance_rad=math.radians(4.0),
            control_yaw=False,
        )
        self.assertFalse(done)
        self.assertFalse(diagnostics["yaw_control_enabled"])
        self.assertEqual(command[2], 0.0)
        self.assertNotEqual(command[:2], (0.0, 0.0))

    def test_translation_only_pan_settles_despite_yaw_drift(self):
        control = _new_pan_xy_control_state()
        done = False
        for step in range(PAN_SETTLE_WINDOW_FRAMES + PAN_SETTLE_STEPS - 1):
            command, done, diagnostics = _pan_world_xy_control_step(
                control,
                current_pose=[
                    0.01,
                    -0.02,
                    math.radians(20.0 + 0.5 * step),
                ],
                target_pose=[0.0, 0.0, 0.0],
                base_lin_vel=[0.0, 0.0, 0.0],
                position_tolerance_m=0.05,
                yaw_tolerance_rad=math.radians(4.0),
                control_yaw=False,
            )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0, 0.0))
        self.assertFalse(diagnostics["yaw_control_enabled"])
        self.assertGreater(
            diagnostics["trend_yaw_rate_radps"], PAN_SETTLE_YAW_RATE
        )

    def test_pan_xy_pose_trend_rejects_sustained_drift(self):
        control = _new_pan_xy_control_state()
        done = False
        for step in range(PAN_SETTLE_WINDOW_FRAMES + PAN_SETTLE_STEPS):
            _, done, diagnostics = _pan_world_xy_control_step(
                control,
                current_pose=[0.002 * step, 0.0, 0.0],
                target_pose=[0.0, 0.0, 0.0],
                base_lin_vel=[0.0, 0.0, 0.0],
                position_tolerance_m=0.05,
                yaw_tolerance_rad=math.radians(3.0),
            )
        self.assertFalse(done)
        self.assertGreater(
            diagnostics["trend_speed_mps"], 0.09
        )

    def test_directional_pan_rejects_outside_and_accepts_inward_band(self):
        outside_control = _new_pan_xy_control_state()
        command, done, diagnostics = _pan_world_xy_control_step(
            outside_control,
            current_pose=[0.98, 0.0, 0.0],
            target_pose=[1.0, 0.0, 0.0],
            base_lin_vel=[0.0, 0.0, 0.0],
            position_tolerance_m=0.05,
            yaw_tolerance_rad=math.radians(4.0),
            approach_direction=[1.0, 0.0],
            inward_tolerance_m=0.05,
        )
        self.assertFalse(done)
        self.assertGreater(diagnostics["longitudinal_error_m"], 0.0)
        self.assertGreater(command[0], 0.0)

        inward_control = _new_pan_xy_control_state()
        for _ in range(PAN_SETTLE_WINDOW_FRAMES + PAN_SETTLE_STEPS - 1):
            command, done, diagnostics = _pan_world_xy_control_step(
                inward_control,
                current_pose=[1.03, 0.01, 0.0],
                target_pose=[1.0, 0.0, 0.0],
                base_lin_vel=[0.0, 0.0, 0.0],
                position_tolerance_m=0.05,
                yaw_tolerance_rad=math.radians(4.0),
                approach_direction=[1.0, 0.0],
                inward_tolerance_m=0.05,
            )
        self.assertTrue(done)
        self.assertEqual(command, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(diagnostics["longitudinal_error_m"], -0.03)

    def test_directional_terminal_capture_tolerates_small_pose_drift(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
        }
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.025, 0.0, math.radians(-3.0)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertEqual(command, (0.0, 0.0, 0.0))

        # This is outside the inner capture region, but remains inside the
        # outer [-6, 0] cm / +/-6 cm / +/-8 deg hold region.  It must not
        # restart either XY or yaw correction.
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.055, -0.055, math.radians(-4.8)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertFalse(diagnostics["inside_inner_position"])
        self.assertTrue(diagnostics["inside_outer_position"])
        self.assertEqual(command, (0.0, 0.0, 0.0))

    def test_directional_terminal_capture_exits_outer_position_region(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
        }
        _pan_world_xy_control_step(
            control, current_pose=[1.025, 0.0, 0.0], **common
        )
        command, _, diagnostics = _pan_world_xy_control_step(
            control, current_pose=[1.025, -0.061, 0.0], **common
        )
        self.assertFalse(diagnostics["terminal_capture"])
        self.assertFalse(diagnostics["inside_outer_position"])
        self.assertNotEqual(command[:2], (0.0, 0.0))

    def test_directional_capture_cancels_residual_yaw_command(self):
        control = _new_pan_xy_control_state()
        # Reproduce the state seen in the failing run: geometry is already
        # good, but the preceding P alignment has a residual command.
        control["cmd_wz"] = -0.12
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
        }
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.025, 0.0, math.radians(3.25)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertAlmostEqual(command[2], -0.06)

        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.025, 0.0, math.radians(3.25)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertAlmostEqual(command[2], 0.0)

    @staticmethod
    def _terminal_region():
        covariance = np.diag([0.01, 0.01, 0.10])
        return {
            "observation_pose_world": np.zeros(3),
            "reference_pose_observation": np.array([1.0, 0.0, 0.0]),
            "component_index": 0,
            "covariance": covariance,
            "precision": np.linalg.inv(covariance),
            "enter_radius": 1.0,
            "hold_radius": 1.5,
            "slow_radius": 2.0,
        }

    def test_n2m_terminal_region_keeps_joint_xy_yaw_control(self):
        control = _new_pan_xy_control_state()
        command, done, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[0.90, 0.0, math.radians(10.0)],
            target_pose=[1.0, 0.0, 0.0],
            base_lin_vel=[0.0, 0.0, 0.0],
            position_tolerance_m=0.05,
            yaw_tolerance_rad=math.radians(4.0),
            approach_direction=[1.0, 0.0],
            inward_tolerance_m=0.05,
            terminal_region=self._terminal_region(),
        )
        self.assertFalse(done)
        self.assertTrue(diagnostics["yaw_control_enabled"])
        self.assertTrue(diagnostics["n2m_terminal_enabled"])
        self.assertGreater(command[0], 0.0)
        self.assertLess(command[2], 0.0)
        self.assertLess(diagnostics["terminal_speed_scale"], 1.0)

    def test_n2m_terminal_capture_holds_small_boundary_drift(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
            "terminal_region": self._terminal_region(),
        }
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.03, 0.04, math.radians(5.0)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertTrue(diagnostics["inside_inner_position"])
        self.assertEqual(command, (0.0, 0.0, 0.0))

        # Outside the capture contour but inside the wider hold contour.  The
        # controller must remain latched instead of issuing another gait kick.
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.06, 0.06, math.radians(7.0)],
            **common,
        )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertFalse(diagnostics["inside_inner_position"])
        self.assertTrue(diagnostics["inside_outer_position"])
        self.assertEqual(command, (0.0, 0.0, 0.0))

    def test_n2m_terminal_capture_requires_persistent_hold_exit(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
            "terminal_region": self._terminal_region(),
        }
        _pan_world_xy_control_step(
            control, current_pose=[1.03, 0.0, 0.0], **common
        )
        for _ in range(N2M_TERMINAL_EXIT_FRAMES - 1):
            command, _, diagnostics = _pan_world_xy_control_step(
                control, current_pose=[1.10, 0.10, 0.0], **common
            )
        self.assertTrue(diagnostics["terminal_capture"])
        self.assertEqual(command, (0.0, 0.0, 0.0))

        command, _, diagnostics = _pan_world_xy_control_step(
            control, current_pose=[1.10, 0.10, 0.0], **common
        )
        self.assertFalse(diagnostics["terminal_capture"])
        self.assertNotEqual(command[:2], (0.0, 0.0))

    def test_sample_terminal_region_uses_most_likely_component_covariance(self):
        prediction = {
            "observation_pose_world": [2.0, 3.0, 0.2],
            "modes": [
                {
                    "pose_observation": [0.0, 0.0, 0.0],
                    "weight": 0.5,
                    "covariance": np.diag([0.01, 0.01, 0.01]).tolist(),
                },
                {
                    "pose_observation": [1.0, 0.0, 0.0],
                    "weight": 0.5,
                    "covariance": np.diag([0.04, 0.04, 0.04]).tolist(),
                },
            ],
        }
        queue_row = {
            "selection": {
                "kind": "sample",
                "index": 3,
                "pose_observation": [0.9, 0.0, 0.0],
            }
        }
        region = _n2m_candidate_terminal_region(prediction, queue_row)
        self.assertEqual(region["component_index"], 1)
        np.testing.assert_allclose(
            region["reference_pose_observation"], [0.9, 0.0, 0.0]
        )

    def test_directional_pan_never_rotates_and_requests_safe_realign(self):
        control = _new_pan_xy_control_state()
        common = {
            "target_pose": [1.0, 0.0, 0.0],
            "base_lin_vel": [0.0, 0.0, 0.0],
            "position_tolerance_m": 0.05,
            "yaw_tolerance_rad": math.radians(4.0),
            "approach_direction": [1.0, 0.0],
            "inward_tolerance_m": 0.05,
        }
        command, _, diagnostics = _pan_world_xy_control_step(
            control,
            current_pose=[1.025, 0.0, math.radians(6.5)],
            **common,
        )
        self.assertFalse(diagnostics["terminal_capture"])
        self.assertTrue(diagnostics["terminal_yaw_realign_required"])
        self.assertEqual(command[2], 0.0)

    def test_directional_pan_escalates_persistent_terminal_yaw_to_p(self):
        class FakeEnv:
            dt = 0.02

            @staticmethod
            def get_robot_state():
                return {
                    "pos": np.array([1.025, 0.0, 0.5]),
                    "yaw": math.radians(6.5),
                    "lin_vel": np.zeros(3),
                }

        runner = SimpleNamespace(
            _arm_q=lambda: np.zeros(6, dtype=np.float64)
        )
        interactive = SimpleNamespace(quit=False)
        with patch(
            "custom_envs.scripts.navigation.navigate_mujoco_n2m."
            "_checked_base_motion_step",
            return_value=np.zeros(16, dtype=np.float32),
        ), patch("builtins.print"):
            with self.assertRaises(EpisodeFailure) as caught:
                pan_xy_to_pose(
                    name="M_DOCK_PAN_XY",
                    env=FakeEnv(),
                    runner=runner,
                    policy=None,
                    device=None,
                    interactive=interactive,
                    target_pose=[1.0, 0.0, 0.0],
                    position_tolerance_m=0.05,
                    yaw_tolerance_rad=math.radians(4.0),
                    max_steps=20,
                    approach_direction=[1.0, 0.0],
                    inward_tolerance_m=0.05,
                )
        self.assertEqual(
            caught.exception.reason, "terminal_yaw_realign_required"
        )


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

    def test_global_plan_feedback_is_exposed_as_a_copy(self):
        bridge = IsaacROS2Bridge()
        bridge._feedback_q.put(
            {
                "global_path": [[1.0, 2.0], [3.0, 4.0]],
                "path_revision": 7,
            }
        )
        path = bridge.get_global_path()
        self.assertEqual(path, [[1.0, 2.0], [3.0, 4.0]])
        path[0][0] = 99.0
        self.assertEqual(bridge.get_global_path()[0][0], 1.0)


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
        self.assertEqual(args.predock_distance, 0.55)
        self.assertEqual(args.predock_rotation_radius, 0.18)
        self.assertEqual(args.predock_align_max_steps, 800)
        self.assertEqual(args.dock_position_tolerance, 0.05)
        self.assertEqual(args.dock_inward_tolerance, 0.05)
        self.assertEqual(args.dock_yaw_tolerance_deg, 4.0)
        self.assertEqual(args.dock_pulse_steps, 8)
        self.assertEqual(args.dock_max_cycles, 120)
        self.assertEqual(args.nav_snapshot_every, 200)
        self.assertTrue(args.teleport_navigation)
        self.assertEqual(args.teleport_every_steps, 5)
        self.assertEqual(args.teleport_step_distance, 0.08)

    def test_teleport_navigation_alias_enables_pose_reset_mode(self):
        args = build_parser().parse_args(
            [
                "--policy_path",
                "policy.pt",
                "--grasp-checkpoint",
                "grasp.tar",
                "--destination",
                "12.0",
                "12.0",
                "--teleport",
                "--teleport-every-steps",
                "7",
                "--teleport-step-distance",
                "0.1",
            ]
        )
        self.assertTrue(args.teleport_navigation)
        self.assertEqual(args.teleport_every_steps, 7)
        self.assertEqual(args.teleport_step_distance, 0.1)

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

    def test_nav_command_conditioner_rejects_alternating_yaw_chatter(self):
        conditioner = NavCommandConditioner(0.02)
        outputs = []
        # Model a 10 Hz bridge command whose steering sign flips on every
        # update while the apparent route remains nearly straight.
        for step in range(200):
            sign = 1.0 if (step // 5) % 2 == 0 else -1.0
            _, wz, _ = conditioner.step(0.5, sign * 0.18)
            outputs.append(wz)
        self.assertLess(max(abs(value) for value in outputs), 0.08)

    def test_nav_command_conditioner_preserves_sustained_small_correction(self):
        conditioner = NavCommandConditioner(0.02)
        outputs = [conditioner.step(0.5, 0.18)[1] for _ in range(200)]
        self.assertGreater(outputs[-1], 0.17)

    def test_nav_command_conditioner_preserves_sustained_turn(self):
        conditioner = NavCommandConditioner(0.02)
        vx = wz = 0.0
        for _ in range(75):
            vx, wz, _ = conditioner.step(0.5, 0.4)
        self.assertAlmostEqual(vx, 0.35, places=2)
        self.assertGreater(wz, 0.35)

    def test_nav_command_conditioner_stops_immediately_on_zero(self):
        conditioner = NavCommandConditioner(0.02)
        for _ in range(50):
            conditioner.step(0.5, 0.4)
        vx, wz, diagnostics = conditioner.step(0.0, 0.0)
        self.assertEqual((vx, wz), (0.0, 0.0))
        self.assertTrue(diagnostics["hard_stop"])

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
