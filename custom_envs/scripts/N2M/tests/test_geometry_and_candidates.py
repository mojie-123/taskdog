from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_envs.scripts.N2M.candidate_generation import (
    generate_candidates,
    nominal_pose_for_edge,
)
from custom_envs.scripts.N2M.adaptive_n2m_recovery import (
    build_candidate_queue,
    gmm_log_density,
)
from custom_envs.scripts.N2M.config import (
    CUBE_SCENES,
    DEFAULT_REFERENCE_BASE,
    DEFAULT_ROBOT_ROOT_PITCH,
    DEFAULT_ROBOT_ROOT_ROLL,
    CandidateGrid,
)
from custom_envs.scripts.N2M.geometry import (
    angular_error,
    joint1_towards_cube,
    relative_se2,
    rpy_to_quat_wxyz,
    transform_se2,
)
from custom_envs.scripts.N2M.manifests import append_jsonl, read_jsonl, write_jsonl
from custom_envs.scripts.N2M.replay_n2m_prediction import build_prediction_candidate
from custom_envs.scripts.N2M.retry_n2m_grasp_until_success import (
    pose_motion,
    quaternion_angle_rad,
)


class GeometryTests(unittest.TestCase):
    def test_se2_relative_round_trip(self):
        origin = np.array([1.3, -0.7, 1.1])
        target = np.array([-2.0, 4.2, -2.4])
        rebuilt = transform_se2(origin, relative_se2(origin, target))
        np.testing.assert_allclose(rebuilt[:2], target[:2], atol=1e-12)
        self.assertAlmostEqual(angular_error(rebuilt[2], target[2]), 0.0, places=12)

    def test_joint1_is_cube_bearing_in_base_frame(self):
        target = joint1_towards_cube((1.0, 2.0, math.pi / 2), (2.0, 2.0, 0.0))
        self.assertAlmostEqual(target, -math.pi / 2, places=12)

    def test_recorded_reference_orientation_round_trip(self):
        expected = np.array([0.697786, 0.010601, 0.013125, 0.716107])
        expected /= np.linalg.norm(expected)
        actual = rpy_to_quat_wxyz(
            DEFAULT_ROBOT_ROOT_ROLL,
            DEFAULT_ROBOT_ROOT_PITCH,
            DEFAULT_REFERENCE_BASE[2],
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)


class CandidateTests(unittest.TestCase):
    @staticmethod
    def _adaptive_prediction():
        observation = {
            "observation_id": "o-adaptive",
            "scene_id": "short_left",
            "approach_edge": "left",
            "observation_pose_world": [1.0, 2.0, 0.4],
            "cube_pose_world": [5.3, 3.65, 0.475, 1.0, 0.0, 0.0, 0.0],
        }
        relative_rows = [
            [1.0, -0.2, 0.3],
            [1.01, -0.19, 0.31],
            [1.20, -0.2, 0.3],
            [1.01, -0.19, 0.60],
        ]
        prediction = {
            "observation_id": observation["observation_id"],
            "observation_pose_world": observation["observation_pose_world"],
            "modes": [
                {
                    "pose_observation": relative_rows[0],
                    "pose_world": transform_se2(
                        observation["observation_pose_world"], relative_rows[0]
                    ).tolist(),
                    "weight": 1.0,
                    "covariance": [
                        [0.04, 0.0, 0.0],
                        [0.0, 0.04, 0.0],
                        [0.0, 0.0, 0.09],
                    ],
                }
            ],
            "samples": [
                {
                    "pose_observation": pose,
                    "pose_world": transform_se2(
                        observation["observation_pose_world"], pose
                    ).tolist(),
                }
                for pose in relative_rows[1:]
            ],
        }
        return observation, prediction

    def test_adaptive_queue_starts_at_single_gaussian_mean(self):
        observation, prediction = self._adaptive_prediction()
        queue, diagnostics = build_candidate_queue(
            prediction,
            observation,
            min_xy_m=0.05,
            min_yaw_rad=math.radians(10.0),
        )
        self.assertEqual(queue[0]["selection"]["kind"], "mode")
        self.assertEqual(queue[0]["selection"]["index"], 0)
        self.assertEqual(diagnostics["raw_candidate_count"], 4)

    def test_adaptive_queue_suppresses_only_xy_and_yaw_near_copies(self):
        observation, prediction = self._adaptive_prediction()
        queue, diagnostics = build_candidate_queue(
            prediction,
            observation,
            min_xy_m=0.05,
            min_yaw_rad=math.radians(10.0),
        )
        # The near-identical first sample is removed.  A translated sample and
        # a same-position/different-yaw sample both remain useful alternatives.
        self.assertEqual(len(queue), 3)
        self.assertEqual(diagnostics["suppressed_near_duplicate_count"], 1)
        selections = {(row["selection"]["kind"], row["selection"]["index"]) for row in queue}
        self.assertIn(("sample", 1), selections)
        self.assertIn(("sample", 2), selections)

    def test_gmm_density_decreases_away_from_single_mode(self):
        _, prediction = self._adaptive_prediction()
        center = gmm_log_density([1.0, -0.2, 0.3], prediction["modes"])
        offset = gmm_log_density([1.4, -0.2, 0.3], prediction["modes"])
        self.assertGreater(center, offset)

    def test_prediction_replay_uses_highest_weight_mode_and_checks_transform(self):
        observation = {
            "observation_id": "o0",
            "scene_id": "short_left",
            "approach_edge": "left",
            "observation_pose_world": [1.0, 2.0, math.pi / 2],
            "cube_pose_world": [5.3, 3.65, 0.475, 1.0, 0.0, 0.0, 0.0],
        }
        prediction = {
            "observation_id": "o0",
            "observation_pose_world": observation["observation_pose_world"],
            "modes": [
                {
                    "pose_observation": [1.0, 0.0, 0.1],
                    "pose_world": transform_se2(
                        observation["observation_pose_world"], [1.0, 0.0, 0.1]
                    ).tolist(),
                    "weight": 0.1,
                },
                {
                    "pose_observation": [2.0, 0.0, -0.2],
                    "pose_world": transform_se2(
                        observation["observation_pose_world"], [2.0, 0.0, -0.2]
                    ).tolist(),
                    "weight": 0.9,
                },
            ],
            "samples": [],
        }
        candidate, selection = build_prediction_candidate(prediction, observation)
        self.assertEqual(selection["kind"], "mode")
        self.assertEqual(selection["index"], 1)
        np.testing.assert_allclose(candidate["base_pose"], [1.0, 4.0, math.pi / 2 - 0.2])
        self.assertEqual(candidate["source"], "n2m_prediction")

    def test_prediction_replay_can_select_saved_sample(self):
        observation = {
            "observation_id": "o0",
            "scene_id": "short_left",
            "approach_edge": "left",
            "observation_pose_world": [0.0, 0.0, 0.0],
            "cube_pose_world": [5.3, 3.65, 0.475, 1.0, 0.0, 0.0, 0.0],
        }
        prediction = {
            "observation_id": "o0",
            "observation_pose_world": [0.0, 0.0, 0.0],
            "modes": [],
            "samples": [
                {"pose_observation": [0.5, -0.2, 0.3], "pose_world": [0.5, -0.2, 0.3]}
            ],
        }
        candidate, selection = build_prediction_candidate(
            prediction, observation, sample_index=0
        )
        self.assertEqual(selection["kind"], "sample")
        np.testing.assert_allclose(candidate["base_pose"], [0.5, -0.2, 0.3])

    def test_persistent_retry_pose_motion_handles_quaternion_sign(self):
        before = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        after = [1.003, 1.996, 3.0, -1.0, 0.0, 0.0, 0.0]
        translation, rotation = pose_motion(before, after)
        self.assertAlmostEqual(translation, 0.005, places=12)
        self.assertAlmostEqual(rotation, 0.0, places=12)

    def test_quaternion_angle_reports_known_rotation(self):
        half = math.radians(15.0)
        rotated = [math.cos(half), 0.0, 0.0, math.sin(half)]
        self.assertAlmostEqual(
            quaternion_angle_rad([1.0, 0.0, 0.0, 0.0], rotated),
            math.radians(30.0),
            places=12,
        )

    def test_default_grid_has_100_candidates_per_edge(self):
        candidates = generate_candidates(DEFAULT_REFERENCE_BASE, ["short_left"])
        self.assertEqual(len(candidates), 100)
        self.assertEqual(candidates[0]["candidate_id"], "short_left__left__m0000")
        np.testing.assert_allclose(candidates[0]["base_pose"], DEFAULT_REFERENCE_BASE)
        self.assertEqual(candidates[1]["perturbation"]["normal_m"], -0.05)
        self.assertLess(candidates[1]["base_pose"][0], candidates[0]["base_pose"][0])

    def test_default_nonzero_normal_offsets_only_move_away_from_table(self):
        candidates = generate_candidates(DEFAULT_REFERENCE_BASE, ["short_left"])
        reference_x = DEFAULT_REFERENCE_BASE[0]
        for candidate in candidates:
            normal = candidate["perturbation"]["normal_m"]
            self.assertLessEqual(normal, 0.0)
            if normal < 0.0:
                self.assertLess(candidate["base_pose"][0], reference_x)

    def test_positive_inward_normal_offset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normal offsets must be <= 0"):
            generate_candidates(
                DEFAULT_REFERENCE_BASE,
                ["short_left"],
                CandidateGrid([0.05], [0.0], [0.0]),
            )

    def test_nominal_only_covers_six_scenes_and_eight_edges(self):
        grid = CandidateGrid([0.0], [0.0], [0.0])
        candidates = generate_candidates(DEFAULT_REFERENCE_BASE, list(CUBE_SCENES), grid)
        self.assertEqual(len(candidates), 8)
        self.assertEqual({row["scene_id"] for row in candidates}, set(CUBE_SCENES))
        self.assertEqual(
            {(row["scene_id"], row["approach_edge"]) for row in candidates},
            {
                (scene_id, edge_id)
                for scene_id, scene in CUBE_SCENES.items()
                for edge_id in scene.approach_edges
            },
        )

    def test_transferred_nominal_keeps_edge_relative_geometry(self):
        reference = np.asarray(DEFAULT_REFERENCE_BASE)
        reference_cube = np.asarray(CUBE_SCENES["short_left"].cube_xyz)
        reference_distance = np.linalg.norm(reference[:2] - reference_cube[:2])
        reference_joint1 = joint1_towards_cube(reference, reference_cube)
        for scene in CUBE_SCENES.values():
            for edge_id in scene.approach_edges:
                nominal = nominal_pose_for_edge(reference, scene.cube_xyz, edge_id)
                cube = np.asarray(scene.cube_xyz)
                self.assertAlmostEqual(
                    np.linalg.norm(nominal[:2] - cube[:2]),
                    reference_distance,
                    places=12,
                )
                self.assertAlmostEqual(
                    joint1_towards_cube(nominal, cube), reference_joint1, places=12
                )


class ManifestTests(unittest.TestCase):
    def test_atomic_rewrite_and_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, [{"value": np.int64(1)}])
            append_jsonl(path, {"value": 2})
            self.assertEqual(read_jsonl(path), [{"value": 1}, {"value": 2}])
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
