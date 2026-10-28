from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from custom_envs.scripts.N2M.attached_object_specs import (
    ATTACHED_OBJECT_SPECS,
    attached_box_pose_matrix,
    resolve_attached_box_spec,
)
from custom_envs.scripts.N2M.manipulation_episode import (
    ManipulationEpisodeRunner,
)


class _Pose:
    def __init__(self, position, rotation):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)

    def matrix(self):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform


class _MoveIt:
    def __init__(self):
        self.calls = []

    def attach_box(self, object_id, pose, size_xyz, **kwargs):
        self.calls.append((object_id, pose, size_xyz, kwargs))
        return {"ok": True}


class AttachedObjectSpecTests(unittest.TestCase):
    def test_current_scene_objects_have_valid_specs(self):
        for name in ("cube", "apple", "orange", "mug"):
            spec = ATTACHED_OBJECT_SPECS[name]
            self.assertTrue(all(value > 0.0 for value in spec.size_xyz))
            self.assertTrue(all(math.isfinite(value) for value in spec.center_xyz))

    def test_cube_origin_is_already_its_box_center(self):
        spec = resolve_attached_box_spec("cube")
        self.assertEqual(spec.size_xyz, (0.05, 0.05, 0.05))
        self.assertEqual(spec.center_xyz, (0.0, 0.0, 0.0))

    def test_explicit_overrides_replace_default_values(self):
        spec = resolve_attached_box_spec(
            "apple",
            size_override=(0.1, 0.2, 0.3),
            center_override=(0.01, -0.02, 0.03),
        )
        self.assertEqual(spec.size_xyz, (0.1, 0.2, 0.3))
        self.assertEqual(spec.center_xyz, (0.01, -0.02, 0.03))

    def test_custom_object_requires_an_explicit_size(self):
        with self.assertRaisesRegex(ValueError, "no attached-object box"):
            resolve_attached_box_spec("custom_target")
        spec = resolve_attached_box_spec(
            "custom_target", size_override=(0.04, 0.05, 0.06)
        )
        self.assertEqual(spec.center_xyz, (0.0, 0.0, 0.0))

    def test_center_offset_is_rotated_from_object_frame(self):
        T_tcp_object = np.eye(4, dtype=np.float64)
        T_tcp_object[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T_tcp_object[:3, 3] = [1.0, 2.0, 3.0]
        T_tcp_box = attached_box_pose_matrix(
            T_tcp_object, center_xyz=(0.1, 0.0, 0.2)
        )
        np.testing.assert_allclose(T_tcp_box[:3, 3], [1.0, 2.1, 3.2])
        np.testing.assert_allclose(
            T_tcp_box[:3, :3], T_tcp_object[:3, :3]
        )

    def test_runner_passes_box_center_pose_to_moveit(self):
        runner = object.__new__(ManipulationEpisodeRunner)
        runner.cfg = SimpleNamespace(
            object_name="orange",
            moveit_attached_object_size=None,
            moveit_attached_object_center=None,
        )
        runner.Pose = _Pose
        runner.moveit = _MoveIt()
        runner.attached_id = "test_carried_object"
        runner.attached = False
        runner.pose_to_ros_dict = lambda pose: {
            "position": pose.position.tolist(),
            "rotation": pose.rotation.tolist(),
        }
        object_pose = _Pose([0.1, 0.2, 0.3], np.eye(3))

        result = runner._attach_current_object(
            "LIFT", object_pose_in_tcp=object_pose
        )

        self.assertTrue(runner.attached)
        self.assertEqual(len(runner.moveit.calls), 1)
        _, pose, size_xyz, kwargs = runner.moveit.calls[0]
        spec = ATTACHED_OBJECT_SPECS["orange"]
        np.testing.assert_allclose(
            pose["position"], np.asarray(object_pose.position) + spec.center_xyz
        )
        self.assertEqual(size_xyz, spec.size_xyz)
        self.assertEqual(kwargs["link_name"], "grasp_tcp")
        self.assertIs(result["object_pose_in_tcp"], object_pose)


if __name__ == "__main__":
    unittest.main()
