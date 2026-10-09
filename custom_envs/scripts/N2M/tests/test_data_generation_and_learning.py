from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from custom_envs.scripts.N2M.data_generation.camera_geometry import (
    CameraIntrinsics,
    camera_points_to_frame,
    crop_point_mask,
    depth_to_optical_points,
    voxel_downsample_indices,
)
from custom_envs.scripts.N2M.data_generation.observation_sampling import (
    propose_observation_poses,
    stratified_order,
)
from custom_envs.scripts.N2M.data_generation.capture_observations import (
    _proposal_grid_extension_compatible,
)
from custom_envs.scripts.N2M.data_generation.pair_geometry import (
    assert_target_round_trip,
    target_in_observation_frame,
)
from custom_envs.scripts.N2M.learning.dataset import N2MPairDataset, augment_se2
from custom_envs.scripts.N2M.manifests import write_jsonl


class CameraGeometryTests(unittest.TestCase):
    def test_center_pixel_unprojects_along_optical_z(self):
        intrinsics = CameraIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0
        points, indices = depth_to_optical_points(depth, intrinsics)
        np.testing.assert_allclose(points, [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(indices, [4])

    def test_optical_to_identity_mujoco_frame(self):
        result = camera_points_to_frame(
            [[0.0, 0.0, 2.0]],
            [0.0, 0.0, 0.0],
            np.eye(3),
            [0.0, 0.0, 0.0],
            np.eye(3),
        )
        np.testing.assert_allclose(result, [[0.0, 0.0, -2.0]])

    def test_crop_and_voxel_indices_propagate_semantics(self):
        xyz = np.array(
            [[0.001, 0.0, 0.0], [0.009, 0.0, 0.0], [0.021, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        cube_flags = np.array([True, False, True, False])
        crop = crop_point_mask(xyz, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        keep = voxel_downsample_indices(xyz[crop], 0.01)
        np.testing.assert_array_equal(keep, [0, 2])
        np.testing.assert_array_equal(cube_flags[crop][keep], [True, True])


class ObservationAndPairTests(unittest.TestCase):
    def test_resume_accepts_only_proposal_grid_supersets(self):
        previous = {
            "scene_xml": "scene.xml",
            "seed": 42,
            "ranges": [1.0, 1.2],
            "tangent_offsets": [-0.2, 0.0, 0.2],
            "yaw_offsets_deg": [-10.0, 0.0, 10.0],
        }
        extended = {
            **previous,
            "ranges": [1.0, 1.1, 1.2],
            "tangent_offsets": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "yaw_offsets_deg": [-10.0, -5.0, 0.0, 5.0, 10.0],
        }
        self.assertTrue(_proposal_grid_extension_compatible(previous, extended))
        self.assertFalse(_proposal_grid_extension_compatible(extended, previous))
        self.assertFalse(
            _proposal_grid_extension_compatible(
                previous, {**extended, "scene_xml": "different.xml"}
            )
        )

    def test_default_proposals_cover_parameter_product(self):
        rows = propose_observation_poses(
            "short_left",
            ranges_m=[1.0, 1.5],
            tangent_offsets_m=[-0.25, 0.0, 0.25],
            yaw_offsets_deg=[-15.0, 0.0, 15.0],
        )
        self.assertEqual(len(rows), 18)
        self.assertEqual({row["approach_edge"] for row in rows}, {"left"})

    def test_stratified_order_balances_small_pilot(self):
        rows = propose_observation_poses(
            "short_left",
            ranges_m=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            tangent_offsets_m=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
            yaw_offsets_deg=[-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0],
        )
        ordered = stratified_order(rows, seed=42)
        self.assertEqual(len(ordered), 294)
        self.assertEqual(ordered, stratified_order(rows, seed=42))
        first = ordered[:14]
        for key in ("tangent_m", "yaw_offset_deg"):
            counts = Counter(row["proposal"][key] for row in first)
            self.assertEqual(set(counts.values()), {2})

    def test_pair_label_round_trip(self):
        observation = [1.0, 2.0, math.pi / 2]
        manipulation = [1.5, 3.0, -math.pi / 2]
        target = target_in_observation_frame(observation, manipulation)
        assert_target_round_trip(observation, manipulation, target)

    def test_se2_augmentation_changes_cloud_and_label_together(self):
        xyz = np.array([[1.0, 0.0, 0.2]], dtype=np.float32)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        xyz_augmented, target_augmented = augment_se2(
            xyz, target, math.pi / 2, np.array([0.5, -0.5], dtype=np.float32)
        )
        np.testing.assert_allclose(xyz_augmented[0, :2], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(target_augmented[:2], [0.5, 0.5], atol=1e-6)
        self.assertAlmostEqual(float(target_augmented[2]), math.pi / 2, places=6)


class DatasetTests(unittest.TestCase):
    def test_loader_returns_xyzrgb_and_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "observations").mkdir()
            np.savez_compressed(
                root / "observations" / "o.npz",
                xyz=np.arange(30, dtype=np.float32).reshape(10, 3),
                rgb=np.full((10, 3), 255, dtype=np.uint8),
            )
            write_jsonl(
                root / "pairs.jsonl",
                [
                    {
                        "pair_id": "p0",
                        "observation_id": "o0",
                        "cloud_file": "observations/o.npz",
                        "target_pose_observation": [0.5, -0.2, 0.1],
                        "scene_id": "short_left",
                        "approach_edge": "left",
                        "split": "train",
                    }
                ],
            )
            dataset = N2MPairDataset(root, "train", num_points=8, training=False)
            sample = dataset[0]
            self.assertEqual(sample["points"].shape, (8, 6))
            self.assertEqual(sample["target"].shape, (3,))
            self.assertLessEqual(float(sample["points"][:, 3:].max()), 1.0)


class ModelTests(unittest.TestCase):
    def test_downloaded_modelnet_checkpoint_has_full_backbone_coverage(self):
        try:
            import torch
            if not hasattr(torch, "nn") or not hasattr(torch.nn, "Module"):
                self.skipTest("the active Python has only an incomplete torch namespace")
            from custom_envs.scripts.N2M.vendor_n2m.model.pointbert import PointTransformer
        except ImportError as error:
            self.skipTest(str(error))
        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "checkpoints" / "pointbert" / "PointTransformer_ModelNet8192points.pth"
        )
        if not checkpoint.exists():
            self.skipTest("optional downloaded Point-BERT checkpoint is absent")
        model = PointTransformer(
            {
                "trans_dim": 384,
                "depth": 12,
                "drop_path_rate": 0.1,
                "num_heads": 6,
                "group_size": 32,
                "num_group": 512,
                "encoder_dims": 256,
            }
        )
        report = model.load_checkpoint(checkpoint)
        self.assertEqual(report["element_coverage"], 1.0)
        self.assertTrue(report["adapted_xyz_to_xyzrgb"])
        self.assertEqual(len(report["ignored_tensors"]), 4)
        first_weight = model.encoder.first_conv[0].weight.detach()
        self.assertGreater(float(first_weight[:, :3].abs().sum()), 0.0)
        self.assertEqual(int(torch.count_nonzero(first_weight[:, 3:])), 0)

    def test_small_model_forward_backward(self):
        try:
            import torch
            if not hasattr(torch, "nn") or not hasattr(torch.nn, "Module"):
                self.skipTest("the active Python has only an incomplete torch namespace")
            from custom_envs.scripts.N2M.learning.gmm_loss import GMMPoseLoss
            from custom_envs.scripts.N2M.vendor_n2m import N2MNet
        except ImportError as error:
            self.skipTest(str(error))
        config = {
            "encoder": {
                "name": "PointBERT",
                "config": {
                    "trans_dim": 32,
                    "depth": 2,
                    "drop_path_rate": 0.0,
                    "num_heads": 4,
                    "group_size": 8,
                    "num_group": 8,
                    "encoder_dims": 32,
                },
            },
            "decoder": {
                "num_gaussians": 2,
                "output_dim": 3,
                "layers": [32],
                "dropout": 0.0,
            },
        }
        model = N2MNet(config)
        points = torch.randn(2, 64, 6)
        points[..., 3:] = torch.sigmoid(points[..., 3:])
        means, covariance, weights = model(points)
        self.assertEqual(tuple(means.shape), (2, 2, 3))
        self.assertEqual(tuple(covariance.shape), (2, 2, 3, 3))
        self.assertEqual(tuple(weights.shape), (2, 2))
        loss, _ = GMMPoseLoss()(means, covariance, weights, torch.zeros(2, 3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
