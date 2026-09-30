import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from custom_envs.mujoco.env import MuJoCoEnv
from custom_envs.mujoco.lidar import MuJoCoMid360


@pytest.fixture
def env():
    instance = MuJoCoEnv(render=False, substeps=1)
    try:
        yield instance
    finally:
        instance.close()


def test_mid360_pattern_and_sensor_pose(env):
    lidar = MuJoCoMid360(env.model)
    scan = lidar.scan(env.data)

    assert lidar.num_rays == 40 * 360
    np.testing.assert_allclose(
        np.linalg.norm(scan.directions_w, axis=1), 1.0, atol=1e-12)
    expected_origin = env.data.xpos[lidar.base_body_id] + np.array(
        [0.30, 0.0, 0.55])
    np.testing.assert_allclose(scan.origin_w, expected_origin, atol=1e-12)
    assert scan.valid_mask.any()
    assert scan.hits_w.shape == (int(scan.valid_mask.sum()), 3)


def test_scan_excludes_all_robot_collision_geometries(env):
    lidar = MuJoCoMid360(env.model)
    scan = lidar.scan(env.data)
    hit_geom_ids = scan.geom_ids[scan.valid_mask]

    assert np.intersect1d(hit_geom_ids, lidar.robot_geom_ids).size == 0
    assert np.all(env.model.geom_group[lidar.robot_geom_ids] == 4)


def test_sensor_offset_rotates_with_base(env):
    half = np.pi / 4.0
    yaw_90_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    env.set_root_pose(np.array([2.0, 3.0, 0.58]), yaw_90_quat)
    lidar = MuJoCoMid360(env.model)

    origin, _ = lidar.sensor_pose_world(env.data)
    np.testing.assert_allclose(origin, [2.0, 3.3, 1.13], atol=1e-8)
