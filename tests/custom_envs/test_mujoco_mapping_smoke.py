import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from custom_envs.mujoco.env import MuJoCoEnv
from custom_envs.mujoco.lidar import MuJoCoMid360
from custom_envs.scripts.navigation.convert_map import convert
from custom_envs.utils.occupancy_grid import OccupancyGrid


def _quat_from_yaw(yaw):
    return np.array([
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    ])


def test_mujoco_scan_to_nav2_map_pipeline(tmp_path):
    env = MuJoCoEnv(render=False, substeps=1)
    try:
        lidar = MuJoCoMid360(env.model)
        grid = OccupancyGrid(width=300, height=300, resolution=0.05)
        grid.set_origin(-3.0, -3.0)
        hit_names = set()

        for x, y, yaw in ((0.0, 0.0, 0.0),
                          (1.0, 5.0, 0.0),
                          (4.0, 2.0, np.pi / 2.0)):
            env.set_root_pose(
                np.array([x, y, 0.58]), _quat_from_yaw(yaw))
            scan = lidar.scan(env.data)
            grid.integrate_scan(
                scan.origin_w,
                scan.hits_w,
                obstacle_min_z=0.15,
                min_hits=1,
            )
            for geom_id in scan.geom_ids[scan.valid_mask]:
                name = mujoco.mj_id2name(
                    env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
                if name:
                    hit_names.add(name)

        assert int((grid.grid < -0.5).sum()) > 0
        assert int((grid.grid > 0.5).sum()) > 0
        assert "floor" in hit_names
        assert any(name.startswith("table") for name in hit_names)

        npz_path = tmp_path / "mujoco_smoke.npz"
        grid.save(npz_path)
        pgm_path, yaml_path = convert(
            str(npz_path), str(tmp_path / "mujoco_smoke_nav2"))

        assert os.path.getsize(pgm_path) > grid.width * grid.height
        assert os.path.isfile(yaml_path)
        loaded = OccupancyGrid.load(npz_path)
        np.testing.assert_array_equal(loaded.grid, grid.grid)
    finally:
        env.close()
