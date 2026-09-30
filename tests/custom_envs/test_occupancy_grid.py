import numpy as np
import pytest

from custom_envs.scripts.navigation.convert_map import log_odds_to_pgm_pixel
from custom_envs.utils.nav_utils import world_to_grid as nav_world_to_grid
from custom_envs.utils.occupancy_grid import OccupancyGrid


def test_world_grid_uses_row_y_and_col_x():
    grid = OccupancyGrid(width=9, height=6, resolution=1.0)
    grid.set_origin(0.0, 0.0)

    assert grid.world_to_grid(4.2, 2.2) == (2, 4)
    assert grid.grid_to_world(2, 4) == (4.5, 2.5)
    assert grid.world_to_grid(-0.1, 0.1) == (0, -1)
    assert nav_world_to_grid(-0.1, 0.1, (0.0, 0.0), 1.0) == (0, -1)


def test_ray_cast_does_not_transpose_row_and_col():
    grid = OccupancyGrid(width=9, height=6, resolution=1.0)
    grid._ray_cast(1.2, 1.2, 4.2, 2.2)

    assert grid.grid[2, 4] < 0.0
    assert grid.grid[4, 2] == 0.0


def test_scan_integration_uses_sensor_origin_and_marks_obstacle():
    grid = OccupancyGrid(width=10, height=8, resolution=1.0)
    finite = grid.integrate_scan(
        sensor_origin_w=np.array([1.2, 1.2, 1.0]),
        lidar_hits_w=np.array([
            [4.2, 2.2, 0.5],
            [np.inf, 3.0, 0.5],
        ]),
        obstacle_min_z=0.15,
        min_hits=1,
    )

    assert finite.shape == (1, 3)
    assert grid.grid[1, 1] < 0.0
    assert grid.grid[2, 4] > 0.0
    assert grid.grid[4, 2] == 0.0


def test_outside_ray_is_clipped_to_map():
    grid = OccupancyGrid(width=5, height=3, resolution=1.0)
    grid._ray_cast(-2.0, 1.2, 7.0, 1.2)

    assert np.all(grid.grid[1, :] < 0.0)
    assert np.all(grid.grid[[0, 2], :] == 0.0)


def test_save_load_preserves_hit_counter(tmp_path):
    grid = OccupancyGrid(width=7, height=4, resolution=0.25)
    grid.set_origin(-1.0, -2.0)
    grid.mark_elevated(np.array([[0.1, -1.1]]), min_hits=2)
    map_path = tmp_path / "roundtrip.npz"
    grid.save(map_path)

    loaded = OccupancyGrid.load(map_path)
    assert loaded.width == 7
    assert loaded.height == 4
    assert loaded.origin == (-1.0, -2.0)
    np.testing.assert_array_equal(loaded._elevated_count,
                                  grid._elevated_count)


def test_nav2_image_flips_only_the_image_y_axis():
    values = np.zeros((3, 4), dtype=np.float32)
    values[0, 1] = 1.0   # minimum world Y, occupied
    values[2, 3] = -1.0  # maximum world Y, free

    pgm = log_odds_to_pgm_pixel(values)

    assert pgm[2, 1] == 0
    assert pgm[0, 3] == 254
    assert pgm[1, 0] == 205


def test_invalid_grid_and_hit_threshold_are_rejected():
    with pytest.raises(ValueError):
        OccupancyGrid(width=0, height=4)
    grid = OccupancyGrid(width=4, height=4)
    with pytest.raises(ValueError):
        grid.mark_elevated(np.zeros((0, 2)), min_hits=0)
