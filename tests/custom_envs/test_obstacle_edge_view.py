from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from custom_envs.scripts.map.nav2_map_io import load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap
from custom_envs.scripts.sysnav.config import load_sysnav_config
from custom_envs.scripts.sysnav.obstacle_edge_view import (
    fill_component_holes,
    plan_obstacle_edge_views,
    save_obstacle_edge_view_debug,
)


def test_irregular_component_holes_are_filled_without_mutating_input():
    component = np.zeros((9, 11), dtype=bool)
    component[2, 2:8] = True
    component[3:7, 2] = True
    component[6, 2:9] = True
    component[3:7, 8] = True
    component[2:5, 7] = True
    before = component.copy()

    filled = fill_component_holes(component, closing_iterations=0)

    assert np.array_equal(component, before)
    assert filled[4, 4]
    assert filled[5, 7]
    assert not filled[0, 0]


def test_real_map_selects_cube_nearest_coffee_table_short_edge_without_mutation(
    tmp_path,
):
    config = load_sysnav_config()
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    room_map = RoomMap.load(config.search.room_map_manifest)
    occupied_before = nav_map.occupied_mask.copy()
    free_before = nav_map.free_mask.copy()
    unknown_before = nav_map.unknown_mask.copy()
    pgm_digest_before = hashlib.sha256(nav_map.image_path.read_bytes()).hexdigest()

    target = np.array([5.312095553755776, 3.624983119154656, 0.4768])
    plan = plan_obstacle_edge_views(
        nav_map,
        room_map,
        target,
        np.array([3.975, 6.825, -1.54]),
        preferred_distance_m=1.4,
        min_distance_m=1.2,
        max_distance_m=1.7,
    )

    selected = plan.selected
    assert plan.room_id == "room_0"
    assert plan.raw_component_cells >= 1300
    assert plan.filled_component_cells >= plan.raw_component_cells
    np.testing.assert_allclose(
        plan.component_bounds_world,
        [[4.975, 3.175], [10.025, 4.125]],
        atol=1e-9,
    )
    # The cube is close to the west short edge, so the robot should observe
    # from west of the table while facing approximately east.
    assert selected.pose[0] < target[0]
    assert abs(selected.pose[1] - target[1]) <= 0.06
    assert abs(selected.pose[2]) <= np.deg2rad(3.0)
    assert 1.2 <= selected.target_distance_m <= 1.7
    assert selected.obstacle_clearance_m >= 0.45
    assert selected.edge_distance_m <= 0.35

    assert np.array_equal(nav_map.occupied_mask, occupied_before)
    assert np.array_equal(nav_map.free_mask, free_before)
    assert np.array_equal(nav_map.unknown_mask, unknown_before)
    assert hashlib.sha256(nav_map.image_path.read_bytes()).hexdigest() == pgm_digest_before

    debug_path = save_obstacle_edge_view_debug(
        tmp_path / "round_0001.png",
        nav_map,
        plan,
    )
    debug = cv2.imread(str(debug_path), cv2.IMREAD_COLOR)
    assert debug is not None
    assert debug.shape[:2] == (nav_map.height * 2, nav_map.width * 2)


def test_second_edge_view_avoids_the_first_view_side():
    config = load_sysnav_config()
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    room_map = RoomMap.load(config.search.room_map_manifest)
    target = np.array([5.3121, 3.6250, 0.4768])
    first = plan_obstacle_edge_views(
        nav_map,
        room_map,
        target,
        np.array([3.975, 6.825, -1.54]),
    )
    second = plan_obstacle_edge_views(
        nav_map,
        room_map,
        target,
        np.asarray(first.selected.pose),
        previous_observation_poses=[first.selected.pose],
        preferred_angle_separation_deg=60.0,
    )

    assert second.selected.minimum_prior_view_angle_rad >= np.deg2rad(60.0)
    assert not np.allclose(second.selected.pose[:2], first.selected.pose[:2])
