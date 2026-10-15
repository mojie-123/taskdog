from pathlib import Path

import numpy as np
import yaml

from custom_envs.scripts.map.mujoco_structure import (
    detect_portals,
    extract_wall_segments,
)
from custom_envs.scripts.map.nav2_map_io import load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap, write_room_map
from custom_envs.scripts.map.room_partition import build_room_partition


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"
NAV_MAP = ROOT / "custom_envs" / "maps" / "map_house_mujoco_nav2.yaml"


def _build_current_house():
    nav_map = load_nav2_map(NAV_MAP)
    walls = extract_wall_segments(SCENE)
    portals = detect_portals(walls, min_width=0.7, max_width=1.8)
    result = build_room_partition(nav_map, walls, portals)
    return nav_map, walls, portals, result


def test_current_house_uses_only_structural_walls_and_finds_anonymous_rooms():
    _nav_map, walls, portals, result = _build_current_house()

    assert len(walls) == 10
    assert all(wall.name.startswith("wall_") for wall in walls)
    assert len(portals) == 4
    assert all(np.isclose(portal.width, 1.2) for portal in portals)
    assert len(result.rooms) == 3
    assert sum(not portal["is_exterior"] for portal in result.portals) == 3
    assert {room["id"] for room in result.rooms} == {"room_0", "room_1", "room_2"}


def test_room_map_round_trip(tmp_path: Path):
    nav_map, walls, _portals, result = _build_current_house()
    manifest = tmp_path / "rooms.yaml"
    raster = tmp_path / "rooms.npz"
    write_room_map(
        manifest,
        raster,
        result,
        scene_xml=SCENE,
        nav2_map_yaml=NAV_MAP,
        resolution=nav_map.resolution,
        origin=nav_map.origin,
        structure_sources=[wall.name for wall in walls],
        generation_parameters={},
    )

    room_map = RoomMap.load(manifest)
    assert room_map.room_id_at_world(8.0, 3.0) == "room_0"
    assert room_map.room_id_at_world(4.0, 10.0) == "room_1"
    assert room_map.room_id_at_world(12.0, 10.0) == "room_2"
    assert room_map.neighbors("room_0") == ["room_1", "room_2"]
    assert room_map.searchable_mask("room_0").any()

    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    serialized = yaml.safe_dump(document).lower()
    for forbidden in ("living_room", "bedroom", "kitchen", "coffee_table", "cube"):
        assert forbidden not in serialized
