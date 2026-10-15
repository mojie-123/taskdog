#!/usr/bin/env python3
"""Generate an anonymous room map from structural MuJoCo wall geometry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from custom_envs.scripts.map.mujoco_structure import (  # noqa: E402
    detect_portals,
    extract_wall_segments,
)
from custom_envs.scripts.map.nav2_map_io import Nav2Map, load_nav2_map  # noqa: E402
from custom_envs.scripts.map.room_map_io import write_room_map  # noqa: E402
from custom_envs.scripts.map.room_partition import (  # noqa: E402
    RoomPartitionResult,
    build_room_partition,
)


ROOM_COLORS = np.asarray(
    [
        [66, 135, 245],
        [52, 168, 83],
        [251, 188, 4],
        [171, 71, 188],
        [0, 172, 193],
        [244, 81, 30],
    ],
    dtype=np.uint8,
)


def write_debug_preview(
    path: Path,
    nav_map: Nav2Map,
    result: RoomPartitionResult,
    scale: int = 2,
) -> None:
    """Write a coloured inspection image; it is never a runtime input."""

    rgb = np.repeat(nav_map.image[:, :, None], 3, axis=2).astype(np.float32)
    for room in result.rooms:
        label = int(room["label_value"])
        mask = result.room_region_id_map == label
        color = ROOM_COLORS[(label - 1) % len(ROOM_COLORS)].astype(np.float32)
        rgb[mask] = 0.58 * rgb[mask] + 0.42 * color
    rgb[result.wall_mask.astype(bool)] = np.array([20, 20, 20], dtype=np.float32)
    rgb[result.portal_id_map > 0] = np.array([235, 52, 235], dtype=np.float32)

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    scale = max(1, int(scale))
    if scale != 1:
        image = image.resize(
            (nav_map.width * scale, nav_map.height * scale), Image.Resampling.NEAREST
        )
    draw = ImageDraw.Draw(image)
    for room in result.rooms:
        x, y = room["center"]
        row, col = nav_map.world_to_grid(float(x), float(y))
        position = (int(col * scale), int(row * scale))
        text = str(room["id"])
        box = draw.textbbox(position, text)
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text(position, text, fill=(0, 0, 0))
    for portal in result.portals:
        x, y = portal["center"]
        row, col = nav_map.world_to_grid(float(x), float(y))
        position = (int(col * scale), int(row * scale))
        draw.text(position, str(portal["id"]), fill=(255, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate anonymous room IDs from wall_* MuJoCo geoms. "
            "Furniture, task objects and XML comments are not inspected."
        )
    )
    parser.add_argument("--scene", type=Path, required=True, help="MuJoCo scene XML")
    parser.add_argument("--map-yaml", type=Path, required=True, help="Nav2 map YAML")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output prefix; .yaml, .npz and _debug.png are appended",
    )
    parser.add_argument("--door-min-width", type=float, default=0.70)
    parser.add_argument("--door-max-width", type=float, default=1.80)
    parser.add_argument("--line-angle-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--line-offset-tolerance", type=float, default=0.15)
    parser.add_argument("--portal-overlap", type=float, default=0.06)
    parser.add_argument("--barrier-close-cells", type=int, default=1)
    parser.add_argument("--min-room-area", type=float, default=2.0)
    parser.add_argument("--search-clearance", type=float, default=0.52)
    parser.add_argument("--entry-offset", type=float, default=0.85)
    parser.add_argument("--preview-scale", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_xml = args.scene.expanduser().resolve()
    map_yaml = args.map_yaml.expanduser().resolve()
    out_prefix = args.out_prefix.expanduser().resolve()
    manifest_path = out_prefix.with_suffix(".yaml")
    raster_path = out_prefix.with_suffix(".npz")
    preview_path = out_prefix.parent / f"{out_prefix.name}_debug.png"

    nav_map = load_nav2_map(map_yaml)
    walls = extract_wall_segments(scene_xml)
    portals = detect_portals(
        walls,
        min_width=args.door_min_width,
        max_width=args.door_max_width,
        angle_tolerance_degrees=args.line_angle_tolerance_deg,
        line_offset_tolerance=args.line_offset_tolerance,
    )
    result = build_room_partition(
        nav_map,
        walls,
        portals,
        min_room_area_m2=args.min_room_area,
        portal_overlap_m=args.portal_overlap,
        barrier_close_cells=args.barrier_close_cells,
        search_clearance_m=args.search_clearance,
        entry_offset_m=args.entry_offset,
    )
    generation_parameters = {
        "door_min_width_m": float(args.door_min_width),
        "door_max_width_m": float(args.door_max_width),
        "line_angle_tolerance_deg": float(args.line_angle_tolerance_deg),
        "line_offset_tolerance_m": float(args.line_offset_tolerance),
        "portal_overlap_m": float(args.portal_overlap),
        "barrier_close_cells": int(args.barrier_close_cells),
        "min_room_area_m2": float(args.min_room_area),
        "search_clearance_m": float(args.search_clearance),
        "entry_offset_m": float(args.entry_offset),
    }
    write_room_map(
        manifest_path,
        raster_path,
        result,
        scene_xml=scene_xml,
        nav2_map_yaml=map_yaml,
        resolution=nav_map.resolution,
        origin=nav_map.origin,
        structure_sources=[wall.name for wall in walls],
        generation_parameters=generation_parameters,
    )
    write_debug_preview(preview_path, nav_map, result, scale=args.preview_scale)

    internal_portals = sum(not bool(portal["is_exterior"]) for portal in result.portals)
    print(f"[room-map] structural walls: {len(walls)}")
    print(f"[room-map] detected portals: {len(portals)} ({internal_portals} internal)")
    print(f"[room-map] anonymous rooms: {len(result.rooms)}")
    print(f"[room-map] manifest: {manifest_path}")
    print(f"[room-map] raster:   {raster_path}")
    print(f"[room-map] preview:  {preview_path}")


if __name__ == "__main__":
    main()
