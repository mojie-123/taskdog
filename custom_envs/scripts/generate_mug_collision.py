#!/usr/bin/env python3
"""Generate a hollow MuJoCo collision shell for the red mug.

The complete mug OBJ remains visual-only.  This script creates an MJCF fragment
made from convex primitives: a 24-box main wall, a 24-box flared rim, and one
thin cylinder bottom.  The handle is deliberately visual-only for the first
MoveIt validation.  Because the wall is a ring rather than a solid cylinder,
one gripper finger can enter the cup while the other remains outside.

The input visual OBJ is expected to have the cup-body axis at X=Y=0 and its
lowest visual point at Z=0.  ``export_usd_mesh_to_obj.py --translate`` is used
to establish that coordinate frame for the current asset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL = ROOT / "custom_envs/mujoco/meshes/mug/textured.obj"
DEFAULT_FRAGMENT = ROOT / "custom_envs/mujoco/meshes/mug/collision_geoms.xml"
DEFAULT_SCENE = ROOT / "custom_envs/mujoco/scene.xml"
BEGIN = "<!-- MUG_COLLISION_BEGIN -->"
END = "<!-- MUG_COLLISION_END -->"


def read_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def ring_boxes(
    *,
    prefix: str,
    count: int,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    friction: str,
) -> list[str]:
    center_radius = (inner_radius + outer_radius) / 2.0
    radial_half = (outer_radius - inner_radius) / 2.0
    # Tangential boxes overlap by 5%, preventing cracks between adjacent convex
    # geoms while keeping the inner cavity open.
    tangential_half = outer_radius * math.tan(math.pi / count) * 1.05
    z_center = (z_min + z_max) / 2.0
    z_half = (z_max - z_min) / 2.0

    lines = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        x = center_radius * math.cos(theta)
        y = center_radius * math.sin(theta)
        qw = math.cos(theta / 2.0)
        qz = math.sin(theta / 2.0)
        lines.append(
            f'      <geom name="{prefix}_{index:02d}" type="box" group="3" '
            f'pos="{x:.7f} {y:.7f} {z_center:.7f}" '
            f'quat="{qw:.8f} 0 0 {qz:.8f}" '
            f'size="{radial_half:.7f} {tangential_half:.7f} {z_half:.7f}" '
            f'mass="0" friction="{friction}" '
            f'solref="0.004 1" solimp="0.99 0.999 0.001" '
            f'rgba="0.55 0.02 0.02 1" />'
        )
    return lines


def build_fragment(args, visual_height: float) -> tuple[str, dict[str, float]]:
    main_outer = args.source_body_outer_radius * args.xy_scale
    main_inner = main_outer - args.wall_thickness
    rim_outer = args.source_rim_outer_radius * args.xy_scale
    rim_inner = rim_outer - args.rim_thickness
    if min(main_inner, rim_inner) <= 0.0:
        raise ValueError("wall/rim thickness leaves no inner cavity")
    if args.bottom_thickness + args.rim_height >= visual_height:
        raise ValueError("bottom plus rim height exceeds the scaled mug height")

    rim_z_min = visual_height - args.rim_height
    lines = [BEGIN]
    lines.append(
        f'      <geom name="mug_bottom_geom" type="cylinder" group="3" '
        f'pos="0 0 {args.bottom_thickness / 2.0:.7f}" '
        f'size="{main_inner:.7f} {args.bottom_thickness / 2.0:.7f}" '
        f'mass="0" friction="{args.friction}" condim="6" '
        f'solref="0.004 1" solimp="0.99 0.999 0.001" '
        f'rgba="0.55 0.02 0.02 1" />'
    )
    lines.extend(ring_boxes(
        prefix="mug_wall",
        count=args.segments,
        inner_radius=main_inner,
        outer_radius=main_outer,
        z_min=args.bottom_thickness,
        z_max=rim_z_min,
        friction=args.friction,
    ))
    lines.extend(ring_boxes(
        prefix="mug_rim",
        count=args.segments,
        inner_radius=rim_inner,
        outer_radius=rim_outer,
        z_min=rim_z_min,
        z_max=visual_height,
        friction=args.friction,
    ))
    lines.append(END)
    values = {
        "main_inner": main_inner,
        "main_outer": main_outer,
        "rim_inner": rim_inner,
        "rim_outer": rim_outer,
        "height": visual_height,
        "rim_z_min": rim_z_min,
    }
    return "\n".join(lines) + "\n", values


def update_scene(path: Path, fragment: str) -> None:
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^[ \t]*{re.escape(BEGIN)}.*?^[ \t]*{re.escape(END)}[ \t]*$",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(source):
        raise ValueError(f"{path} does not contain {BEGIN} / {END}")
    path.write_text(pattern.sub(fragment.rstrip(), source), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--update-scene", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--xy-scale", type=float, default=0.62,
                        help="visual/collision scale in X and Y")
    parser.add_argument("--z-scale", type=float, default=0.72,
                        help="visual scale in Z; kept larger to preserve side-wall height")
    parser.add_argument("--source-body-outer-radius", type=float, default=0.0410,
                        help="main body outer radius in the unscaled source mesh (m)")
    parser.add_argument("--source-rim-outer-radius", type=float, default=0.0461,
                        help="flared rim outer radius in the unscaled source mesh (m)")
    parser.add_argument("--wall-thickness", type=float, default=0.0030,
                        help="physical collision wall thickness after scaling (m)")
    parser.add_argument("--rim-thickness", type=float, default=0.0050,
                        help="physical collision rim thickness after scaling (m)")
    parser.add_argument("--bottom-thickness", type=float, default=0.0030)
    parser.add_argument("--rim-height", type=float, default=0.0045)
    parser.add_argument("--friction", default="1.2 0.01 0.01")
    args = parser.parse_args()

    if args.segments < 8:
        raise SystemExit("--segments must be at least 8")
    vertices = read_vertices(args.visual)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    if abs(lower[2]) > 1e-5:
        raise SystemExit(
            f"visual OBJ is not grounded (min z={lower[2]:.6f}); "
            "export it in the normalized mug frame"
        )
    visual_height = float(upper[2] - lower[2]) * args.z_scale
    fragment, values = build_fragment(args, visual_height)

    print(f"visual source bbox (mm): {(upper - lower) * 1000.0}")
    print(f"scaled visual height (mm): {values['height'] * 1000.0:.2f}")
    print(
        "main wall inner/outer diameter (mm): "
        f"{2.0 * values['main_inner'] * 1000.0:.2f} / "
        f"{2.0 * values['main_outer'] * 1000.0:.2f}"
    )
    print(
        "rim inner/outer diameter (mm): "
        f"{2.0 * values['rim_inner'] * 1000.0:.2f} / "
        f"{2.0 * values['rim_outer'] * 1000.0:.2f}"
    )
    print(f"collision geoms: {1 + 2 * args.segments}")

    if args.dry_run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")
    print(f"wrote {args.output}")
    if args.update_scene:
        update_scene(args.scene, fragment)
        print(f"updated {args.scene}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
