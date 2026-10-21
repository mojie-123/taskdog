#!/usr/bin/env python3
"""Generate a stable, gripper-friendly convex collision mesh for the orange.

The visual orange is almost spherical and 74 mm wide. A raw spherical collision
shape is wider than the 70 mm gripper and tends to roll or receive vertical
force components from the fingers. This generator follows the proven apple
setup: it builds an ellipsoid from the visual bbox, clips X/Y to create vertical
gripping faces, and shapes the lower cap into a shallow dome with a unique
lowest point for stable table contact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "custom_envs/mujoco/meshes/orange/textured.obj"
DEFAULT_DST = ROOT / "custom_envs/mujoco/meshes/orange/collision_flat.obj"


def read_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def ellipsoid(center: np.ndarray, axes: np.ndarray, nlat: int):
    nlon = 2 * nlat
    vertices = [center + np.array([0.0, 0.0, axes[2]])]
    for i in range(1, nlat):
        theta = np.pi * i / nlat
        for j in range(nlon):
            phi = 2.0 * np.pi * j / nlon
            vertices.append(center + np.array([
                axes[0] * np.sin(theta) * np.cos(phi),
                axes[1] * np.sin(theta) * np.sin(phi),
                axes[2] * np.cos(theta),
            ]))
    vertices.append(center + np.array([0.0, 0.0, -axes[2]]))

    def vertex_id(i: int, j: int) -> int:
        # Return a zero-based index for latitude ring i=1..nlat-1.
        return 1 + (i - 1) * nlon + (j % nlon)

    faces = []
    for j in range(nlon):
        faces.append((0, vertex_id(1, j + 1), vertex_id(1, j)))
    for i in range(1, nlat - 1):
        for j in range(nlon):
            a = vertex_id(i, j)
            b = vertex_id(i + 1, j)
            c = vertex_id(i + 1, j + 1)
            d = vertex_id(i, j + 1)
            faces.extend(((a, c, b), (a, d, c)))
    south = len(vertices) - 1
    for j in range(nlon):
        faces.append((south, vertex_id(nlat - 1, j), vertex_id(nlat - 1, j + 1)))
    return np.asarray(vertices, dtype=np.float64), faces


def boundary_points(center: np.ndarray, axes: np.ndarray, half: float,
                    flat_height: float, samples: int = 96) -> np.ndarray:
    """Add exact plane boundaries; MuJoCo forms a convex hull from all vertices."""
    phi = np.arange(samples) * 2.0 * np.pi / samples
    rings = []
    for sign in (1.0, -1.0):
        kx = np.sqrt(max(1.0 - (half / axes[0]) ** 2, 0.0))
        rings.append(np.stack([
            np.full(samples, center[0] + sign * half),
            center[1] + axes[1] * kx * np.sin(phi),
            center[2] + axes[2] * kx * np.cos(phi),
        ], axis=1))
        ky = np.sqrt(max(1.0 - (half / axes[1]) ** 2, 0.0))
        rings.append(np.stack([
            center[0] + axes[0] * ky * np.sin(phi),
            np.full(samples, center[1] + sign * half),
            center[2] + axes[2] * ky * np.cos(phi),
        ], axis=1))

    z = center[2] - axes[2] + flat_height
    kz = np.sqrt(max(1.0 - ((z - center[2]) / axes[2]) ** 2, 0.0))
    rings.append(np.stack([
        center[0] + axes[0] * kz * np.cos(phi),
        center[1] + axes[1] * kz * np.sin(phi),
        np.full(samples, z),
    ], axis=1))
    return np.vstack(rings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output", type=Path, default=DEFAULT_DST)
    parser.add_argument("--half", type=float, default=0.0315,
                        help="unscaled half-width of the X/Y gripping faces (m)")
    parser.add_argument("--flat-height", type=float, default=0.012,
                        help="height above the bbox bottom included in the support cap (m)")
    parser.add_argument("--bottom-dome", type=float, default=0.0025,
                        help="edge rise of the shallow support dome before MJCF scaling (m)")
    parser.add_argument("--scale", type=float, default=0.8,
                        help="MJCF collision scale, used for diagnostics only")
    parser.add_argument("--nlat", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_vertices = read_vertices(args.source)
    lower = source_vertices.min(axis=0)
    upper = source_vertices.max(axis=0)
    center = (lower + upper) / 2.0
    axes = (upper - lower) / 2.0
    if args.half <= 0.0 or args.half >= min(axes[0], axes[1]):
        raise SystemExit(
            f"--half must be between 0 and {min(axes[0], axes[1]):.6f} m"
        )
    if args.flat_height <= 0.0 or args.flat_height >= axes[2]:
        raise SystemExit(f"--flat-height must be between 0 and {axes[2]:.6f} m")

    vertices, faces = ellipsoid(center, axes, args.nlat)
    vertices = np.vstack([
        vertices,
        boundary_points(center, axes, args.half, args.flat_height),
    ])

    for axis in (0, 1):
        offset = vertices[:, axis] - center[axis]
        vertices[:, axis] = center[axis] + np.clip(offset, -args.half, args.half)

    bottom = lower[2]
    support = vertices[:, 2] <= bottom + args.flat_height + 1e-12
    radial = np.hypot(vertices[support, 0] - center[0],
                      vertices[support, 1] - center[1])
    max_radius = float(radial.max())
    vertices[support, 2] = bottom + args.bottom_dome * (radial / max_radius) ** 2

    physical_dims = (vertices.max(axis=0) - vertices.min(axis=0)) * args.scale
    print(f"visual bbox (mm): {(upper - lower) * 1000.0}")
    print(f"collision bbox after x{args.scale:g} (mm): {physical_dims * 1000.0}")
    print(f"gripper clearance per side (mm): {(0.070 - max(physical_dims[:2])) * 500.0:.2f}")
    for axis, other, name in ((0, 1, "X"), (1, 0, "Y")):
        plane = np.isclose(np.abs(vertices[:, axis] - center[axis]), args.half,
                           atol=1e-9)
        tangential = np.ptp(vertices[plane, other]) * args.scale
        vertical = np.ptp(vertices[plane, 2]) * args.scale
        print(f"{name} gripping face after scale (mm): "
              f"{tangential * 1000.0:.2f} x {vertical * 1000.0:.2f}")

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# Convex orange collision mesh: clipped gripping faces + domed support\n")
        output.write("o orange_collision\n")
        for point in vertices:
            output.write(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        for face in faces:
            output.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print(f"wrote {args.output}: {len(vertices)} vertices, {len(faces)} faces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
