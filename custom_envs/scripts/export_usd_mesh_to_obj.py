#!/usr/bin/env python3
"""Export one USD mesh to a triangulated OBJ.

Run this script with the IsaacLab Python environment, which provides ``pxr``::

    /home/mojie/anaconda3/envs/env_isaaclab/bin/python \
        custom_envs/scripts/export_usd_mesh_to_obj.py INPUT.usd OUTPUT.obj

The mesh's local-to-world transform is baked into the OBJ vertices. Vertex and
face-varying ``st`` texture coordinates are preserved when present. Materials
are intentionally not exported: MuJoCo assigns the visual material in MJCF.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _select_mesh(stage, prim_path: str | None):
    from pxr import UsdGeom

    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"{prim_path!r} is not a USD Mesh prim")
        return UsdGeom.Mesh(prim)

    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        paths = ", ".join(str(mesh.GetPath()) for mesh in meshes) or "none"
        raise ValueError(
            f"expected exactly one mesh, found {len(meshes)} ({paths}); "
            "pass --mesh-prim explicitly"
        )
    return meshes[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_usd", type=Path)
    parser.add_argument("output_obj", type=Path)
    parser.add_argument("--mesh-prim", default=None,
                        help="USD mesh prim path; auto-detected when the stage has one mesh")
    parser.add_argument(
        "--translate", nargs=3, type=float, default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="translation baked into exported vertices after the USD world transform",
    )
    args = parser.parse_args()

    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise SystemExit(
            "pxr is unavailable; run with "
            "/home/mojie/anaconda3/envs/env_isaaclab/bin/python"
        ) from exc

    stage = Usd.Stage.Open(str(args.input_usd))
    if stage is None:
        raise SystemExit(f"failed to open USD stage: {args.input_usd}")
    mesh = _select_mesh(stage, args.mesh_prim)

    points = mesh.GetPointsAttr().Get() or []
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if not points or not counts or sum(counts) != len(indices):
        raise SystemExit(f"invalid or empty mesh at {mesh.GetPath()}")

    transform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translate = tuple(float(value) for value in args.translate)
    world_points = [
        transform.Transform(point) + translate
        for point in points
    ]

    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uv_values = list(st.ComputeFlattened() or []) if st else []
    uv_interp = st.GetInterpolation() if st else None

    # Keep the original polygon corner metadata, then triangulate every polygon
    # with a fan. The current orange asset is already all triangles, but this
    # also makes the exporter reusable for other single-mesh USD objects.
    polygons: list[list[tuple[int, int | None]]] = []
    cursor = 0
    for face_index, count in enumerate(counts):
        corners: list[tuple[int, int | None]] = []
        for local_corner in range(count):
            point_index = int(indices[cursor + local_corner])
            uv_index = None
            if uv_values:
                if uv_interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
                    uv_index = point_index
                elif uv_interp == UsdGeom.Tokens.faceVarying:
                    uv_index = cursor + local_corner
                elif uv_interp == UsdGeom.Tokens.uniform:
                    uv_index = face_index
                elif uv_interp == UsdGeom.Tokens.constant:
                    uv_index = 0
            corners.append((point_index, uv_index))
        cursor += count
        if len(corners) >= 3:
            for i in range(1, len(corners) - 1):
                polygons.append([corners[0], corners[i], corners[i + 1]])

    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        polygons = [[tri[0], tri[2], tri[1]] for tri in polygons]

    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    with args.output_obj.open("w", encoding="utf-8") as output:
        output.write(f"# Exported from {args.input_usd}\n")
        output.write(f"# USD mesh prim: {mesh.GetPath()}\n")
        output.write("o visual\n")
        for point in world_points:
            output.write(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
        for uv in uv_values:
            output.write(f"vt {uv[0]:.9f} {uv[1]:.9f}\n")
        for triangle in polygons:
            fields = []
            for point_index, uv_index in triangle:
                if uv_index is None:
                    fields.append(str(point_index + 1))
                else:
                    fields.append(f"{point_index + 1}/{uv_index + 1}")
            output.write("f " + " ".join(fields) + "\n")

    print(
        f"wrote {args.output_obj}: {len(world_points)} vertices, "
        f"{len(uv_values)} UVs, {len(polygons)} triangles"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
