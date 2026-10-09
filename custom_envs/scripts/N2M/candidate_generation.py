"""Generate deterministic manipulation-base candidates from one reference pose."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from custom_envs.scripts.N2M.config import (
    CUBE_SCENES,
    EDGES,
    REFERENCE_EDGE_ID,
    REFERENCE_SCENE_ID,
    CandidateGrid,
)
from custom_envs.scripts.N2M.geometry import (
    joint1_towards_cube,
    matrix_to_se2,
    se2_to_matrix,
    wrap_angle,
)


TABLE_BOUNDS = {
    "x_min": 5.0,
    "x_max": 10.0,
    "y_min": 3.2,
    "y_max": 4.1,
}


def edge_projection(cube_xyz: Iterable[float], edge_id: str) -> np.ndarray:
    cube = np.asarray(cube_xyz, dtype=np.float64)
    if edge_id == "left":
        return np.array([TABLE_BOUNDS["x_min"], cube[1]], dtype=np.float64)
    if edge_id == "right":
        return np.array([TABLE_BOUNDS["x_max"], cube[1]], dtype=np.float64)
    if edge_id == "bottom":
        return np.array([cube[0], TABLE_BOUNDS["y_min"]], dtype=np.float64)
    if edge_id == "top":
        return np.array([cube[0], TABLE_BOUNDS["y_max"]], dtype=np.float64)
    raise KeyError(f"Unknown edge: {edge_id}")


def edge_frame_world(cube_xyz: Iterable[float], edge_id: str) -> np.ndarray:
    """Return ^W T_E with +x_E pointing out of the table.

    +y_E is chosen as z cross +x_E, so the frame remains right handed.
    """
    origin = edge_projection(cube_xyz, edge_id)
    outward = np.asarray(EDGES[edge_id].outward_xy, dtype=np.float64)
    yaw = math.atan2(outward[1], outward[0])
    return se2_to_matrix((origin[0], origin[1], yaw))


def reference_pose_in_edge(reference_base: Sequence[float]) -> np.ndarray:
    ref_scene = CUBE_SCENES[REFERENCE_SCENE_ID]
    T_w_edge = edge_frame_world(ref_scene.cube_xyz, REFERENCE_EDGE_ID)
    return np.linalg.inv(T_w_edge) @ se2_to_matrix(reference_base)


def nominal_pose_for_edge(
    reference_base: Sequence[float],
    cube_xyz: Sequence[float],
    edge_id: str,
) -> np.ndarray:
    T_edge_m = reference_pose_in_edge(reference_base)
    return matrix_to_se2(edge_frame_world(cube_xyz, edge_id) @ T_edge_m)


def _candidate_id(scene_id: str, edge_id: str, index: int) -> str:
    return f"{scene_id}__{edge_id}__m{index:04d}"


def generate_candidates(
    reference_base: Sequence[float],
    scene_ids: Sequence[str] | None = None,
    grid: CandidateGrid | None = None,
) -> List[dict]:
    grid = grid or CandidateGrid()
    if any(float(offset) > 0.0 for offset in grid.normal_offsets):
        raise ValueError(
            "normal offsets must be <= 0: zero is the reference M and "
            "negative values move away from the table"
        )
    if scene_ids is None:
        scene_ids = list(CUBE_SCENES)

    candidates: List[dict] = []
    for scene_id in scene_ids:
        if scene_id not in CUBE_SCENES:
            raise KeyError(f"Unknown cube scene {scene_id!r}")
        scene = CUBE_SCENES[scene_id]
        scene_index = 0
        for edge_id in scene.approach_edges:
            nominal = nominal_pose_for_edge(reference_base, scene.cube_xyz, edge_id)
            outward = np.asarray(EDGES[edge_id].outward_xy, dtype=np.float64)
            inward = -outward
            tangent = np.array([-outward[1], outward[0]], dtype=np.float64)

            # Keep local variants adjacent: nominal first, then outward/inward
            # alternatives for the same tangent/yaw.  This makes a bounded
            # --max-candidates search useful and reaches collision-free
            # outward poses early.
            for dt in grid.tangent_offsets:
                for dyaw_deg in grid.yaw_offsets_deg:
                    for dn in grid.normal_offsets:
                        # User-facing normal convention: positive points into
                        # the tabletop, negative points away from it.
                        xy = nominal[:2] + float(dn) * inward + float(dt) * tangent
                        yaw = wrap_angle(nominal[2] + math.radians(float(dyaw_deg)))
                        base_pose = np.array([xy[0], xy[1], yaw], dtype=np.float64)
                        j1 = joint1_towards_cube(base_pose, scene.cube_xyz)
                        candidates.append({
                            "candidate_id": _candidate_id(scene_id, edge_id, scene_index),
                            "scene_id": scene_id,
                            "category": scene.category,
                            "approach_edge": edge_id,
                            "cube_pose": list(scene.cube_xyz) + [1.0, 0.0, 0.0, 0.0],
                            "base_pose": base_pose.tolist(),
                            "joint1_target": float(j1),
                            "source": "reference_edge_transfer",
                            "reference_scene_id": REFERENCE_SCENE_ID,
                            "reference_edge_id": REFERENCE_EDGE_ID,
                            "reference_base_pose": [float(x) for x in reference_base],
                            "nominal_base_pose": nominal.tolist(),
                            "perturbation": {
                                "normal_m": float(dn),
                                "tangent_m": float(dt),
                                "yaw_deg": float(dyaw_deg),
                            },
                            "grid": asdict(grid),
                        })
                        scene_index += 1
    return candidates


def select_candidates(
    candidates: Sequence[dict],
    candidate_id: str | None = None,
    max_candidates: int | None = None,
) -> List[dict]:
    selected = list(candidates)
    if candidate_id:
        selected = [row for row in selected if row["candidate_id"] == candidate_id]
        if not selected:
            raise KeyError(f"Candidate not found: {candidate_id}")
    if max_candidates is not None:
        selected = selected[:max(0, int(max_candidates))]
    return selected
