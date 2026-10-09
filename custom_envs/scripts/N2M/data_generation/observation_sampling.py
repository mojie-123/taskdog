"""Deterministic, stratified NAV-end observation-pose proposals."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from custom_envs.scripts.N2M.config import CUBE_SCENES, EDGES
from custom_envs.scripts.N2M.geometry import wrap_angle


def propose_observation_poses(
    scene_id: str,
    ranges_m: Sequence[float],
    tangent_offsets_m: Sequence[float],
    yaw_offsets_deg: Sequence[float],
) -> list[dict]:
    """Propose base poses outside each approach edge, looking toward the cube."""
    if scene_id not in CUBE_SCENES:
        raise KeyError(f"unknown cube scene: {scene_id}")
    scene = CUBE_SCENES[scene_id]
    cube_xy = np.asarray(scene.cube_xyz[:2], dtype=np.float64)
    rows: list[dict] = []
    for edge_id in scene.approach_edges:
        outward = np.asarray(EDGES[edge_id].outward_xy, dtype=np.float64)
        tangent = np.array([-outward[1], outward[0]], dtype=np.float64)
        combinations = itertools.product(ranges_m, tangent_offsets_m, yaw_offsets_deg)
        for range_m, tangent_m, yaw_deg in combinations:
            if float(range_m) <= 0.0:
                raise ValueError("observation range must be positive")
            position = cube_xy + outward * float(range_m) + tangent * float(tangent_m)
            cube_bearing = math.atan2(cube_xy[1] - position[1], cube_xy[0] - position[0])
            yaw = wrap_angle(cube_bearing + math.radians(float(yaw_deg)))
            rows.append(
                {
                    "scene_id": scene_id,
                    "approach_edge": edge_id,
                    "observation_pose_world": [float(position[0]), float(position[1]), yaw],
                    "proposal": {
                        "range_m": float(range_m),
                        "tangent_m": float(tangent_m),
                        "yaw_offset_deg": float(yaw_deg),
                    },
                }
            )
    return rows


def select_stratified(rows: Iterable[dict], maximum: int | None, seed: int) -> list[dict]:
    """Select a reproducible subset while preserving edge/parameter coverage."""
    ordered = stratified_order(rows, seed)
    if maximum is None or maximum <= 0:
        return ordered
    return ordered[: int(maximum)]


def _balanced_parameter_order(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Greedily order one edge so every proposal dimension stays balanced.

    Random shuffling alone can make a small pilot contain only near views or a
    narrow yaw range.  This greedy ordering always prefers parameter values
    used least often so far, with a seeded shuffle used only to break ties.
    """
    remaining = list(rows)
    rng.shuffle(remaining)
    counts = {
        "range_m": Counter(),
        "tangent_m": Counter(),
        "yaw_offset_deg": Counter(),
    }
    ordered: list[dict] = []
    while remaining:
        def score(row: dict) -> tuple[int, int]:
            proposal = row["proposal"]
            used = [counts[key][proposal[key]] for key in counts]
            return max(used), sum(used)

        best_index = min(range(len(remaining)), key=lambda index: score(remaining[index]))
        row = remaining.pop(best_index)
        ordered.append(row)
        for key, counter in counts.items():
            counter[row["proposal"][key]] += 1
    return ordered


def stratified_order(rows: Iterable[dict], seed: int) -> list[dict]:
    """Return every proposal in reproducible, edge/parameter-balanced order."""
    rows = list(rows)
    if not rows:
        return []
    rng = np.random.default_rng(int(seed))
    by_edge: dict[str, list[dict]] = {}
    for row in rows:
        by_edge.setdefault(row["approach_edge"], []).append(row)
    edge_orders = {
        edge_id: _balanced_parameter_order(edge_rows, rng)
        for edge_id, edge_rows in sorted(by_edge.items())
    }
    selected: list[dict] = []
    edge_ids = sorted(edge_orders)
    while True:
        progressed = False
        for edge_id in edge_ids:
            if edge_orders[edge_id]:
                selected.append(edge_orders[edge_id].pop(0))
                progressed = True
        if not progressed:
            break
    return selected
