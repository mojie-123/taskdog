"""Plan N2M observation poses from the nearest exterior obstacle edge.

The Nav2 raster is an immutable input.  Furniture closing, hole filling and
boundary extraction are performed on derived boolean arrays only; neither the
``Nav2Map`` arrays nor its source PGM are modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage

from custom_envs.scripts.map.nav2_map_io import Nav2Map
from custom_envs.scripts.map.room_map_io import RoomMap


class ObstacleEdgePlanningError(RuntimeError):
    """Raised when no trustworthy obstacle-edge observation pose exists."""


def _angular_separation(first: float, second: float) -> float:
    return abs((float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi)


def _grid_points_world(
    nav_map: Nav2Map, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    x = nav_map.origin[0] + (cols + 0.5) * nav_map.resolution
    y = (
        nav_map.origin[1]
        + (float(nav_map.height - 1) - rows + 0.5) * nav_map.resolution
    )
    return np.column_stack((x, y))


def fill_component_holes(
    component_mask: np.ndarray, *, closing_iterations: int = 1
) -> np.ndarray:
    """Return a closed, hole-filled copy of one obstacle component."""

    source = np.asarray(component_mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("component_mask must be a 2-D array")
    if closing_iterations < 0:
        raise ValueError("closing_iterations must be non-negative")
    working = source.copy()
    if closing_iterations:
        working = ndimage.binary_closing(
            working,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(closing_iterations),
            border_value=0,
        )
    # Padding provides an unambiguous exterior seed without assuming anything
    # about the component's (possibly irregular) shape.
    padded = np.pad(working, 1, mode="constant", constant_values=False)
    filled = ndimage.binary_fill_holes(padded)
    return np.asarray(filled[1:-1, 1:-1], dtype=bool)


@dataclass(frozen=True)
class ObstacleEdgeViewCandidate:
    pose: tuple[float, float, float]
    grid_cell: tuple[int, int]
    edge_point_world: tuple[float, float]
    edge_distance_m: float
    target_distance_m: float
    obstacle_clearance_m: float
    travel_distance_m: float
    view_side_angle_rad: float
    minimum_prior_view_angle_rad: float
    score: float

    def to_dict(self) -> dict:
        return {
            "pose": list(self.pose),
            "grid_cell": list(self.grid_cell),
            "edge_point_world": list(self.edge_point_world),
            "edge_distance_m": float(self.edge_distance_m),
            "target_distance_m": float(self.target_distance_m),
            "obstacle_clearance_m": float(self.obstacle_clearance_m),
            "travel_distance_m": float(self.travel_distance_m),
            "view_side_angle_deg": math.degrees(self.view_side_angle_rad),
            "minimum_prior_view_angle_deg": math.degrees(
                self.minimum_prior_view_angle_rad
            ),
            "score": float(self.score),
        }


@dataclass(frozen=True)
class ObstacleEdgeViewPlan:
    target_position_world: tuple[float, float, float]
    target_grid_cell: tuple[int, int]
    room_id: str
    component_label: int
    raw_component_cells: int
    filled_component_cells: int
    component_area_m2: float
    component_bounds_world: tuple[tuple[float, float], tuple[float, float]]
    selected: ObstacleEdgeViewCandidate
    candidates: tuple[ObstacleEdgeViewCandidate, ...]
    preferred_angle_separation_deg: float
    used_angle_fallback: bool
    raw_component_mask: np.ndarray
    filled_component_mask: np.ndarray
    outer_boundary_mask: np.ndarray

    def to_dict(self, *, candidate_limit: int = 50) -> dict:
        return {
            "schema_version": 1,
            "planner": "obstacle_edge_view",
            "map_mutation": "none_derived_masks_only",
            "target_position_world": list(self.target_position_world),
            "target_grid_cell": list(self.target_grid_cell),
            "room_id": self.room_id,
            "component_label": int(self.component_label),
            "raw_component_cells": int(self.raw_component_cells),
            "filled_component_cells": int(self.filled_component_cells),
            "component_area_m2": float(self.component_area_m2),
            "component_bounds_world": [
                list(self.component_bounds_world[0]),
                list(self.component_bounds_world[1]),
            ],
            "preferred_angle_separation_deg": float(
                self.preferred_angle_separation_deg
            ),
            "used_angle_fallback": bool(self.used_angle_fallback),
            "candidate_count": len(self.candidates),
            "selected": self.selected.to_dict(),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates[: max(0, int(candidate_limit))]
            ],
        }


def _select_support_component(
    nav_map: Nav2Map,
    target_xy: np.ndarray,
    *,
    search_radius_m: float,
    minimum_area_m2: float,
    maximum_area_m2: float,
    maximum_extent_m: float,
    closing_iterations: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    occupied = np.asarray(nav_map.occupied_mask, dtype=bool)
    labels, label_count = ndimage.label(
        occupied,
        structure=np.ones((3, 3), dtype=bool),
    )
    if label_count == 0:
        raise ObstacleEdgePlanningError("occupancy map has no obstacle components")

    target_row, target_col = nav_map.world_to_grid(*target_xy)
    if not (0 <= target_row < nav_map.height and 0 <= target_col < nav_map.width):
        raise ObstacleEdgePlanningError("target lies outside the Nav2 raster")

    radius_cells = max(1, int(math.ceil(search_radius_m / nav_map.resolution)))
    row_min = max(0, target_row - radius_cells)
    row_max = min(nav_map.height, target_row + radius_cells + 1)
    col_min = max(0, target_col - radius_cells)
    col_max = min(nav_map.width, target_col + radius_cells + 1)
    nearby_labels = np.unique(labels[row_min:row_max, col_min:col_max])
    nearby_labels = nearby_labels[nearby_labels > 0]
    target_label = int(labels[target_row, target_col])

    options: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for label_value in nearby_labels:
        raw = labels == int(label_value)
        raw_count = int(raw.sum())
        area = raw_count * nav_map.resolution**2
        if area < minimum_area_m2 or area > maximum_area_m2:
            continue
        rows, cols = np.nonzero(raw)
        if len(rows) == 0:
            continue
        extent_x = (float(cols.max() - cols.min()) + 1.0) * nav_map.resolution
        extent_y = (float(rows.max() - rows.min()) + 1.0) * nav_map.resolution
        if max(extent_x, extent_y) > maximum_extent_m:
            continue
        if (
            rows.min() == 0
            or rows.max() == nav_map.height - 1
            or cols.min() == 0
            or cols.max() == nav_map.width - 1
        ):
            continue

        filled = fill_component_holes(
            raw,
            closing_iterations=closing_iterations,
        )
        filled_rows, filled_cols = np.nonzero(filled)
        filled_world = _grid_points_world(nav_map, filled_rows, filled_cols)
        distance = float(np.linalg.norm(filled_world - target_xy, axis=1).min())
        contains_target = bool(filled[target_row, target_col])
        # Prefer a component whose filled footprint contains the target, then
        # use geometric distance.  The tiny bias keeps the choice deterministic.
        cost = distance + (0.0 if contains_target else search_radius_m)
        if int(label_value) == target_label:
            cost -= 1e-6
        options.append((cost, int(label_value), raw, filled))

    if not options:
        raise ObstacleEdgePlanningError(
            "no furniture-sized obstacle component was found near the target"
        )
    _, label_value, raw, filled = min(options, key=lambda item: item[0])
    return label_value, raw, filled


def plan_obstacle_edge_views(
    nav_map: Nav2Map,
    room_map: RoomMap,
    target_position_world: Iterable[float],
    current_robot_pose: Iterable[float],
    *,
    previous_observation_poses: Iterable[Iterable[float]] = (),
    preferred_distance_m: float = 1.4,
    min_distance_m: float = 1.2,
    max_distance_m: float = 1.7,
    preferred_angle_separation_deg: float = 60.0,
    minimum_clearance_m: float = 0.45,
    component_search_radius_m: float = 0.75,
    minimum_component_area_m2: float = 0.15,
    maximum_component_area_m2: float = 30.0,
    maximum_component_extent_m: float = 8.0,
    closing_iterations: int = 1,
) -> ObstacleEdgeViewPlan:
    """Return ranked, collision-clear views around the target's support obstacle."""

    target = np.asarray(tuple(target_position_world), dtype=np.float64).reshape(3)
    current = np.asarray(tuple(current_robot_pose), dtype=np.float64).reshape(3)
    previous = [
        np.asarray(tuple(pose), dtype=np.float64).reshape(3)
        for pose in previous_observation_poses
    ]
    if not 0.0 < min_distance_m <= preferred_distance_m <= max_distance_m:
        raise ValueError("observation distances must satisfy 0 < min <= preferred <= max")
    if minimum_clearance_m < 0.0:
        raise ValueError("minimum_clearance_m must be non-negative")

    # Snapshot the read-only inputs so accidental mutation is detected even if
    # this function is changed later.
    occupied_before = np.asarray(nav_map.occupied_mask, dtype=bool).copy()
    free_before = np.asarray(nav_map.free_mask, dtype=bool).copy()
    unknown_before = np.asarray(nav_map.unknown_mask, dtype=bool).copy()

    room_id = room_map.room_id_at_world(float(target[0]), float(target[1]))
    if room_id is None:
        raise ObstacleEdgePlanningError("target lies outside the anonymous room map")
    target_grid = nav_map.world_to_grid(float(target[0]), float(target[1]))
    component_label, raw_component, filled_component = _select_support_component(
        nav_map,
        target[:2],
        search_radius_m=float(component_search_radius_m),
        minimum_area_m2=float(minimum_component_area_m2),
        maximum_area_m2=float(maximum_component_area_m2),
        maximum_extent_m=float(maximum_component_extent_m),
        closing_iterations=int(closing_iterations),
    )

    eroded = ndimage.binary_erosion(
        filled_component,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    outer_boundary = filled_component & ~eroded
    boundary_rows, boundary_cols = np.nonzero(outer_boundary)
    if len(boundary_rows) == 0:
        raise ObstacleEdgePlanningError("filled obstacle component has no boundary")
    boundary_world = _grid_points_world(nav_map, boundary_rows, boundary_cols)

    # The filled footprint is only a private planning obstacle.  It prevents a
    # speckled free pixel inside the tabletop from becoming an observation pose.
    planning_obstacles = (
        occupied_before | unknown_before | filled_component
    )
    clearance = (
        ndimage.distance_transform_edt(~planning_obstacles) * nav_map.resolution
    )
    searchable = room_map.searchable_mask(room_id)
    prior_angles = [
        math.atan2(float(pose[1] - target[1]), float(pose[0] - target[0]))
        for pose in previous
    ]

    by_cell: dict[tuple[int, int], ObstacleEdgeViewCandidate] = {}
    for edge_xy in boundary_world:
        edge_offset = edge_xy - target[:2]
        edge_distance = float(np.linalg.norm(edge_offset))
        if edge_distance <= nav_map.resolution * 0.25:
            continue
        direction = edge_offset / edge_distance
        nominal_xy = target[:2] + direction * float(preferred_distance_m)
        row, col = nav_map.world_to_grid(float(nominal_xy[0]), float(nominal_xy[1]))
        if not (0 <= row < nav_map.height and 0 <= col < nav_map.width):
            continue
        if not free_before[row, col] or planning_obstacles[row, col]:
            continue
        if not searchable[row, col]:
            continue
        candidate_xy = np.asarray(nav_map.grid_to_world(row, col), dtype=np.float64)
        if room_map.room_id_at_world(*candidate_xy) != room_id:
            continue
        target_distance = float(np.linalg.norm(candidate_xy - target[:2]))
        if not float(min_distance_m) <= target_distance <= float(max_distance_m):
            continue
        candidate_clearance = float(clearance[row, col])
        if candidate_clearance < float(minimum_clearance_m):
            continue

        view_angle = math.atan2(
            float(candidate_xy[1] - target[1]),
            float(candidate_xy[0] - target[0]),
        )
        if prior_angles:
            minimum_prior = min(
                _angular_separation(view_angle, prior) for prior in prior_angles
            )
        else:
            minimum_prior = math.pi
        travel = float(np.linalg.norm(candidate_xy - current[:2]))
        radial_error = abs(target_distance - float(preferred_distance_m))
        score = (
            -6.0 * edge_distance
            - 2.0 * radial_error
            + 0.40 * min(candidate_clearance, 1.5)
            - 0.05 * travel
            + (0.50 * minimum_prior if prior_angles else 0.0)
        )
        yaw = math.atan2(
            float(target[1] - candidate_xy[1]),
            float(target[0] - candidate_xy[0]),
        )
        candidate = ObstacleEdgeViewCandidate(
            pose=(float(candidate_xy[0]), float(candidate_xy[1]), float(yaw)),
            grid_cell=(int(row), int(col)),
            edge_point_world=(float(edge_xy[0]), float(edge_xy[1])),
            edge_distance_m=edge_distance,
            target_distance_m=target_distance,
            obstacle_clearance_m=candidate_clearance,
            travel_distance_m=travel,
            view_side_angle_rad=view_angle,
            minimum_prior_view_angle_rad=minimum_prior,
            score=score,
        )
        existing = by_cell.get((row, col))
        if existing is None or candidate.score > existing.score:
            by_cell[(row, col)] = candidate

    candidates = list(by_cell.values())
    if not candidates:
        raise ObstacleEdgePlanningError(
            "the support obstacle has no free, clear observation pose"
        )

    requested_separation = math.radians(float(preferred_angle_separation_deg))
    preferred = [
        candidate
        for candidate in candidates
        if candidate.minimum_prior_view_angle_rad >= requested_separation
    ]
    used_angle_fallback = bool(prior_angles and not preferred)
    pool = preferred if preferred else candidates
    selected = max(pool, key=lambda candidate: candidate.score)
    ranked = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate is selected,
                candidate.minimum_prior_view_angle_rad >= requested_separation,
                candidate.score,
            ),
            reverse=True,
        )
    )

    component_rows, component_cols = np.nonzero(filled_component)
    component_world = _grid_points_world(nav_map, component_rows, component_cols)
    bounds = (
        (float(component_world[:, 0].min()), float(component_world[:, 1].min())),
        (float(component_world[:, 0].max()), float(component_world[:, 1].max())),
    )

    if not (
        np.array_equal(occupied_before, nav_map.occupied_mask)
        and np.array_equal(free_before, nav_map.free_mask)
        and np.array_equal(unknown_before, nav_map.unknown_mask)
    ):
        raise RuntimeError("obstacle-edge planning mutated the source Nav2 map")

    return ObstacleEdgeViewPlan(
        target_position_world=tuple(float(value) for value in target),
        target_grid_cell=(int(target_grid[0]), int(target_grid[1])),
        room_id=str(room_id),
        component_label=int(component_label),
        raw_component_cells=int(raw_component.sum()),
        filled_component_cells=int(filled_component.sum()),
        component_area_m2=float(filled_component.sum() * nav_map.resolution**2),
        component_bounds_world=bounds,
        selected=selected,
        candidates=ranked,
        preferred_angle_separation_deg=float(preferred_angle_separation_deg),
        used_angle_fallback=used_angle_fallback,
        raw_component_mask=raw_component,
        filled_component_mask=filled_component,
        outer_boundary_mask=outer_boundary,
    )


def save_obstacle_edge_view_debug(
    path: Path | str,
    nav_map: Nav2Map,
    plan: ObstacleEdgeViewPlan,
    *,
    previous_observation_poses: Iterable[Iterable[float]] = (),
    scale: int = 2,
) -> Path:
    """Save a diagnostic overlay without changing any runtime map arrays."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.repeat(nav_map.image[:, :, None], 3, axis=2).astype(np.float32)
    filled = plan.filled_component_mask
    boundary = plan.outer_boundary_mask
    rgb[filled] = 0.55 * rgb[filled] + 0.45 * np.array([255, 165, 0])
    rgb[boundary] = np.array([0, 200, 255])
    image = np.clip(rgb, 0, 255).astype(np.uint8)

    def pixel(xy: Iterable[float]) -> tuple[int, int]:
        point = tuple(xy)
        row, col = nav_map.world_to_grid(float(point[0]), float(point[1]))
        return int(col), int(row)

    for candidate in plan.candidates[:40]:
        cv2.circle(image, pixel(candidate.pose[:2]), 2, (40, 90, 230), -1)
    for pose in previous_observation_poses:
        cv2.circle(image, pixel(tuple(pose)[:2]), 4, (220, 50, 220), -1)

    target_pixel = pixel(plan.target_position_world[:2])
    edge_pixel = pixel(plan.selected.edge_point_world)
    selected_pixel = pixel(plan.selected.pose[:2])
    cv2.circle(image, target_pixel, 5, (255, 0, 0), -1)
    cv2.circle(image, edge_pixel, 4, (255, 255, 0), -1)
    cv2.circle(image, selected_pixel, 6, (0, 180, 0), -1)
    cv2.arrowedLine(image, target_pixel, edge_pixel, (255, 0, 0), 2, tipLength=0.25)
    cv2.arrowedLine(image, selected_pixel, target_pixel, (0, 120, 0), 2, tipLength=0.15)

    scale = max(1, int(scale))
    if scale != 1:
        image = cv2.resize(
            image,
            (nav_map.width * scale, nav_map.height * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    # OpenCV writes BGR; the working image is RGB.
    if not cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write obstacle-edge debug image: {output}")
    return output
