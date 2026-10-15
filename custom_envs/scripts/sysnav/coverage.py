"""Classical room-internal viewpoint coverage over the static 2-D map."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from custom_envs.scripts.map.nav2_map_io import Nav2Map
from custom_envs.scripts.map.room_map_io import RoomMap

from .config import CoverageConfig


def _wrap_angle(value: np.ndarray | float):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _line_is_clear(
    blocked: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> bool:
    row0, col0 = start
    row1, col1 = end
    count = max(abs(row1 - row0), abs(col1 - col0)) + 1
    rows = np.rint(np.linspace(row0, row1, count)).astype(np.int32)
    cols = np.rint(np.linspace(col0, col1, count)).astype(np.int32)
    if count <= 2:
        return True
    return not bool(blocked[rows[1:-1], cols[1:-1]].any())


@dataclass(frozen=True)
class CoverageViewpoint:
    index: int
    room_id: str
    pose: tuple[float, float, float]
    grid_cell: tuple[int, int]
    visible_target_indices: np.ndarray = field(repr=False, compare=False)


class RoomCoveragePlan:
    """A dynamic greedy set-cover plan for one anonymous room."""

    def __init__(
        self,
        room_id: str,
        viewpoints: list[CoverageViewpoint],
        target_cells: np.ndarray,
        target_world_xy: np.ndarray,
        config: CoverageConfig,
    ):
        self.room_id = room_id
        self.viewpoints = viewpoints
        self.target_cells = np.asarray(target_cells, dtype=np.int32).reshape(-1, 2)
        self.target_world_xy = np.asarray(target_world_xy, dtype=np.float64).reshape(-1, 2)
        self.config = config
        self.covered = np.zeros(len(self.target_cells), dtype=bool)
        self.visited_viewpoints: set[int] = set()

    @property
    def coverage(self) -> float:
        return float(self.covered.mean()) if len(self.covered) else 1.0

    @property
    def exhausted(self) -> bool:
        return self.next_viewpoint((0.0, 0.0, 0.0)) is None

    def mark_visited(self, viewpoint_index: int) -> None:
        viewpoint = self.viewpoints[int(viewpoint_index)]
        self.covered[viewpoint.visible_target_indices] = True
        self.visited_viewpoints.add(int(viewpoint_index))

    def next_viewpoint(
        self,
        current_pose: tuple[float, float, float] | np.ndarray,
        *,
        interest_points_world: list[np.ndarray] | None = None,
    ) -> CoverageViewpoint | None:
        """Select maximum new coverage, penalising travel and boosting known anchors."""

        current = np.asarray(current_pose, dtype=np.float64).reshape(-1)
        if current.size < 2:
            raise ValueError("current_pose must contain at least x and y")
        best: CoverageViewpoint | None = None
        best_score = -math.inf
        half_fov = math.radians(self.config.horizontal_fov_deg) / 2.0
        interests = [np.asarray(value, dtype=np.float64).reshape(-1)[:2] for value in (interest_points_world or [])]
        for viewpoint in self.viewpoints:
            if viewpoint.index in self.visited_viewpoints:
                continue
            visible = viewpoint.visible_target_indices
            new_count = int((~self.covered[visible]).sum())
            if new_count < self.config.min_new_coverage_cells:
                continue
            position = np.asarray(viewpoint.pose[:2], dtype=np.float64)
            travel = float(np.linalg.norm(position - current[:2]))
            semantic_boost = 0.0
            for point in interests:
                delta = point - position
                distance = float(np.linalg.norm(delta))
                angle = math.atan2(float(delta[1]), float(delta[0]))
                if abs(float(_wrap_angle(angle - viewpoint.pose[2]))) <= half_fov:
                    semantic_boost += self.config.semantic_interest_weight / (0.25 + distance)
            score = (
                float(new_count)
                + semantic_boost
                - self.config.travel_cost_weight * travel
            )
            if score > best_score:
                best_score = score
                best = viewpoint
        return best


class CoveragePlanner:
    """Generate collision-free, occlusion-aware front-camera viewpoints."""

    def __init__(self, room_map: RoomMap, nav_map: Nav2Map, config: CoverageConfig):
        self.room_map = room_map
        self.nav_map = nav_map
        self.config = config
        if room_map.room_region_id_map.shape != nav_map.free_mask.shape:
            raise ValueError("room and Nav2 rasters have different shapes")
        if not np.isclose(room_map.resolution, nav_map.resolution) or not np.allclose(
            room_map.origin, nav_map.origin
        ):
            raise ValueError("room and Nav2 map transforms disagree")

    def _sample_cells(self, mask: np.ndarray, spacing_m: float) -> np.ndarray:
        stride = max(1, int(round(float(spacing_m) / self.nav_map.resolution)))
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return np.empty((0, 2), dtype=np.int32)
        row_origin, col_origin = int(rows.min()), int(cols.min())
        selected = (
            ((rows - row_origin) % stride == 0)
            & ((cols - col_origin) % stride == 0)
        )
        cells = np.column_stack((rows[selected], cols[selected])).astype(np.int32)
        if len(cells) == 0:
            midpoint = len(rows) // 2
            cells = np.asarray([[rows[midpoint], cols[midpoint]]], dtype=np.int32)
        return cells

    def _world_xy(self, cells: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.nav_map.grid_to_world(int(row), int(col)) for row, col in cells],
            dtype=np.float64,
        )

    def build_plan(self, room_id: str) -> RoomCoveragePlan:
        room_mask = self.room_map.searchable_mask(room_id) & self.nav_map.free_mask
        candidate_cells = self._sample_cells(
            room_mask, self.config.candidate_spacing_m
        )
        target_cells = self._sample_cells(room_mask, self.config.target_spacing_m)
        if len(candidate_cells) == 0 or len(target_cells) == 0:
            raise ValueError(f"room {room_id} has no searchable coverage cells")
        candidate_world = self._world_xy(candidate_cells)
        target_world = self._world_xy(target_cells)
        blocked = self.nav_map.occupied_mask | self.nav_map.unknown_mask
        max_range = float(self.config.max_view_range_m)
        half_fov = math.radians(self.config.horizontal_fov_deg) / 2.0
        yaw_values = np.linspace(
            -math.pi,
            math.pi,
            num=self.config.yaw_bins,
            endpoint=False,
            dtype=np.float64,
        )
        viewpoints: list[CoverageViewpoint] = []
        for candidate_cell, candidate_xy in zip(
            candidate_cells, candidate_world, strict=True
        ):
            delta = target_world - candidate_xy
            distance = np.linalg.norm(delta, axis=1)
            bearings = np.arctan2(delta[:, 1], delta[:, 0])
            nearby = np.flatnonzero(distance <= max_range)
            line_visible = np.asarray(
                [
                    target_index
                    for target_index in nearby
                    if _line_is_clear(
                        blocked,
                        (int(candidate_cell[0]), int(candidate_cell[1])),
                        (int(target_cells[target_index, 0]), int(target_cells[target_index, 1])),
                    )
                ],
                dtype=np.int32,
            )
            for yaw in yaw_values:
                if len(line_visible) == 0:
                    continue
                angular = np.abs(_wrap_angle(bearings[line_visible] - yaw))
                visible = line_visible[angular <= half_fov]
                if len(visible) < self.config.min_new_coverage_cells:
                    continue
                viewpoints.append(
                    CoverageViewpoint(
                        index=len(viewpoints),
                        room_id=room_id,
                        pose=(float(candidate_xy[0]), float(candidate_xy[1]), float(yaw)),
                        grid_cell=(int(candidate_cell[0]), int(candidate_cell[1])),
                        visible_target_indices=visible,
                    )
                )
        if not viewpoints:
            raise RuntimeError(f"no viable coverage viewpoint generated for {room_id}")
        return RoomCoveragePlan(
            room_id=room_id,
            viewpoints=viewpoints,
            target_cells=target_cells,
            target_world_xy=target_world,
            config=self.config,
        )
