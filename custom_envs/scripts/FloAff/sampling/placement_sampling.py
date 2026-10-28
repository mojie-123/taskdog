"""Sample non-overlapping target placements and edge-conditioned instances."""

from __future__ import annotations

from collections import Counter

import numpy as np

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.FloAff.geometry import left_tangent
from custom_envs.scripts.FloAff.schemas import InstanceRecord, PlacementRecord
from custom_envs.scripts.FloAff.sampling.common import latin_hypercube, scale_unit


EDGE_OUTWARDS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
}


def _target_pose(x: float, y: float, z: float) -> tuple[float, ...]:
    return (float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0)


def edge_distances(x: float, y: float, config: DataGenerationConfig) -> dict[str, float]:
    table = config.table
    return {
        "left": float(x - table.x_min),
        "right": float(table.x_max - x),
        "bottom": float(y - table.y_min),
        "top": float(table.y_max - y),
    }


def _edge_coordinate(edge: str, distance: np.ndarray, config: DataGenerationConfig) -> np.ndarray:
    table = config.table
    if edge == "left":
        return table.x_min + distance
    if edge == "right":
        return table.x_max - distance
    if edge == "bottom":
        return table.y_min + distance
    if edge == "top":
        return table.y_max - distance
    raise KeyError(f"unknown edge: {edge}")


def _short_placements(config: DataGenerationConfig, rng: np.random.Generator) -> list[PlacementRecord]:
    count = int(config.placements.short_count)
    if count <= 0:
        return []
    edge = config.placements.train_short_edge
    if edge not in {"left", "right"}:
        raise ValueError("train_short_edge must be left or right for this table")
    unit = latin_hypercube(count, 2, rng)
    distance = scale_unit(
        unit[:, 0],
        config.placements.edge_distance_min_m,
        config.placements.edge_distance_max_m,
    )
    x = _edge_coordinate(edge, distance, config)
    y = scale_unit(
        unit[:, 1],
        config.table.y_min + config.placements.region_separation_m,
        config.table.y_max - config.placements.region_separation_m,
    )
    z = config.table.top_z + config.scene.target_half_height_m
    return [
        PlacementRecord(
            placement_id=f"short_{edge}_{index:04d}",
            category="short_edge",
            target_pose_world=_target_pose(x[index], y[index], z),
            edge_distances_m=edge_distances(x[index], y[index], config),
            source_region=f"short_{edge}",
            approach_edges=(edge,),
        )
        for index in range(count)
    ]


def _corner_placements(config: DataGenerationConfig, rng: np.random.Generator) -> list[PlacementRecord]:
    count = int(config.placements.corner_count)
    if count <= 0:
        return []
    first, second = config.placements.train_corner_edges
    if {first, second}.issubset({"left", "right"}) or {first, second}.issubset({"top", "bottom"}):
        raise ValueError("train_corner_edges must contain one short and one long edge")
    unit = latin_hypercube(count, 2, rng)
    first_distance = scale_unit(
        unit[:, 0],
        config.placements.edge_distance_min_m,
        config.placements.edge_distance_max_m,
    )
    second_distance = scale_unit(
        unit[:, 1],
        config.placements.edge_distance_min_m,
        config.placements.edge_distance_max_m,
    )
    coordinates: dict[str, np.ndarray] = {}
    for edge, distance in ((first, first_distance), (second, second_distance)):
        axis = "x" if edge in {"left", "right"} else "y"
        coordinates[axis] = _edge_coordinate(edge, distance, config)
    x, y = coordinates["x"], coordinates["y"]
    z = config.table.top_z + config.scene.target_half_height_m
    token = f"{first}_{second}"
    return [
        PlacementRecord(
            placement_id=f"corner_{token}_{index:04d}",
            category="corner",
            target_pose_world=_target_pose(x[index], y[index], z),
            edge_distances_m=edge_distances(x[index], y[index], config),
            source_region=f"corner_{token}",
            approach_edges=(first, second),
        )
        for index in range(count)
    ]


def _long_placements(config: DataGenerationConfig, rng: np.random.Generator) -> list[PlacementRecord]:
    total = int(config.placements.long_count)
    if total <= 0:
        return []
    edges = tuple(config.placements.long_edges)
    if not edges or any(edge not in {"bottom", "top"} for edge in edges):
        raise ValueError("long_edges must contain bottom/top")
    counts = Counter({edge: total // len(edges) for edge in edges})
    for edge in edges[: total % len(edges)]:
        counts[edge] += 1
    rows: list[PlacementRecord] = []
    z = config.table.top_z + config.scene.target_half_height_m
    for edge in edges:
        count = counts[edge]
        if count <= 0:
            continue
        unit = latin_hypercube(count, 2, rng)
        distance = scale_unit(
            unit[:, 0],
            config.placements.edge_distance_min_m,
            config.placements.edge_distance_max_m,
        )
        y = _edge_coordinate(edge, distance, config)
        x = scale_unit(
            unit[:, 1],
            config.table.x_min + config.placements.region_separation_m,
            config.table.x_max - config.placements.region_separation_m,
        )
        rows.extend(
            PlacementRecord(
                placement_id=f"long_{edge}_{index:04d}",
                category="long_edge",
                target_pose_world=_target_pose(x[index], y[index], z),
                edge_distances_m=edge_distances(x[index], y[index], config),
                source_region=f"long_{edge}",
                approach_edges=(edge,),
            )
            for index in range(count)
        )
    return rows


def _transfer_placements(config: DataGenerationConfig, rng: np.random.Generator) -> list[PlacementRecord]:
    """Optional final-test regions; zero counts leave the training pilot unchanged."""
    rows: list[PlacementRecord] = []
    minimum = config.placements.edge_distance_min_m
    maximum = config.placements.edge_distance_max_m
    separation = config.placements.region_separation_m
    z = config.table.top_z + config.scene.target_half_height_m

    short_count = int(config.placements.transfer_short_count)
    short_edge = config.placements.transfer_short_edge
    if short_count:
        if short_edge not in {"left", "right"}:
            raise ValueError("transfer_short_edge must be left or right")
        unit = latin_hypercube(short_count, 2, rng)
        x = _edge_coordinate(short_edge, scale_unit(unit[:, 0], minimum, maximum), config)
        y = scale_unit(unit[:, 1], config.table.y_min + separation, config.table.y_max - separation)
        rows.extend(
            PlacementRecord(
                placement_id=f"transfer_short_{short_edge}_{index:04d}",
                category="short_edge",
                target_pose_world=_target_pose(x[index], y[index], z),
                edge_distances_m=edge_distances(x[index], y[index], config),
                source_region=f"transfer_short_{short_edge}",
                approach_edges=(short_edge,),
            )
            for index in range(short_count)
        )

    corner_count = int(config.placements.transfer_corner_count_per_region)
    for region_index, edge_pair in enumerate(config.placements.transfer_corner_edges):
        if corner_count <= 0:
            break
        first, second = edge_pair
        if {first, second}.issubset({"left", "right"}) or {first, second}.issubset({"top", "bottom"}):
            raise ValueError(f"invalid transfer corner: {edge_pair}")
        unit = latin_hypercube(corner_count, 2, rng)
        coordinates: dict[str, np.ndarray] = {}
        for edge, values in zip((first, second), (unit[:, 0], unit[:, 1])):
            axis = "x" if edge in {"left", "right"} else "y"
            coordinates[axis] = _edge_coordinate(
                edge, scale_unit(values, minimum, maximum), config
            )
        x, y = coordinates["x"], coordinates["y"]
        rows.extend(
            PlacementRecord(
                placement_id=f"transfer_corner_{first}_{second}_{index:04d}",
                category="corner",
                target_pose_world=_target_pose(x[index], y[index], z),
                edge_distances_m=edge_distances(x[index], y[index], config),
                source_region=f"transfer_corner_{first}_{second}",
                approach_edges=(first, second),
            )
            for index in range(corner_count)
        )

    long_count = int(config.placements.transfer_long_count_per_edge)
    for edge in config.placements.transfer_long_edges:
        if long_count <= 0:
            break
        if edge not in {"bottom", "top"}:
            raise ValueError(f"invalid transfer long edge: {edge}")
        unit = latin_hypercube(long_count, 2, rng)
        y = _edge_coordinate(edge, scale_unit(unit[:, 0], minimum, maximum), config)
        x = scale_unit(unit[:, 1], config.table.x_min + separation, config.table.x_max - separation)
        rows.extend(
            PlacementRecord(
                placement_id=f"transfer_long_{edge}_{index:04d}",
                category="long_edge",
                target_pose_world=_target_pose(x[index], y[index], z),
                edge_distances_m=edge_distances(x[index], y[index], config),
                source_region=f"transfer_long_{edge}",
                approach_edges=(edge,),
            )
            for index in range(long_count)
        )
    return rows


def _instance(placement: PlacementRecord, edge: str) -> InstanceRecord:
    outward = np.asarray(EDGE_OUTWARDS[edge], dtype=np.float64)
    tangent = left_tangent(outward)
    if placement.category == "corner":
        profile = "corner"
    elif placement.category == "long_edge" and not placement.placement_id.startswith(
        "transfer_"
    ):
        # Development long-edge O uses the narrower +/-35 degree core and
        # +/-45 degree tails.  Held-out transfer long edges intentionally keep
        # the original wider long_edge profile for a harder final test.
        profile = "long_edge_train"
    else:
        profile = placement.category
    other_corner_edge = None
    if placement.category == "corner":
        other_corner_edge = next(item for item in placement.approach_edges if item != edge)
    return InstanceRecord(
        instance_id=f"{placement.placement_id}__{edge}",
        placement_id=placement.placement_id,
        category=placement.category,
        approach_edge=edge,
        outward_xy_world=(float(outward[0]), float(outward[1])),
        tangent_xy_world=(float(tangent[0]), float(tangent[1])),
        target_pose_world=placement.target_pose_world,
        observation_profile=profile,
        corner_other_edge=other_corner_edge,
    )


def generate_placements_and_instances(
    config: DataGenerationConfig,
) -> tuple[list[PlacementRecord], list[InstanceRecord]]:
    seed_sequence = np.random.SeedSequence(int(config.seed))
    short_seed, corner_seed, long_seed, transfer_seed = seed_sequence.spawn(4)
    placements = [
        *_short_placements(config, np.random.default_rng(short_seed)),
        *_corner_placements(config, np.random.default_rng(corner_seed)),
        *_long_placements(config, np.random.default_rng(long_seed)),
        *_transfer_placements(config, np.random.default_rng(transfer_seed)),
    ]
    instances: list[InstanceRecord] = []
    for placement in placements:
        instances.extend(_instance(placement, edge) for edge in placement.approach_edges)
    return placements, instances
