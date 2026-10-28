"""Configuration for FloAff dataset generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]


@dataclass
class TableConfig:
    body_name: str = "living_coffee_table"
    x_min: float = 5.0
    x_max: float = 10.0
    y_min: float = 3.2
    y_max: float = 4.1
    top_z: float = 0.45


@dataclass
class SceneConfig:
    scene_xml: str = str(PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml")
    target_body: str = "cube"
    target_freejoint: str = "cube_freejoint"
    target_half_height_m: float = 0.025
    anchor_body: str = "living_coffee_table"
    robot_root_body: str = "base_link"
    camera_name: str = "front_realsense"
    camera_width: int = 640
    camera_height: int = 480
    robot_root_z: float = 0.563874


@dataclass
class PlacementConfig:
    short_count: int = 20
    corner_count: int = 30
    long_count: int = 0
    edge_distance_min_m: float = 0.10
    edge_distance_max_m: float = 0.30
    region_separation_m: float = 0.30
    train_short_edge: str = "left"
    train_corner_edges: tuple[str, str] = ("left", "top")
    long_edges: tuple[str, ...] = ("bottom", "top")
    transfer_short_count: int = 0
    transfer_short_edge: str = "right"
    transfer_corner_count_per_region: int = 0
    transfer_corner_edges: tuple[tuple[str, str], ...] = (
        ("left", "bottom"),
        ("right", "bottom"),
        ("right", "top"),
    )
    transfer_long_count_per_edge: int = 0
    transfer_long_edges: tuple[str, ...] = ("bottom", "top")


@dataclass
class ManipulationSamplingConfig:
    candidates_per_instance: int = 100
    transfer_candidates_per_instance: int = 40
    outward_min_m: float = 0.25
    outward_max_m: float = 0.85
    tangent_min_m: float = -0.50
    tangent_max_m: float = 0.50
    initial_attempts: int = 3
    ambiguous_extra_attempts: int = 2


@dataclass
class ObservationAngleProfile:
    core_min_deg: float
    core_max_deg: float
    negative_tail_min_deg: float
    negative_tail_max_deg: float
    positive_tail_min_deg: float
    positive_tail_max_deg: float
    core_fraction: float = 0.80

    def quotas(self, total: int) -> dict[str, int]:
        core = int(round(int(total) * float(self.core_fraction)))
        tail = int(total) - core
        negative = tail // 2
        return {"core": core, "negative_tail": negative, "positive_tail": tail - negative}

    def bounds(self, stratum: str) -> tuple[float, float]:
        if stratum == "core":
            return self.core_min_deg, self.core_max_deg
        if stratum == "negative_tail":
            return self.negative_tail_min_deg, self.negative_tail_max_deg
        if stratum == "positive_tail":
            return self.positive_tail_min_deg, self.positive_tail_max_deg
        raise KeyError(f"unknown observation stratum: {stratum}")


def _corner_profile() -> ObservationAngleProfile:
    return ObservationAngleProfile(-25.0, 25.0, -35.0, -25.0, 25.0, 35.0)


def _short_profile() -> ObservationAngleProfile:
    return ObservationAngleProfile(-35.0, 35.0, -45.0, -35.0, 35.0, 45.0)


def _long_profile() -> ObservationAngleProfile:
    return ObservationAngleProfile(-50.0, 50.0, -65.0, -50.0, 50.0, 65.0)


def _long_training_profile() -> ObservationAngleProfile:
    """Narrower long-edge views used only by development/training placements."""
    return ObservationAngleProfile(-35.0, 35.0, -45.0, -35.0, 35.0, 45.0)


@dataclass
class ObservationSamplingConfig:
    valid_per_instance: int = 40
    transfer_valid_per_instance: int = 20
    camera_range_min_m: float = 0.8
    camera_range_max_m: float = 2.0
    yaw_offset_min_deg: float = -20.0
    yaw_offset_max_deg: float = 20.0
    proposal_multiplier: int = 12
    corner: ObservationAngleProfile = field(default_factory=_corner_profile)
    short_edge: ObservationAngleProfile = field(default_factory=_short_profile)
    long_edge_train: ObservationAngleProfile = field(default_factory=_long_training_profile)
    long_edge: ObservationAngleProfile = field(default_factory=_long_profile)

    def profile(self, name: str) -> ObservationAngleProfile:
        if name == "corner":
            return self.corner
        if name == "short_edge":
            return self.short_edge
        if name == "long_edge_train":
            return self.long_edge_train
        if name == "long_edge":
            return self.long_edge
        raise KeyError(f"unknown observation profile: {name}")


@dataclass
class PointCloudConfig:
    voxel_size_m: float = 0.01
    max_depth_m: float = 3.0
    min_target_pixels: int = 80
    min_target_points: int = 15
    min_anchor_points: int = 100
    anchor_max_behind_target_m: float = 0.50
    target_sample_points: int = 1024
    anchor_sample_points: int = 3072
    axis_anchor_points: int = 4096


@dataclass
class AffordanceLabelConfig:
    outward_min_m: float = 0.25
    outward_max_m: float = 0.85
    tangent_min_m: float = -0.50
    tangent_max_m: float = 0.50
    resolution_m: float = 0.05
    kernel_sigma_m: float = 0.08
    support_radius_m: float = 0.16


@dataclass
class SplitConfig:
    axis_train_fraction: float = 0.40
    seed: int = 42


@dataclass
class DataGenerationConfig:
    seed: int = 42
    output_dir: str = str(PACKAGE_DIR / "data" / "dataset_v1")
    table: TableConfig = field(default_factory=TableConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    placements: PlacementConfig = field(default_factory=PlacementConfig)
    manipulation: ManipulationSamplingConfig = field(default_factory=ManipulationSamplingConfig)
    observations: ObservationSamplingConfig = field(default_factory=ObservationSamplingConfig)
    point_cloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    affordance: AffordanceLabelConfig = field(default_factory=AffordanceLabelConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)

    def validate(self) -> None:
        if not self.table.x_min < self.table.x_max or not self.table.y_min < self.table.y_max:
            raise ValueError("invalid table bounds")
        if not 0.0 < self.placements.edge_distance_min_m < self.placements.edge_distance_max_m:
            raise ValueError("invalid target edge-distance interval")
        observation_counts = {
            "valid_per_instance": self.observations.valid_per_instance,
            "transfer_valid_per_instance": self.observations.transfer_valid_per_instance,
        }
        for name, value in observation_counts.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
            if int(value) % 10:
                raise ValueError(f"{name} must be divisible by 10 for the 80/10/10 split")
        manipulation_counts = {
            "candidates_per_instance": self.manipulation.candidates_per_instance,
            "transfer_candidates_per_instance": self.manipulation.transfer_candidates_per_instance,
        }
        for name, value in manipulation_counts.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.observations.camera_range_min_m < self.observations.camera_range_max_m:
            raise ValueError("invalid camera range")
        if not 0.0 < self.splits.axis_train_fraction < 1.0:
            raise ValueError("axis_train_fraction must be in (0, 1)")
        if self.affordance.resolution_m <= 0.0:
            raise ValueError("affordance resolution must be positive")
        placement_counts = (
            self.placements.short_count,
            self.placements.corner_count,
            self.placements.long_count,
            self.placements.transfer_short_count,
            self.placements.transfer_corner_count_per_region,
            self.placements.transfer_long_count_per_edge,
        )
        if any(int(value) < 0 for value in placement_counts):
            raise ValueError("placement counts cannot be negative")
        for name in ("corner", "short_edge", "long_edge_train", "long_edge"):
            profile = self.observations.profile(name)
            bounds = [
                profile.negative_tail_min_deg,
                profile.negative_tail_max_deg,
                profile.core_min_deg,
                profile.core_max_deg,
                profile.positive_tail_min_deg,
                profile.positive_tail_max_deg,
            ]
            if bounds != sorted(bounds):
                raise ValueError(f"overlapping or unordered {name} observation profile: {bounds}")
        if max(abs(self.observations.corner.negative_tail_min_deg), abs(self.observations.corner.positive_tail_max_deg)) >= 45.0:
            raise ValueError("corner observation families must stay below 45 degrees to remain disjoint")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance, values: dict[str, Any]) -> None:
    known = {item.name: item for item in fields(instance)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"unknown config keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)


def load_data_generation_config(path: Path | str | None = None) -> DataGenerationConfig:
    config = DataGenerationConfig()
    if path is not None:
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise TypeError("data-generation config must contain a YAML mapping")
        _update_dataclass(config, values)
    config.validate()
    return config
