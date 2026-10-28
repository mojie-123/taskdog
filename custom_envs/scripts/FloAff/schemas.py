"""Versioned records shared by the FloAff data-generation stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlacementRecord:
    placement_id: str
    category: str
    target_pose_world: tuple[float, float, float, float, float, float, float]
    edge_distances_m: dict[str, float]
    source_region: str
    approach_edges: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    placement_id: str
    category: str
    approach_edge: str
    outward_xy_world: tuple[float, float]
    tangent_xy_world: tuple[float, float]
    target_pose_world: tuple[float, float, float, float, float, float, float]
    observation_profile: str
    corner_other_edge: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ManipulationCandidate:
    candidate_id: str
    instance_id: str
    placement_id: str
    scene_id: str
    category: str
    approach_edge: str
    cube_pose: tuple[float, float, float, float, float, float, float]
    base_pose: tuple[float, float, float]
    joint1_target: float
    proposal: dict[str, float]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        row = asdict(self)
        # Compatibility with N2M ManipulationEpisodeRunner.
        row["candidate_id"] = self.candidate_id
        row["scene_id"] = self.scene_id
        return row


@dataclass(frozen=True)
class ObservationProposal:
    proposal_id: str
    instance_id: str
    placement_id: str
    approach_edge: str
    stratum: str
    camera_range_m: float
    alpha_deg: float
    yaw_offset_deg: float
    desired_camera_pose_world: tuple[float, float, float]
    base_pose_world: tuple[float, float, float]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, **asdict(self)}
