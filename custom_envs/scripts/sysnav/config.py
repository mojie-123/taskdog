"""Configuration loading for the taskdog SysNav component package."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "custom_envs" / "config" / "sysnav.yaml"


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class VLMConfig:
    api_key_env: str
    base_url: str
    reasoning_model: str
    fast_model: str
    timeout_s: float

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"VLM API key is not set; export {self.api_key_env} before running"
            )
        return value


@dataclass(frozen=True)
class PerceptionConfig:
    yoloe_weights: Path
    mobileclip_weights: Path
    sam2_checkpoint: Path
    sam2_model_config: str
    device: str
    confidence: float
    iou: float
    max_detections: int
    min_mask_pixels: int
    max_depth_m: float
    min_localization_points: int


@dataclass(frozen=True)
class CoverageConfig:
    candidate_spacing_m: float
    target_spacing_m: float
    max_view_range_m: float
    horizontal_fov_deg: float
    yaw_bins: int
    min_new_coverage_cells: int
    travel_cost_weight: float
    semantic_interest_weight: float


@dataclass(frozen=True)
class SearchConfig:
    room_map_manifest: Path
    nav2_map_yaml: Path
    camera_name: str
    object_merge_distance_m: float
    target_accept_confidence: float
    early_stop_min_room_coverage: float
    early_stop_min_ordinary_viewpoints: int
    early_stop_max_current_images: int


@dataclass(frozen=True)
class SysNavConfig:
    vlm: VLMConfig
    perception: PerceptionConfig
    coverage: CoverageConfig
    search: SearchConfig

    def validate_files(self, *, require_models: bool = True) -> None:
        paths = [self.search.room_map_manifest, self.search.nav2_map_yaml]
        if require_models:
            paths.extend(
                [
                    self.perception.yoloe_weights,
                    self.perception.mobileclip_weights,
                    self.perception.sam2_checkpoint,
                ]
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing SysNav files: " + ", ".join(missing))


def _required(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid '{name}' section")
    return value


def load_sysnav_config(path: Path | str = DEFAULT_CONFIG) -> SysNavConfig:
    """Load the checked, path-resolved configuration without reading an API key."""

    config_path = Path(path).expanduser().resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if int(document.get("version", -1)) != 1:
        raise ValueError(f"unsupported SysNav config version in {config_path}")
    vlm = _required(document, "vlm")
    perception = _required(document, "perception")
    coverage = _required(document, "coverage")
    search = _required(document, "search")
    root = PROJECT_ROOT
    result = SysNavConfig(
        vlm=VLMConfig(
            api_key_env=str(vlm["api_key_env"]),
            base_url=str(vlm["base_url"]),
            reasoning_model=str(vlm["reasoning_model"]),
            fast_model=str(vlm["fast_model"]),
            timeout_s=float(vlm["timeout_s"]),
        ),
        perception=PerceptionConfig(
            yoloe_weights=_resolve_path(perception["yoloe_weights"], root),
            mobileclip_weights=_resolve_path(perception["mobileclip_weights"], root),
            sam2_checkpoint=_resolve_path(perception["sam2_checkpoint"], root),
            sam2_model_config=str(perception["sam2_model_config"]),
            device=str(perception["device"]),
            confidence=float(perception["confidence"]),
            iou=float(perception["iou"]),
            max_detections=int(perception["max_detections"]),
            min_mask_pixels=int(perception["min_mask_pixels"]),
            max_depth_m=float(perception["max_depth_m"]),
            min_localization_points=int(perception["min_localization_points"]),
        ),
        coverage=CoverageConfig(
            candidate_spacing_m=float(coverage["candidate_spacing_m"]),
            target_spacing_m=float(coverage["target_spacing_m"]),
            max_view_range_m=float(coverage["max_view_range_m"]),
            horizontal_fov_deg=float(coverage["horizontal_fov_deg"]),
            yaw_bins=int(coverage["yaw_bins"]),
            min_new_coverage_cells=int(coverage["min_new_coverage_cells"]),
            travel_cost_weight=float(coverage["travel_cost_weight"]),
            semantic_interest_weight=float(coverage["semantic_interest_weight"]),
        ),
        search=SearchConfig(
            room_map_manifest=_resolve_path(search["room_map_manifest"], root),
            nav2_map_yaml=_resolve_path(search["nav2_map_yaml"], root),
            camera_name=str(search["camera_name"]),
            object_merge_distance_m=float(search["object_merge_distance_m"]),
            target_accept_confidence=float(search["target_accept_confidence"]),
            early_stop_min_room_coverage=float(
                search["early_stop_min_room_coverage"]
            ),
            early_stop_min_ordinary_viewpoints=int(
                search["early_stop_min_ordinary_viewpoints"]
            ),
            early_stop_max_current_images=int(
                search.get("early_stop_max_current_images", 4)
            ),
        ),
    )
    if not 0.0 < result.perception.confidence <= 1.0:
        raise ValueError("perception.confidence must lie in (0, 1]")
    if result.coverage.yaw_bins <= 0 or result.coverage.candidate_spacing_m <= 0:
        raise ValueError("coverage spacing and yaw_bins must be positive")
    if not 0.0 <= result.search.early_stop_min_room_coverage <= 1.0:
        raise ValueError("early_stop_min_room_coverage must lie in [0, 1]")
    if result.search.early_stop_min_ordinary_viewpoints <= 0:
        raise ValueError("early_stop_min_ordinary_viewpoints must be positive")
    if result.search.early_stop_max_current_images <= 0:
        raise ValueError("early_stop_max_current_images must be positive")
    return result
