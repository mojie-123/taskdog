"""Composition root for a future ``navigate_mujoco_sysnav.py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from custom_envs.scripts.map.nav2_map_io import Nav2Map, load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap

from .config import DEFAULT_CONFIG, SysNavConfig, load_sysnav_config
from .coverage import CoveragePlanner
from .perception import YOLOESAM2Perceptor
from .rgbd import DepthLocalizer, RGBDFrame
from .room_graph import AnonymousRoomGraph
from .scene_memory import ObjectRecord, SceneMemory, ViewpointRecord
from .search_policy import SysNavSearchPolicy
from .types import Detection2D, LocalizedDetection
from .vlm import QwenVLMClient


@dataclass(frozen=True)
class ObservationUpdate:
    viewpoint: ViewpointRecord
    detections: tuple[Detection2D, ...]
    localized_detections: tuple[LocalizedDetection, ...]
    coverage: float | None


@dataclass
class SysNavComponents:
    config: SysNavConfig
    room_map: RoomMap
    nav_map: Nav2Map
    room_graph: AnonymousRoomGraph
    scene_memory: SceneMemory
    coverage_planner: CoveragePlanner
    vlm: QwenVLMClient | None
    perception: YOLOESAM2Perceptor
    localizer: DepthLocalizer
    search_policy: SysNavSearchPolicy

    def infer_observation(
        self, frame: RGBDFrame
    ) -> tuple[tuple[Detection2D, ...], tuple[LocalizedDetection, ...]]:
        """Infer one RGB-D frame without changing coverage or scene memory."""

        detections = tuple(self.perception.detect(frame.rgb))
        localized = tuple(self.localizer.localize_all(frame, detections))
        return detections, localized

    def commit_observation(
        self,
        frame: RGBDFrame,
        robot_pose_world: tuple[float, float, float] | np.ndarray,
        room_id: str,
        detections: tuple[Detection2D, ...],
        localized: tuple[LocalizedDetection, ...],
        *,
        planned_viewpoint_index: int | None = None,
        image_path: str = "",
    ) -> ObservationUpdate:
        """Attach a viewpoint and coverage, but keep proposals out of memory.

        ``localized`` is returned for logging and VLM verification.  It is
        intentionally not attached to the persistent Object layer here.
        """

        viewpoint = self.scene_memory.add_viewpoint(
            room_id,
            robot_pose_world,
            [],
            timestamp_s=frame.timestamp_s,
            image_path=image_path,
        )
        coverage = None
        if planned_viewpoint_index is not None:
            coverage = self.search_policy.mark_viewpoint_complete(
                room_id, planned_viewpoint_index
            )
        return ObservationUpdate(
            viewpoint=viewpoint,
            detections=detections,
            localized_detections=localized,
            coverage=coverage,
        )

    def commit_verified_detection(
        self, viewpoint_id: str, detection: LocalizedDetection
    ) -> ObjectRecord:
        """Persist one object only after the semantic verifier accepts it."""

        return self.scene_memory.attach_verified_detection(viewpoint_id, detection)

    def process_observation(
        self,
        frame: RGBDFrame,
        robot_pose_world: tuple[float, float, float] | np.ndarray,
        room_id: str,
        *,
        planned_viewpoint_index: int | None = None,
        image_path: str = "",
    ) -> ObservationUpdate:
        """Run perception/localization and persist only room/viewpoint state."""

        detections, localized = self.infer_observation(frame)
        return self.commit_observation(
            frame,
            robot_pose_world,
            room_id,
            detections,
            localized,
            planned_viewpoint_index=planned_viewpoint_index,
            image_path=image_path,
        )

    def close(self) -> None:
        self.perception.close()


def build_sysnav_components(
    config_path: Path | str = DEFAULT_CONFIG,
    *,
    enable_vlm: bool = True,
    refine_with_sam2: bool = True,
) -> SysNavComponents:
    """Build lazy model clients and eagerly validate all map/model paths."""

    config = load_sysnav_config(config_path)
    config.validate_files(require_models=True)
    room_map = RoomMap.load(config.search.room_map_manifest)
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    graph = AnonymousRoomGraph(room_map)
    memory = SceneMemory(
        graph, merge_distance_m=config.search.object_merge_distance_m
    )
    coverage = CoveragePlanner(room_map, nav_map, config.coverage)
    vlm = QwenVLMClient(config.vlm) if enable_vlm else None
    perception = YOLOESAM2Perceptor(
        config.perception, refine_with_sam2=refine_with_sam2
    )
    localizer = DepthLocalizer(config.perception, room_map)
    policy = SysNavSearchPolicy(
        graph,
        coverage,
        memory,
        config.search,
        reasoner=vlm,
    )
    return SysNavComponents(
        config=config,
        room_map=room_map,
        nav_map=nav_map,
        room_graph=graph,
        scene_memory=memory,
        coverage_planner=coverage,
        vlm=vlm,
        perception=perception,
        localizer=localizer,
        search_policy=policy,
    )
