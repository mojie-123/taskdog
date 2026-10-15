"""Lightweight Room--Viewpoint--Object structured scene memory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from .room_graph import AnonymousRoomGraph
from .types import LocalizedDetection, RoomDescription, json_safe


@dataclass
class RoomState:
    room_id: str
    neighbors: tuple[str, ...]
    area_m2: float
    viewpoint_ids: list[str] = field(default_factory=list)
    object_ids: list[str] = field(default_factory=list)
    semantic_description: str = ""
    semantic_confidence: float = 0.0
    semantic_cues: tuple[str, ...] = ()
    representative_image_path: str = ""
    observed_labels: tuple[str, ...] = ()
    visited: bool = False
    exhausted: bool = False
    coverage: float = 0.0

@dataclass
class ViewpointRecord:
    viewpoint_id: str
    room_id: str
    pose_world: tuple[float, float, float]
    timestamp_s: float
    image_path: str = ""
    object_ids: list[str] = field(default_factory=list)


@dataclass
class ObjectRecord:
    object_id: str
    label: str
    room_id: str
    position_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    best_confidence: float
    observation_count: int = 1
    viewpoint_ids: list[str] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)
    last_seen_s: float = 0.0

    def __post_init__(self) -> None:
        self.position_world = np.asarray(self.position_world, dtype=np.float64).reshape(3)
        self.bbox_min_world = np.asarray(self.bbox_min_world, dtype=np.float64).reshape(3)
        self.bbox_max_world = np.asarray(self.bbox_max_world, dtype=np.float64).reshape(3)
        if not self.label_votes:
            self.label_votes[self.label] += 1


class SceneMemory:
    """Persistent relations equivalent to SysNav's three representation layers.

    Room nodes come from structural geometry.  Viewpoint nodes are robot poses
    that were actually observed.  Object nodes are fused YOLOE/SAM2 instances.
    No furniture or target truth is read from MJCF.
    """

    def __init__(self, graph: AnonymousRoomGraph, *, merge_distance_m: float = 0.45):
        if merge_distance_m <= 0.0:
            raise ValueError("merge_distance_m must be positive")
        self.graph = graph
        self.merge_distance_m = float(merge_distance_m)
        self.rooms: dict[str, RoomState] = {
            room_id: RoomState(
                room_id=room_id,
                neighbors=node.neighbors,
                area_m2=node.area_m2,
            )
            for room_id, node in graph.nodes.items()
        }
        self.viewpoints: dict[str, ViewpointRecord] = {}
        self.objects: dict[str, ObjectRecord] = {}
        self._viewpoint_counter = 0
        self._object_counter = 0

    def observed_labels(self, room_id: str) -> tuple[str, ...]:
        labels = [self.objects[obj_id].label for obj_id in self.rooms[room_id].object_ids]
        return tuple(sorted(set(labels)))

    def update_room_description(
        self,
        room_id: str,
        description: RoomDescription,
        *,
        representative_image_path: str = "",
    ) -> None:
        state = self.rooms[room_id]
        if description.confidence >= state.semantic_confidence:
            state.semantic_description = description.room_type
            state.semantic_confidence = float(description.confidence)
            state.semantic_cues = tuple(description.visual_cues)
            if representative_image_path:
                state.representative_image_path = representative_image_path

    def update_room_progress(
        self, room_id: str, coverage: float, *, exhausted: bool | None = None
    ) -> None:
        state = self.rooms[room_id]
        state.visited = True
        state.coverage = max(state.coverage, float(np.clip(coverage, 0.0, 1.0)))
        if exhausted is not None:
            state.exhausted = bool(exhausted)

    def _match_object(self, detection: LocalizedDetection, room_id: str) -> ObjectRecord | None:
        best = None
        best_distance = self.merge_distance_m
        for record in self.objects.values():
            if record.room_id != room_id or record.label != detection.detection.label:
                continue
            distance = float(np.linalg.norm(record.position_world - detection.position_world))
            if distance < best_distance:
                best, best_distance = record, distance
        return best

    def _add_or_update_object(
        self,
        detection: LocalizedDetection,
        room_id: str,
        viewpoint_id: str,
        timestamp_s: float,
    ) -> ObjectRecord:
        record = self._match_object(detection, room_id)
        if record is None:
            object_id = f"object_{self._object_counter}"
            self._object_counter += 1
            record = ObjectRecord(
                object_id=object_id,
                label=detection.detection.label,
                room_id=room_id,
                position_world=detection.position_world.copy(),
                bbox_min_world=detection.bbox_min_world.copy(),
                bbox_max_world=detection.bbox_max_world.copy(),
                best_confidence=detection.detection.confidence,
                viewpoint_ids=[viewpoint_id],
                last_seen_s=timestamp_s,
            )
            self.objects[object_id] = record
            self.rooms[room_id].object_ids.append(object_id)
            self.rooms[room_id].observed_labels = self.observed_labels(room_id)
            return record

        old_count = record.observation_count
        new_weight = max(0.05, detection.detection.confidence)
        old_weight = float(old_count)
        record.position_world = (
            record.position_world * old_weight
            + detection.position_world * new_weight
        ) / (old_weight + new_weight)
        record.bbox_min_world = np.minimum(record.bbox_min_world, detection.bbox_min_world)
        record.bbox_max_world = np.maximum(record.bbox_max_world, detection.bbox_max_world)
        record.best_confidence = max(
            record.best_confidence, detection.detection.confidence
        )
        record.observation_count += 1
        record.label_votes[detection.detection.label] += 1
        record.label = record.label_votes.most_common(1)[0][0]
        self.rooms[room_id].observed_labels = self.observed_labels(room_id)
        record.last_seen_s = timestamp_s
        if viewpoint_id not in record.viewpoint_ids:
            record.viewpoint_ids.append(viewpoint_id)
        return record

    def add_viewpoint(
        self,
        room_id: str,
        pose_world: tuple[float, float, float] | np.ndarray,
        detections: list[LocalizedDetection],
        *,
        timestamp_s: float | None = None,
        image_path: str = "",
    ) -> ViewpointRecord:
        if room_id not in self.rooms:
            raise KeyError(f"unknown room id: {room_id}")
        timestamp = float(time.time() if timestamp_s is None else timestamp_s)
        pose = tuple(float(value) for value in np.asarray(pose_world).reshape(3))
        viewpoint_id = f"viewpoint_{self._viewpoint_counter}"
        self._viewpoint_counter += 1
        record = ViewpointRecord(
            viewpoint_id=viewpoint_id,
            room_id=room_id,
            pose_world=pose,
            timestamp_s=timestamp,
            image_path=str(image_path),
        )
        self.viewpoints[viewpoint_id] = record
        room = self.rooms[room_id]
        room.viewpoint_ids.append(viewpoint_id)
        room.visited = True
        for detection in detections:
            assigned_room = detection.room_id or room_id
            if assigned_room not in self.rooms:
                assigned_room = room_id
            obj = self._add_or_update_object(
                detection, assigned_room, viewpoint_id, timestamp
            )
            record.object_ids.append(obj.object_id)
        return record

    def objects_by_label(self, label: str, *, room_id: str | None = None) -> list[ObjectRecord]:
        normalised = " ".join(str(label).strip().lower().split())
        result = [record for record in self.objects.values() if record.label == normalised]
        if room_id is not None:
            result = [record for record in result if record.room_id == room_id]
        return sorted(result, key=lambda item: item.best_confidence, reverse=True)

    def nearby_labels(
        self, position_world: np.ndarray, radius_m: float, *, exclude_id: str | None = None
    ) -> tuple[str, ...]:
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        labels = {
            record.label
            for record in self.objects.values()
            if record.object_id != exclude_id
            and np.linalg.norm(record.position_world - position) <= float(radius_m)
        }
        return tuple(sorted(labels))

    def to_dict(self) -> dict:
        rooms = {}
        for room_id, state in self.rooms.items():
            payload = json_safe(state)
            rooms[room_id] = payload
        return {
            "version": 1,
            "rooms": rooms,
            "viewpoints": {
                key: json_safe(value) for key, value in self.viewpoints.items()
            },
            "objects": {key: json_safe(value) for key, value in self.objects.items()},
        }

    def save(self, path: Path | str) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
