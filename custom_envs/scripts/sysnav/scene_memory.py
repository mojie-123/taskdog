"""Lightweight Room--Viewpoint--Object structured scene memory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

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
    semantic_status: str = "unknown"
    semantic_revision: int = 0
    semantic_label_revision: int = 0
    semantic_evidence_count: int = 0
    semantic_ordinary_evidence_count: int = 0
    semantic_evidence_tokens: tuple[str, ...] = ()
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
    observation_kind: str = "ordinary"
    evidence_group: str = ""
    camera_position_world: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    camera_rotation_world: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    camera_intrinsics: dict[str, float | int] = field(default_factory=dict)
    object_bboxes_xyxy: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.camera_position_world is not None:
            self.camera_position_world = np.asarray(
                self.camera_position_world, dtype=np.float64
            ).reshape(3)
        if self.camera_rotation_world is not None:
            self.camera_rotation_world = np.asarray(
                self.camera_rotation_world, dtype=np.float64
            ).reshape(3, 3)


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
    best_image_path: str = ""
    best_viewpoint_id: str = ""
    best_bbox_xyxy: tuple[float, float, float, float] | None = None
    best_image_score: int = 0
    best_image_revision: int = 0
    evidence_revision: int = 0
    confirmed_label: str = ""
    type_verified: bool = False
    type_verdict: str = ""
    type_reason: str = ""
    type_queried_best_revision: int = -1
    spatial_verdict: str = ""
    spatial_reason: str = ""
    spatial_query_key: tuple[Any, ...] | None = None
    is_considered: bool = False
    is_considered_strong: bool = False
    query_in_flight: bool = False
    latest_detection: LocalizedDetection | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.position_world = np.asarray(self.position_world, dtype=np.float64).reshape(3)
        self.bbox_min_world = np.asarray(self.bbox_min_world, dtype=np.float64).reshape(3)
        self.bbox_max_world = np.asarray(self.bbox_max_world, dtype=np.float64).reshape(3)
        if not self.label_votes:
            self.label_votes[self.label] += 1

    def to_dict(self) -> dict:
        """Serialize persistent state without embedding full-resolution masks."""

        return {
            "object_id": self.object_id,
            "label": self.label,
            "room_id": self.room_id,
            "position_world": self.position_world.tolist(),
            "bbox_min_world": self.bbox_min_world.tolist(),
            "bbox_max_world": self.bbox_max_world.tolist(),
            "best_confidence": float(self.best_confidence),
            "observation_count": int(self.observation_count),
            "viewpoint_ids": list(self.viewpoint_ids),
            "label_votes": dict(self.label_votes),
            "last_seen_s": float(self.last_seen_s),
            "best_image_path": self.best_image_path,
            "best_viewpoint_id": self.best_viewpoint_id,
            "best_bbox_xyxy": self.best_bbox_xyxy,
            "best_image_score": int(self.best_image_score),
            "best_image_revision": int(self.best_image_revision),
            "evidence_revision": int(self.evidence_revision),
            "confirmed_label": self.confirmed_label,
            "type_verified": bool(self.type_verified),
            "type_verdict": self.type_verdict,
            "type_reason": self.type_reason,
            "type_queried_best_revision": int(self.type_queried_best_revision),
            "spatial_verdict": self.spatial_verdict,
            "spatial_reason": self.spatial_reason,
            "spatial_query_key": self.spatial_query_key,
            "is_considered": bool(self.is_considered),
            "is_considered_strong": bool(self.is_considered_strong),
            "query_in_flight": bool(self.query_in_flight),
        }


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
        evidence_count: int = 1,
        ordinary_evidence_count: int = 0,
        evidence_tokens: tuple[str, ...] = (),
        stable_min_ordinary_views: int = 2,
        stable_confidence: float = 0.75,
    ) -> bool:
        """Replace room semantics with the latest multi-view interpretation.

        The newest interpretation is deliberately allowed to revise an older
        label.  Stability describes the amount and quality of current evidence;
        it is not a permanent lock on the first classification.
        """

        state = self.rooms[room_id]
        evidence_count = max(1, int(evidence_count))
        ordinary_evidence_count = max(0, int(ordinary_evidence_count))
        room_type = description.room_type or "unknown"
        if room_type == "unknown":
            status = "unknown"
        elif (
            ordinary_evidence_count >= int(stable_min_ordinary_views)
            and description.confidence >= float(stable_confidence)
        ):
            status = "stable"
        else:
            status = "tentative"

        label_changed = room_type != state.semantic_description
        changed = bool(
            label_changed
            or float(description.confidence) != state.semantic_confidence
            or tuple(description.visual_cues) != state.semantic_cues
            or status != state.semantic_status
            or evidence_count != state.semantic_evidence_count
            or ordinary_evidence_count != state.semantic_ordinary_evidence_count
        )
        state.semantic_description = room_type
        state.semantic_confidence = float(description.confidence)
        state.semantic_cues = tuple(description.visual_cues)
        state.semantic_status = status
        state.semantic_evidence_count = evidence_count
        state.semantic_ordinary_evidence_count = ordinary_evidence_count
        state.semantic_evidence_tokens = tuple(str(value) for value in evidence_tokens)
        state.semantic_revision += 1
        if label_changed:
            state.semantic_label_revision += 1
            # Upstream SysNav allows objects in a room to be reconsidered when
            # the room label changes, but not for confidence-only revisions.
            for object_id in state.object_ids:
                record = self.objects.get(object_id)
                if record is not None and not record.is_considered_strong:
                    record.is_considered = False
        if representative_image_path:
            state.representative_image_path = representative_image_path
        return changed

    def update_room_progress(
        self, room_id: str, coverage: float, *, exhausted: bool | None = None
    ) -> None:
        state = self.rooms[room_id]
        state.visited = True
        state.coverage = max(state.coverage, float(np.clip(coverage, 0.0, 1.0)))
        if exhausted is not None:
            state.exhausted = bool(exhausted)

    @staticmethod
    def _aabb_iou(
        minimum_a: np.ndarray,
        maximum_a: np.ndarray,
        minimum_b: np.ndarray,
        maximum_b: np.ndarray,
    ) -> float:
        intersection = np.maximum(
            0.0, np.minimum(maximum_a, maximum_b) - np.maximum(minimum_a, minimum_b)
        )
        intersection_volume = float(np.prod(intersection))
        volume_a = float(np.prod(np.maximum(0.0, maximum_a - minimum_a)))
        volume_b = float(np.prod(np.maximum(0.0, maximum_b - minimum_b)))
        union = volume_a + volume_b - intersection_volume
        return 0.0 if union <= 0.0 else intersection_volume / union

    def _match_object(self, detection: LocalizedDetection, room_id: str) -> ObjectRecord | None:
        best = None
        best_score = -np.inf
        for record in self.objects.values():
            if record.room_id != room_id or record.label != detection.detection.label:
                continue
            distance = float(np.linalg.norm(record.position_world - detection.position_world))
            iou = self._aabb_iou(
                record.bbox_min_world,
                record.bbox_max_world,
                detection.bbox_min_world,
                detection.bbox_max_world,
            )
            record_contains = bool(
                np.all(detection.position_world >= record.bbox_min_world - 0.05)
                and np.all(detection.position_world <= record.bbox_max_world + 0.05)
            )
            detection_contains = bool(
                np.all(record.position_world >= detection.bbox_min_world - 0.05)
                and np.all(record.position_world <= detection.bbox_max_world + 0.05)
            )
            if distance >= self.merge_distance_m and iou <= 0.0:
                continue
            score = -distance + 0.35 * iou + 0.10 * float(
                record_contains or detection_contains
            )
            if score > best_score:
                best, best_score = record, score
        return best

    @staticmethod
    def _is_new_semantic_view(
        record: ObjectRecord,
        viewpoint: ViewpointRecord,
        viewpoints: dict[str, ViewpointRecord],
    ) -> bool:
        if not record.viewpoint_ids:
            return True
        if viewpoint.observation_kind != "transit":
            return viewpoint.viewpoint_id not in record.viewpoint_ids
        # Continuous frames from one navigation leg correspond to one semantic
        # viewpoint in the upstream representation.  They still refine the 3-D
        # object and best crop, but do not repeatedly clear is_considered.
        return not any(
            viewpoints[viewpoint_id].observation_kind == "transit"
            and viewpoints[viewpoint_id].evidence_group == viewpoint.evidence_group
            for viewpoint_id in record.viewpoint_ids
            if viewpoint_id in viewpoints
        )

    def _add_or_update_object(
        self,
        detection: LocalizedDetection,
        room_id: str,
        viewpoint: ViewpointRecord,
    ) -> ObjectRecord:
        viewpoint_id = viewpoint.viewpoint_id
        timestamp_s = viewpoint.timestamp_s
        record = self._match_object(detection, room_id)
        mask_score = int(np.asarray(detection.detection.mask, dtype=bool).sum())
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
                best_image_path=viewpoint.image_path,
                best_viewpoint_id=viewpoint_id,
                best_bbox_xyxy=detection.detection.bbox_xyxy,
                best_image_score=mask_score,
                best_image_revision=1,
                evidence_revision=1,
                latest_detection=detection,
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
        record.latest_detection = detection
        if self._is_new_semantic_view(record, viewpoint, self.viewpoints):
            record.viewpoint_ids.append(viewpoint_id)
            record.evidence_revision += 1
            if not record.is_considered_strong:
                record.is_considered = False
        if mask_score > record.best_image_score and viewpoint.image_path:
            record.best_image_path = viewpoint.image_path
            record.best_viewpoint_id = viewpoint_id
            record.best_bbox_xyxy = detection.detection.bbox_xyxy
            record.best_image_score = mask_score
            record.best_image_revision += 1
            # This mirrors SingleObject.save_best_image(): a better close-up
            # makes the object eligible for a fresh category review.
            record.type_verified = False
            record.is_considered = False
        return record

    def add_viewpoint(
        self,
        room_id: str,
        pose_world: tuple[float, float, float] | np.ndarray,
        detections: list[LocalizedDetection],
        *,
        timestamp_s: float | None = None,
        image_path: str = "",
        observation_kind: str = "ordinary",
        evidence_group: str = "",
        camera_position_world: np.ndarray | None = None,
        camera_rotation_world: np.ndarray | None = None,
        camera_intrinsics: dict[str, float | int] | None = None,
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
            observation_kind=str(observation_kind),
            evidence_group=str(evidence_group),
            camera_position_world=camera_position_world,
            camera_rotation_world=camera_rotation_world,
            camera_intrinsics=dict(camera_intrinsics or {}),
        )
        self.viewpoints[viewpoint_id] = record
        room = self.rooms[room_id]
        room.viewpoint_ids.append(viewpoint_id)
        room.visited = True
        for detection in detections:
            self.attach_detection(viewpoint_id, detection)
        return record

    def attach_detection(
        self, viewpoint_id: str, detection: LocalizedDetection
    ) -> ObjectRecord:
        """Fuse a localized proposal into the persistent provisional Object layer."""

        if viewpoint_id not in self.viewpoints:
            raise KeyError(f"unknown viewpoint id: {viewpoint_id}")
        viewpoint = self.viewpoints[viewpoint_id]
        assigned_room = detection.room_id or viewpoint.room_id
        if assigned_room not in self.rooms:
            assigned_room = viewpoint.room_id
        obj = self._add_or_update_object(detection, assigned_room, viewpoint)
        viewpoint.object_bboxes_xyxy[obj.object_id] = detection.detection.bbox_xyxy
        if obj.object_id not in viewpoint.object_ids:
            viewpoint.object_ids.append(obj.object_id)
        return obj

    def attach_verified_detection(
        self, viewpoint_id: str, detection: LocalizedDetection
    ) -> ObjectRecord:
        """Compatibility helper: fuse an observation and mark it confirmed."""

        obj = self.attach_detection(viewpoint_id, detection)
        obj.confirmed_label = detection.detection.label
        obj.type_verified = True
        obj.type_verdict = "confirmed"
        obj.spatial_verdict = "confirmed"
        obj.is_considered = True
        return obj

    def objects_by_label(self, label: str, *, room_id: str | None = None) -> list[ObjectRecord]:
        normalised = " ".join(str(label).strip().lower().split())
        result = [
            record
            for record in self.objects.values()
            if (record.confirmed_label or record.label) == normalised
        ]
        if room_id is not None:
            result = [record for record in result if record.room_id == room_id]
        return sorted(result, key=lambda item: item.best_confidence, reverse=True)

    def nearby_labels(
        self, position_world: np.ndarray, radius_m: float, *, exclude_id: str | None = None
    ) -> tuple[str, ...]:
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        labels = {
            (record.confirmed_label or record.label)
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
            "version": 3,
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
