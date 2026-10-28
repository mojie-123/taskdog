"""Executable room-search state machine, independent of a concrete base controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from custom_envs.scripts.N2M.manifests import append_jsonl, write_json

from .async_vlm import AsyncRoomVLMManager, AsyncVLMResult
from .object_verification import ObjectVerificationManager, VerificationResult
from .rgbd import RGBDFrame
from .runtime import ObservationUpdate, SysNavComponents
from .target_reobservation import plan_target_reobservation_pose
from .transit_detection import (
    TransitObservation,
    TransitReobserveRequested,
    TransitTargetFound,
)
from .types import (
    EarlyStopDecision,
    LocalizedDetection,
    RoomDescription,
    RoomSelection,
    SearchInstruction,
)


def _wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


class SearchNavigator(Protocol):
    def current_pose(self) -> np.ndarray: ...

    def move_to(self, pose: tuple[float, float, float], *, name: str) -> dict: ...


@dataclass(frozen=True)
class SearchResult:
    found: bool
    reason: str
    target_position_world: np.ndarray | None
    target_room_id: str | None
    detection_confidence: float | None
    searched_viewpoints: int
    searched_rooms: tuple[str, ...]
    final_robot_pose: tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "reason": self.reason,
            "target_position_world": (
                None
                if self.target_position_world is None
                else self.target_position_world.astype(float).tolist()
            ),
            "target_room_id": self.target_room_id,
            "detection_confidence": self.detection_confidence,
            "searched_viewpoints": self.searched_viewpoints,
            "searched_rooms": list(self.searched_rooms),
            "final_robot_pose": list(self.final_robot_pose),
        }


@dataclass(frozen=True)
class RoomImageEvidence:
    """One saved room view and the metadata needed to audit VLM decisions."""

    viewed_room_id: str
    metric_room_id: str
    token: str
    image_path: str
    image_rgb: np.ndarray
    pose_world: tuple[float, float, float]
    coverage: float | None
    detection_labels: tuple[str, ...]
    observation_kind: str

    def event_dict(self) -> dict:
        return {
            "viewed_room_id": self.viewed_room_id,
            "metric_room_id": self.metric_room_id,
            "token": self.token,
            "image_path": self.image_path,
            "pose_world": list(self.pose_world),
            "coverage": self.coverage,
            "detection_labels": list(self.detection_labels),
            "observation_kind": self.observation_kind,
        }


@dataclass(frozen=True)
class _RoomDescriptionRequest:
    room_id: str
    total_count: int
    ordinary_count: int
    evidence: tuple[RoomImageEvidence, ...]


@dataclass(frozen=True)
class _RoomSelectionRequest:
    current_room_id: str
    candidate_room_ids: tuple[str, ...]
    state_signature: tuple


@dataclass(frozen=True)
class _EarlyStopRequest:
    current_room_id: str
    nearby_room_id: str
    current_semantic_revision: int
    nearby_semantic_revision: int
    current_images: tuple[RoomImageEvidence, ...]
    nearby_images: tuple[RoomImageEvidence, ...]


class SysNavSearchRunner:
    """Alternate VLM room decisions with classical in-room coverage."""

    def __init__(
        self,
        components: SysNavComponents,
        navigator: SearchNavigator,
        capture_frame: Callable[[], RGBDFrame],
        output_dir: Path | str,
        *,
        enable_room_semantics: bool = True,
        enable_early_stop: bool = True,
        max_viewpoints: int = 120,
    ):
        if max_viewpoints <= 0:
            raise ValueError("max_viewpoints must be positive")
        self.components = components
        self.navigator = navigator
        self.capture_frame = capture_frame
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_room_semantics = bool(enable_room_semantics)
        self.enable_early_stop = bool(enable_early_stop)
        self.max_viewpoints = int(max_viewpoints)
        # Keep the most recent image for the general room-selection API, and
        # retain the full history for multi-view early-stop comparisons.
        self.room_images: dict[str, np.ndarray] = {}
        self.room_image_history: dict[str, list[RoomImageEvidence]] = {}
        self._peeked_edges: set[tuple[str, str]] = set()
        self._searched_rooms: list[str] = []
        self._viewpoint_count = 0
        self._ordinary_viewpoint_counts: dict[str, int] = {}
        self._room_semantic_last_counts: dict[str, tuple[int, int]] = {}
        self._verification_manager: ObjectVerificationManager | None = None
        self._verified_target: LocalizedDetection | None = None
        self._instruction: SearchInstruction | None = None
        self._reobserve_attempts: dict[str, int] = {}
        self._reobserve_in_progress = False
        self._room_vlm_manager: AsyncRoomVLMManager | None = None
        self._room_selection_cache: dict[
            str, tuple[_RoomSelectionRequest, RoomSelection]
        ] = {}
        self._room_selection_consumed: set[str] = set()
        self._pending_room_switch: tuple[str, str] | None = None

    def _record_event(self, event: str, **payload) -> None:
        append_jsonl(
            self.output_dir / "events.jsonl",
            {"event": event, **payload},
        )

    def _save_image(self, frame: RGBDFrame, token: str) -> str:
        path = self.output_dir / "images" / f"{token}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame.rgb, mode="RGB").save(path)
        return str(path)

    @staticmethod
    def _room_semantic_snapshot(state) -> dict:
        return {
            "room_type": state.semantic_description,
            "confidence": float(state.semantic_confidence),
            "status": state.semantic_status,
            "revision": int(state.semantic_revision),
            "visual_cues": list(state.semantic_cues),
        }

    def _room_selection_snapshot(
        self, current_room_id: str
    ) -> tuple[list[dict], dict[str, np.ndarray], tuple]:
        candidates = self.components.room_graph.candidate_summaries(
            current_room_id, self.components.scene_memory.rooms
        )
        # Preselection runs while the current room is still being searched.
        # Its output is for the *next* room, so never offer the current room to
        # the VLM even though it is not yet marked exhausted.
        candidates = [
            item for item in candidates if item["room_id"] != current_room_id
        ]
        candidate_ids = {str(item["room_id"]) for item in candidates}
        images = {
            room_id: np.ascontiguousarray(image).copy()
            for room_id, image in self.room_images.items()
            if room_id in candidate_ids
        }
        signature = tuple(
            (
                str(item["room_id"]),
                int(item["runtime_description_revision"]),
                tuple(str(value) for value in item["observed_objects"]),
                bool(item["visited"]),
                (
                    self.room_image_history[item["room_id"]][-1].token
                    if self.room_image_history.get(item["room_id"])
                    else ""
                ),
            )
            for item in candidates
        )
        return candidates, images, signature

    def _ensure_room_preselection(self, current_room_id: str) -> None:
        manager = getattr(self, "_room_vlm_manager", None)
        reasoner = getattr(self.components.search_policy, "reasoner", None)
        instruction = getattr(self, "_instruction", None)
        if manager is None or reasoner is None or instruction is None:
            return
        key = ("room_selection", current_room_id)
        if manager.is_active(key):
            return
        candidates, images, signature = self._room_selection_snapshot(
            current_room_id
        )
        cached = self._room_selection_cache.get(current_room_id)
        if cached is not None:
            request, _ = cached
            if request.state_signature == signature:
                return
            self._room_selection_cache.pop(current_room_id, None)
            self._record_event(
                "room_preselection_discarded",
                current_room=current_room_id,
                reason="candidate room state changed after preselection completed",
            )
        if not candidates:
            return
        informative = any(
            item["runtime_description"] or item["observed_objects"]
            for item in candidates
        ) or bool(images)
        if not informative:
            return
        request = _RoomSelectionRequest(
            current_room_id=current_room_id,
            candidate_room_ids=tuple(str(item["room_id"]) for item in candidates),
            state_signature=signature,
        )
        submitted = manager.submit(
            "room_selection",
            key,
            request,
            reasoner.select_room,
            instruction,
            candidates,
            room_images=images,
        )
        if submitted:
            self._room_selection_consumed.discard(current_room_id)
            self._record_event(
                "room_preselection_enqueued",
                current_room=current_room_id,
                candidate_room_ids=list(request.candidate_room_ids),
                state_signature=list(request.state_signature),
            )
            print(
                f"[SYSNAV][ROOM_PRESELECT_ENQUEUED] current={current_room_id} "
                f"candidates={list(request.candidate_room_ids)}",
                flush=True,
            )

    def _fallback_room_selection(
        self, current_room_id: str, *, reason: str
    ) -> RoomSelection | None:
        candidates, _, _ = self._room_selection_snapshot(current_room_id)
        if not candidates:
            return None
        selected = min(
            candidates,
            key=lambda item: (float(item["travel_cost"]), str(item["room_id"])),
        )
        return RoomSelection(
            room_id=str(selected["room_id"]), reason=str(reason), used_vlm=False
        )

    def _take_room_preselection(self, current_room_id: str) -> RoomSelection | None:
        self._poll_room_vlm_results()
        cached = self._room_selection_cache.pop(current_room_id, None)
        _, _, signature = self._room_selection_snapshot(current_room_id)
        if cached is not None:
            request, selection = cached
            if (
                request.state_signature == signature
                and selection.room_id in request.candidate_room_ids
            ):
                self._room_selection_consumed.add(current_room_id)
                return selection
            self._record_event(
                "room_preselection_discarded",
                current_room=current_room_id,
                reason="candidate room state changed before selection was consumed",
            )
        self._room_selection_consumed.add(current_room_id)
        return self._fallback_room_selection(
            current_room_id,
            reason=(
                "asynchronous VLM preselection was not ready; used travel cost"
            ),
        )

    def _apply_room_description(
        self, request: _RoomDescriptionRequest, description: RoomDescription
    ) -> LocalizedDetection | None:
        room_id = request.room_id
        state = self.components.scene_memory.rooms[room_id]
        previous = {
            "room_type": state.semantic_description,
            "confidence": state.semantic_confidence,
            "status": state.semantic_status,
            "revision": state.semantic_revision,
        }
        evidence = request.evidence
        self._room_semantic_last_counts[room_id] = (
            request.total_count,
            request.ordinary_count,
        )
        changed = self.components.scene_memory.update_room_description(
            room_id,
            description,
            representative_image_path=evidence[-1].image_path,
            evidence_count=len(evidence),
            ordinary_evidence_count=sum(
                item.observation_kind == "ordinary" for item in evidence
            ),
            evidence_tokens=tuple(item.token for item in evidence),
            stable_min_ordinary_views=(
                self.components.config.search.room_semantic_stable_min_ordinary_views
            ),
            stable_confidence=(
                self.components.config.search.room_semantic_stable_confidence
            ),
        )
        self._record_event(
            "room_described",
            room_id=room_id,
            room_type=description.room_type,
            confidence=description.confidence,
            visual_cues=list(description.visual_cues),
            semantic_status=state.semantic_status,
            semantic_revision=state.semantic_revision,
            changed=changed,
            previous=previous,
            evidence=[item.event_dict() for item in evidence],
            asynchronous=True,
        )
        print(
            f"[SYSNAV][ROOM_SEMANTIC_UPDATE] room={room_id} "
            f"revision={state.semantic_revision} status={state.semantic_status} "
            f"type={state.semantic_description!r} "
            f"confidence={state.semantic_confidence:.2f} "
            f"evidence={state.semantic_evidence_count} "
            f"ordinary={state.semantic_ordinary_evidence_count} "
            f"previous={previous['room_type']!r}/{previous['status']} async=True",
            flush=True,
        )
        self._submit_object_records(self.components.scene_memory.objects.values())
        target = self._poll_object_verification()
        room_graph = getattr(self.components, "room_graph", None)
        navigator = getattr(self, "navigator", None)
        if room_graph is not None and navigator is not None:
            pose = navigator.current_pose()
            current_room = room_graph.room_at(float(pose[0]), float(pose[1]))
            if current_room is not None:
                self._ensure_room_preselection(current_room)
        return target

    def _apply_room_vlm_result(
        self, result: AsyncVLMResult
    ) -> LocalizedDetection | None:
        if result.error:
            self._record_event(
                f"{result.kind}_failed",
                key=list(result.key),
                error=result.error,
                asynchronous=True,
            )
            print(
                f"[SYSNAV][ASYNC_VLM_FAILED] kind={result.kind} "
                f"key={list(result.key)} error={result.error}",
                flush=True,
            )
            return None

        if result.kind == "room_description":
            return self._apply_room_description(result.context, result.value)

        if result.kind == "room_selection":
            request: _RoomSelectionRequest = result.context
            if request.current_room_id in self._room_selection_consumed:
                self._record_event(
                    "room_preselection_discarded",
                    current_room=request.current_room_id,
                    reason="fallback or another selection was already committed",
                )
                return None
            _, _, signature = self._room_selection_snapshot(
                request.current_room_id
            )
            selection: RoomSelection = result.value
            if (
                signature != request.state_signature
                or selection.room_id not in request.candidate_room_ids
            ):
                self._record_event(
                    "room_preselection_discarded",
                    current_room=request.current_room_id,
                    room_id=selection.room_id,
                    reason="candidate room state changed while VLM was running",
                )
                return None
            self._room_selection_cache[request.current_room_id] = (
                request,
                selection,
            )
            self._record_event(
                "room_preselection_ready",
                current_room=request.current_room_id,
                room_id=selection.room_id,
                reason=selection.reason,
            )
            print(
                f"[SYSNAV][ROOM_PRESELECT_READY] current={request.current_room_id} "
                f"next={selection.room_id}",
                flush=True,
            )
            return None

        if result.kind == "early_stop":
            request: _EarlyStopRequest = result.context
            decision: EarlyStopDecision = result.value
            pose = self.navigator.current_pose()
            metric_room = self.components.room_graph.room_at(
                float(pose[0]), float(pose[1])
            )
            stale = metric_room != request.current_room_id
            if stale:
                self._record_event(
                    "early_stop_decision_discarded",
                    current_room=request.current_room_id,
                    nearby_room=request.nearby_room_id,
                    reason="robot already left the room while VLM was running",
                )
                return None
            self._record_event(
                "early_stop_decision",
                current_room=request.current_room_id,
                nearby_room=request.nearby_room_id,
                switch_room=decision.switch_room,
                reason=decision.reason,
                asynchronous=True,
            )
            if decision.switch_room:
                self._pending_room_switch = (
                    request.current_room_id,
                    request.nearby_room_id,
                )
                print(
                    f"[SYSNAV][EARLY_STOP_READY] current={request.current_room_id} "
                    f"next={request.nearby_room_id}",
                    flush=True,
                )
            return None

        raise RuntimeError(f"unknown asynchronous VLM result kind: {result.kind}")

    def _poll_room_vlm_results(self) -> LocalizedDetection | None:
        manager = getattr(self, "_room_vlm_manager", None)
        if manager is None:
            return self._poll_object_verification()
        target = None
        for result in manager.poll():
            candidate = self._apply_room_vlm_result(result)
            if candidate is not None:
                target = candidate
        object_target = self._poll_object_verification()
        return target if target is not None else object_target

    def _take_pending_room_switch(self, current_room_id: str) -> str | None:
        pending = getattr(self, "_pending_room_switch", None)
        if pending is None or pending[0] != current_room_id:
            return None
        self._pending_room_switch = None
        self._room_selection_consumed.add(current_room_id)
        return pending[1]

    def _start_object_verification(self, instruction: SearchInstruction) -> None:
        self._instruction = instruction
        self._reobserve_attempts.clear()
        self._reobserve_in_progress = False
        self._room_vlm_manager = AsyncRoomVLMManager(max_workers=3)
        self._room_selection_cache.clear()
        self._room_selection_consumed.clear()
        self._pending_room_switch = None
        self._verification_manager = ObjectVerificationManager(
            self.components.scene_memory,
            self.components.vlm,
            instruction,
            min_best_image_pixels=(
                self.components.config.search.object_vlm_min_mask_pixels
            ),
        )
        detector = getattr(self.navigator, "transit_detector", None)
        if detector is not None:
            detector.set_observation_callback(
                self._ingest_transit_observation,
                verified_target_provider=self._take_verified_target,
                interrupt_on_candidate=False,
            )

    def _consume_verification_results(
        self, results: list[VerificationResult]
    ) -> LocalizedDetection | None:
        for result in results:
            snapshot = result.snapshot
            spatial = result.spatial_decision
            self._record_event(
                "object_instance_reviewed",
                object_id=snapshot.object_id,
                role=snapshot.role,
                query_key=list(snapshot.query_key),
                best_image_revision=snapshot.best_image_revision,
                evidence_revision=snapshot.evidence_revision,
                type_verdict=result.type_decision.verdict,
                type_reason=result.type_decision.reason,
                spatial_verdict=None if spatial is None else spatial.verdict,
                spatial_reason=None if spatial is None else spatial.reason,
                room_label=snapshot.room_label,
                viewpoint_count=len(snapshot.viewpoint_images_rgb),
                error=result.error,
            )
            print(
                f"[SYSNAV][OBJECT_REVIEW] object={snapshot.object_id} "
                f"role={snapshot.role} type={result.type_decision.verdict} "
                f"spatial={None if spatial is None else spatial.verdict} "
                f"views={len(snapshot.viewpoint_images_rgb)} "
                f"room={snapshot.room_label or 'unknown'}",
                flush=True,
            )
            if result.target_confirmed and snapshot.detection is not None:
                self._verified_target = snapshot.detection
        return self._verified_target

    def _poll_object_verification(self) -> LocalizedDetection | None:
        manager = getattr(self, "_verification_manager", None)
        if manager is None:
            return getattr(self, "_verified_target", None)
        return self._consume_verification_results(manager.poll())

    def _take_verified_target(self) -> LocalizedDetection | None:
        self._poll_object_verification()
        target = self._verified_target
        self._verified_target = None
        return target

    def _submit_object_records(self, records) -> list[str]:
        manager = self._verification_manager
        if manager is None:
            return []
        submitted = manager.consider(records)
        for object_id in submitted:
            record = self.components.scene_memory.objects[object_id]
            self._record_event(
                "object_instance_query_enqueued",
                object_id=object_id,
                label=record.label,
                room_id=record.room_id,
                evidence_revision=record.evidence_revision,
                best_image_revision=record.best_image_revision,
                viewpoint_ids=list(record.viewpoint_ids),
            )
            print(
                f"[SYSNAV][OBJECT_QUERY] object={object_id} label={record.label} "
                f"evidence_revision={record.evidence_revision} "
                f"views={len(record.viewpoint_ids)}",
                flush=True,
            )
        return submitted

    def _provisional_reobserve_request(self, records) -> TransitReobserveRequested | None:
        """Return a stable, still-too-small target that merits a close-up."""

        config = self.components.config.search
        instruction = self._instruction
        if (
            not config.provisional_reobserve_enabled
            or self._reobserve_in_progress
            or instruction is None
            or self._verified_target is not None
        ):
            return None

        eligible = []
        for record in records:
            label = record.confirmed_label or record.label
            if label != instruction.target_object:
                continue
            if record.is_considered_strong or record.query_in_flight:
                continue
            if record.type_verdict == "rejected":
                continue
            if (
                self._reobserve_attempts.get(record.object_id, 0)
                >= config.provisional_reobserve_max_attempts
            ):
                continue
            if record.observation_count < config.provisional_reobserve_min_observations:
                continue
            if not (
                config.provisional_reobserve_min_mask_pixels
                <= record.best_image_score
                < config.object_vlm_min_mask_pixels
            ):
                continue
            if record.best_confidence < config.provisional_reobserve_min_confidence:
                continue
            latest = record.latest_detection
            if latest is None:
                continue
            residual = float(
                np.linalg.norm(latest.position_world - record.position_world)
            )
            if residual > config.provisional_reobserve_max_position_residual_m:
                continue
            eligible.append((record.best_image_score, record.best_confidence, record))

        if not eligible:
            return None
        _, _, record = max(eligible, key=lambda item: (item[0], item[1]))
        return TransitReobserveRequested(
            object_id=record.object_id,
            room_id=record.room_id,
            position_world=record.position_world,
            best_image_score=record.best_image_score,
            observation_count=record.observation_count,
        )

    def _ingest_transit_observation(self, observation: TransitObservation) -> None:
        self._poll_room_vlm_results()
        room_id = observation.metric_room_id
        if room_id is None:
            self._record_event(
                "transit_scene_commit_skipped",
                sequence=observation.sequence,
                reason="robot pose is outside the anonymous room map",
            )
            return
        update = self.components.commit_observation(
            observation.frame,
            observation.robot_pose_world,
            room_id,
            observation.detections,
            observation.localized_detections,
            planned_viewpoint_index=None,
            image_path=observation.image_path,
            observation_kind="transit",
            evidence_group=observation.navigation_name,
        )
        self._record_event(
            "transit_scene_updated",
            sequence=observation.sequence,
            navigation_name=observation.navigation_name,
            viewpoint_id=update.viewpoint.viewpoint_id,
            object_ids=[record.object_id for record in update.object_records],
        )
        self._submit_object_records(update.object_records)
        self._poll_room_vlm_results()
        request = self._provisional_reobserve_request(update.object_records)
        if request is not None:
            self._record_event(
                "provisional_target_reobserve_requested",
                object_id=request.object_id,
                room_id=request.room_id,
                position_world=request.position_world.tolist(),
                best_image_score=request.best_image_score,
                observation_count=request.observation_count,
                sequence=observation.sequence,
                navigation_name=observation.navigation_name,
            )
            print(
                f"[SYSNAV][TARGET_REOBSERVE_REQUEST] object={request.object_id} "
                f"observations={request.observation_count} "
                f"mask_pixels={request.best_image_score} "
                f"position={np.round(request.position_world, 3).tolist()}",
                flush=True,
            )
            raise request

    def _perform_target_reobservation(
        self, request: TransitReobserveRequested
    ) -> None:
        """Move to a safe close-up pose, review it, then resume navigation."""

        config = self.components.config.search
        attempt = self._reobserve_attempts.get(request.object_id, 0) + 1
        self._reobserve_attempts[request.object_id] = attempt
        preferred = max(
            config.provisional_reobserve_min_standoff_m,
            config.provisional_reobserve_preferred_standoff_m
            - 0.20 * float(attempt - 1),
        )
        current = self.navigator.current_pose()
        planned = plan_target_reobservation_pose(
            self.components.room_map,
            self.components.nav_map,
            request.room_id,
            request.position_world,
            current,
            preferred_standoff_m=preferred,
            min_standoff_m=config.provisional_reobserve_min_standoff_m,
            max_standoff_m=config.provisional_reobserve_max_standoff_m,
        )
        if planned is None:
            self._reobserve_attempts[request.object_id] = (
                config.provisional_reobserve_max_attempts
            )
            self._record_event(
                "provisional_target_reobserve_skipped",
                object_id=request.object_id,
                room_id=request.room_id,
                reason="no searchable Nav2-free pose in the standoff annulus",
                position_world=request.position_world.tolist(),
            )
            print(
                f"[SYSNAV][TARGET_REOBSERVE_SKIPPED] object={request.object_id} "
                "reason=no_safe_pose",
                flush=True,
            )
            return

        token = f"target_reobserve_{request.object_id}_{attempt:02d}"
        self._record_event(
            "provisional_target_reobserve_started",
            object_id=request.object_id,
            room_id=request.room_id,
            attempt=attempt,
            source_position_world=request.position_world.tolist(),
            source_mask_pixels=request.best_image_score,
            source_observation_count=request.observation_count,
            start_pose=np.asarray(current, dtype=float).tolist(),
            goal_pose=list(planned.pose),
            target_distance_m=planned.target_distance_m,
            travel_distance_m=planned.travel_distance_m,
            bearing_change_deg=planned.bearing_change_deg,
        )
        print(
            f"[SYSNAV][TARGET_REOBSERVE] object={request.object_id} "
            f"attempt={attempt}/{config.provisional_reobserve_max_attempts} "
            f"goal={np.round(planned.pose, 3).tolist()} "
            f"standoff={planned.target_distance_m:.2f}m",
            flush=True,
        )

        self._reobserve_in_progress = True
        try:
            navigation = self.navigator.move_to(
                planned.pose, name=f"TARGET_REOBSERVE_{request.object_id}_{attempt:02d}"
            )
            frame = self.capture_frame()
            image_path = self._save_image(frame, token)
            pose = self.navigator.current_pose()
            metric_room = self.components.room_graph.room_at(
                float(pose[0]), float(pose[1])
            )
            room_id = metric_room or request.room_id
            update = self.components.process_observation(
                frame,
                pose,
                room_id,
                planned_viewpoint_index=None,
                image_path=image_path,
                observation_kind="target_reobserve",
                evidence_group=token,
            )
            self._submit_object_records(update.object_records)
            target = self._poll_object_verification()
            record = self.components.scene_memory.objects.get(request.object_id)
            manager = self._verification_manager
            # At a deliberately selected close-up it is better to wait for
            # the one resulting review than to immediately walk away again.
            if (
                target is None
                and manager is not None
                and record is not None
                and record.query_in_flight
            ):
                target = self._consume_verification_results(manager.wait())
            self._record_event(
                "provisional_target_reobserve_completed",
                object_id=request.object_id,
                room_id=room_id,
                attempt=attempt,
                image_path=image_path,
                navigation=navigation,
                localized_object_ids=[
                    item.object_id for item in update.object_records
                ],
                best_image_score=(
                    None if record is None else record.best_image_score
                ),
                type_verdict=(None if record is None else record.type_verdict),
                spatial_verdict=(
                    None if record is None else record.spatial_verdict
                ),
                target_verified=target is not None,
            )
            print(
                f"[SYSNAV][TARGET_REOBSERVE_DONE] object={request.object_id} "
                f"mask_pixels={None if record is None else record.best_image_score} "
                f"verified={target is not None}",
                flush=True,
            )
            if target is not None:
                raise TransitTargetFound(target)
        finally:
            self._reobserve_in_progress = False

    def _navigate_with_transit_detection(
        self, pose: tuple[float, float, float], *, name: str
    ) -> dict:
        """Execute a goal, temporarily diverting for useful target close-ups."""

        while True:
            try:
                return self.navigator.move_to(pose, name=name)
            except TransitReobserveRequested as request:
                self._perform_target_reobservation(request)
                self._record_event(
                    "navigation_resumed_after_target_reobserve",
                    navigation_name=name,
                    destination_pose=list(pose),
                    object_id=request.object_id,
                )

    def _remember_room_image(
        self,
        *,
        viewed_room_id: str,
        metric_room_id: str,
        token: str,
        image_path: str,
        image_rgb: np.ndarray,
        pose_world: np.ndarray,
        update: ObservationUpdate,
        planned_viewpoint_index: int | None,
    ) -> RoomImageEvidence:
        rgb = np.ascontiguousarray(image_rgb).copy()
        pose = tuple(float(value) for value in np.asarray(pose_world).reshape(3))
        evidence = RoomImageEvidence(
            viewed_room_id=viewed_room_id,
            metric_room_id=metric_room_id,
            token=token,
            image_path=image_path,
            image_rgb=rgb,
            pose_world=pose,
            coverage=(None if update.coverage is None else float(update.coverage)),
            detection_labels=tuple(item.label for item in update.detections),
            observation_kind=(
                "ordinary"
                if planned_viewpoint_index is not None
                and viewed_room_id == metric_room_id
                else "peek"
            ),
        )
        self.room_images[viewed_room_id] = rgb
        self.room_image_history.setdefault(viewed_room_id, []).append(evidence)
        return evidence

    def _semantic_room_evidence(
        self, room_id: str
    ) -> tuple[RoomImageEvidence, ...]:
        """Select a bounded ordinary-first multi-view room evidence set."""

        history = list(getattr(self, "room_image_history", {}).get(room_id, ()))
        if not history:
            return ()
        limit = int(self.components.config.search.room_semantic_max_images)
        ordinary = [item for item in history if item.observation_kind == "ordinary"]
        partial = [item for item in history if item.observation_kind != "ordinary"]
        if not ordinary:
            return tuple(partial[-limit:])

        # Preserve up to two doorway/context views, but let observations made
        # inside the room dominate the multi-view classification.
        partial_limit = min(2, len(partial), max(0, limit - 1))
        ordinary_limit = limit - partial_limit
        selected = ordinary[-ordinary_limit:]
        if partial_limit:
            selected.extend(partial[-partial_limit:])
        order = {id(item): index for index, item in enumerate(history)}
        selected.sort(key=lambda item: order[id(item)])
        return tuple(selected)

    def _room_semantic_update_due(self, room_id: str) -> bool:
        history = list(getattr(self, "room_image_history", {}).get(room_id, ()))
        if not history:
            return False
        total_count = len(history)
        ordinary_count = sum(
            item.observation_kind == "ordinary" for item in history
        )
        last_total, last_ordinary = getattr(
            self, "_room_semantic_last_counts", {}
        ).get(room_id, (0, 0))
        state = self.components.scene_memory.rooms[room_id]
        if state.semantic_revision == 0:
            return True
        if last_ordinary == 0 and ordinary_count > 0:
            return True
        stable_min = int(
            self.components.config.search.room_semantic_stable_min_ordinary_views
        )
        if last_ordinary < stable_min <= ordinary_count:
            return True
        return (
            total_count - last_total
            >= self.components.config.search.room_semantic_refresh_min_new_images
        )

    def _early_stop_current_evidence(
        self, room_id: str
    ) -> tuple[RoomImageEvidence, ...]:
        """Return recent ordinary views, excluding stale doorway peeks."""

        history = list(getattr(self, "room_image_history", {}).get(room_id, ()))
        ordinary = [
            item
            for item in history
            if item.observation_kind == "ordinary"
            and item.metric_room_id == room_id
        ]
        pool = ordinary or history
        limit = int(self.components.config.search.early_stop_max_current_images)
        if pool:
            return tuple(pool[-limit:])

        # Preserve compatibility with lightweight controllers/tests that only
        # populate the former single-image cache.
        image = self.room_images.get(room_id)
        if image is None:
            return ()
        return (
            RoomImageEvidence(
                viewed_room_id=room_id,
                metric_room_id=room_id,
                token="legacy_room_image",
                image_path="",
                image_rgb=np.asarray(image),
                pose_world=(0.0, 0.0, 0.0),
                coverage=float(self.components.scene_memory.rooms[room_id].coverage),
                detection_labels=(),
                observation_kind="ordinary",
            ),
        )

    def _describe_room(
        self,
        room_id: str,
        frame: RGBDFrame,
        update: ObservationUpdate,
        image_path: str,
    ) -> LocalizedDetection | None:
        del frame, update, image_path
        target = self._poll_room_vlm_results()
        vlm = self.components.vlm
        if not self.enable_room_semantics or vlm is None:
            return target
        manager = self._room_vlm_manager
        if manager is None:
            manager = AsyncRoomVLMManager(max_workers=3)
            self._room_vlm_manager = manager
        key = ("room_description", room_id)
        if manager.is_active(key):
            return target
        if not self._room_semantic_update_due(room_id):
            return target
        evidence = self._semantic_room_evidence(room_id)
        if not evidence:
            return target
        history = self.room_image_history.get(room_id, ())
        total_count = len(history)
        ordinary_count = sum(
            item.observation_kind == "ordinary" for item in history
        )
        request = _RoomDescriptionRequest(
            room_id=room_id,
            total_count=total_count,
            ordinary_count=ordinary_count,
            evidence=evidence,
        )
        submitted = manager.submit(
            "room_description",
            key,
            request,
            vlm.describe_room_multiview,
            room_id,
            tuple(np.ascontiguousarray(item.image_rgb).copy() for item in evidence),
            observation_kinds=tuple(
                item.observation_kind for item in evidence
            ),
            detected_labels=tuple(
                sorted(
                    {
                        label
                        for item in evidence
                        for label in item.detection_labels
                    }
                )
            ),
        )
        if submitted:
            self._record_event(
                "room_description_enqueued",
                room_id=room_id,
                evidence=[item.event_dict() for item in evidence],
                total_count=total_count,
                ordinary_count=ordinary_count,
                asynchronous=True,
            )
            print(
                f"[SYSNAV][ROOM_SEMANTIC_ENQUEUED] room={room_id} "
                f"evidence={len(evidence)} ordinary={ordinary_count}",
                flush=True,
            )
        return target

    def _observe(
        self,
        room_id: str,
        *,
        token: str,
        planned_viewpoint_index: int | None,
        semantic_room_id: str | None = None,
    ) -> tuple[RGBDFrame, ObservationUpdate, LocalizedDetection | None]:
        # The robot can remain on the current side of a doorway while the
        # camera looks into its neighbour.  Keep the viewpoint attached to the
        # robot's metric room, but attach the image-derived room semantics to
        # the room actually being viewed.
        viewed_room = semantic_room_id or room_id
        frame = self.capture_frame()
        image_path = self._save_image(frame, token)
        pose = self.navigator.current_pose()
        update = self.components.process_observation(
            frame,
            pose,
            room_id,
            planned_viewpoint_index=planned_viewpoint_index,
            image_path=image_path,
            observation_kind=(
                "ordinary" if planned_viewpoint_index is not None else "peek"
            ),
            evidence_group=token,
        )
        self._remember_room_image(
            viewed_room_id=viewed_room,
            metric_room_id=room_id,
            token=token,
            image_path=image_path,
            image_rgb=frame.rgb,
            pose_world=pose,
            update=update,
            planned_viewpoint_index=planned_viewpoint_index,
        )
        target = self._describe_room(
            viewed_room, frame, update, image_path
        )
        self._submit_object_records(update.object_records)
        if target is None:
            target = self._poll_room_vlm_results()
        if planned_viewpoint_index is not None:
            self._ensure_room_preselection(room_id)
        self._record_event(
            "observation",
            room_id=room_id,
            semantic_room_id=viewed_room,
            token=token,
            image_path=image_path,
            coverage=update.coverage,
            detections=[
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox_xyxy": list(item.bbox_xyxy),
                }
                for item in update.detections
            ],
            localized=[
                {
                    "label": item.detection.label,
                    "confidence": item.detection.confidence,
                    "position_world": item.position_world.astype(float).tolist(),
                    "room_id": item.room_id,
                    "point_count": item.point_count,
                }
                for item in update.localized_detections
            ],
        )
        return frame, update, target

    def _move_between_rooms(self, current_room: str, target_room: str) -> str:
        if current_room == target_room:
            return current_room
        path, distance = self.components.room_graph.shortest_path(
            current_room, target_room
        )
        if not path:
            raise RuntimeError(f"no room path from {current_room} to {target_room}")
        self._record_event(
            "room_route",
            source_room=current_room,
            target_room=target_room,
            path=path,
            cost=distance,
        )
        active = current_room
        for next_room in path[1:]:
            anchor = self.components.room_graph.execution_anchor(active, next_room)
            metrics = self._navigate_with_transit_detection(
                anchor, name=f"ROOM_{active}_TO_{next_room}"
            )
            self._record_event(
                "room_entered",
                source_room=active,
                room_id=next_room,
                anchor=list(anchor),
                navigation=metrics,
            )
            active = next_room
        return active

    def _portal_peek_pose(
        self, current_room: str, nearby_room: str
    ) -> tuple[float, float, float]:
        current_side = self.components.room_graph.execution_anchor(
            nearby_room, current_room
        )
        return (
            current_side[0],
            current_side[1],
            _wrap_angle(current_side[2] + math.pi),
        )

    def _consider_adjacent_rooms(
        self, current_room: str
    ) -> tuple[str | None, LocalizedDetection | None]:
        target = self._poll_room_vlm_results()
        if target is not None:
            return None, target
        pending_switch = self._take_pending_room_switch(current_room)
        if pending_switch is not None:
            return pending_switch, None
        if (
            not self.enable_early_stop
            or self.components.vlm is None
            or current_room not in self.room_images
        ):
            return None, None
        state = self.components.scene_memory.rooms[current_room]
        if state.coverage < self.components.config.search.early_stop_min_room_coverage:
            return None, None
        ordinary_count = self._ordinary_viewpoint_counts.get(current_room, 0)
        if (
            ordinary_count
            < self.components.config.search.early_stop_min_ordinary_viewpoints
        ):
            return None, None
        if self._viewpoint_count >= self.max_viewpoints:
            return None, None
        for nearby_room in self.components.room_graph.nodes[current_room].neighbors:
            edge = (current_room, nearby_room)
            if edge in self._peeked_edges:
                continue
            self._peeked_edges.add(edge)
            pose = self._portal_peek_pose(current_room, nearby_room)
            metrics = self._navigate_with_transit_detection(
                pose, name=f"PEEK_{current_room}_TO_{nearby_room}"
            )
            self._viewpoint_count += 1
            frame, update, target = self._observe(
                current_room,
                token=f"peek_{current_room}_{nearby_room}_{self._viewpoint_count:04d}",
                planned_viewpoint_index=None,
                semantic_room_id=nearby_room,
            )
            self._record_event(
                "portal_peek",
                current_room=current_room,
                nearby_room=nearby_room,
                pose=list(pose),
                navigation=metrics,
            )
            if target is not None:
                return None, target
            # The doorway image is often the first visual evidence for a
            # candidate room, so start next-room preselection immediately
            # instead of waiting for another ordinary observation.
            self._ensure_room_preselection(current_room)
            current_evidence = self._early_stop_current_evidence(current_room)
            nearby_history = getattr(self, "room_image_history", {}).get(
                nearby_room, ()
            )
            if nearby_history:
                nearby_evidence = (nearby_history[-1],)
            else:
                nearby_evidence = (
                    RoomImageEvidence(
                        viewed_room_id=nearby_room,
                        metric_room_id=current_room,
                        token=(
                            f"peek_{current_room}_{nearby_room}_"
                            f"{self._viewpoint_count:04d}"
                        ),
                        image_path="",
                        image_rgb=np.asarray(frame.rgb),
                        pose_world=tuple(float(value) for value in pose),
                        coverage=None,
                        detection_labels=tuple(
                            item.label for item in getattr(update, "detections", ())
                        ),
                        observation_kind="peek",
                    ),
                )
            self._record_event(
                "early_stop_vlm_inputs",
                current_room=current_room,
                nearby_room=nearby_room,
                current_images=[item.event_dict() for item in current_evidence],
                nearby_images=[item.event_dict() for item in nearby_evidence],
            )
            manager = getattr(self, "_room_vlm_manager", None)
            reasoner = getattr(self.components.search_policy, "reasoner", None)
            instruction = getattr(self, "_instruction", None)
            if manager is None or reasoner is None or instruction is None:
                return None, None
            current_state = self.components.scene_memory.rooms[current_room]
            nearby_state = self.components.scene_memory.rooms[nearby_room]
            request = _EarlyStopRequest(
                current_room_id=current_room,
                nearby_room_id=nearby_room,
                current_semantic_revision=int(current_state.semantic_revision),
                nearby_semantic_revision=int(nearby_state.semantic_revision),
                current_images=current_evidence,
                nearby_images=nearby_evidence,
            )
            submitted = manager.submit(
                "early_stop",
                ("early_stop", current_room, nearby_room),
                request,
                reasoner.should_switch_room,
                instruction,
                current_room,
                tuple(
                    np.ascontiguousarray(item.image_rgb).copy()
                    for item in current_evidence
                ),
                nearby_room,
                tuple(
                    np.ascontiguousarray(item.image_rgb).copy()
                    for item in nearby_evidence
                ),
                current_room_semantics=self._room_semantic_snapshot(current_state),
                nearby_room_semantics=self._room_semantic_snapshot(nearby_state),
            )
            if submitted:
                self._record_event(
                    "early_stop_enqueued",
                    current_room=current_room,
                    nearby_room=nearby_room,
                    asynchronous=True,
                )
                print(
                    f"[SYSNAV][EARLY_STOP_ENQUEUED] current={current_room} "
                    f"nearby={nearby_room}",
                    flush=True,
                )
            # The decision runs in the background. Resume ordinary coverage
            # immediately; a positive result is consumed at the next safe
            # navigation boundary.
            return None, None
        return None, None

    def _result(
        self,
        *,
        found: bool,
        reason: str,
        target: LocalizedDetection | None = None,
    ) -> SearchResult:
        pose = tuple(float(value) for value in self.navigator.current_pose())
        result = SearchResult(
            found=found,
            reason=reason,
            target_position_world=(
                None if target is None else target.position_world.copy()
            ),
            target_room_id=None if target is None else target.room_id,
            detection_confidence=(
                None if target is None else target.detection.confidence
            ),
            searched_viewpoints=self._viewpoint_count,
            searched_rooms=tuple(self._searched_rooms),
            final_robot_pose=pose,
        )
        self.components.scene_memory.save(self.output_dir / "scene_memory.json")
        write_json(self.output_dir / "result.json", result.to_dict())
        return result

    def _wait_for_final_target(self) -> LocalizedDetection | None:
        """Drain semantic snapshots before the final object-review barrier."""

        room_manager = self._room_vlm_manager
        target = None
        if room_manager is not None and room_manager.has_pending:
            for result in room_manager.wait():
                candidate = self._apply_room_vlm_result(result)
                if candidate is not None:
                    target = candidate
        object_manager = self._verification_manager
        if target is None and object_manager is not None and object_manager.has_pending:
            target = self._consume_verification_results(object_manager.wait())
        return target

    def run(self, instruction: SearchInstruction) -> SearchResult:
        try:
            return self._run_search(instruction)
        except TransitTargetFound as verified:
            return self._result(
                found=True,
                reason="target verified asynchronously during navigation",
                target=verified.target,
            )
        finally:
            detector = getattr(self.navigator, "transit_detector", None)
            if detector is not None:
                detector.save_summary()
            if self._verification_manager is not None:
                self._verification_manager.close()
                self.components.scene_memory.save(
                    self.output_dir / "scene_memory.json"
                )
            if self._room_vlm_manager is not None:
                # A verified target must not be held back by unrelated room
                # selection requests that are still in flight.
                self._room_vlm_manager.close(wait=False)

    def _run_search(self, instruction: SearchInstruction) -> SearchResult:
        self.components.search_policy.start(instruction)
        self.components.perception.set_prompts(instruction.detector_prompts)
        self._start_object_verification(instruction)
        pose = self.navigator.current_pose()
        current_room = self.components.room_graph.room_at(pose[0], pose[1])
        if current_room is None:
            return self._result(
                found=False,
                reason="initial robot pose is outside every anonymous room",
            )
        self._record_event(
            "search_started",
            instruction=instruction.to_dict(),
            initial_pose=pose.astype(float).tolist(),
            initial_room=current_room,
        )
        requested_room: str | None = current_room
        while self._viewpoint_count < self.max_viewpoints:
            if requested_room is None:
                selection = self._take_room_preselection(current_room)
                if selection is None:
                    target = self._wait_for_final_target()
                    if target is not None:
                        return self._result(
                            found=True,
                            reason="target verified after final pending semantic review",
                            target=target,
                        )
                    return self._result(
                        found=False,
                        reason="all anonymous rooms exhausted without a verified target",
                    )
                requested_room = selection.room_id
                self._record_event(
                    "room_selected",
                    current_room=current_room,
                    room_id=selection.room_id,
                    reason=selection.reason,
                    used_vlm=selection.used_vlm,
                )
            current_room = self._move_between_rooms(current_room, requested_room)
            requested_room = None
            if current_room not in self._searched_rooms:
                self._searched_rooms.append(current_room)

            while self._viewpoint_count < self.max_viewpoints:
                target = self._poll_room_vlm_results()
                if target is not None:
                    return self._result(
                        found=True,
                        reason="target verified by an asynchronous semantic review",
                        target=target,
                    )
                requested_room = self._take_pending_room_switch(current_room)
                if requested_room is not None:
                    break
                viewpoint = self.components.search_policy.next_viewpoint(
                    current_room, self.navigator.current_pose()
                )
                if viewpoint is None:
                    self._record_event(
                        "room_exhausted",
                        room_id=current_room,
                        coverage=self.components.scene_memory.rooms[current_room].coverage,
                    )
                    break
                navigation = self._navigate_with_transit_detection(
                    viewpoint.pose,
                    name=f"SEARCH_{current_room}_{viewpoint.index:04d}",
                )
                target = self._poll_room_vlm_results()
                if target is not None:
                    return self._result(
                        found=True,
                        reason="target verified during navigation to a coverage viewpoint",
                        target=target,
                    )
                requested_room = self._take_pending_room_switch(current_room)
                if requested_room is not None:
                    self._record_event(
                        "coverage_observation_skipped_for_async_room_switch",
                        room_id=current_room,
                        viewpoint_index=viewpoint.index,
                        requested_room=requested_room,
                        navigation=navigation,
                    )
                    break
                self._viewpoint_count += 1
                _, _, target = self._observe(
                    current_room,
                    token=f"view_{current_room}_{self._viewpoint_count:04d}",
                    planned_viewpoint_index=viewpoint.index,
                )
                self._ordinary_viewpoint_counts[current_room] = (
                    self._ordinary_viewpoint_counts.get(current_room, 0) + 1
                )
                self._record_event(
                    "viewpoint_reached",
                    room_id=current_room,
                    viewpoint_index=viewpoint.index,
                    pose=list(viewpoint.pose),
                    navigation=navigation,
                    ordinary_viewpoints_completed=(
                        self._ordinary_viewpoint_counts[current_room]
                    ),
                )
                if target is not None:
                    return self._result(
                        found=True,
                        reason="target verified from coverage observation",
                        target=target,
                    )
                requested_room, peek_target = self._consider_adjacent_rooms(
                    current_room
                )
                if peek_target is not None:
                    return self._result(
                        found=True,
                        reason="target verified while looking into an adjacent room",
                        target=peek_target,
                    )
                if requested_room is not None:
                    break
            if requested_room is None:
                self.components.scene_memory.rooms[current_room].exhausted = True

        target = self._wait_for_final_target()
        if target is not None:
            return self._result(
                found=True,
                reason="target verified after final pending semantic review",
                target=target,
            )
        return self._result(
            found=False,
            reason=f"search viewpoint limit reached ({self.max_viewpoints})",
        )
