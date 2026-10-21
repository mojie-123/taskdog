"""Executable room-search state machine, independent of a concrete base controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image

from custom_envs.scripts.N2M.manifests import append_jsonl, write_json

from .rgbd import RGBDFrame
from .runtime import ObservationUpdate, SysNavComponents
from .transit_detection import ConfirmedTransitCandidate, TransitCandidate
from .types import LocalizedDetection, SearchInstruction


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


class _TransitTargetVerified(RuntimeError):
    """Internal signal used to escape room/coverage loops after verification."""

    def __init__(self, target: LocalizedDetection):
        super().__init__("target verified from an in-motion observation")
        self.target = target


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


@dataclass
class _PendingTarget:
    """Visually verified target waiting only for Room-node confirmation."""

    detection: LocalizedDetection
    viewpoint_id: str
    source: str
    visual_reason: str
    last_room_revision: int = -1


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
        self._pending_targets: list[_PendingTarget] = []

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

    def _take_transit_candidate(self) -> TransitCandidate | None:
        take = getattr(self.navigator, "take_transit_candidate", None)
        return None if take is None else take()

    def _commit_confirmed_transit(
        self, confirmed: ConfirmedTransitCandidate
    ) -> ObservationUpdate | None:
        observation = confirmed.representative
        room_id = observation.metric_room_id or confirmed.target.room_id
        if room_id is None:
            self._record_event(
                "transit_scene_commit_skipped",
                reason="robot and target are outside the anonymous room map",
                sequence=observation.sequence,
            )
            return None
        representative_target = (
            confirmed.first.target
            if observation is confirmed.first.observation
            else confirmed.confirmation.target
        )
        localized = tuple(
            confirmed.target if item is representative_target else item
            for item in observation.localized_detections
        )
        return self.components.commit_observation(
            observation.frame,
            observation.robot_pose_world,
            room_id,
            observation.detections,
            localized,
            planned_viewpoint_index=None,
            image_path=observation.image_path,
        )

    def _verify_transit_candidate(
        self, candidate: TransitCandidate, *, navigation_name: str
    ) -> str | None:
        detector = getattr(self.navigator, "transit_detector", None)
        if detector is None:
            return
        self._record_event(
            "transit_candidate_detected",
            navigation_name=navigation_name,
            sequence=candidate.observation.sequence,
            image_path=candidate.observation.image_path,
            robot_pose_world=candidate.observation.robot_pose_world.tolist(),
            position_world=candidate.target.position_world.tolist(),
            confidence=candidate.target.detection.confidence,
        )
        confirmed = detector.confirm(
            candidate,
            self.navigator.current_pose(),
            navigation_name=navigation_name,
        )
        if confirmed is None:
            self._record_event(
                "transit_candidate_confirmation_failed",
                navigation_name=navigation_name,
                sequence=candidate.observation.sequence,
            )
            return "not_confirmed"

        # Persist only the observation viewpoint before verification.  The
        # candidate itself must not enter the Object layer unless the VLM
        # accepts the complete target instruction below.
        update = self._commit_confirmed_transit(confirmed)
        try:
            decision = self.components.search_policy.verify_candidate(
                confirmed.target,
                confirmed.representative.frame.rgb,
            )
        except Exception as exc:
            detector.defer(
                confirmed.target.position_world,
                confirmed.representative.robot_pose_world,
                verdict="not_confirmed",
                reason=f"VLM verification failed: {type(exc).__name__}: {exc}",
            )
            self._record_event(
                "transit_target_verification_failed",
                navigation_name=navigation_name,
                error=f"{type(exc).__name__}: {exc}",
            )
            return "not_confirmed"
        scene_object_id = None
        if decision.is_target and update is not None:
            scene_object = self.components.commit_verified_detection(
                update.viewpoint.viewpoint_id,
                confirmed.target,
            )
            scene_object_id = scene_object.object_id
        elif (
            update is not None
            and decision.visual_verdict == "confirmed"
            and decision.room_verdict == "uncertain"
        ):
            self._remember_pending_target(
                confirmed.target,
                update.viewpoint.viewpoint_id,
                source=f"transit:{navigation_name}",
                visual_reason=decision.reason,
            )
        self._record_event(
            "transit_target_verified",
            navigation_name=navigation_name,
            first_sequence=confirmed.first.observation.sequence,
            confirmation_sequence=confirmed.confirmation.observation.sequence,
            separation_m=confirmed.separation_m,
            image_path=confirmed.representative.image_path,
            position_world=confirmed.target.position_world.tolist(),
            confidence=confirmed.target.detection.confidence,
            room_id=confirmed.target.room_id,
            is_target=decision.is_target,
            verdict=decision.verdict,
            visual_verdict=decision.visual_verdict,
            room_verdict=decision.room_verdict,
            reason=decision.reason,
            scene_viewpoint_id=(
                None if update is None else update.viewpoint.viewpoint_id
            ),
            scene_object_id=scene_object_id,
        )
        if decision.is_target:
            raise _TransitTargetVerified(confirmed.target)
        # The VLM keeps a three-way diagnostic verdict, but navigation has
        # only two semantic outcomes: confirmed or not confirmed.  Any
        # negative/ambiguous view is deduplicated from the same observer pose,
        # never converted into a persistent spatial blacklist.
        detector.defer(
            confirmed.target.position_world,
            confirmed.representative.robot_pose_world,
            verdict=decision.verdict,
            reason=f"VLM did not confirm candidate: {decision.reason}",
        )
        return "not_confirmed"

    def _navigate_with_transit_detection(
        self, pose: tuple[float, float, float], *, name: str
    ) -> dict:
        """Resume the same goal after non-confirmed transit candidates."""

        interruptions: list[dict] = []
        while True:
            metrics = self.navigator.move_to(pose, name=name)
            candidate = self._take_transit_candidate()
            if candidate is None:
                if interruptions:
                    result = dict(metrics)
                    result["transit_interruptions"] = interruptions
                    return result
                return metrics
            interruptions.append(
                {
                    "sequence": candidate.observation.sequence,
                    "position_world": candidate.target.position_world.tolist(),
                    "confidence": candidate.target.detection.confidence,
                }
            )
            verdict = self._verify_transit_candidate(candidate, navigation_name=name)
            self._record_event(
                "navigation_resumed_after_transit_review",
                navigation_name=name,
                goal=list(pose),
                interruption_count=len(interruptions),
                verdict=verdict,
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
        vlm = self.components.vlm
        state = self.components.scene_memory.rooms[room_id]
        if not self.enable_room_semantics or vlm is None:
            return None
        if not self._room_semantic_update_due(room_id):
            return None
        evidence = self._semantic_room_evidence(room_id)
        if not evidence:
            return None
        history = self.room_image_history.get(room_id, ())
        total_count = len(history)
        ordinary_count = sum(
            item.observation_kind == "ordinary" for item in history
        )
        previous = {
            "room_type": state.semantic_description,
            "confidence": state.semantic_confidence,
            "status": state.semantic_status,
            "revision": state.semantic_revision,
        }
        try:
            description = vlm.describe_room_multiview(
                room_id,
                tuple(item.image_rgb for item in evidence),
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
        except Exception as exc:
            self._record_event(
                "room_description_failed",
                room_id=room_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None
        self._room_semantic_last_counts[room_id] = (
            total_count,
            ordinary_count,
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
        )
        print(
            f"[SYSNAV][ROOM_SEMANTIC_UPDATE] room={room_id} "
            f"revision={state.semantic_revision} status={state.semantic_status} "
            f"type={state.semantic_description!r} "
            f"confidence={state.semantic_confidence:.2f} "
            f"evidence={state.semantic_evidence_count} "
            f"ordinary={state.semantic_ordinary_evidence_count} "
            f"previous={previous['room_type']!r}/{previous['status']}",
            flush=True,
        )
        return self._resolve_pending_target(room_id)

    def _remember_pending_target(
        self,
        detection: LocalizedDetection,
        viewpoint_id: str,
        *,
        source: str,
        visual_reason: str,
    ) -> None:
        room_id = detection.room_id
        for pending in self._pending_targets:
            if pending.detection.room_id != room_id:
                continue
            separation = float(
                np.linalg.norm(
                    pending.detection.position_world[:2]
                    - detection.position_world[:2]
                )
            )
            if separation > self.components.config.transit_detection.candidate_dedup_radius_m:
                continue
            if (
                detection.detection.confidence
                > pending.detection.detection.confidence
            ):
                pending.detection = detection
                pending.viewpoint_id = viewpoint_id
                pending.source = source
                pending.visual_reason = visual_reason
            return
        self._pending_targets.append(
            _PendingTarget(
                detection=detection,
                viewpoint_id=viewpoint_id,
                source=source,
                visual_reason=visual_reason,
            )
        )
        self._record_event(
            "target_pending_room_semantics",
            source=source,
            viewpoint_id=viewpoint_id,
            room_id=room_id,
            position_world=detection.position_world.astype(float).tolist(),
            confidence=detection.detection.confidence,
            reason=visual_reason,
        )
        print(
            f"[SYSNAV][TARGET_PENDING_ROOM] room={room_id} "
            f"position={detection.position_world.astype(float).tolist()} "
            f"confidence={detection.detection.confidence:.3f} source={source}",
            flush=True,
        )

    def _resolve_pending_target(self, room_id: str) -> LocalizedDetection | None:
        state = self.components.scene_memory.rooms[room_id]
        for pending in list(self._pending_targets):
            if pending.detection.room_id != room_id:
                continue
            if pending.last_room_revision == state.semantic_revision:
                continue
            pending.last_room_revision = state.semantic_revision
            room_decision = self.components.search_policy.verify_room_condition(room_id)
            self._record_event(
                "pending_target_room_reviewed",
                source=pending.source,
                room_id=room_id,
                semantic_revision=state.semantic_revision,
                room_type=state.semantic_description,
                semantic_status=state.semantic_status,
                room_verdict=room_decision.room_verdict,
                reason=room_decision.reason,
            )
            if not room_decision.is_target:
                continue
            scene_object = self.components.commit_verified_detection(
                pending.viewpoint_id, pending.detection
            )
            self._pending_targets.remove(pending)
            self._record_event(
                "pending_target_resolved",
                source=pending.source,
                viewpoint_id=pending.viewpoint_id,
                object_id=scene_object.object_id,
                room_id=room_id,
                semantic_revision=state.semantic_revision,
                position_world=scene_object.position_world.astype(float).tolist(),
            )
            print(
                f"[SYSNAV][TARGET_RESOLVED_FROM_ROOM] room={room_id} "
                f"revision={state.semantic_revision} object={scene_object.object_id} "
                f"position={scene_object.position_world.astype(float).tolist()}",
                flush=True,
            )
            return pending.detection
        return None

    def _verify_targets(
        self,
        frame: RGBDFrame,
        detections: tuple[LocalizedDetection, ...],
        *,
        viewpoint_id: str,
    ) -> LocalizedDetection | None:
        instruction = self.components.search_policy.instruction
        assert instruction is not None
        candidates = sorted(
            (
                item
                for item in detections
                if item.detection.label == instruction.target_object
            ),
            key=lambda item: item.detection.confidence,
            reverse=True,
        )
        for candidate in candidates:
            try:
                decision = self.components.search_policy.verify_candidate(
                    candidate, frame.rgb
                )
            except Exception as exc:
                self._record_event(
                    "target_verification_failed",
                    label=candidate.detection.label,
                    confidence=candidate.detection.confidence,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            self._record_event(
                "target_verified",
                label=candidate.detection.label,
                confidence=candidate.detection.confidence,
                room_id=candidate.room_id,
                is_target=decision.is_target,
                verdict=decision.verdict,
                visual_verdict=decision.visual_verdict,
                room_verdict=decision.room_verdict,
                reason=decision.reason,
                position_world=candidate.position_world.astype(float).tolist(),
            )
            if decision.is_target:
                return candidate
            if (
                decision.visual_verdict == "confirmed"
                and decision.room_verdict == "uncertain"
            ):
                self._remember_pending_target(
                    candidate,
                    viewpoint_id,
                    source="coverage",
                    visual_reason=decision.reason,
                )
        return None

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
        resolved_target = self._describe_room(
            viewed_room, frame, update, image_path
        )
        if resolved_target is None:
            target = self._verify_targets(
                frame,
                update.localized_detections,
                viewpoint_id=update.viewpoint.viewpoint_id,
            )
        else:
            # _resolve_pending_target() already attached the previously seen
            # detection to its original viewpoint. Do not duplicate that
            # object observation on the current room-semantics frame.
            target = resolved_target
        if target is not None and resolved_target is None:
            scene_object = self.components.commit_verified_detection(
                update.viewpoint.viewpoint_id,
                target,
            )
            self._record_event(
                "verified_object_committed",
                viewpoint_id=update.viewpoint.viewpoint_id,
                object_id=scene_object.object_id,
                label=scene_object.label,
                room_id=scene_object.room_id,
                position_world=scene_object.position_world.astype(float).tolist(),
            )
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
            try:
                decision = self.components.search_policy.consider_early_stop(
                    current_room,
                    nearby_room,
                    tuple(item.image_rgb for item in current_evidence),
                    tuple(item.image_rgb for item in nearby_evidence),
                )
            except Exception as exc:
                self._record_event(
                    "early_stop_failed",
                    current_room=current_room,
                    nearby_room=nearby_room,
                    error=f"{type(exc).__name__}: {exc}",
                )
                # One completed ordinary viewpoint may trigger at most one
                # doorway peek.  A failed VLM call must not cascade into a
                # tour of every other doorway in the same iteration.
                return None, None
            self._record_event(
                "early_stop_decision",
                current_room=current_room,
                nearby_room=nearby_room,
                switch_room=decision.switch_room,
                reason=decision.reason,
            )
            if decision.switch_room:
                return nearby_room, None
            # Resume ordinary coverage before considering another adjacent
            # room, even when this doorway did not justify an early switch.
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

    def run(self, instruction: SearchInstruction) -> SearchResult:
        try:
            return self._run_search(instruction)
        except _TransitTargetVerified as verified:
            return self._result(
                found=True,
                reason="target verified from in-motion observation",
                target=verified.target,
            )
        finally:
            detector = getattr(self.navigator, "transit_detector", None)
            if detector is not None:
                detector.save_summary()

    def _run_search(self, instruction: SearchInstruction) -> SearchResult:
        self.components.search_policy.start(instruction)
        self.components.perception.set_prompts(instruction.detector_prompts)
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
                selection = self.components.search_policy.choose_next_room(
                    current_room, room_images=self.room_images
                )
                if selection is None:
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

        return self._result(
            found=False,
            reason=f"search viewpoint limit reached ({self.max_viewpoints})",
        )
