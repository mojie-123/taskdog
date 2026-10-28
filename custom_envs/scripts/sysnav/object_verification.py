"""Asynchronous, object-instance-centric VLM review for SysNav.

This mirrors the upstream SysNav state machine: raw observations first update a
persistent Object node, category review is tied to the node's best close-up,
and the complete instruction is reviewed once per semantic evidence revision.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .scene_memory import ObjectRecord, SceneMemory
from .search_policy import SemanticReasoner
from .types import LocalizedDetection, SearchInstruction, TargetVerification
from .view_projection import annotate_viewpoint, best_object_images


@dataclass(frozen=True)
class VerificationSnapshot:
    object_id: str
    requested_label: str
    role: str
    query_key: tuple
    best_image_revision: int
    evidence_revision: int
    type_already_verified: bool
    outlined_image_rgb: np.ndarray
    clean_crop_rgb: np.ndarray
    viewpoint_images_rgb: tuple[np.ndarray, ...]
    room_label: str
    clipped_by_image_edge: bool
    detection: LocalizedDetection | None


@dataclass(frozen=True)
class VerificationResult:
    snapshot: VerificationSnapshot
    type_decision: TargetVerification
    spatial_decision: TargetVerification | None
    error: str = ""

    @property
    def target_confirmed(self) -> bool:
        return bool(
            self.snapshot.role == "target"
            and self.type_decision.is_target
            and self.spatial_decision is not None
            and self.spatial_decision.is_target
        )


class ObjectVerificationManager:
    """Deduplicate object queries and run the remote VLM off the navigation loop."""

    def __init__(
        self,
        memory: SceneMemory,
        reasoner: SemanticReasoner | None,
        instruction: SearchInstruction,
        *,
        min_best_image_pixels: int = 500,
    ):
        self.memory = memory
        self.reasoner = reasoner
        self.instruction = instruction
        self.min_best_image_pixels = max(1, int(min_best_image_pixels))
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sysnav-object-vlm"
        )
        self._futures: dict[Future, VerificationSnapshot] = {}
        self._closed = False

    def _role(self, record: ObjectRecord) -> str | None:
        label = record.confirmed_label or record.label
        if label == self.instruction.target_object:
            return "target"
        if self.instruction.anchor_object and label == self.instruction.anchor_object:
            return "anchor"
        return None

    def _query_key(self, record: ObjectRecord, role: str) -> tuple:
        room = self.memory.rooms[record.room_id]
        return (
            role,
            int(record.evidence_revision),
            str(room.semantic_description or "unknown"),
            self.instruction.raw_text,
        )

    def _snapshot(
        self, record: ObjectRecord, role: str, query_key: tuple
    ) -> VerificationSnapshot | None:
        best_viewpoint = self.memory.viewpoints.get(record.best_viewpoint_id)
        if best_viewpoint is None:
            return None
        best_images = best_object_images(best_viewpoint, record)
        if best_images is None:
            return None
        outlined, crop = best_images
        historical: list[np.ndarray] = []
        for viewpoint_id in record.viewpoint_ids:
            viewpoint = self.memory.viewpoints.get(viewpoint_id)
            if viewpoint is None:
                continue
            annotated = annotate_viewpoint(viewpoint, record)
            if annotated is not None:
                historical.append(annotated)
        bbox = record.best_bbox_xyxy
        clipped = False
        if bbox is not None:
            height, width = outlined.shape[:2]
            x1, y1, x2, y2 = bbox
            clipped = bool(
                x1 <= 2.0 or y1 <= 2.0 or x2 >= width - 2.0 or y2 >= height - 2.0
            )
        room_label = self.memory.rooms[record.room_id].semantic_description
        return VerificationSnapshot(
            object_id=record.object_id,
            requested_label=(
                self.instruction.target_object
                if role == "target"
                else self.instruction.anchor_object
            ),
            role=role,
            query_key=query_key,
            best_image_revision=int(record.best_image_revision),
            evidence_revision=int(record.evidence_revision),
            type_already_verified=bool(
                record.type_verified
                and record.type_queried_best_revision == record.best_image_revision
            ),
            outlined_image_rgb=outlined,
            clean_crop_rgb=crop,
            viewpoint_images_rgb=tuple(historical),
            room_label=room_label,
            clipped_by_image_edge=clipped,
            detection=record.latest_detection,
        )

    def consider(self, records: Iterable[ObjectRecord]) -> list[str]:
        """Submit eligible Object nodes, marking them considered immediately."""

        if self._closed:
            return []
        submitted: list[str] = []
        for record in records:
            role = self._role(record)
            if role is None or record.is_considered_strong or record.query_in_flight:
                continue
            if record.best_image_score < self.min_best_image_pixels:
                continue
            query_key = self._query_key(record, role)
            if record.is_considered and record.spatial_query_key == query_key:
                continue
            snapshot = self._snapshot(record, role, query_key)
            if snapshot is None:
                continue
            # Exactly as upstream CheckObjectFound(): set considered when the
            # query is published, rather than waiting for the remote answer.
            record.is_considered = True
            record.spatial_query_key = query_key
            record.query_in_flight = True
            future = self._executor.submit(self._run_query, snapshot)
            self._futures[future] = snapshot
            submitted.append(record.object_id)
        return submitted

    def consider_all(self) -> list[str]:
        return self.consider(self.memory.objects.values())

    def _run_query(self, snapshot: VerificationSnapshot) -> VerificationResult:
        try:
            if snapshot.type_already_verified:
                type_decision = TargetVerification(
                    True,
                    "object type retained from the current best close-up",
                    verdict="confirmed",
                    visual_verdict="confirmed",
                )
            elif self.reasoner is None:
                type_decision = TargetVerification(
                    True,
                    "detector category accepted because no semantic reasoner is configured",
                    verdict="confirmed",
                    visual_verdict="confirmed",
                )
            else:
                type_decision = self.reasoner.verify_object_type(
                    snapshot.requested_label,
                    snapshot.outlined_image_rgb,
                    snapshot.clean_crop_rgb,
                    candidate_clipped_by_image_edge=(
                        snapshot.clipped_by_image_edge
                    ),
                )
            if not type_decision.is_target:
                return VerificationResult(snapshot, type_decision, None)

            if snapshot.role == "anchor":
                if self.reasoner is None:
                    anchor_decision = TargetVerification(
                        not bool(
                            self.instruction.room_condition
                            or self.instruction.anchor_attribute
                        ),
                        (
                            "anchor category accepted without additional constraints"
                            if not (
                                self.instruction.room_condition
                                or self.instruction.anchor_attribute
                            )
                            else "anchor constraints require a semantic reasoner"
                        ),
                        verdict=(
                            "confirmed"
                            if not (
                                self.instruction.room_condition
                                or self.instruction.anchor_attribute
                            )
                            else "rejected"
                        ),
                    )
                else:
                    anchor_decision = self.reasoner.verify_anchor_object(
                        self.instruction,
                        snapshot.requested_label,
                        snapshot.clean_crop_rgb,
                        room_label=snapshot.room_label,
                    )
                return VerificationResult(
                    snapshot, type_decision, anchor_decision
                )

            if self.reasoner is None:
                if self.instruction.needs_vlm_verification:
                    spatial = TargetVerification(
                        False,
                        "complete instruction requires a semantic reasoner",
                        verdict="rejected",
                    )
                else:
                    spatial = TargetVerification(
                        True,
                        "class-only instruction accepted",
                        verdict="confirmed",
                        visual_verdict="confirmed",
                        room_verdict="not_required",
                    )
            else:
                spatial = self.reasoner.verify_spatial_target(
                    self.instruction,
                    snapshot.requested_label,
                    snapshot.clean_crop_rgb,
                    snapshot.viewpoint_images_rgb,
                    room_label=snapshot.room_label,
                )
            return VerificationResult(snapshot, type_decision, spatial)
        except Exception as exc:
            return VerificationResult(
                snapshot,
                TargetVerification(
                    False,
                    f"VLM query failed: {type(exc).__name__}: {exc}",
                    verdict="uncertain",
                ),
                None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def poll(self) -> list[VerificationResult]:
        completed: list[VerificationResult] = []
        for future in list(self._futures):
            if not future.done():
                continue
            snapshot = self._futures.pop(future)
            try:
                result = future.result()
            except Exception as exc:  # defensive; _run_query normally captures it
                result = VerificationResult(
                    snapshot,
                    TargetVerification(
                        False,
                        f"verification worker failed: {type(exc).__name__}: {exc}",
                        verdict="uncertain",
                    ),
                    None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            record = self.memory.objects.get(snapshot.object_id)
            if record is not None:
                record.query_in_flight = False
                record.type_queried_best_revision = snapshot.best_image_revision
                record.type_verdict = result.type_decision.verdict
                record.type_reason = result.type_decision.reason
                if result.type_decision.is_target:
                    record.confirmed_label = snapshot.requested_label
                    record.type_verified = (
                        record.best_image_revision == snapshot.best_image_revision
                    )
                if result.spatial_decision is not None:
                    record.spatial_verdict = result.spatial_decision.verdict
                    record.spatial_reason = result.spatial_decision.reason
                    if result.spatial_decision.is_target:
                        record.is_considered_strong = True
                # A newer semantic viewpoint or changed room label may have
                # arrived while the query was in flight. Preserve that reset.
                current_key = self._query_key(record, snapshot.role)
                if current_key != snapshot.query_key and not record.is_considered_strong:
                    record.is_considered = False
            completed.append(result)
        return completed

    @property
    def has_pending(self) -> bool:
        return bool(self._futures)

    def wait(self) -> list[VerificationResult]:
        """Wait for currently published queries and apply every result."""

        for future in tuple(self._futures):
            try:
                future.result()
            except Exception:
                pass
        return self.poll()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.poll()
