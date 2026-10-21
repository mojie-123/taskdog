"""High-level search decisions without simulation or navigation side effects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import re
from typing import Protocol

import cv2
import numpy as np

from .config import SearchConfig
from .coverage import CoveragePlanner, CoverageViewpoint, RoomCoveragePlan
from .perception import annotate_detection
from .room_graph import AnonymousRoomGraph
from .scene_memory import SceneMemory
from .types import (
    EarlyStopDecision,
    LocalizedDetection,
    RoomSelection,
    SearchInstruction,
    TargetVerification,
)


class SemanticReasoner(Protocol):
    def parse_instruction(self, instruction: str) -> SearchInstruction: ...

    def select_room(
        self,
        instruction: SearchInstruction,
        candidates: list[dict],
        room_images: dict[str, np.ndarray] | None = None,
    ) -> RoomSelection: ...

    def should_switch_room(
        self,
        instruction: SearchInstruction,
        current_room_id: str,
        current_images_rgb: Sequence[np.ndarray],
        nearby_room_id: str,
        nearby_images_rgb: Sequence[np.ndarray],
        *,
        current_room_semantics: dict | None = None,
        nearby_room_semantics: dict | None = None,
    ) -> EarlyStopDecision: ...

    def verify_target(
        self,
        instruction: SearchInstruction,
        candidate_label: str,
        image_rgb: np.ndarray,
        *,
        nearby_objects=(),
        candidate_crop_rgb: np.ndarray | None = None,
        candidate_clipped_by_image_edge: bool = False,
    ) -> TargetVerification: ...


_ROOM_TYPE_ALIASES = {
    "客厅": "living room",
    "起居室": "living room",
    "会客厅": "living room",
    "livingroom": "living room",
    "lounge": "living room",
    "厨房": "kitchen",
    "卧室": "bedroom",
    "睡房": "bedroom",
    "卫生间": "bathroom",
    "洗手间": "bathroom",
    "浴室": "bathroom",
    "厕所": "bathroom",
    "餐厅": "dining room",
    "饭厅": "dining room",
    "书房": "study",
    "办公室": "office",
    "走廊": "hallway",
    "过道": "hallway",
}


def _canonical_room_type(value: str) -> str:
    text = " ".join(str(value).strip().lower().replace("_", " ").split())
    compact = re.sub(r"[\s\-_/]+", "", text)
    return _ROOM_TYPE_ALIASES.get(text, _ROOM_TYPE_ALIASES.get(compact, text))


def _room_semantic_payload(state) -> dict:
    return {
        "room_type": state.semantic_description,
        "confidence": float(state.semantic_confidence),
        "status": state.semantic_status,
        "revision": int(state.semantic_revision),
        "visual_cues": list(state.semantic_cues),
    }


def _candidate_crop(
    outlined_image_rgb: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a context crop for VLM inspection, without applying a size rule."""

    image = np.asarray(outlined_image_rgb)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    crop_width = max(64.0, 1.5 * box_width)
    crop_height = max(64.0, 1.5 * box_height)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    left = max(0, int(np.floor(center_x - 0.5 * crop_width)))
    right = min(width, int(np.ceil(center_x + 0.5 * crop_width)))
    top = max(0, int(np.floor(center_y - 0.5 * crop_height)))
    bottom = min(height, int(np.ceil(center_y + 0.5 * crop_height)))
    crop = np.ascontiguousarray(image[top:bottom, left:right])
    if crop.size == 0:
        return np.ascontiguousarray(image).copy()
    # Upscaling only improves VLM visibility; it is not used as an acceptance
    # criterion and therefore introduces no object-size prior.
    scale = max(1.0, 256.0 / float(max(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    return crop


class SysNavSearchPolicy:
    """Coordinate room semantics and classical coverage planning.

    The future executable owns movement and observation timing.  This class
    only returns room ids, metric viewpoints and target decisions.
    """

    def __init__(
        self,
        graph: AnonymousRoomGraph,
        coverage_planner: CoveragePlanner,
        memory: SceneMemory,
        config: SearchConfig,
        *,
        reasoner: SemanticReasoner | None = None,
    ):
        self.graph = graph
        self.coverage_planner = coverage_planner
        self.memory = memory
        self.config = config
        self.reasoner = reasoner
        self.instruction: SearchInstruction | None = None
        self.coverage_plans: dict[str, RoomCoveragePlan] = {}

    def start(self, instruction: SearchInstruction) -> None:
        self.instruction = instruction
        self.coverage_plans.clear()
        for room in self.memory.rooms.values():
            room.exhausted = False
            room.coverage = 0.0

    def parse_and_start(self, instruction_text: str) -> SearchInstruction:
        if self.reasoner is None:
            raise RuntimeError("a semantic reasoner is required to parse free text")
        instruction = self.reasoner.parse_instruction(instruction_text)
        self.start(instruction)
        return instruction

    def plan_for_room(self, room_id: str) -> RoomCoveragePlan:
        plan = self.coverage_plans.get(room_id)
        if plan is None:
            plan = self.coverage_planner.build_plan(room_id)
            self.coverage_plans[room_id] = plan
        return plan

    def next_viewpoint(
        self, room_id: str, current_pose: tuple[float, float, float] | np.ndarray
    ) -> CoverageViewpoint | None:
        plan = self.plan_for_room(room_id)
        interest_points: list[np.ndarray] = []
        if self.instruction is not None and self.instruction.anchor_object:
            interest_points = [
                record.position_world
                for record in self.memory.objects_by_label(
                    self.instruction.anchor_object, room_id=room_id
                )
            ]
        viewpoint = plan.next_viewpoint(
            current_pose, interest_points_world=interest_points
        )
        if viewpoint is None:
            self.memory.update_room_progress(room_id, plan.coverage, exhausted=True)
        return viewpoint

    def mark_viewpoint_complete(self, room_id: str, viewpoint_index: int) -> float:
        plan = self.plan_for_room(room_id)
        plan.mark_visited(viewpoint_index)
        self.memory.update_room_progress(room_id, plan.coverage)
        return plan.coverage

    def choose_next_room(
        self,
        current_room_id: str,
        *,
        room_images: dict[str, np.ndarray] | None = None,
    ) -> RoomSelection | None:
        if self.instruction is None:
            raise RuntimeError("start a search instruction first")
        candidates = self.graph.candidate_summaries(
            current_room_id, self.memory.rooms
        )
        candidates = [item for item in candidates if not self.memory.rooms[item["room_id"]].exhausted]
        if not candidates:
            return None
        informative = any(
            item["runtime_description"] or item["observed_objects"]
            for item in candidates
        ) or bool(room_images)
        if self.reasoner is not None and informative:
            try:
                return self.reasoner.select_room(
                    self.instruction, candidates, room_images=room_images
                )
            except Exception as exc:
                fallback_reason = f"VLM room selection failed ({type(exc).__name__}); used travel cost"
            else:
                fallback_reason = ""
        else:
            fallback_reason = "no runtime semantic evidence yet; used travel cost"
        selected = min(candidates, key=lambda item: (item["travel_cost"], item["room_id"]))
        return RoomSelection(
            room_id=selected["room_id"], reason=fallback_reason, used_vlm=False
        )

    def consider_early_stop(
        self,
        current_room_id: str,
        nearby_room_id: str,
        current_images_rgb: Sequence[np.ndarray],
        nearby_images_rgb: Sequence[np.ndarray],
    ) -> EarlyStopDecision:
        """Expose the SysNav adjacent-room interruption decision explicitly."""

        if self.instruction is None:
            raise RuntimeError("start a search instruction first")
        if nearby_room_id not in self.graph.nodes[current_room_id].neighbors:
            raise ValueError(f"{nearby_room_id} is not adjacent to {current_room_id}")
        coverage = self.memory.rooms[current_room_id].coverage
        if coverage < self.config.early_stop_min_room_coverage:
            return EarlyStopDecision(
                switch_room=False,
                target_room_id=None,
                reason=(
                    f"current room coverage {coverage:.3f} is below the early-stop gate "
                    f"{self.config.early_stop_min_room_coverage:.3f}"
                ),
            )
        if self.reasoner is None:
            return EarlyStopDecision(False, None, "no semantic reasoner configured")
        return self.reasoner.should_switch_room(
            self.instruction,
            current_room_id,
            current_images_rgb,
            nearby_room_id,
            nearby_images_rgb,
            current_room_semantics=_room_semantic_payload(
                self.memory.rooms[current_room_id]
            ),
            nearby_room_semantics=_room_semantic_payload(
                self.memory.rooms[nearby_room_id]
            ),
        )

    def verify_room_condition(self, room_id: str | None) -> TargetVerification:
        """Evaluate the instruction's room clause from persistent Room state."""

        if self.instruction is None:
            raise RuntimeError("start a search instruction first")
        requested = _canonical_room_type(self.instruction.room_condition)
        if not requested:
            return TargetVerification(
                True,
                "no room condition",
                verdict="confirmed",
                room_verdict="not_required",
            )
        if not room_id or room_id not in self.memory.rooms:
            return TargetVerification(
                False,
                "room condition uncertain: target is outside the anonymous room map",
                verdict="uncertain",
                room_verdict="uncertain",
            )
        state = self.memory.rooms[room_id]
        observed = _canonical_room_type(state.semantic_description)
        if state.semantic_status != "stable":
            return TargetVerification(
                False,
                (
                    "room condition uncertain: "
                    f"{room_id} semantics are {state.semantic_status} "
                    f"({state.semantic_description or 'unclassified'}, "
                    f"confidence={state.semantic_confidence:.2f}, "
                    f"revision={state.semantic_revision})"
                ),
                verdict="uncertain",
                room_verdict="uncertain",
            )
        if observed != requested:
            return TargetVerification(
                False,
                (
                    "room condition uncertain: stable Room node currently describes "
                    f"{room_id} as {state.semantic_description!r}, not "
                    f"{self.instruction.room_condition!r}; retain the candidate while "
                    "later multi-view evidence may revise the room"
                ),
                verdict="uncertain",
                room_verdict="uncertain",
            )
        return TargetVerification(
            True,
            (
                f"room condition confirmed by stable Room node {room_id}: "
                f"{state.semantic_description} (confidence={state.semantic_confidence:.2f}, "
                f"revision={state.semantic_revision})"
            ),
            verdict="confirmed",
            room_verdict="confirmed",
        )

    def verify_candidate(
        self,
        candidate: LocalizedDetection,
        frame_rgb: np.ndarray,
    ) -> TargetVerification:
        if self.instruction is None:
            raise RuntimeError("start a search instruction first")
        detection = candidate.detection
        if detection.label != self.instruction.target_object:
            return TargetVerification(False, "detector label does not match target object")
        if detection.confidence < self.config.target_accept_confidence:
            return TargetVerification(False, "detector confidence is below acceptance threshold")
        if self.reasoner is None:
            if not self.instruction.needs_vlm_verification:
                return TargetVerification(
                    True, "class-only instruction accepted without a configured VLM"
                )
            return TargetVerification(False, "task constraints require VLM verification")
        nearby = tuple(
            label
            for label in self.memory.nearby_labels(candidate.position_world, 1.5)
            if label != self.instruction.target_object
        )
        outlined = annotate_detection(frame_rgb, detection, show_label=False)
        height, width = np.asarray(frame_rgb).shape[:2]
        x1, y1, x2, y2 = detection.bbox_xyxy
        edge_tolerance_px = 2.0
        clipped_by_edge = bool(
            x1 <= edge_tolerance_px
            or y1 <= edge_tolerance_px
            or x2 >= width - edge_tolerance_px
            or y2 >= height - edge_tolerance_px
        )
        # Room semantics are intentionally removed from the per-image VLM
        # request.  The Room node below is the sole source for that condition.
        visual_instruction = replace(self.instruction, room_condition="")
        visual = self.reasoner.verify_target(
            visual_instruction,
            detection.label,
            outlined,
            nearby_objects=nearby,
            # Keep the location cue in the full scene, but do not let the
            # artificial green contour dominate a tiny target's clean crop.
            candidate_crop_rgb=_candidate_crop(frame_rgb, detection.bbox_xyxy),
            candidate_clipped_by_image_edge=clipped_by_edge,
        )
        if not visual.is_target:
            return TargetVerification(
                False,
                visual.reason,
                verdict=visual.verdict,
                visual_verdict=visual.verdict,
            )

        room = self.verify_room_condition(candidate.room_id)
        if not room.is_target:
            return TargetVerification(
                False,
                f"{visual.reason}; {room.reason}",
                verdict="uncertain",
                visual_verdict="confirmed",
                room_verdict=room.room_verdict,
            )
        return TargetVerification(
            True,
            f"{visual.reason}; {room.reason}",
            verdict="confirmed",
            visual_verdict="confirmed",
            room_verdict=room.room_verdict,
        )
