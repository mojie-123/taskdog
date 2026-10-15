"""High-level search decisions without simulation or navigation side effects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

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
    ) -> EarlyStopDecision: ...

    def verify_target(
        self,
        instruction: SearchInstruction,
        candidate_label: str,
        image_rgb: np.ndarray,
        *,
        room_description: str = "",
        nearby_objects=(),
    ) -> TargetVerification: ...


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
        if not self.instruction.needs_vlm_verification:
            return TargetVerification(True, "class-only instruction accepted by detector")
        if self.reasoner is None:
            return TargetVerification(False, "task constraints require VLM verification")
        room_id = candidate.room_id or ""
        room_description = (
            self.memory.rooms[room_id].semantic_description
            if room_id in self.memory.rooms
            else ""
        )
        nearby = self.memory.nearby_labels(candidate.position_world, 1.5)
        return self.reasoner.verify_target(
            self.instruction,
            detection.label,
            annotate_detection(frame_rgb, detection),
            room_description=room_description,
            nearby_objects=nearby,
        )
