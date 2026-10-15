"""OpenAI-compatible VLM decisions used by the lightweight SysNav policy.

The model only returns discrete semantic decisions.  It never produces metric
coordinates or motion commands: room ids originate in ``room_id_map`` and are
resolved to executable geometry by :mod:`room_graph`.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from collections.abc import Sequence
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .config import VLMConfig
from .types import (
    EarlyStopDecision,
    RoomDescription,
    RoomSelection,
    SearchInstruction,
    TargetVerification,
    json_safe,
)


class _InstructionResult(BaseModel):
    target_object: str
    room_condition: str = ""
    spatial_condition: str = ""
    attribute_condition: str = ""
    anchor_object: str = ""
    anchor_attribute: str = ""


class _RoomDescriptionResult(BaseModel):
    room_type: str
    visual_cues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class _RoomSelectionResult(BaseModel):
    room_id: str
    reason: str


class _EarlyStopResult(BaseModel):
    switch_room: bool
    reason: str


class _TargetIdentityResult(BaseModel):
    verdict: Literal["confirmed", "rejected", "uncertain"]
    reason: str


class _TargetRelationResult(BaseModel):
    verdict: Literal["confirmed", "rejected", "uncertain"]
    reason: str


def _image_data_url(image_rgb: np.ndarray, *, quality: int = 88) -> str:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="JPEG", quality=quality)
    payload = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class QwenVLMClient:
    """Small synchronous semantic API; safe to construct before a key exists.

    The OpenAI client and API key are read lazily on the first remote request,
    which keeps map generation and unit tests completely offline.
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key(),
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                max_retries=2,
            )
        return self._client

    def _parse(self, model: str, response_format: type[BaseModel], messages: list[dict]):
        completion = self._get_client().beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            refusal = completion.choices[0].message.refusal
            raise RuntimeError(f"VLM returned no structured result: {refusal or 'unknown'}")
        return parsed

    def parse_instruction(self, instruction: str) -> SearchInstruction:
        """Extract target, room, attribute and relation slots from free text."""

        raw = str(instruction).strip()
        if not raw:
            raise ValueError("instruction must not be empty")
        result = self._parse(
            self.config.fast_model,
            _InstructionResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Extract an indoor object-search instruction. Preserve the "
                        "user's object nouns, but make every returned field concise. "
                        "Use empty strings for absent constraints. anchor_object is "
                        "the reference object in relations such as 'on the table'."
                    ),
                },
                {"role": "user", "content": raw},
            ],
        )
        return SearchInstruction(raw_text=raw, **result.model_dump())

    def describe_room(
        self,
        room_id: str,
        image_rgb: np.ndarray,
        detected_labels: Iterable[str] = (),
    ) -> RoomDescription:
        """Infer runtime room semantics from what the robot actually observed."""

        labels = sorted({str(value).strip().lower() for value in detected_labels if value})
        result = self._parse(
            self.config.reasoning_model,
            _RoomDescriptionResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the visually observed indoor room from the RGB image. "
                        "The supplied room id is an anonymous map id, not a semantic "
                        "label. Infer room_type from directly visible room layout, fixed "
                        "features, and large room-defining furniture. Detector proposals "
                        "are unverified hints only: never claim an object is present merely "
                        "because its name appears in that list. Do not identify or record "
                        "small movable task targets in visual_cues. If the image does not "
                        "support a room type, return room_type='unknown' with low confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"room_id={room_id}; "
                                f"unverified_detector_proposals={labels}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_rgb)}},
                    ],
                },
            ],
        )
        return RoomDescription(
            room_type=result.room_type,
            visual_cues=tuple(result.visual_cues),
            confidence=result.confidence,
        )

    def select_room(
        self,
        instruction: SearchInstruction,
        candidates: list[dict[str, Any]],
        room_images: dict[str, np.ndarray] | None = None,
    ) -> RoomSelection:
        """Choose one candidate id; geometry remains outside the model."""

        if not candidates:
            raise ValueError("at least one room candidate is required")
        valid_ids = {str(candidate["room_id"]) for candidate in candidates}
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Instruction:\n"
                    + json.dumps(instruction.to_dict(), ensure_ascii=False)
                    + "\nCandidate rooms:\n"
                    + json.dumps(json_safe(candidates), ensure_ascii=False)
                ),
            }
        ]
        for room_id, image in (room_images or {}).items():
            if room_id not in valid_ids:
                continue
            content.extend(
                [
                    {"type": "text", "text": f"Observed image for {room_id}:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                ]
            )
        result = self._parse(
            self.config.reasoning_model,
            _RoomSelectionResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Select the candidate room most likely to advance the object "
                        "search. Consider visual evidence, detected objects, previous "
                        "coverage and travel_cost. Return exactly one supplied room_id."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        if result.room_id not in valid_ids:
            raise RuntimeError(
                f"VLM selected unknown room {result.room_id!r}; valid ids={sorted(valid_ids)}"
            )
        return RoomSelection(room_id=result.room_id, reason=result.reason, used_vlm=True)

    def should_switch_room(
        self,
        instruction: SearchInstruction,
        current_room_id: str,
        current_images_rgb: Sequence[np.ndarray],
        nearby_room_id: str,
        nearby_images_rgb: Sequence[np.ndarray],
    ) -> EarlyStopDecision:
        """Compare doorway evidence with the room currently being covered."""

        current_images = tuple(current_images_rgb)
        nearby_images = tuple(nearby_images_rgb)
        if not current_images:
            raise ValueError("current room evidence must contain at least one image")
        if not nearby_images:
            raise ValueError("nearby room evidence must contain at least one image")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Instruction: " + json.dumps(instruction.to_dict(), ensure_ascii=False),
            },
            {
                "type": "text",
                "text": (
                    f"Current room {current_room_id}: consider all "
                    f"{len(current_images)} views jointly."
                ),
            },
        ]
        for index, image in enumerate(current_images, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Current room view {index}/{len(current_images)}:",
                    },
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"Nearby room {nearby_room_id}: consider all "
                    f"{len(nearby_images)} doorway views jointly."
                ),
            }
        )
        for index, image in enumerate(nearby_images, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Nearby room view {index}/{len(nearby_images)}:",
                    },
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                ]
            )

        result = self._parse(
            self.config.reasoning_model,
            _EarlyStopResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Decide whether to interrupt coverage of the current room and "
                        "search the nearby room. Switch only when the nearby view shows "
                        "the target or substantially stronger task-relevant evidence. "
                        "Evaluate every supplied view of each room jointly; do not let "
                        "one weak view override stronger evidence from the other views."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        return EarlyStopDecision(
            switch_room=result.switch_room,
            target_room_id=nearby_room_id if result.switch_room else None,
            reason=result.reason,
        )

    def verify_target(
        self,
        instruction: SearchInstruction,
        candidate_label: str,
        image_rgb: np.ndarray,
        *,
        room_description: str = "",
        nearby_objects: Iterable[str] = (),
        candidate_crop_rgb: np.ndarray | None = None,
        candidate_clipped_by_image_edge: bool = False,
    ) -> TargetVerification:
        """Verify visual identity first, then the instruction constraints."""

        del candidate_label  # The detector's class prediction must not bias identity.
        crop = image_rgb if candidate_crop_rgb is None else candidate_crop_rgb
        identity = self._parse(
            self.config.reasoning_model,
            _TargetIdentityResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Stage 1: independently verify object identity. The green-outlined "
                        "region is an untrusted detector proposal. The green line is an "
                        "artificial annotation overlay, not part of the object's color, texture, "
                        "or silhouette. Use the outlined full scene to locate the candidate and "
                        "the clean enlarged crop to inspect its unannotated pixels. Judge only "
                        "that physical candidate against the requested target category; ignore "
                        "detector labels, confidence, room type, anchor objects, relations, and "
                        "target-like objects elsewhere. Return exactly one verdict. MANDATORY "
                        "CLIPPING RULE: when candidate_clipped_by_image_edge=true, never reject "
                        "because the candidate looks flat or elongated, lacks depth or faces, or "
                        "has an incomplete silhouette; clipping makes all of those observations "
                        "unreliable. In that case reject only when independent visible semantic "
                        "evidence identifies background or a different object category. Otherwise "
                        "return uncertain rather than rejected. Confirm when "
                        "the visible appearance and scene contact support a discrete physical "
                        "instance of the category. Reject when visible evidence shows background, "
                        "a continuous part of a wall, floor or large furniture item, a different "
                        "category, or shape proportions incompatible with the requested category. "
                        "These visible proportions and surface-continuity cues are allowed; do not "
                        "use assumed metric dimensions or a memorized real-world object size. For "
                        "a cube, a compact approximately square face belonging to a discrete object "
                        "can support confirmation even when viewed far away or nearly face-on; do "
                        "not require multiple visible faces or visible depth. Conversely, a clearly "
                        "elongated rectangle or a region visibly continuous with room structure is "
                        "evidence against a cube, not merely missing evidence. Check that the clean "
                        "crop interpretation agrees with the candidate's silhouette and structural "
                        "context in the full scene; never confirm from the crop alone when the full "
                        "scene shows an elongated architectural or furniture element. If the "
                        "candidate touches an image boundary and is visibly clipped, do not reject "
                        "it from the partial silhouette or proportions; return uncertain unless the "
                        "remaining visible evidence independently identifies it. The supplied "
                        "candidate_clipped_by_image_edge flag is authoritative visibility metadata: "
                        "when true, absent depth, incomplete shape, or clipped proportions cannot "
                        "justify rejection. Return uncertain unless the visible pixels positively "
                        "identify background or a different category. Return uncertain "
                        "only when image quality, clipping, occlusion, proposal alignment, or "
                        "viewpoint leaves both the category-consistent and category-inconsistent "
                        "explanations genuinely plausible. Do not turn missing evidence into "
                        "rejection, and do not guess it into confirmation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "requested_target_category": instruction.target_object,
                                    "detector_proposal_is_untrusted": True,
                                    "candidate_clipped_by_image_edge": bool(
                                        candidate_clipped_by_image_edge
                                    ),
                                    "mandatory_clipping_policy": (
                                        "when clipped, flatness, aspect ratio, missing depth, "
                                        "and incomplete silhouette cannot justify rejection"
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        },
                        {
                            "type": "text",
                            "text": "Full scene; inspect only the green-outlined candidate:",
                        },
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_rgb)}},
                        {
                            "type": "text",
                            "text": (
                                "Clean enlarged crop of the same candidate; this crop has no "
                                "green annotation overlay:"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": _image_data_url(crop)}},
                    ],
                },
            ],
        )
        if candidate_clipped_by_image_edge and identity.verdict == "rejected":
            return TargetVerification(
                is_target=False,
                reason=(
                    "identity uncertain after clipping guard: the candidate is "
                    "incomplete at the image boundary, so the VLM rejection is "
                    f"not a conclusive mismatch ({identity.reason})"
                ),
                verdict="uncertain",
            )
        if identity.verdict != "confirmed":
            return TargetVerification(
                is_target=False,
                reason=f"identity {identity.verdict}: {identity.reason}",
                verdict=identity.verdict,
            )
        if not instruction.needs_vlm_verification:
            return TargetVerification(
                is_target=True,
                reason=f"identity confirmed: {identity.reason}; no relation constraints",
                verdict="confirmed",
            )

        relation_context = {
            "requested_target_category": instruction.target_object,
            "room_condition": instruction.room_condition,
            "spatial_condition": instruction.spatial_condition,
            "attribute_condition": instruction.attribute_condition,
            "anchor_object": instruction.anchor_object,
            "anchor_attribute": instruction.anchor_attribute,
            "runtime_room_description": room_description,
            "untrusted_nearby_detector_labels": sorted(set(nearby_objects)),
        }
        relation = self._parse(
            self.config.reasoning_model,
            _TargetRelationResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Stage 2: verify every explicit non-identity constraint for the same "
                        "green-outlined candidate whose identity is confirmed. Return exactly "
                        "one verdict for all explicit room, attribute, and spatial constraints: "
                        "confirmed only when the supplied image directly supports every "
                        "constraint; rejected only when a visible constraint is contradicted; "
                        "uncertain when an anchor, relation, attribute, or room cue is not "
                        "visible enough to decide. The outlined candidate itself must "
                        "participate in the requested relation; do not substitute another "
                        "object elsewhere. Merely seeing an anchor nearby is insufficient. "
                        "Visually identify the requested anchor from category-defining evidence; "
                        "do not rename an arbitrary support surface as the anchor just because the "
                        "candidate rests on it. For example, a brown horizontal region is not by "
                        "itself proof of a coffee table: require directly visible table structure "
                        "or sufficiently clear whole-scene evidence, otherwise return uncertain. "
                        "The green outline is an artificial annotation and is not part of the "
                        "candidate's appearance. "
                        "Treat runtime room text and detector-provided nearby labels as "
                        "unverified supporting hints, never as proof. Do not use object-size "
                        "assumptions. Never turn missing evidence into either rejection or "
                        "confirmation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(relation_context, ensure_ascii=False),
                        },
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_rgb)}},
                    ],
                },
            ],
        )
        if relation.verdict != "confirmed":
            return TargetVerification(
                is_target=False,
                reason=(
                    f"identity confirmed: {identity.reason}; "
                    f"constraints {relation.verdict}: {relation.reason}"
                ),
                verdict=relation.verdict,
            )
        return TargetVerification(
            is_target=True,
            reason=(
                f"identity confirmed: {identity.reason}; "
                f"constraints confirmed: {relation.reason}"
            ),
            verdict="confirmed",
        )
