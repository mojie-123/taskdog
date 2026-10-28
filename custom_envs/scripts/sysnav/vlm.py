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
import threading
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


class _TargetSpatialResult(BaseModel):
    is_target: bool
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
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
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
                        "Normalize an explicit room_condition to a concise canonical "
                        "English room type such as 'living room', 'kitchen', or "
                        "'bedroom', regardless of the input language. "
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
        """Compatibility wrapper for a single-view room observation."""

        return self.describe_room_multiview(
            room_id,
            (image_rgb,),
            observation_kinds=("ordinary",),
            detected_labels=detected_labels,
        )

    def describe_room_multiview(
        self,
        room_id: str,
        images_rgb: Sequence[np.ndarray],
        *,
        observation_kinds: Sequence[str] = (),
        detected_labels: Iterable[str] = (),
    ) -> RoomDescription:
        """Infer one revisable room description from diverse views of that room."""

        images = tuple(images_rgb)
        if not images:
            raise ValueError("room evidence must contain at least one image")
        kinds = tuple(str(value).strip().lower() for value in observation_kinds)
        if kinds and len(kinds) != len(images):
            raise ValueError("observation_kinds must align with images_rgb")
        if not kinds:
            kinds = ("ordinary",) * len(images)
        labels = sorted({str(value).strip().lower() for value in detected_labels if value})
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"room_id={room_id}; image_count={len(images)}; "
                    f"unverified_detector_proposals={labels}. All images show the "
                    "same anonymous geometric room from different viewpoints."
                ),
            }
        ]
        for index, (image, kind) in enumerate(zip(images, kinds), start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Room evidence view {index}/{len(images)}; "
                            f"observation_kind={kind}:"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    },
                ]
            )
        result = self._parse(
            self.config.reasoning_model,
            _RoomDescriptionResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Classify one indoor room jointly from every supplied RGB view. "
                        "The supplied room id is an anonymous map id, not a semantic "
                        "label. The views belong to the same geometric room; doorway PEEK "
                        "views are partial evidence and ordinary in-room views are stronger. "
                        "Do not let one narrow or ambiguous view override consistent evidence "
                        "from the other views. Infer room_type from directly visible room "
                        "layout, fixed features, and large room-defining furniture. Return a "
                        "concise canonical English room type such as 'living room', 'kitchen', "
                        "'bedroom', 'bathroom', or 'dining room'. Detector proposals "
                        "are unverified hints only: never claim an object is present merely "
                        "because its name appears in that list. Do not identify or record "
                        "small movable task targets in visual_cues. If the image does not "
                        "support a room type, return room_type='unknown' with low confidence. "
                        "The user task and any previous room label are intentionally absent; "
                        "classify only from the supplied visual evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
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
        *,
        current_room_semantics: dict[str, Any] | None = None,
        nearby_room_semantics: dict[str, Any] | None = None,
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
                    "Shared multi-view Room-node semantics (the status and revision "
                    "indicate whether each description is tentative or stable):\n"
                    + json.dumps(
                        {
                            current_room_id: current_room_semantics or {},
                            nearby_room_id: nearby_room_semantics or {},
                        },
                        ensure_ascii=False,
                    )
                ),
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
                        "Use the supplied multi-view Room-node semantics as the shared "
                        "room interpretation; tentative labels are revisable and stable "
                        "labels summarize stronger multi-view evidence. "
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

    def verify_object_type(
        self,
        requested_label: str,
        outlined_image_rgb: np.ndarray,
        clean_crop_rgb: np.ndarray,
        *,
        candidate_clipped_by_image_edge: bool = False,
    ) -> TargetVerification:
        """Perform the upstream SysNav object-label review for one Object node."""

        result = self._parse(
            self.config.fast_model,
            _TargetIdentityResult,
            [
                {
                    "role": "system",
                    "content": (
                        "You are provided with an RGB scene containing one object "
                        "inside a green bounding box and a clean close-up of the same "
                        "object. Verify whether that physical object is the requested "
                        "category. The detector proposal is untrusted and the green "
                        "line is an annotation. Return confirmed, rejected, or uncertain. "
                        "When the box touches an image boundary, incomplete shape alone "
                        "cannot justify rejection; return uncertain unless visible evidence "
                        "identifies another category or background."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "requested_category": str(requested_label),
                                    "candidate_clipped_by_image_edge": bool(
                                        candidate_clipped_by_image_edge
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        },
                        {
                            "type": "text",
                            "text": "Full scene with the candidate marked in green:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(outlined_image_rgb)},
                        },
                        {"type": "text", "text": "Clean close-up:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(clean_crop_rgb)},
                        },
                    ],
                },
            ],
        )
        verdict = result.verdict
        if candidate_clipped_by_image_edge and verdict == "rejected":
            verdict = "uncertain"
        return TargetVerification(
            is_target=verdict == "confirmed",
            verdict=verdict,
            visual_verdict=verdict,
            reason=f"object type {verdict}: {result.reason}",
        )

    def verify_spatial_target(
        self,
        instruction: SearchInstruction,
        object_label: str,
        closeup_image_rgb: np.ndarray,
        viewpoint_images_rgb: Sequence[np.ndarray],
        *,
        room_label: str = "",
    ) -> TargetVerification:
        """Jointly review attributes, spatial relation and room as upstream SysNav."""

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Instruction and decomposed conditions:\n"
                    + json.dumps(instruction.to_dict(), ensure_ascii=False)
                    + "\nCandidate description:\n"
                    + json.dumps(
                        {
                            "object_type": object_label,
                            "room_type": room_label or "Unknown",
                        },
                        ensure_ascii=False,
                    )
                ),
            },
            {"type": "text", "text": "Close-up image of the candidate object:"},
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(closeup_image_rgb)},
            },
        ]
        views = tuple(viewpoint_images_rgb)
        if views:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "The following saved viewpoints show the candidate and its "
                        "surroundings; the green box marks the same 3-D object."
                    ),
                }
            )
            for index, image in enumerate(views, start=1):
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Surrounding viewpoint {index}/{len(views)}:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image)},
                        },
                    ]
                )
        result = self._parse(
            self.config.fast_model,
            _TargetSpatialResult,
            [
                {
                    "role": "system",
                    "content": (
                        "You are given an instruction specifying object type, room "
                        "condition, spatial condition, target attributes, an anchor "
                        "object and anchor attributes. You are also given a close-up "
                        "of one candidate and several images of that object and its "
                        "surroundings from different viewpoints. Decide whether the "
                        "candidate matches the complete instruction. Consider target "
                        "attributes, anchor identity, spatial relation and room condition "
                        "jointly. If room_type is Unknown, infer it from the images. If "
                        "any requested condition is unsatisfied or cannot be clearly "
                        "determined, return is_target=false. Return true only when all "
                        "requested conditions are satisfied."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        verdict = "confirmed" if result.is_target else "rejected"
        room_verdict = (
            "not_required"
            if not instruction.room_condition
            else ("confirmed" if result.is_target else "uncertain")
        )
        return TargetVerification(
            is_target=bool(result.is_target),
            verdict=verdict,
            visual_verdict=verdict,
            room_verdict=room_verdict,
            reason=result.reason,
        )

    def verify_anchor_object(
        self,
        instruction: SearchInstruction,
        object_label: str,
        closeup_image_rgb: np.ndarray,
        *,
        room_label: str = "",
    ) -> TargetVerification:
        """Confirm the anchor's attributes and room before it guides search."""

        result = self._parse(
            self.config.fast_model,
            _TargetSpatialResult,
            [
                {
                    "role": "system",
                    "content": (
                        "You are given a requested anchor object, its attribute "
                        "condition and room condition, plus a close-up image of one "
                        "candidate. Decide whether this is the requested anchor. If "
                        "the supplied room type is Unknown, infer it from visible "
                        "context. If any requested condition is not satisfied or "
                        "cannot be clearly determined, return is_target=false."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "anchor_object": instruction.anchor_object,
                                    "anchor_attribute": instruction.anchor_attribute,
                                    "room_condition": instruction.room_condition,
                                    "candidate_label": object_label,
                                    "room_type": room_label or "Unknown",
                                },
                                ensure_ascii=False,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(closeup_image_rgb)},
                        },
                    ],
                },
            ],
        )
        verdict = "confirmed" if result.is_target else "rejected"
        room_verdict = (
            "not_required"
            if not instruction.room_condition
            else ("confirmed" if result.is_target else "uncertain")
        )
        return TargetVerification(
            is_target=bool(result.is_target),
            verdict=verdict,
            visual_verdict=verdict,
            room_verdict=room_verdict,
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
        """Verify identity and local visual relations, never room semantics.

        ``room_description`` remains accepted for source compatibility, but is
        intentionally ignored.  The search policy evaluates room constraints
        against the independently maintained Room node.
        """

        del candidate_label  # The detector's class prediction must not bias identity.
        del room_description
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
        needs_local_relation = bool(
            instruction.spatial_condition
            or instruction.attribute_condition
            or instruction.anchor_object
            or instruction.anchor_attribute
        )
        if not needs_local_relation:
            return TargetVerification(
                is_target=True,
                reason=(
                    f"identity confirmed: {identity.reason}; "
                    "no local relation constraints"
                ),
                verdict="confirmed",
                visual_verdict="confirmed",
            )

        relation_context = {
            "requested_target_category": instruction.target_object,
            "spatial_condition": instruction.spatial_condition,
            "attribute_condition": instruction.attribute_condition,
            "anchor_object": instruction.anchor_object,
            "anchor_attribute": instruction.anchor_attribute,
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
                        "green-outlined candidate whose identity is confirmed. The candidate's "
                        "identity has already been confirmed by Stage 1 and is a fixed premise "
                        "for this stage. Do not re-identify, question, discuss, or contradict the "
                        "candidate's object category, even if its appearance seems ambiguous in "
                        "this image. Judge only anchor identity, spatial relation, and "
                        "non-category attributes. Room type is deliberately evaluated elsewhere "
                        "from the multi-view Room node: do not infer, discuss, confirm, or reject "
                        "the room type in this stage. Return exactly one verdict for all supplied "
                        "local attribute and spatial constraints: "
                        "confirmed only when the supplied image directly supports every "
                        "constraint; rejected only when a visible constraint is contradicted; "
                        "uncertain when an anchor, relation, or attribute is not "
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
                        "Treat detector-provided nearby labels as unverified supporting hints, "
                        "never as proof. Do not use object-size "
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
                visual_verdict=relation.verdict,
            )
        return TargetVerification(
            is_target=True,
            reason=(
                f"identity confirmed: {identity.reason}; "
                f"constraints confirmed: {relation.reason}"
            ),
            verdict="confirmed",
            visual_verdict="confirmed",
        )
