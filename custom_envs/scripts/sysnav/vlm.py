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
from typing import Any, Iterable

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


class _TargetResult(BaseModel):
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
                        "Describe the visually observed indoor room. The supplied "
                        "room id is an anonymous map id, not a semantic label. Base "
                        "the answer only on the image and detected object names."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"room_id={room_id}; detections={labels}"},
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
    ) -> TargetVerification:
        """Verify attributes/room/spatial constraints on a detected candidate."""

        context = {
            "instruction": instruction.to_dict(),
            "candidate_label": candidate_label,
            "runtime_room_description": room_description,
            "nearby_detected_objects": sorted(set(nearby_objects)),
        }
        result = self._parse(
            self.config.reasoning_model,
            _TargetResult,
            [
                {
                    "role": "system",
                    "content": (
                        "Verify whether the outlined candidate satisfies the complete "
                        "instruction. Reject it when an explicit attribute, room or "
                        "spatial relation is contradicted or not visible enough to verify."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(context, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_rgb)}},
                    ],
                },
            ],
        )
        return TargetVerification(is_target=result.is_target, reason=result.reason)
