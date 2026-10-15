"""Shared value objects for the SysNav integration boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _normalise_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class SearchInstruction:
    """Structured form of one natural-language object-search instruction."""

    raw_text: str
    target_object: str
    room_condition: str = ""
    spatial_condition: str = ""
    attribute_condition: str = ""
    anchor_object: str = ""
    anchor_attribute: str = ""

    def __post_init__(self) -> None:
        if not str(self.raw_text).strip():
            raise ValueError("raw_text must not be empty")
        for name in (
            "target_object",
            "room_condition",
            "spatial_condition",
            "attribute_condition",
            "anchor_object",
            "anchor_attribute",
        ):
            object.__setattr__(self, name, _normalise_text(getattr(self, name)))
        if not self.target_object:
            raise ValueError("target_object must not be empty")

    @property
    def detector_prompts(self) -> tuple[str, ...]:
        """Open-vocabulary labels required by the current task."""

        values = [self.target_object]
        if self.anchor_object and self.anchor_object not in values:
            values.append(self.anchor_object)
        return tuple(values)

    @property
    def needs_vlm_verification(self) -> bool:
        return bool(
            self.room_condition
            or self.spatial_condition
            or self.attribute_condition
            or self.anchor_attribute
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Detection2D:
    """One open-vocabulary detection and its refined binary mask."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        label = _normalise_text(self.label)
        if not label:
            raise ValueError("detection label must not be empty")
        object.__setattr__(self, "label", label)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must be an ordered (x1, y1, x2, y2)")
        object.__setattr__(self, "bbox_xyxy", bbox)
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be a two-dimensional array")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class LocalizedDetection:
    """A 2-D detection localized in the shared MuJoCo/Nav2 world frame."""

    detection: Detection2D
    position_world: np.ndarray = field(repr=False, compare=False)
    bbox_min_world: np.ndarray = field(repr=False, compare=False)
    bbox_max_world: np.ndarray = field(repr=False, compare=False)
    point_count: int
    room_id: str | None

    def __post_init__(self) -> None:
        for name in ("position_world", "bbox_min_world", "bbox_max_world"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value)
        if int(self.point_count) <= 0:
            raise ValueError("point_count must be positive")


@dataclass(frozen=True)
class RoomSelection:
    room_id: str
    reason: str
    used_vlm: bool = True


@dataclass(frozen=True)
class RoomDescription:
    """Runtime visual semantics for an otherwise anonymous room id."""

    room_type: str
    visual_cues: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "room_type", _normalise_text(self.room_type))
        object.__setattr__(
            self,
            "visual_cues",
            tuple(_normalise_text(value) for value in self.visual_cues if str(value).strip()),
        )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")


@dataclass(frozen=True)
class EarlyStopDecision:
    switch_room: bool
    target_room_id: str | None
    reason: str


@dataclass(frozen=True)
class TargetVerification:
    is_target: bool
    reason: str


def json_safe(value: Any) -> Any:
    """Recursively convert component values into JSON-serialisable data."""

    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
