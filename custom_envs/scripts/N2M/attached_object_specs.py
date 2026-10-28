"""Shared MoveIt attached-object box specifications.

The box center is expressed in the MuJoCo target body's local frame.  It is
not necessarily the body origin: several tabletop objects use a free-body
origin at (or near) the support surface while their collision geometry extends
upward from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AttachedBoxSpec:
    """One oriented box used to approximate a carried object in MoveIt."""

    size_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float]


# These values bound the MuJoCo collision geometry, not the visual mesh.  The
# banana and bowl entries retain compatibility with the older test scenes even
# though the current scene.xml uses orange and mug in their places.
ATTACHED_OBJECT_SPECS: dict[str, AttachedBoxSpec] = {
    "cube": AttachedBoxSpec(
        size_xyz=(0.050000, 0.050000, 0.050000),
        center_xyz=(0.0, 0.0, 0.0),
    ),
    "apple": AttachedBoxSpec(
        size_xyz=(0.052800, 0.052800, 0.058442),
        center_xyz=(0.000687, -0.002797, 0.028428),
    ),
    "orange": AttachedBoxSpec(
        size_xyz=(0.050400, 0.050400, 0.057082),
        center_xyz=(-0.005548, -0.014688, 0.028342),
    ),
    "mug": AttachedBoxSpec(
        size_xyz=(0.057300, 0.057300, 0.058538),
        center_xyz=(0.0, 0.0, 0.029269),
    ),
    "banana": AttachedBoxSpec(
        size_xyz=(0.197263, 0.075175, 0.037305),
        center_xyz=(-0.003177, -0.014535, 0.0),
    ),
    "bowl": AttachedBoxSpec(
        size_xyz=(0.162011, 0.161463, 0.055657),
        center_xyz=(0.000001, -0.000127, 0.000049),
    ),
}


def _validated_triplet(
    values: Iterable[float],
    *,
    field_name: str,
    require_positive: bool,
) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain three finite values")
    if require_positive and np.any(array <= 0.0):
        raise ValueError(f"{field_name} values must all be positive")
    return tuple(float(value) for value in array)


def resolve_attached_box_spec(
    object_name: str,
    *,
    size_override: Iterable[float] | None = None,
    center_override: Iterable[float] | None = None,
) -> AttachedBoxSpec:
    """Resolve one object's box while preserving explicit config overrides."""

    normalized_name = str(object_name).strip().lower()
    default = ATTACHED_OBJECT_SPECS.get(normalized_name)
    if default is None and size_override is None:
        supported = ", ".join(sorted(ATTACHED_OBJECT_SPECS))
        raise ValueError(
            f"no attached-object box is defined for {object_name!r}; "
            f"supported objects: {supported}; provide "
            "moveit_attached_object_size and optionally "
            "moveit_attached_object_center for a custom object"
        )

    size = _validated_triplet(
        size_override if size_override is not None else default.size_xyz,
        field_name="moveit_attached_object_size",
        require_positive=True,
    )
    center = _validated_triplet(
        center_override
        if center_override is not None
        else default.center_xyz if default is not None else (0.0, 0.0, 0.0),
        field_name="moveit_attached_object_center",
        require_positive=False,
    )
    return AttachedBoxSpec(size_xyz=size, center_xyz=center)


def attached_box_pose_matrix(
    T_tcp_object: np.ndarray,
    center_xyz: Iterable[float],
) -> np.ndarray:
    """Return ``T_tcp_box`` for a center expressed in the object frame."""

    transform = np.asarray(T_tcp_object, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_tcp_object must be a finite 4x4 transform")
    center = np.asarray(
        _validated_triplet(
            center_xyz,
            field_name="attached-object box center",
            require_positive=False,
        ),
        dtype=np.float64,
    )
    T_object_box = np.eye(4, dtype=np.float64)
    T_object_box[:3, 3] = center
    return transform @ T_object_box
