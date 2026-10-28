"""Build the multi-view image evidence used by SysNav object review.

The upstream SysNav implementation projects an object's 3-D bounding box into
saved viewpoint images before asking the VLM about spatial relations.  TaskDog
uses an ordinary pinhole camera rather than a panoramic camera, so views whose
projected box lies outside the image are simply omitted.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .scene_memory import ObjectRecord, ViewpointRecord


def _bbox_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    minimum = np.asarray(minimum, dtype=np.float64).reshape(3)
    maximum = np.asarray(maximum, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def project_bbox_world(
    bbox_min_world: np.ndarray,
    bbox_max_world: np.ndarray,
    viewpoint: ViewpointRecord,
) -> tuple[float, float, float, float] | None:
    """Project an axis-aligned world box into one saved perspective image."""

    if viewpoint.camera_position_world is None:
        return None
    if viewpoint.camera_rotation_world is None or not viewpoint.camera_intrinsics:
        return None
    intrinsics = viewpoint.camera_intrinsics
    required = {"width", "height", "fx", "fy", "cx", "cy"}
    if not required.issubset(intrinsics):
        return None

    corners_world = _bbox_corners(bbox_min_world, bbox_max_world)
    camera_position = np.asarray(
        viewpoint.camera_position_world, dtype=np.float64
    ).reshape(3)
    camera_rotation = np.asarray(
        viewpoint.camera_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    # Row-vector inverse of: p_world = p_mujoco_camera @ R.T + t.
    corners_camera = (corners_world - camera_position) @ camera_rotation
    # MuJoCo camera local (+X right, +Y up, -Z forward) -> optical.
    optical = np.column_stack(
        (corners_camera[:, 0], -corners_camera[:, 1], -corners_camera[:, 2])
    )
    in_front = optical[:, 2] > 1e-4
    if not np.any(in_front):
        return None
    optical = optical[in_front]
    u = float(intrinsics["fx"]) * optical[:, 0] / optical[:, 2] + float(
        intrinsics["cx"]
    )
    v = float(intrinsics["fy"]) * optical[:, 1] / optical[:, 2] + float(
        intrinsics["cy"]
    )
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    x1 = float(np.clip(np.min(u), 0.0, max(0, width - 1)))
    y1 = float(np.clip(np.min(v), 0.0, max(0, height - 1)))
    x2 = float(np.clip(np.max(u), 0.0, max(0, width - 1)))
    y2 = float(np.clip(np.max(v), 0.0, max(0, height - 1)))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return (x1, y1, x2, y2)


def load_rgb(path: str) -> np.ndarray | None:
    source = Path(path)
    if not path or not source.is_file():
        return None
    return np.asarray(Image.open(source).convert("RGB"))


def annotate_viewpoint(
    viewpoint: ViewpointRecord,
    object_record: ObjectRecord,
) -> np.ndarray | None:
    """Load one viewpoint and mark the object location with a green box."""

    image = load_rgb(viewpoint.image_path)
    if image is None:
        return None
    bbox = viewpoint.object_bboxes_xyxy.get(object_record.object_id)
    if bbox is None:
        bbox = project_bbox_world(
            object_record.bbox_min_world,
            object_record.bbox_max_world,
            viewpoint,
        )
    if bbox is None:
        return None
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    annotated = np.ascontiguousarray(image).copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return annotated


def best_object_images(
    viewpoint: ViewpointRecord,
    object_record: ObjectRecord,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the annotated full scene and an enlarged clean context crop."""

    image = load_rgb(viewpoint.image_path)
    if image is None:
        return None
    bbox = viewpoint.object_bboxes_xyxy.get(object_record.object_id)
    if bbox is None:
        bbox = object_record.best_bbox_xyxy
    if bbox is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding = max(80.0, box_width, box_height)
    left = max(0, int(np.floor(x1 - padding)))
    right = min(width, int(np.ceil(x2 + padding)))
    top = max(0, int(np.floor(y1 - padding)))
    bottom = min(height, int(np.ceil(y2 + padding)))
    crop = np.ascontiguousarray(image[top:bottom, left:right]).copy()
    if crop.size == 0:
        crop = np.ascontiguousarray(image).copy()
    scale = max(1.0, 256.0 / float(max(crop.shape[:2])))
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    annotated = np.ascontiguousarray(image).copy()
    cv2.rectangle(
        annotated,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (0, 255, 0),
        3,
    )
    return annotated, crop
