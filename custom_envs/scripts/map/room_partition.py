"""Rasterise structural walls and partition the enclosed space into rooms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy import ndimage

from .mujoco_structure import Portal, WallSegment
from .nav2_map_io import Nav2Map


@dataclass(frozen=True)
class RoomPartitionResult:
    room_region_id_map: np.ndarray
    room_searchable_id_map: np.ndarray
    room_searchable_mask: np.ndarray
    wall_mask: np.ndarray
    portal_id_map: np.ndarray
    rooms: list[dict[str, object]]
    portals: list[dict[str, object]]


def _fill_world_polygon(
    mask: np.ndarray, nav_map: Nav2Map, polygon: np.ndarray, value: int
) -> None:
    image_points = nav_map.world_points_to_image(polygon)
    cv2.fillPoly(mask, [np.rint(image_points).astype(np.int32)], int(value))


def _world_bbox(nav_map: Nav2Map, rows: np.ndarray, cols: np.ndarray) -> list[float]:
    x_min, y_min = nav_map.grid_to_world(int(rows.max()), int(cols.min()))
    x_max, y_max = nav_map.grid_to_world(int(rows.min()), int(cols.max()))
    half = 0.5 * nav_map.resolution
    return [x_min - half, y_min - half, x_max + half, y_max + half]


def _component_polygon(
    nav_map: Nav2Map, component_mask: np.ndarray
) -> list[list[float]]:
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon_pixels = max(1.0, 0.10 / nav_map.resolution)
    simplified = cv2.approxPolyDP(contour, epsilon_pixels, True).reshape(-1, 2)
    polygon: list[list[float]] = []
    for col, row in simplified:
        x, y = nav_map.grid_to_world(int(row), int(col))
        polygon.append([round(x, 4), round(y, 4)])
    return polygon


def _label_at_world(nav_map: Nav2Map, labels: np.ndarray, point: np.ndarray) -> int:
    row, col = nav_map.world_to_grid(float(point[0]), float(point[1]))
    if row < 0 or row >= nav_map.height or col < 0 or col >= nav_map.width:
        return -2
    return int(labels[row, col])


def _nearest_searchable_point(
    nav_map: Nav2Map,
    searchable_id_map: np.ndarray,
    room_label: int,
    requested: np.ndarray,
) -> np.ndarray | None:
    rows, cols = np.nonzero(searchable_id_map == room_label)
    if rows.size == 0:
        return None
    xs = nav_map.origin[0] + (cols.astype(np.float64) + 0.5) * nav_map.resolution
    ys = (
        nav_map.origin[1]
        + (nav_map.height - 1 - rows.astype(np.float64) + 0.5) * nav_map.resolution
    )
    distances = np.square(xs - requested[0]) + np.square(ys - requested[1])
    index = int(np.argmin(distances))
    return np.array([xs[index], ys[index]], dtype=np.float64)


def build_room_partition(
    nav_map: Nav2Map,
    walls: list[WallSegment],
    portals: list[Portal],
    min_room_area_m2: float = 2.0,
    portal_overlap_m: float = 0.06,
    barrier_close_cells: int = 1,
    search_clearance_m: float = 0.52,
    entry_offset_m: float = 0.85,
) -> RoomPartitionResult:
    """Build anonymous room labels using only structural wall geometry."""

    shape = nav_map.image.shape
    wall_mask_u8 = np.zeros(shape, dtype=np.uint8)
    for wall in walls:
        _fill_world_polygon(wall_mask_u8, nav_map, wall.footprint(), 1)
    wall_mask = wall_mask_u8.astype(bool)

    portal_id_map = np.zeros(shape, dtype=np.int16)
    for portal_index, portal in enumerate(portals, start=1):
        _fill_world_polygon(
            portal_id_map,
            nav_map,
            portal.closure_footprint(overlap=portal_overlap_m),
            portal_index,
        )
    portal_mask = portal_id_map > 0
    barrier = (wall_mask | portal_mask).astype(np.uint8)
    if barrier_close_cells > 0:
        kernel_size = 2 * int(barrier_close_cells) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel)
    open_space = barrier == 0

    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    open_labels, _ = ndimage.label(open_space, structure=connectivity)
    border_labels = np.unique(
        np.concatenate(
            [
                open_labels[0, :],
                open_labels[-1, :],
                open_labels[:, 0],
                open_labels[:, -1],
            ]
        )
    )
    exterior = np.isin(open_labels, border_labels)
    interior = open_space & ~exterior
    raw_rooms, raw_count = ndimage.label(interior, structure=connectivity)

    min_cells = max(1, int(math.ceil(float(min_room_area_m2) / nav_map.resolution**2)))
    components: list[tuple[int, int, float, float]] = []
    for raw_label in range(1, raw_count + 1):
        rows, cols = np.nonzero(raw_rooms == raw_label)
        if rows.size < min_cells:
            continue
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())
        centroid_x = nav_map.origin[0] + (centroid_col + 0.5) * nav_map.resolution
        centroid_y = (
            nav_map.origin[1]
            + (nav_map.height - 1 - centroid_row + 0.5) * nav_map.resolution
        )
        components.append((raw_label, int(rows.size), centroid_x, centroid_y))
    components.sort(key=lambda item: (item[3], item[2]))

    region_id_map = np.full(shape, -2, dtype=np.int16)
    region_id_map[wall_mask] = -1
    rooms: list[dict[str, object]] = []
    for stable_label, (raw_label, cell_count, centroid_x, centroid_y) in enumerate(
        components, start=1
    ):
        component_mask = raw_rooms == raw_label
        region_id_map[component_mask] = stable_label
        rows, cols = np.nonzero(component_mask)
        rooms.append(
            {
                "id": f"room_{stable_label - 1}",
                "label_value": stable_label,
                "area_m2": round(cell_count * nav_map.resolution**2, 4),
                "center": [round(centroid_x, 4), round(centroid_y, 4)],
                "bbox": [round(value, 4) for value in _world_bbox(nav_map, rows, cols)],
                "polygon": _component_polygon(nav_map, component_mask),
                "neighbors": [],
                "portals": [],
                "entry_poses": [],
            }
        )

    if not rooms:
        raise ValueError(
            "No enclosed room larger than the minimum area was found; "
            "check the wall overlay and portal width parameters"
        )

    # Portal pixels stay anonymous. Structural wall pixels retain -1 even
    # where the closure overlaps a wall endpoint.
    region_id_map[portal_mask & ~wall_mask] = 0

    distance_to_nav_obstacle = ndimage.distance_transform_edt(nav_map.free_mask)
    distance_to_nav_obstacle *= nav_map.resolution
    searchable = (
        (region_id_map > 0)
        & nav_map.free_mask
        & (distance_to_nav_obstacle >= float(search_clearance_m))
    )
    searchable_id_map = np.where(searchable, region_id_map, 0).astype(np.int16)
    room_searchable_mask = searchable.astype(np.uint8)

    room_by_label = {int(room["label_value"]): room for room in rooms}
    portal_documents: list[dict[str, object]] = []
    for portal in portals:
        normal = portal.normal
        first_probe = portal.center + normal * float(entry_offset_m)
        second_probe = portal.center - normal * float(entry_offset_m)
        first_label = _label_at_world(nav_map, region_id_map, first_probe)
        second_label = _label_at_world(nav_map, region_id_map, second_probe)
        connected_labels = sorted(
            {label for label in (first_label, second_label) if label > 0}
        )
        connected_ids = [room_by_label[label]["id"] for label in connected_labels]
        portal_document: dict[str, object] = {
            "id": portal.id,
            "center": [round(float(value), 4) for value in portal.center],
            "endpoints": [
                [round(float(value), 4) for value in endpoint]
                for endpoint in portal.endpoints
            ],
            "width_m": round(float(portal.width), 4),
            "connects": connected_ids,
            "is_exterior": len(connected_ids) < 2,
            "source_walls": list(portal.source_walls),
        }
        portal_documents.append(portal_document)

        if len(connected_labels) == 2:
            first_room, second_room = (
                room_by_label[label] for label in connected_labels
            )
            first_room["neighbors"].append(second_room["id"])
            second_room["neighbors"].append(first_room["id"])
        for probe, label in ((first_probe, first_label), (second_probe, second_label)):
            if label <= 0 or label not in room_by_label:
                continue
            entry = _nearest_searchable_point(nav_map, searchable_id_map, label, probe)
            if entry is None:
                continue
            yaw = math.atan2(entry[1] - portal.center[1], entry[0] - portal.center[0])
            room_document = room_by_label[label]
            room_document["portals"].append(portal.id)
            room_document["entry_poses"].append(
                {
                    "portal": portal.id,
                    "pose": [
                        round(float(entry[0]), 4),
                        round(float(entry[1]), 4),
                        round(float(yaw), 6),
                    ],
                }
            )

    for room in rooms:
        room["neighbors"] = sorted(set(room["neighbors"]))
        room["portals"] = sorted(set(room["portals"]))
        room["entry_poses"] = sorted(
            room["entry_poses"], key=lambda item: item["portal"]
        )

    return RoomPartitionResult(
        room_region_id_map=region_id_map,
        room_searchable_id_map=searchable_id_map,
        room_searchable_mask=room_searchable_mask,
        wall_mask=wall_mask.astype(np.uint8),
        portal_id_map=portal_id_map,
        rooms=rooms,
        portals=portal_documents,
    )
