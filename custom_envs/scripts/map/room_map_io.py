"""Persist and load the anonymous room map consumed by navigation code."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .room_partition import RoomPartitionResult


def _relative_path(target: Path, base: Path) -> str:
    return os.path.relpath(str(target.resolve()), str(base.resolve()))


def write_room_map(
    manifest_path: Path,
    raster_path: Path,
    result: RoomPartitionResult,
    *,
    scene_xml: Path,
    nav2_map_yaml: Path,
    resolution: float,
    origin: tuple[float, float],
    structure_sources: list[str],
    generation_parameters: dict[str, Any],
) -> None:
    """Write the runtime NPZ plus a human-readable YAML manifest."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raster_path,
        room_region_id_map=result.room_region_id_map.astype(np.int16),
        room_searchable_id_map=result.room_searchable_id_map.astype(np.int16),
        room_searchable_mask=result.room_searchable_mask.astype(np.uint8),
        wall_mask=result.wall_mask.astype(np.uint8),
        portal_id_map=result.portal_id_map.astype(np.int16),
        resolution=np.asarray(float(resolution), dtype=np.float64),
        origin=np.asarray(origin, dtype=np.float64),
    )
    document = {
        "version": 1,
        "frame_id": "map",
        "source": {
            "scene_xml": _relative_path(scene_xml, manifest_path.parent),
            "nav2_map": _relative_path(nav2_map_yaml, manifest_path.parent),
            "structure_geoms": structure_sources,
        },
        "raster_file": _relative_path(raster_path, manifest_path.parent),
        "resolution": float(resolution),
        "origin": [float(origin[0]), float(origin[1])],
        "width": int(result.room_region_id_map.shape[1]),
        "height": int(result.room_region_id_map.shape[0]),
        "labels": {
            "outside_or_unknown": -2,
            "structural_wall": -1,
            "portal_or_unassigned": 0,
        },
        "generation": generation_parameters,
        "rooms": {
            str(room["id"]): {k: v for k, v in room.items() if k != "id"}
            for room in result.rooms
        },
        "portals": {
            str(portal["id"]): {k: v for k, v in portal.items() if k != "id"}
            for portal in result.portals
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


@dataclass
class RoomMap:
    """Runtime view over an anonymous room-map manifest and its raster data."""

    manifest_path: Path
    document: dict[str, Any]
    room_region_id_map: np.ndarray
    room_searchable_id_map: np.ndarray
    room_searchable_mask: np.ndarray
    wall_mask: np.ndarray
    portal_id_map: np.ndarray
    resolution: float
    origin: tuple[float, float]

    @classmethod
    def load(cls, manifest: Path | str) -> "RoomMap":
        manifest_path = Path(manifest).expanduser().resolve()
        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if int(document.get("version", -1)) != 1:
            raise ValueError(f"Unsupported room-map version in {manifest_path}")
        raster_path = Path(str(document["raster_file"])).expanduser()
        if not raster_path.is_absolute():
            raster_path = manifest_path.parent / raster_path
        with np.load(raster_path.resolve(), allow_pickle=False) as raster:
            arrays = {
                name: np.asarray(raster[name]).copy()
                for name in (
                    "room_region_id_map",
                    "room_searchable_id_map",
                    "room_searchable_mask",
                    "wall_mask",
                    "portal_id_map",
                )
            }
            resolution = float(raster["resolution"])
            origin_array = np.asarray(raster["origin"], dtype=np.float64)
        shape = arrays["room_region_id_map"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError(f"Room-map arrays do not share one shape in {raster_path}")
        if list(shape) != [int(document["height"]), int(document["width"])]:
            raise ValueError(f"Room-map raster shape disagrees with {manifest_path}")
        yaml_resolution = float(document["resolution"])
        yaml_origin = np.asarray(document["origin"], dtype=np.float64)
        if not np.isclose(resolution, yaml_resolution) or not np.allclose(
            origin_array, yaml_origin
        ):
            raise ValueError(f"Room-map raster metadata disagrees with {manifest_path}")
        return cls(
            manifest_path=manifest_path,
            document=document,
            resolution=resolution,
            origin=(float(origin_array[0]), float(origin_array[1])),
            **arrays,
        )

    @property
    def height(self) -> int:
        return int(self.room_region_id_map.shape[0])

    @property
    def width(self) -> int:
        return int(self.room_region_id_map.shape[1])

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.floor((float(x) - self.origin[0]) / self.resolution))
        y_index = int(np.floor((float(y) - self.origin[1]) / self.resolution))
        return self.height - 1 - y_index, col

    def room_id_at_world(self, x: float, y: float) -> str | None:
        row, col = self._world_to_grid(x, y)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        label = int(self.room_region_id_map[row, col])
        if label <= 0:
            return None
        for room_id, room in self.document["rooms"].items():
            if int(room["label_value"]) == label:
                return str(room_id)
        return None

    def searchable_mask(self, room_id: str) -> np.ndarray:
        room = self.document["rooms"].get(room_id)
        if room is None:
            raise KeyError(f"Unknown room id: {room_id}")
        return self.room_searchable_id_map == int(room["label_value"])

    def entry_poses(self, room_id: str) -> list[dict[str, Any]]:
        room = self.document["rooms"].get(room_id)
        if room is None:
            raise KeyError(f"Unknown room id: {room_id}")
        return list(room.get("entry_poses", []))

    def neighbors(self, room_id: str) -> list[str]:
        room = self.document["rooms"].get(room_id)
        if room is None:
            raise KeyError(f"Unknown room id: {room_id}")
        return [str(value) for value in room.get("neighbors", [])]
