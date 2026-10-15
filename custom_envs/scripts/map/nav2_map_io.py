"""Read a Nav2 PGM/YAML map and convert between grid and world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Nav2Map:
    """A Nav2 occupancy map in its original image-row convention."""

    yaml_path: Path
    image_path: Path
    image: np.ndarray
    resolution: float
    origin: tuple[float, float]
    free_mask: np.ndarray
    occupied_mask: np.ndarray
    unknown_mask: np.ndarray

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Return ``(row, col)`` for a world point.

        Nav2's origin is the lower-left map corner, while image row zero is
        the top row. Points outside the raster are returned as out-of-range
        indices so callers can decide whether to clamp or reject them.
        """

        col = int(np.floor((float(x) - self.origin[0]) / self.resolution))
        y_index = int(np.floor((float(y) - self.origin[1]) / self.resolution))
        row = self.height - 1 - y_index
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return the world coordinate at the centre of a grid cell."""

        x = self.origin[0] + (float(col) + 0.5) * self.resolution
        y = self.origin[1] + (float(self.height - 1 - row) + 0.5) * self.resolution
        return x, y

    def world_points_to_image(self, points: Iterable[Iterable[float]]) -> np.ndarray:
        """Convert world XY polygon vertices to OpenCV image coordinates."""

        points_array = np.asarray(list(points), dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        cols = (points_array[:, 0] - self.origin[0]) / self.resolution - 0.5
        rows = (
            self.height - (points_array[:, 1] - self.origin[1]) / self.resolution - 0.5
        )
        return np.column_stack((cols, rows))


def load_nav2_map(path: Path | str) -> Nav2Map:
    """Load a Nav2 map without changing its row orientation."""

    yaml_path = Path(path).expanduser().resolve()
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    required = {
        "image",
        "resolution",
        "origin",
        "negate",
        "occupied_thresh",
        "free_thresh",
    }
    missing = sorted(required.difference(document))
    if missing:
        raise ValueError(f"Nav2 map YAML is missing fields: {', '.join(missing)}")

    image_path = Path(str(document["image"])).expanduser()
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image_path = image_path.resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read Nav2 map image: {image_path}")

    resolution = float(document["resolution"])
    if resolution <= 0.0:
        raise ValueError(f"Map resolution must be positive, got {resolution}")
    raw_origin = document["origin"]
    if not isinstance(raw_origin, (list, tuple)) or len(raw_origin) < 2:
        raise ValueError("Nav2 map origin must contain at least x and y")
    origin = (float(raw_origin[0]), float(raw_origin[1]))

    pixels = image.astype(np.float64) / 255.0
    occupancy_probability = pixels if int(document["negate"]) else 1.0 - pixels
    occupied_mask = occupancy_probability > float(document["occupied_thresh"])
    free_mask = occupancy_probability < float(document["free_thresh"])
    unknown_mask = ~(occupied_mask | free_mask)

    return Nav2Map(
        yaml_path=yaml_path,
        image_path=image_path,
        image=image,
        resolution=resolution,
        origin=origin,
        free_mask=free_mask,
        occupied_mask=occupied_mask,
        unknown_mask=unknown_mask,
    )
