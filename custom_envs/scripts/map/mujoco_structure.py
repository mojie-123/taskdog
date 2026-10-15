"""Extract structural walls and anonymous portals from a MuJoCo XML file.

This module deliberately inspects only direct ``worldbody`` box geoms whose
names start with ``wall_``. XML comments, bodies, furniture and task objects
are outside the accepted input surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


STRUCTURE_GEOM_PREFIX = "wall_"


@dataclass(frozen=True)
class WallSegment:
    name: str
    center: np.ndarray
    direction: np.ndarray
    half_length: float
    half_thickness: float
    z_center: float
    z_half_size: float

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]], dtype=np.float64)

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        delta = self.direction * self.half_length
        return self.center - delta, self.center + delta

    def footprint(self, extra_thickness: float = 0.0) -> np.ndarray:
        along = self.direction * self.half_length
        across = self.normal * (self.half_thickness + float(extra_thickness))
        return np.asarray(
            [
                self.center - along - across,
                self.center + along - across,
                self.center + along + across,
                self.center - along + across,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Portal:
    id: str
    center: np.ndarray
    direction: np.ndarray
    width: float
    half_thickness: float
    endpoints: tuple[np.ndarray, np.ndarray]
    source_walls: tuple[str, ...]

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]], dtype=np.float64)

    def closure_footprint(self, overlap: float = 0.0) -> np.ndarray:
        half_length = 0.5 * self.width + float(overlap)
        half_thickness = self.half_thickness + float(overlap)
        along = self.direction * half_length
        across = self.normal * half_thickness
        return np.asarray(
            [
                self.center - along - across,
                self.center + along - across,
                self.center + along + across,
                self.center - along + across,
            ],
            dtype=np.float64,
        )


def _parse_vector(element: ET.Element, attribute: str, expected: int) -> np.ndarray:
    raw = element.get(attribute)
    if raw is None:
        raise ValueError(
            f"<{element.tag} name={element.get('name')!r}> lacks {attribute!r}"
        )
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.size != expected:
        raise ValueError(
            f"<{element.tag} name={element.get('name')!r}> {attribute!r} "
            f"must contain {expected} values, got {values.size}"
        )
    return values


def _yaw_from_element(element: ET.Element, angles_are_degrees: bool) -> float:
    if element.get("quat") is not None:
        w, x, y, z = _parse_vector(element, "quat", 4)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    if element.get("euler") is not None:
        yaw = float(_parse_vector(element, "euler", 3)[2])
        return math.radians(yaw) if angles_are_degrees else yaw
    return 0.0


def extract_wall_segments(scene_xml: Path | str) -> list[WallSegment]:
    """Read only top-level structural wall boxes from a MuJoCo scene."""

    xml_path = Path(scene_xml).expanduser().resolve()
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"
    angles_are_degrees = angle_unit == "degree"
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo XML has no <worldbody>: {xml_path}")

    walls: list[WallSegment] = []
    for geom in worldbody.findall("geom"):
        name = geom.get("name", "")
        if not name.startswith(STRUCTURE_GEOM_PREFIX):
            continue
        if geom.get("type", "sphere") != "box":
            raise ValueError(f"Structural geom {name!r} must be a box")
        if geom.get("fromto") is not None:
            raise ValueError(f"Structural geom {name!r} uses unsupported fromto")

        pos = _parse_vector(geom, "pos", 3)
        size = _parse_vector(geom, "size", 3)
        yaw = _yaw_from_element(geom, angles_are_degrees)
        if size[0] >= size[1]:
            major_axis_yaw = yaw
            half_length = float(size[0])
            half_thickness = float(size[1])
        else:
            major_axis_yaw = yaw + 0.5 * math.pi
            half_length = float(size[1])
            half_thickness = float(size[0])
        major_axis_yaw %= math.pi
        direction = np.array(
            [math.cos(major_axis_yaw), math.sin(major_axis_yaw)], dtype=np.float64
        )
        walls.append(
            WallSegment(
                name=name,
                center=pos[:2].copy(),
                direction=direction,
                half_length=half_length,
                half_thickness=half_thickness,
                z_center=float(pos[2]),
                z_half_size=float(size[2]),
            )
        )

    if not walls:
        raise ValueError(
            f"No direct <worldbody> box geoms named {STRUCTURE_GEOM_PREFIX}* in {xml_path}"
        )
    names = [wall.name for wall in walls]
    if len(names) != len(set(names)):
        raise ValueError("Structural wall geom names must be unique")
    return walls


def _angle_distance_mod_pi(first: float, second: float) -> float:
    return abs((first - second + 0.5 * math.pi) % math.pi - 0.5 * math.pi)


def detect_portals(
    walls: list[WallSegment],
    min_width: float,
    max_width: float,
    angle_tolerance_degrees: float = 5.0,
    line_offset_tolerance: float = 0.15,
) -> list[Portal]:
    """Detect door-sized gaps between collinear structural wall segments."""

    if not 0.0 < min_width <= max_width:
        raise ValueError("Portal width range must satisfy 0 < min <= max")
    angle_tolerance = math.radians(float(angle_tolerance_degrees))
    groups: list[dict[str, object]] = []

    for wall in sorted(walls, key=lambda item: item.name):
        angle = math.atan2(wall.direction[1], wall.direction[0]) % math.pi
        matched: dict[str, object] | None = None
        for group in groups:
            group_angle = float(group["angle"])
            if _angle_distance_mod_pi(angle, group_angle) > angle_tolerance:
                continue
            direction = np.asarray(group["direction"], dtype=np.float64)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            offset = float(np.dot(wall.center, normal))
            if abs(offset - float(group["offset"])) <= line_offset_tolerance:
                matched = group
                break
        if matched is None:
            direction = wall.direction.copy()
            if direction[0] < -1e-12 or (
                abs(direction[0]) <= 1e-12 and direction[1] < 0.0
            ):
                direction *= -1.0
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            matched = {
                "angle": math.atan2(direction[1], direction[0]) % math.pi,
                "direction": direction,
                "offset": float(np.dot(wall.center, normal)),
                "walls": [],
            }
            groups.append(matched)
        matched["walls"].append(wall)  # type: ignore[index, union-attr]

    detected: list[Portal] = []
    for group in groups:
        group_walls = list(group["walls"])  # type: ignore[arg-type]
        if len(group_walls) < 2:
            continue
        direction = np.asarray(group["direction"], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        offset = float(np.median([np.dot(wall.center, normal) for wall in group_walls]))
        intervals: list[tuple[float, float, WallSegment]] = []
        for wall in group_walls:
            centre_projection = float(np.dot(wall.center, direction))
            projected_half_length = wall.half_length * abs(
                float(np.dot(wall.direction, direction))
            )
            intervals.append(
                (
                    centre_projection - projected_half_length,
                    centre_projection + projected_half_length,
                    wall,
                )
            )
        intervals.sort(key=lambda item: item[0])

        merged: list[dict[str, object]] = []
        for start, end, wall in intervals:
            if merged and start <= float(merged[-1]["end"]) + 1e-6:
                merged[-1]["end"] = max(float(merged[-1]["end"]), end)
                merged[-1]["walls"].append(wall)  # type: ignore[index, union-attr]
            else:
                merged.append({"start": start, "end": end, "walls": [wall]})

        for left, right in zip(merged, merged[1:]):
            gap_start = float(left["end"])
            gap_end = float(right["start"])
            width = gap_end - gap_start
            if width < min_width - 1e-6 or width > max_width + 1e-6:
                continue
            centre_projection = 0.5 * (gap_start + gap_end)
            center = centre_projection * direction + offset * normal
            endpoints = (
                gap_start * direction + offset * normal,
                gap_end * direction + offset * normal,
            )
            adjacent_walls = list(left["walls"]) + list(right["walls"])  # type: ignore[arg-type]
            detected.append(
                Portal(
                    id="",
                    center=center,
                    direction=direction.copy(),
                    width=width,
                    half_thickness=max(wall.half_thickness for wall in adjacent_walls),
                    endpoints=endpoints,
                    source_walls=tuple(sorted(wall.name for wall in adjacent_walls)),
                )
            )

    detected.sort(key=lambda portal: (float(portal.center[1]), float(portal.center[0])))
    return [
        Portal(
            id=f"portal_{index}",
            center=portal.center,
            direction=portal.direction,
            width=portal.width,
            half_thickness=portal.half_thickness,
            endpoints=portal.endpoints,
            source_walls=portal.source_walls,
        )
        for index, portal in enumerate(detected)
    ]
