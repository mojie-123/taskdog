"""Geometry-only graph over the anonymous room-map ids."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any

import numpy as np

from custom_envs.scripts.map.room_map_io import RoomMap


@dataclass(frozen=True)
class RoomNode:
    room_id: str
    center_xy: tuple[float, float]
    area_m2: float
    neighbors: tuple[str, ...]
    entry_poses: tuple[tuple[float, float, float], ...]


class AnonymousRoomGraph:
    """Map ids and metric anchors without assigning any semantic room names."""

    def __init__(self, room_map: RoomMap):
        self.room_map = room_map
        self.nodes: dict[str, RoomNode] = {}
        for room_id, payload in room_map.document["rooms"].items():
            entries = tuple(
                tuple(float(value) for value in item["pose"])
                for item in payload.get("entry_poses", [])
            )
            self.nodes[str(room_id)] = RoomNode(
                room_id=str(room_id),
                center_xy=tuple(float(value) for value in payload["center"]),
                area_m2=float(payload["area_m2"]),
                neighbors=tuple(str(value) for value in payload.get("neighbors", [])),
                entry_poses=entries,
            )
        self._edge_weights = self._build_edge_weights()

    @classmethod
    def load(cls, manifest: str) -> "AnonymousRoomGraph":
        return cls(RoomMap.load(manifest))

    def _build_edge_weights(self) -> dict[tuple[str, str], float]:
        weights: dict[tuple[str, str], float] = {}
        for portal in self.room_map.document.get("portals", {}).values():
            connected = [str(value) for value in portal.get("connects", [])]
            if len(connected) != 2:
                continue
            first, second = connected
            center = np.asarray(portal["center"], dtype=np.float64)
            first_center = np.asarray(self.nodes[first].center_xy, dtype=np.float64)
            second_center = np.asarray(self.nodes[second].center_xy, dtype=np.float64)
            weight = float(
                np.linalg.norm(first_center - center)
                + np.linalg.norm(second_center - center)
            )
            weights[(first, second)] = weight
            weights[(second, first)] = weight
        for room_id, node in self.nodes.items():
            for neighbor in node.neighbors:
                weights.setdefault(
                    (room_id, neighbor),
                    float(
                        np.linalg.norm(
                            np.asarray(node.center_xy)
                            - np.asarray(self.nodes[neighbor].center_xy)
                        )
                    ),
                )
        return weights

    def room_at(self, x: float, y: float) -> str | None:
        return self.room_map.room_id_at_world(x, y)

    def shortest_path(self, start: str, goal: str) -> tuple[list[str], float]:
        if start not in self.nodes or goal not in self.nodes:
            raise KeyError(f"unknown room path endpoint: {start!r}, {goal!r}")
        queue: list[tuple[float, str]] = [(0.0, start)]
        distance = {start: 0.0}
        previous: dict[str, str] = {}
        while queue:
            cost, room_id = heapq.heappop(queue)
            if room_id == goal:
                break
            if cost > distance[room_id]:
                continue
            for neighbor in self.nodes[room_id].neighbors:
                candidate = cost + self._edge_weights[(room_id, neighbor)]
                if candidate < distance.get(neighbor, math.inf):
                    distance[neighbor] = candidate
                    previous[neighbor] = room_id
                    heapq.heappush(queue, (candidate, neighbor))
        if goal not in distance:
            return [], math.inf
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path, float(distance[goal])

    def room_travel_costs(self, start: str) -> dict[str, float]:
        return {
            room_id: self.shortest_path(start, room_id)[1]
            for room_id in self.nodes
        }

    def execution_anchor(self, source_room: str, target_room: str) -> tuple[float, float, float]:
        """Return the target-side portal entry pose for a neighboring room."""

        if target_room not in self.nodes[source_room].neighbors:
            raise ValueError(f"{target_room} is not adjacent to {source_room}")
        target = self.room_map.document["rooms"][target_room]
        for entry in target.get("entry_poses", []):
            portal = self.room_map.document["portals"][entry["portal"]]
            if source_room in [str(value) for value in portal.get("connects", [])]:
                return tuple(float(value) for value in entry["pose"])
        raise RuntimeError(f"no portal entry from {source_room} to {target_room}")

    def candidate_summaries(
        self,
        current_room: str,
        room_states: dict[str, Any],
        *,
        include_exhausted: bool = False,
    ) -> list[dict[str, Any]]:
        """Create the room-id JSON given to the VLM, never metric commands."""

        costs = self.room_travel_costs(current_room)
        summaries: list[dict[str, Any]] = []
        for room_id, node in self.nodes.items():
            state = room_states.get(room_id)
            exhausted = bool(getattr(state, "exhausted", False)) if state else False
            if exhausted and not include_exhausted:
                continue
            summaries.append(
                {
                    "room_id": room_id,
                    "travel_cost": round(float(costs[room_id]), 3),
                    "area_m2": round(node.area_m2, 3),
                    "coverage": round(float(getattr(state, "coverage", 0.0)), 4)
                    if state
                    else 0.0,
                    "runtime_description": str(
                        getattr(state, "semantic_description", "")
                    )
                    if state
                    else "",
                    "observed_objects": list(getattr(state, "observed_labels", ()))
                    if state
                    else [],
                    "visited": bool(getattr(state, "visited", False)) if state else False,
                }
            )
        summaries.sort(key=lambda item: (item["travel_cost"], item["room_id"]))
        return summaries
