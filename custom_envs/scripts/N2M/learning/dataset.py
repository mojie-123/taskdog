"""Leakage-safe point-cloud dataset backed by ``pairs.jsonl``."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from custom_envs.scripts.N2M.manifests import read_jsonl


def wrap_angle_array(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def augment_se2(
    xyz: np.ndarray,
    target: np.ndarray,
    rotation_rad: float,
    translation_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one rigid XY augmentation to both cloud and pose label."""
    cosine, sine = math.cos(rotation_rad), math.sin(rotation_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    result_xyz = np.asarray(xyz, dtype=np.float32).copy()
    result_target = np.asarray(target, dtype=np.float32).copy()
    result_xyz[:, :2] = result_xyz[:, :2] @ rotation.T + translation_xy
    result_target[:2] = result_target[:2] @ rotation.T + translation_xy
    result_target[2] = wrap_angle_array(result_target[2] + rotation_rad)
    return result_xyz, result_target


class N2MPairDataset:
    """PyTorch-compatible dataset without importing torch at module import time."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        num_points: int = 8192,
        training: bool = False,
        rotation_degrees: float = 0.0,
        translation_radius_m: float = 0.0,
        seed: int = 42,
        cache_size: int = 64,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        all_pairs = read_jsonl(self.dataset_dir / "pairs.jsonl")
        self.rows = [row for row in all_pairs if row.get("split") == split]
        if not self.rows:
            raise RuntimeError(f"dataset has no pairs for split={split!r}")
        self.split = split
        self.num_points = int(num_points)
        self.training = bool(training)
        self.rotation_radians = math.radians(float(rotation_degrees))
        self.translation_radius_m = float(translation_radius_m)
        self.seed = int(seed)
        self.cache_size = max(0, int(cache_size))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_order: list[str] = []

    def __len__(self) -> int:
        return len(self.rows)

    def _load_cloud(self, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        if relative_path in self._cache:
            return self._cache[relative_path]
        with np.load(self.dataset_dir / relative_path) as cloud:
            xyz = cloud["xyz"].astype(np.float32)
            rgb = cloud["rgb"].astype(np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
            raise ValueError(f"invalid point cloud shapes for {relative_path}")
        if rgb.size and float(rgb.max()) > 1.0:
            rgb /= 255.0
        if self.cache_size:
            self._cache[relative_path] = (xyz, rgb)
            self._cache_order.append(relative_path)
            if len(self._cache_order) > self.cache_size:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
        return xyz, rgb

    def _rng(self, index: int) -> np.random.Generator:
        if self.training:
            return np.random.default_rng()
        pair_id = self.rows[index]["pair_id"]
        digest = hashlib.sha256(f"{self.seed}:{pair_id}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        xyz, rgb = self._load_cloud(row["cloud_file"])
        if len(xyz) == 0:
            raise ValueError(f"empty point cloud: {row['cloud_file']}")
        rng = self._rng(index)
        replace = len(xyz) < self.num_points
        indices = rng.choice(len(xyz), self.num_points, replace=replace)
        xyz = xyz[indices].copy()
        rgb = rgb[indices].copy()
        target = np.asarray(row["target_pose_observation"], dtype=np.float32)

        if self.training and (self.rotation_radians > 0.0 or self.translation_radius_m > 0.0):
            rotation = rng.uniform(-self.rotation_radians, self.rotation_radians)
            radius = self.translation_radius_m * math.sqrt(rng.uniform())
            bearing = rng.uniform(-math.pi, math.pi)
            translation = np.array(
                [radius * math.cos(bearing), radius * math.sin(bearing)], dtype=np.float32
            )
            xyz, target = augment_se2(xyz, target, rotation, translation)

        points = np.concatenate((xyz, rgb), axis=1).astype(np.float32)
        return {
            "points": points,
            "target": target,
            "pair_id": row["pair_id"],
            "observation_id": row["observation_id"],
            "scene_id": row["scene_id"],
            "approach_edge": row["approach_edge"],
        }

    def balanced_sample_weights(self) -> list[float]:
        groups = [(row["scene_id"], row["approach_edge"]) for row in self.rows]
        counts = Counter(groups)
        return [1.0 / counts[group] for group in groups]

