"""Dataset and placement-safe splits for AnchorAxisNet."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import read_jsonl


def load_axis_rows(dataset_root: str | Path, roles: set[str] | None = None) -> list[dict]:
    root = Path(dataset_root)
    rows = read_jsonl(root / "processed" / "axis" / "manifest.jsonl")
    if roles is not None:
        rows = [row for row in rows if row["split_role"] in roles]
    return rows


def split_rows_by_placement(
    rows: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Stratify by category while keeping all sibling instances together."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    category_placements: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        category_placements[row["category"]].add(row["placement_id"])
    validation: set[str] = set()
    rng = np.random.default_rng(int(seed))
    for placements in category_placements.values():
        values = np.asarray(sorted(placements), dtype=object)
        rng.shuffle(values)
        count = int(round(len(values) * float(validation_fraction)))
        if len(values) >= 2:
            count = min(max(count, 1), len(values) - 1)
        validation.update(str(value) for value in values[:count])
    train_rows = [row for row in rows if row["placement_id"] not in validation]
    validation_rows = [row for row in rows if row["placement_id"] in validation]
    if not train_rows or not validation_rows:
        raise RuntimeError("placement split produced an empty train or validation set")
    return train_rows, validation_rows


def rotate_xy(values: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=values.dtype, device=values.device)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    result = values.clone()
    x, y = values[..., 0], values[..., 1]
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


class AnchorAxisDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        rows: list[dict],
        *,
        augment: bool = False,
        yaw_augmentation_deg: float = 180.0,
        jitter_std_m: float = 0.003,
        dropout_probability: float = 0.05,
    ):
        self.root = Path(dataset_root).resolve()
        self.rows = list(rows)
        self.augment = bool(augment)
        self.yaw_augmentation_rad = np.deg2rad(float(yaw_augmentation_deg))
        self.jitter_std_m = float(jitter_std_m)
        self.dropout_probability = float(dropout_probability)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        sample_path = self.root / "processed" / "axis" / row["sample_file"]
        with np.load(sample_path) as sample:
            points = torch.from_numpy(
                sample["anchor_xyz_centered_base"].astype(np.float32, copy=True)
            )
            axis = torch.from_numpy(
                sample["axis_outward_gt_base"].astype(np.float32, copy=True)
            )
        axis = axis / axis.norm().clamp_min(1e-8)

        if self.augment:
            angle = (
                2.0 * torch.rand((), dtype=points.dtype) - 1.0
            ) * self.yaw_augmentation_rad
            points = rotate_xy(points, angle)
            axis = rotate_xy(axis, angle)
            if self.jitter_std_m > 0.0:
                points = points + torch.randn_like(points) * self.jitter_std_m
            if self.dropout_probability > 0.0:
                dropped = torch.rand(len(points)) < self.dropout_probability
                if dropped.any():
                    replacements = torch.randint(0, len(points), (int(dropped.sum()),))
                    points[dropped] = points[replacements]

        return {
            "anchor_xyz": points,
            "axis_gt": axis,
            "observation_id": row["observation_id"],
            "placement_id": row["placement_id"],
            "instance_id": row["instance_id"],
            "category": row["category"],
            "split_role": row["split_role"],
        }
