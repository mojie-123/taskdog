"""Dataset adapter for predicted-axis canonical FloAff samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .axis_dataset import split_rows_by_placement
from .common import read_jsonl


def load_floaff_rows(dataset_root: str | Path, roles: set[str] | None = None) -> list[dict]:
    root = Path(dataset_root)
    rows = read_jsonl(root / "processed" / "floaff" / "manifest.jsonl")
    if roles is not None:
        rows = [row for row in rows if row["split_role"] in roles]
    return rows


class FloAffDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        rows: list[dict],
        *,
        spatial_scale_m: float = 2.0,
        augment: bool = False,
        jitter_std_m: float = 0.003,
        dropout_probability: float = 0.03,
    ):
        self.root = Path(dataset_root).resolve()
        self.rows = list(rows)
        self.spatial_scale_m = float(spatial_scale_m)
        if self.spatial_scale_m <= 0.0:
            raise ValueError("spatial_scale_m must be positive")
        self.augment = bool(augment)
        self.jitter_std_m = float(jitter_std_m)
        self.dropout_probability = float(dropout_probability)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self.root / "processed" / "floaff" / row["sample_file"]
        with np.load(path) as sample:
            scene = torch.from_numpy(sample["scene_points"].astype(np.float32, copy=True))
            query_m = torch.from_numpy(sample["query_xyz"].astype(np.float32, copy=True))
            labels = torch.from_numpy(sample["affordance"].astype(np.float32, copy=True))
            support = torch.from_numpy(
                sample["label_support_mask"].astype(bool, copy=True)
            )

        if self.augment:
            if self.jitter_std_m > 0.0:
                scene[:, :3] += torch.randn_like(scene[:, :3]) * self.jitter_std_m
            if self.dropout_probability > 0.0:
                dropped = torch.rand(len(scene)) < self.dropout_probability
                if dropped.any():
                    replacements = torch.randint(0, len(scene), (int(dropped.sum()),))
                    scene[dropped] = scene[replacements]

        scene[:, :3] /= self.spatial_scale_m
        query = query_m / self.spatial_scale_m
        return {
            "scene_points": scene,
            "query_xyz": query,
            "query_xyz_m": query_m,
            "affordance": labels,
            "support_mask": support,
            "observation_id": row["observation_id"],
            "placement_id": row["placement_id"],
            "instance_id": row["instance_id"],
            "category": row["category"],
            "split_role": row["split_role"],
        }


__all__ = ["FloAffDataset", "load_floaff_rows", "split_rows_by_placement"]
