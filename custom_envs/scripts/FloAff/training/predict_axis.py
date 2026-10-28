#!/usr/bin/env python
"""Generate frozen AnchorAxisNet predictions consumed by FloAff construction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_envs.scripts.FloAff.models import AnchorAxisNet
from custom_envs.scripts.FloAff.training.axis_dataset import (
    AnchorAxisDataset,
    load_axis_rows,
    rotate_xy,
)
from custom_envs.scripts.FloAff.training.common import (
    dataloader_generator,
    load_checkpoint,
    load_yaml,
    resolve_device,
    resolve_path,
    seed_worker,
    write_json,
    write_jsonl,
)
from custom_envs.scripts.FloAff.training.metrics import axis_metrics
from custom_envs.scripts.FloAff.training.train_axis import DEFAULT_CONFIG


@torch.no_grad()
def tta_predict(model, points: torch.Tensor, angles_deg: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    restored = []
    for angle_deg in angles_deg:
        angle = math.radians(float(angle_deg))
        rotated = rotate_xy(points, angle)
        prediction = model(rotated)
        restored.append(rotate_xy(prediction, -angle))
    stacked = torch.stack(restored, dim=0)
    mean = stacked.mean(dim=0)
    confidence = mean.norm(dim=-1).clamp(0.0, 1.0)
    direction = torch.nn.functional.normalize(mean, dim=-1, eps=1e-8)
    return direction, confidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["dataset_root"])
    roles = set(args.role or cfg["inference"].get(
        "roles", ["axis_eval_floaff_train", "transfer_test"]
    ))
    rows = load_axis_rows(root, roles)
    if not rows:
        raise RuntimeError(f"no AnchorAxisNet samples found for roles {sorted(roles)}")
    dataset = AnchorAxisDataset(root, rows, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["inference"].get("batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        worker_init_fn=seed_worker,
        generator=dataloader_generator(int(cfg.get("seed", 42))),
    )
    device = resolve_device(cfg.get("device", "cuda:0"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = AnchorAxisNet(**checkpoint.get("model_config", cfg["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    angles = [float(value) for value in cfg["inference"].get("tta_angles_deg", [0.0])]

    output_rows, all_prediction, all_target = [], [], []
    for batch in loader:
        points = batch["anchor_xyz"].to(device, non_blocking=True)
        prediction, confidence = tta_predict(model, points, angles)
        target = batch["axis_gt"]
        prediction_cpu = prediction.cpu().numpy()
        confidence_cpu = confidence.cpu().numpy()
        all_prediction.append(prediction_cpu)
        all_target.append(target.numpy())
        for index, observation_id in enumerate(batch["observation_id"]):
            output_rows.append(
                {
                    "schema_version": 1,
                    "observation_id": observation_id,
                    "placement_id": batch["placement_id"][index],
                    "instance_id": batch["instance_id"][index],
                    "split_role": batch["split_role"][index],
                    "axis_xy_base": prediction_cpu[index].tolist(),
                    "confidence": float(confidence_cpu[index]),
                }
            )
    output_rows.sort(key=lambda row: row["observation_id"])
    output = args.output or root / "axis_predictions" / "anchor_axis_predictions.jsonl"
    write_jsonl(output, output_rows)
    metrics = axis_metrics(np.concatenate(all_prediction), np.concatenate(all_target))
    write_json(Path(output).with_suffix(".metrics.json"), metrics)
    print(f"[AnchorAxisNet] wrote {len(output_rows)} predictions to {output}")
    print(f"[AnchorAxisNet] prediction metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
