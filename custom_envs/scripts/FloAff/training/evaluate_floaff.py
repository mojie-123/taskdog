#!/usr/bin/env python
"""Evaluate and optionally visualize a trained FloAff query network."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_envs.scripts.FloAff.models import FloAffNet
from custom_envs.scripts.FloAff.training.axis_dataset import split_rows_by_placement
from custom_envs.scripts.FloAff.training.common import (
    dataloader_generator,
    load_checkpoint,
    load_yaml,
    resolve_device,
    resolve_path,
    seed_worker,
    write_json,
)
from custom_envs.scripts.FloAff.training.floaff_dataset import FloAffDataset, load_floaff_rows
from custom_envs.scripts.FloAff.training.train_floaff import DEFAULT_CONFIG, evaluate


def select_rows(root: Path, cfg: dict, split: str) -> list[dict]:
    development = load_floaff_rows(
        root, {cfg["data"].get("train_role", "axis_eval_floaff_train")}
    )
    train_rows, validation_rows = split_rows_by_placement(
        development,
        float(cfg["data"].get("validation_fraction", 0.2)),
        int(cfg.get("seed", 42)),
    )
    if split == "train":
        return train_rows
    if split == "validation":
        return validation_rows
    if split == "transfer":
        return load_floaff_rows(
            root, {cfg["data"].get("transfer_role", "transfer_test")}
        )
    if split == "all":
        return load_floaff_rows(root)
    raise ValueError(split)


@torch.no_grad()
def save_heatmaps(model, loader, device: torch.device, directory: Path, limit: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    for batch in loader:
        prediction = model.predict(
            batch["scene_points"].to(device), batch["query_xyz"].to(device)
        ).cpu().numpy()
        target = batch["affordance"].numpy()
        support = batch["support_mask"].numpy()
        query = batch["query_xyz_m"].numpy()
        for index, observation_id in enumerate(batch["observation_id"]):
            if saved >= limit:
                return
            x_values = np.unique(query[index, :, 0])
            y_values = np.unique(query[index, :, 1])
            shape = (len(x_values), len(y_values))
            pred_map = prediction[index].reshape(shape)
            gt_map = target[index].reshape(shape)
            mask_map = support[index].reshape(shape)
            pred_map = np.where(mask_map, pred_map, np.nan)
            gt_map = np.where(mask_map, gt_map, np.nan)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            extent = [y_values.min(), y_values.max(), x_values.min(), x_values.max()]
            for axis, values, title in zip(axes, [gt_map, pred_map], ["GT", "Prediction"]):
                image = axis.imshow(
                    values,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    vmin=0.0,
                    vmax=1.0,
                    cmap="RdYlGn",
                )
                axis.set_title(title)
                axis.set_xlabel("canonical tangent y (m)")
                axis.set_ylabel("canonical outward x (m)")
                fig.colorbar(image, ax=axis)
            fig.suptitle(observation_id)
            fig.savefig(directory / f"{observation_id}.png", dpi=140)
            plt.close(fig)
            saved += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "transfer", "all"], default="validation"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--heatmap-limit", type=int, default=20)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["dataset_root"])
    rows = select_rows(root, cfg, args.split)
    if not rows:
        raise RuntimeError(f"no FloAff samples found for split {args.split}")
    dataset = FloAffDataset(
        root,
        rows,
        spatial_scale_m=float(cfg["data"].get("spatial_scale_m", 2.0)),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        worker_init_fn=seed_worker,
        generator=dataloader_generator(int(cfg.get("seed", 42))),
    )
    device = resolve_device(cfg.get("device", "cuda:0"))
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = FloAffNet(**checkpoint.get("model_config", cfg["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, metrics = evaluate(
        model,
        loader,
        device,
        dice_weight=float(cfg["training"].get("dice_weight", 1.0)),
    )
    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    write_json(output, {"split": args.split, "metrics": metrics})
    if args.save_heatmaps:
        save_heatmaps(
            model,
            loader,
            device,
            args.checkpoint.parent / f"{args.split}_heatmaps",
            args.heatmap_limit,
        )
    print(f"[FloAff] split={args.split} metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
