#!/usr/bin/env python
"""Train AnchorAxisNet without using the held-out FloAff placements."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from custom_envs.scripts.FloAff.models import AnchorAxisNet
from custom_envs.scripts.FloAff.training.axis_dataset import (
    AnchorAxisDataset,
    load_axis_rows,
    split_rows_by_placement,
)
from custom_envs.scripts.FloAff.training.common import (
    append_jsonl,
    dataloader_generator,
    load_checkpoint,
    load_yaml,
    make_run_dir,
    make_summary_writer,
    optimizer_to,
    resolve_device,
    resolve_path,
    save_checkpoint,
    seed_everything,
    seed_worker,
    write_json,
)
from custom_envs.scripts.FloAff.training.losses import directed_axis_loss
from custom_envs.scripts.FloAff.training.metrics import axis_metrics


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "axis_training.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def make_dataset(root: Path, rows: list[dict], cfg: dict, augment: bool):
    augmentation = cfg["data"].get("augmentation", {})
    return AnchorAxisDataset(
        root,
        rows,
        augment=augment,
        yaw_augmentation_deg=augmentation.get("yaw_deg", 180.0),
        jitter_std_m=augmentation.get("jitter_std_m", 0.003),
        dropout_probability=augmentation.get("dropout_probability", 0.05),
    )


def make_loader(dataset, cfg: dict, *, shuffle: bool, seed: int, drop_last: bool):
    training = cfg["training"]
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 4)),
        pin_memory=bool(training.get("pin_memory", True)),
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(seed),
        persistent_workers=int(training.get("num_workers", 4)) > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses, predictions, targets = [], [], []
    for batch in loader:
        points = batch["anchor_xyz"].to(device, non_blocking=True)
        target = batch["axis_gt"].to(device, non_blocking=True)
        prediction = model(points)
        loss = directed_axis_loss(prediction, target)
        losses.append(float(loss.item()) * len(points))
        predictions.append(prediction.cpu().numpy())
        targets.append(target.cpu().numpy())
    if not predictions:
        raise RuntimeError("cannot evaluate an empty AnchorAxisNet dataset")
    metrics = axis_metrics(np.concatenate(predictions), np.concatenate(targets))
    metrics["loss"] = float(sum(losses) / len(loader.dataset))
    return metrics["loss"], metrics


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch: int,
    cfg: dict,
    metrics: dict,
    train_rows: list[dict],
    validation_rows: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "stage": "anchor_axis",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": cfg["model"],
        "training_config": cfg,
        "metrics": metrics,
        "train_placements": sorted({row["placement_id"] for row in train_rows}),
        "validation_placements": sorted(
            {row["placement_id"] for row in validation_rows}
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    root = resolve_path(cfg["dataset_root"])
    run_dir = make_run_dir(cfg["run_root"], args.run_dir)
    write_json(run_dir / "resolved_config.json", cfg)

    axis_train_rows = load_axis_rows(root, {cfg["data"].get("train_role", "axis_train")})
    train_rows, validation_rows = split_rows_by_placement(
        axis_train_rows,
        float(cfg["data"].get("validation_fraction", 0.2)),
        seed,
    )
    evaluation_rows = load_axis_rows(
        root, {cfg["data"].get("evaluation_role", "axis_eval_floaff_train")}
    )
    write_json(
        run_dir / "data_split.json",
        {
            "train_observations": [row["observation_id"] for row in train_rows],
            "validation_observations": [row["observation_id"] for row in validation_rows],
            "evaluation_observations": [row["observation_id"] for row in evaluation_rows],
        },
    )

    train_dataset = make_dataset(root, train_rows, cfg, augment=True)
    validation_dataset = make_dataset(root, validation_rows, cfg, augment=False)
    evaluation_dataset = make_dataset(root, evaluation_rows, cfg, augment=False)
    train_loader = make_loader(
        train_dataset, cfg, shuffle=True, seed=seed, drop_last=True
    )
    validation_loader = make_loader(
        validation_dataset, cfg, shuffle=False, seed=seed + 1, drop_last=False
    )
    evaluation_loader = make_loader(
        evaluation_dataset, cfg, shuffle=False, seed=seed + 2, drop_last=False
    )

    device = resolve_device(cfg.get("device", "cuda:0"))
    model = AnchorAxisNet(**cfg["model"]).to(device)
    training = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(training.get("minimum_learning_rate", 1e-6))
    )
    start_epoch, best_error = 0, float("inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_error = float(checkpoint.get("metrics", {}).get("best_error_deg", best_error))

    writer = make_summary_writer(run_dir / "tensorboard")
    global_step = start_epoch * max(len(train_loader), 1)
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum, sample_count = 0.0, 0
        for batch in train_loader:
            points = batch["anchor_xyz"].to(device, non_blocking=True)
            target = batch["axis_gt"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(points)
            loss = directed_axis_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip_norm", 5.0))
            )
            optimizer.step()
            count = len(points)
            train_loss_sum += float(loss.item()) * count
            sample_count += count
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            global_step += 1

        scheduler.step()
        train_loss = train_loss_sum / max(sample_count, 1)
        _, validation_metrics = evaluate(model, validation_loader, device)
        current_error = float(validation_metrics["angular_mean_deg"])
        best_error = min(best_error, current_error)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
            "best_error_deg": best_error,
        }
        append_jsonl(run_dir / "history.jsonl", epoch_metrics)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        for key, value in validation_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"validation/{key}", value, epoch)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            cfg,
            epoch_metrics,
            train_rows,
            validation_rows,
        )
        save_checkpoint(run_dir / "last.pt", payload)
        if current_error <= best_error:
            save_checkpoint(run_dir / "best.pt", payload)
        print(
            f"[AnchorAxisNet] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_error={current_error:.3f}deg best={best_error:.3f}deg",
            flush=True,
        )

    writer.close()
    best = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(best["model_state"])
    _, evaluation_metrics = evaluate(model, evaluation_loader, device)
    write_json(run_dir / "axis_eval_metrics.json", evaluation_metrics)
    write_json(Path(cfg["run_root"]).resolve() / "latest.json", {"run_dir": str(run_dir)})
    print(f"[AnchorAxisNet] complete run_dir={run_dir}")
    print(f"[AnchorAxisNet] held-out evaluation={evaluation_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
