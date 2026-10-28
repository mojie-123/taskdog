#!/usr/bin/env python
"""Train the predicted-axis canonical FloAff query network."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from custom_envs.scripts.FloAff.models import FloAffNet
from custom_envs.scripts.FloAff.training.axis_dataset import split_rows_by_placement
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
from custom_envs.scripts.FloAff.training.floaff_dataset import (
    FloAffDataset,
    load_floaff_rows,
)
from custom_envs.scripts.FloAff.training.losses import floaff_loss
from custom_envs.scripts.FloAff.training.metrics import affordance_metrics


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "floaff_training.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def make_dataset(root: Path, rows: list[dict], cfg: dict, augment: bool):
    data = cfg["data"]
    augmentation = data.get("augmentation", {})
    return FloAffDataset(
        root,
        rows,
        spatial_scale_m=float(data.get("spatial_scale_m", 2.0)),
        augment=augment,
        jitter_std_m=float(augmentation.get("jitter_std_m", 0.003)),
        dropout_probability=float(augmentation.get("dropout_probability", 0.03)),
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
def evaluate(model, loader, device: torch.device, dice_weight: float):
    model.eval()
    loss_sum, count = 0.0, 0
    predictions, targets, masks, queries = [], [], [], []
    for batch in loader:
        scene = batch["scene_points"].to(device, non_blocking=True)
        query = batch["query_xyz"].to(device, non_blocking=True)
        target = batch["affordance"].to(device, non_blocking=True)
        support = batch["support_mask"].to(device, non_blocking=True)
        logits = model(scene, query)
        loss, parts = floaff_loss(
            logits, target, support, dice_weight=float(dice_weight)
        )
        batch_size = len(scene)
        loss_sum += float(loss.item()) * batch_size
        count += batch_size
        predictions.append(parts["prediction"].cpu().numpy())
        targets.append(target.cpu().numpy())
        masks.append(support.cpu().numpy())
        queries.append(batch["query_xyz_m"].numpy())
    if not predictions:
        raise RuntimeError("cannot evaluate an empty FloAff dataset")
    metrics = affordance_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(masks),
        np.concatenate(queries),
    )
    metrics["loss"] = float(loss_sum / max(count, 1))
    return metrics["loss"], metrics


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    cfg: dict,
    metrics: dict,
    train_rows: list[dict],
    validation_rows: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "stage": "floaff",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
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

    development_rows = load_floaff_rows(
        root, {cfg["data"].get("train_role", "axis_eval_floaff_train")}
    )
    train_rows, validation_rows = split_rows_by_placement(
        development_rows,
        float(cfg["data"].get("validation_fraction", 0.2)),
        seed,
    )
    transfer_rows = load_floaff_rows(
        root, {cfg["data"].get("transfer_role", "transfer_test")}
    )
    write_json(
        run_dir / "data_split.json",
        {
            "train_observations": [row["observation_id"] for row in train_rows],
            "validation_observations": [row["observation_id"] for row in validation_rows],
            "transfer_observations": [row["observation_id"] for row in transfer_rows],
        },
    )

    train_dataset = make_dataset(root, train_rows, cfg, augment=True)
    validation_dataset = make_dataset(root, validation_rows, cfg, augment=False)
    train_loader = make_loader(
        train_dataset, cfg, shuffle=True, seed=seed, drop_last=True
    )
    validation_loader = make_loader(
        validation_dataset, cfg, shuffle=False, seed=seed + 1, drop_last=False
    )
    transfer_loader = None
    if transfer_rows:
        transfer_loader = make_loader(
            make_dataset(root, transfer_rows, cfg, augment=False),
            cfg,
            shuffle=False,
            seed=seed + 2,
            drop_last=False,
        )

    device = resolve_device(cfg.get("device", "cuda:0"))
    model = FloAffNet(**cfg["model"]).to(device)
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
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_loss, best_sim = 0, float("inf"), -float("inf")
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        previous = checkpoint.get("metrics", {})
        best_loss = float(previous.get("best_loss", best_loss))
        best_sim = float(previous.get("best_sim", best_sim))

    dice_weight = float(training.get("dice_weight", 1.0))
    writer = make_summary_writer(run_dir / "tensorboard")
    global_step = start_epoch * max(len(train_loader), 1)
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum, sample_count = 0.0, 0
        for batch in train_loader:
            scene = batch["scene_points"].to(device, non_blocking=True)
            query = batch["query_xyz"].to(device, non_blocking=True)
            target = batch["affordance"].to(device, non_blocking=True)
            support = batch["support_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(scene, query)
                loss, parts = floaff_loss(
                    logits, target, support, dice_weight=dice_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("gradient_clip_norm", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(scene)
            loss_sum += float(loss.item()) * count
            sample_count += count
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            writer.add_scalar("train/mse_step", float(parts["mse"].item()), global_step)
            writer.add_scalar("train/dice_step", float(parts["dice"].item()), global_step)
            global_step += 1

        scheduler.step()
        train_loss = loss_sum / max(sample_count, 1)
        _, validation_metrics = evaluate(
            model, validation_loader, device, dice_weight=dice_weight
        )
        current_loss = float(validation_metrics["loss"])
        current_sim = float(validation_metrics["sim"])
        improved_loss = current_loss < best_loss
        improved_sim = current_sim > best_sim
        best_loss = min(best_loss, current_loss)
        best_sim = max(best_sim, current_sim)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
            "best_loss": best_loss,
            "best_sim": best_sim,
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
            scaler,
            epoch,
            cfg,
            epoch_metrics,
            train_rows,
            validation_rows,
        )
        save_checkpoint(run_dir / "last.pt", payload)
        if improved_loss:
            save_checkpoint(run_dir / "best_loss.pt", payload)
        if improved_sim:
            save_checkpoint(run_dir / "best_sim.pt", payload)
        print(
            f"[FloAff] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={current_loss:.6f} sim={current_sim:.4f}",
            flush=True,
        )

    writer.close()
    if transfer_loader is not None:
        checkpoint = load_checkpoint(run_dir / "best_loss.pt", device)
        model.load_state_dict(checkpoint["model_state"])
        _, transfer_metrics = evaluate(
            model, transfer_loader, device, dice_weight=dice_weight
        )
        write_json(run_dir / "transfer_metrics.json", transfer_metrics)
        print(f"[FloAff] transfer evaluation={transfer_metrics}")
    write_json(Path(cfg["run_root"]).resolve() / "latest.json", {"run_dir": str(run_dir)})
    print(f"[FloAff] complete run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
