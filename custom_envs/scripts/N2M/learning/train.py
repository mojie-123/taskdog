#!/usr/bin/env python3
"""Train the locally vendored N2M model on an assembled dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from custom_envs.scripts.N2M.learning.dataset import N2MPairDataset
from custom_envs.scripts.N2M.learning.gmm_loss import GMMPoseLoss, wrapped_pose_difference
from custom_envs.scripts.N2M.manifests import write_json
from custom_envs.scripts.N2M.config import PROJECT_ROOT
from custom_envs.scripts.N2M.vendor_n2m import N2MNet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(means, target) -> tuple[torch.Tensor, torch.Tensor]:
    difference = wrapped_pose_difference(target, means)
    xy = torch.linalg.vector_norm(difference[..., :2], dim=-1)
    closest = xy.argmin(dim=1)
    batch = torch.arange(len(target), device=target.device)
    return xy[batch, closest].mean(), difference[batch, closest, 2].abs().mean()


def _run_epoch(model, loader, loss_function, device, optimizer=None, scaler=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "nll": 0.0, "xy_m": 0.0, "yaw_rad": 0.0}
    count = 0
    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            means, covariance, weights = model(points)
            loss, parts = loss_function(means, covariance, weights, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            xy, yaw = _metrics(means, target)
        batch_size = len(target)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["nll"] += float(parts["nll"]) * batch_size
        totals["xy_m"] += float(xy) * batch_size
        totals["yaw_rad"] += float(yaw) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true", help="run only two epochs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.dataset:
        config["dataset"]["path"] = str(args.dataset)
    if args.output:
        config["training"]["output"] = str(args.output)
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.smoke:
        config["training"]["epochs"] = 2
        config["training"]["early_stopping_patience"] = 2
    encoder_checkpoint = config["model"]["encoder"].get("ckpt")
    if encoder_checkpoint:
        encoder_checkpoint = Path(encoder_checkpoint)
        if not encoder_checkpoint.is_absolute():
            encoder_checkpoint = PROJECT_ROOT / encoder_checkpoint
        config["model"]["encoder"]["ckpt"] = str(encoder_checkpoint.resolve())
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset_config = config["dataset"]
    dataset_dir = Path(dataset_config["path"]).resolve()
    common_dataset = dict(
        dataset_dir=dataset_dir,
        num_points=dataset_config.get("num_points", 8192),
        seed=seed,
        cache_size=dataset_config.get("cache_size", 64),
    )
    train_dataset = N2MPairDataset(
        split="train",
        training=True,
        rotation_degrees=dataset_config.get("rotation_degrees", 0.0),
        translation_radius_m=dataset_config.get("translation_radius_m", 0.0),
        **common_dataset,
    )
    validation_dataset = N2MPairDataset(split="val", training=False, **common_dataset)
    training_config = config["training"]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        train_dataset.balanced_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )
    loader_common = dict(
        batch_size=int(training_config.get("batch_size", 8)),
        num_workers=int(training_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_common)

    model_config = copy.deepcopy(config["model"])
    if args.resume:
        # A complete N2M checkpoint overwrites the encoder; do not redundantly
        # reload the Point-BERT initialization before restoring it.
        model_config["encoder"]["ckpt"] = None
    model = N2MNet(model_config).to(device)
    loss_function = GMMPoseLoss(**config.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    epochs = int(training_config.get("epochs", 150))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(training_config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output = Path(training_config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["dataset"]["path"] = str(dataset_dir)
    config["training"]["output"] = str(output)
    write_json(output / "resolved_config.json", config)
    metadata = {
        "pairs_sha256": _sha256(dataset_dir / "pairs.jsonl"),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    write_json(output / "run_metadata.json", metadata)

    start_epoch, best_validation, stale_epochs = 0, float("inf"), 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])

    metrics_path = output / "metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 0
    with metrics_path.open("a" if not write_header else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch", "lr", "train_loss", "train_nll", "train_xy_m",
                "train_yaw_deg", "val_loss", "val_nll", "val_xy_m", "val_yaw_deg",
            ],
        )
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, epochs):
            started = time.time()
            train_metrics = _run_epoch(
                model,
                train_loader,
                loss_function,
                device,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip=float(training_config.get("grad_clip", 1.0)),
            )
            with torch.no_grad():
                validation_metrics = _run_epoch(
                    model, validation_loader, loss_function, device, scaler=scaler
                )
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_nll": train_metrics["nll"],
                "train_xy_m": train_metrics["xy_m"],
                "train_yaw_deg": np.degrees(train_metrics["yaw_rad"]),
                "val_loss": validation_metrics["loss"],
                "val_nll": validation_metrics["nll"],
                "val_xy_m": validation_metrics["xy_m"],
                "val_yaw_deg": np.degrees(validation_metrics["yaw_rad"]),
            }
            writer.writerow(row)
            stream.flush()
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation": best_validation,
                "config": config,
                "metadata": metadata,
            }
            torch.save(state, output / "last.pt")
            if validation_metrics["nll"] < best_validation:
                best_validation = validation_metrics["nll"]
                state["best_validation"] = best_validation
                torch.save(state, output / "best.pt")
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"[N2M][TRAIN] epoch={epoch:03d} train_nll={train_metrics['nll']:.4f} "
                f"val_nll={validation_metrics['nll']:.4f} val_xy={validation_metrics['xy_m']:.3f}m "
                f"val_yaw={np.degrees(validation_metrics['yaw_rad']):.2f}deg "
                f"time={time.time()-started:.1f}s"
            )
            if stale_epochs >= int(training_config.get("early_stopping_patience", 20)):
                print(f"[N2M] early stopping after {stale_epochs} stale epochs")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
