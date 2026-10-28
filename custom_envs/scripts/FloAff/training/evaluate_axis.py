#!/usr/bin/env python
"""Evaluate an AnchorAxisNet checkpoint on one or more manifest roles."""

from __future__ import annotations

import argparse
from pathlib import Path

from custom_envs.scripts.FloAff.models import AnchorAxisNet
from custom_envs.scripts.FloAff.training.axis_dataset import AnchorAxisDataset, load_axis_rows
from custom_envs.scripts.FloAff.training.common import (
    dataloader_generator,
    load_checkpoint,
    load_yaml,
    resolve_device,
    resolve_path,
    seed_worker,
    write_json,
)
from custom_envs.scripts.FloAff.training.train_axis import DEFAULT_CONFIG, evaluate
from torch.utils.data import DataLoader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    root = resolve_path(cfg["dataset_root"])
    roles = set(args.role or [cfg["data"].get("evaluation_role", "axis_eval_floaff_train")])
    rows = load_axis_rows(root, roles)
    if not rows:
        raise RuntimeError(f"no AnchorAxisNet samples found for roles {sorted(roles)}")
    dataset = AnchorAxisDataset(root, rows, augment=False)
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
    model = AnchorAxisNet(**checkpoint.get("model_config", cfg["model"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, metrics = evaluate(model, loader, device)
    output = args.output or args.checkpoint.parent / "evaluation_metrics.json"
    write_json(output, {"roles": sorted(roles), "metrics": metrics})
    print(f"[AnchorAxisNet] {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
