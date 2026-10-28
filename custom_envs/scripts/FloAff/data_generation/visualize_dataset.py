#!/usr/bin/env python
"""Visualize target, table, O camera positions and M success labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/taskdog-floaff-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from custom_envs.scripts.FloAff.config import load_data_generation_config
from custom_envs.scripts.FloAff.data_generation.common import (
    ensure_dataset_config_compatible,
    output_root,
)
from custom_envs.scripts.FloAff.manifests import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--save", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_data_generation_config(args.config)
    root = output_root(config, args.output_dir)
    ensure_dataset_config_compatible(root, config)
    instances = {row["instance_id"]: row for row in read_jsonl(root / "instances.jsonl")}
    instance = instances[args.instance]
    observations = [
        row for row in read_jsonl(root / "observations" / "manifest.jsonl")
        if row["instance_id"] == args.instance
    ]
    summaries = [
        row for row in read_jsonl(root / "manipulation" / "summaries.jsonl")
        if row["instance_id"] == args.instance
    ]
    candidates = {
        row["candidate_id"]: row
        for row in read_jsonl(root / "manipulation" / "candidates.jsonl")
        if row["instance_id"] == args.instance
    }
    target = np.asarray(instance["target_pose_world"][:2], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    table = plt.Rectangle(
        (config.table.x_min, config.table.y_min),
        config.table.x_max - config.table.x_min,
        config.table.y_max - config.table.y_min,
        facecolor="#d6b589",
        edgecolor="black",
        alpha=0.45,
        label="coffee table",
    )
    axis.add_patch(table)
    axis.scatter(*target, marker="*", s=220, color="red", edgecolor="black", label="target")
    if observations:
        camera_xy = np.asarray(
            [row["camera_pose_world"]["position"][:2] for row in observations]
        )
        strata = [row["stratum"] for row in observations]
        colors = {"core": "tab:blue", "negative_tail": "tab:orange", "positive_tail": "tab:green"}
        for stratum, color in colors.items():
            mask = np.array([item == stratum for item in strata])
            if np.any(mask):
                axis.scatter(camera_xy[mask, 0], camera_xy[mask, 1], s=25, color=color, label=f"O {stratum}")
    if summaries:
        rows_with_pose = [
            (
                row.get("base_pose", candidates.get(row["candidate_id"], {}).get("base_pose")),
                row["success_rate"],
            )
            for row in summaries
        ]
        rows_with_pose = [item for item in rows_with_pose if item[0] is not None]
        base_xy = np.asarray([item[0][:2] for item in rows_with_pose])
        rates = np.asarray([item[1] for item in rows_with_pose])
        scatter = axis.scatter(
            base_xy[:, 0], base_xy[:, 1], c=rates, cmap="RdYlGn", vmin=0, vmax=1,
            marker="s", s=30, edgecolors="none", label="M",
        )
        fig.colorbar(scatter, ax=axis, label="M success rate")
    axis.set_aspect("equal")
    axis.set_xlabel("world X [m]")
    axis.set_ylabel("world Y [m]")
    axis.set_title(args.instance)
    axis.legend(loc="best", fontsize=8)
    save = args.save or (root / "previews" / f"{args.instance}.png")
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=160)
    plt.close(fig)
    print(f"[FloAff] preview: {save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
