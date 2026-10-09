#!/usr/bin/env python3
"""Visualize one observation cloud and all paired manipulation poses."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2m-matplotlib")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from custom_envs.scripts.N2M.manifests import read_jsonl


def _footprint(pose, length=0.75, width=0.40):
    corners = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2],
         [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return corners @ rotation.T + np.asarray(pose[:2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    observations = {
        row["observation_id"]: row for row in read_jsonl(root / "observations.jsonl")
    }
    observation = observations.get(args.observation_id)
    if observation is None:
        raise KeyError(f"unknown observation id: {args.observation_id}")
    pairs_path = root / "pairs.jsonl"
    pairs = [
        row for row in read_jsonl(pairs_path)
        if row["observation_id"] == args.observation_id
    ]
    with np.load(root / observation["cloud_file"]) as cloud:
        xyz = cloud["xyz"]
        rgb = cloud["rgb"].astype(np.float32) / 255.0
    if len(xyz) > args.max_points:
        indices = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=1, rasterized=True)
    axes[0].add_patch(Polygon(_footprint([0, 0, 0]), fill=False, color="cyan", lw=2))
    axes[0].arrow(0, 0, 0.35, 0, color="cyan", width=0.01, length_includes_head=True)
    cube = observation.get("cube_center_base")
    if cube:
        axes[0].scatter([cube[0]], [cube[1]], c="yellow", edgecolors="black", s=80, marker="s")
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(pairs))))
    for color, pair in zip(colors, pairs):
        target = pair["target_pose_observation"]
        axes[0].add_patch(
            Polygon(_footprint(target), fill=False, edgecolor=color, alpha=0.65, lw=1)
        )
        axes[0].arrow(
            target[0], target[1], 0.22 * math.cos(target[2]), 0.22 * math.sin(target[2]),
            color=color, width=0.006, length_includes_head=True,
        )
    axes[0].set_title(f"base-frame top view: {len(pairs)} manipulation poses")
    axes[0].set_xlabel("base +X forward [m]")
    axes[0].set_ylabel("base +Y left [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=1, rasterized=True)
    if cube:
        axes[1].scatter([cube[0]], [cube[2]], c="yellow", edgecolors="black", s=80, marker="s")
    axes[1].set_title("base-frame side view")
    axes[1].set_xlabel("base +X forward [m]")
    axes[1].set_ylabel("base +Z up [m]")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.2)

    output = args.output or root / "previews" / f"{args.observation_id}_pointcloud.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"[N2M] point-cloud preview saved: {output}")
    if args.show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
