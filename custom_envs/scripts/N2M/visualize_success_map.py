#!/usr/bin/env python
"""Render a top-down candidate/success/failure map from collector JSONL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/taskdog_n2m_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/taskdog_n2m_cache")
# This module is also imported at the end of unattended/headless collection.
# Force a non-interactive backend before pyplot is imported; otherwise
# Matplotlib may select Qt through OpenCV's bundled xcb plugin and abort when
# there is no display server.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from custom_envs.scripts.N2M.config import CUBE_SCENES, DEFAULT_DATA_DIR
from custom_envs.scripts.N2M.manifests import read_jsonl


def build_plot(scene_id: str, data_dir: Path, output: Path | None = None) -> Path:
    candidates = [
        row for row in read_jsonl(data_dir / "candidates.jsonl")
        if row.get("scene_id") == scene_id
    ]
    rollouts = [
        row for row in read_jsonl(data_dir / "manipulation_rollouts.jsonl")
        if row.get("scene_id") == scene_id
    ]
    summaries = [
        row for row in read_jsonl(data_dir / "candidate_summaries.jsonl")
        if row.get("scene_id") == scene_id
    ]
    latest_rollout = {}
    for row in rollouts:
        latest_rollout[row["candidate_id"]] = row
    latest_summary = {}
    for row in summaries:
        latest_summary[row["candidate_id"]] = row

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.add_patch(Rectangle((5.0, 3.2), 5.0, 0.9, facecolor="#8b5a2b", alpha=0.35,
                           edgecolor="black", linewidth=2.0))
    cube = np.asarray(CUBE_SCENES[scene_id].cube_xyz)
    ax.scatter([cube[0]], [cube[1]], s=140, c="gold", edgecolors="black", marker="s", zorder=5)

    legend_done = set()
    for candidate in candidates:
        pose = np.asarray(candidate["base_pose"], dtype=float)
        summary = latest_summary.get(candidate["candidate_id"])
        result = latest_rollout.get(candidate["candidate_id"])
        if summary is not None and summary.get("stable_success", False):
            color, label = "#2ca02c", "stable success"
        elif summary is not None and summary.get("success_count", 0) > 0:
            color, label = "#ffbf00", "provisional"
        elif summary is not None:
            color, label = "#d62728", "failure"
        elif result is None:
            color, label = "#377eb8", "untested"
        elif result.get("success", False):
            color, label = "#ffbf00", "provisional"
        else:
            color, label = "#d62728", "failure"
        dx, dy = 0.16 * np.cos(pose[2]), 0.16 * np.sin(pose[2])
        ax.arrow(pose[0], pose[1], dx, dy, color=color, width=0.008,
                 head_width=0.05, length_includes_head=True, alpha=0.85,
                 label=label if label not in legend_done else None)
        legend_done.add(label)

    ax.set_aspect("equal")
    ax.set_xlim(4.0, 11.0)
    ax.set_ylim(2.2, 5.1)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(f"Manipulation-pose search: {scene_id}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    output = output or data_dir / f"success_map_{scene_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True, choices=sorted(CUBE_SCENES))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    path = build_plot(args.scene_id, args.data_dir, args.output)
    print(path)


if __name__ == "__main__":
    main()
