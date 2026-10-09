"""Headless quality-control figures for one N2M observation proposal."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2m-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/n2m-matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image


COFFEE_TABLE_XY = (7.50, 3.65)
COFFEE_TABLE_HALF_SIZE_XY = (2.50, 0.45)


def _footprint(pose: Sequence[float], length: float = 0.75, width: float = 0.40):
    corners = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(pose[2])), math.sin(float(pose[2]))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return corners @ rotation.T + np.asarray(pose[:2], dtype=np.float64)


def _image_or_message(axis, image, title: str, *, cmap=None, vmin=None, vmax=None):
    axis.set_title(title)
    axis.axis("off")
    if image is None:
        axis.text(0.5, 0.5, "not available", ha="center", va="center")
        return
    axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)


def _draw_cloud_top(
    axis,
    xyz: np.ndarray | None,
    colors: np.ndarray | None,
    cube_center: Sequence[float] | None,
    cube_centroid: Sequence[float] | None,
):
    if xyz is not None and len(xyz):
        axis.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=1, rasterized=True)
    axis.add_patch(Polygon(_footprint([0.0, 0.0, 0.0]), fill=False, color="cyan", lw=2))
    axis.arrow(0.0, 0.0, 0.35, 0.0, color="cyan", width=0.01, length_includes_head=True)
    if cube_center is not None:
        axis.scatter(
            [cube_center[0]], [cube_center[1]], c="yellow", edgecolors="black",
            s=90, marker="s", label="cube truth", zorder=5,
        )
    if cube_centroid is not None:
        axis.scatter(
            [cube_centroid[0]], [cube_centroid[1]], c="magenta",
            s=65, marker="x", label="visible centroid", zorder=6,
        )
    axis.set_title("base-frame cloud: top")
    axis.set_xlabel("base +X forward [m]")
    axis.set_ylabel("base +Y left [m]")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    if cube_center is not None:
        axis.legend(loc="best", fontsize=7)


def _draw_cloud_side(
    axis,
    xyz: np.ndarray | None,
    colors: np.ndarray | None,
    cube_center: Sequence[float] | None,
    cube_centroid: Sequence[float] | None,
):
    if xyz is not None and len(xyz):
        axis.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=1, rasterized=True)
    if cube_center is not None:
        axis.scatter(
            [cube_center[0]], [cube_center[2]], c="yellow", edgecolors="black",
            s=90, marker="s", label="cube truth", zorder=5,
        )
    if cube_centroid is not None:
        axis.scatter(
            [cube_centroid[0]], [cube_centroid[2]], c="magenta",
            s=65, marker="x", label="visible centroid", zorder=6,
        )
    axis.set_title("base-frame cloud: side")
    axis.set_xlabel("base +X forward [m]")
    axis.set_ylabel("base +Z up [m]")
    axis.axis("equal")
    axis.grid(alpha=0.2)


def _draw_world_map(
    axis,
    observation_pose: Sequence[float],
    cube_pose: Sequence[float],
    manipulation_poses: Iterable[Sequence[float]],
):
    center_x, center_y = COFFEE_TABLE_XY
    half_x, half_y = COFFEE_TABLE_HALF_SIZE_XY
    axis.add_patch(
        Rectangle(
            (center_x - half_x, center_y - half_y),
            2.0 * half_x,
            2.0 * half_y,
            facecolor="#8b5a2b",
            edgecolor="black",
            alpha=0.30,
            linewidth=1.5,
            label="coffee table",
        )
    )
    axis.scatter(
        [cube_pose[0]], [cube_pose[1]], s=100, c="gold", edgecolors="black",
        marker="s", label="cube", zorder=5,
    )
    manipulation_poses = list(manipulation_poses)
    for index, pose in enumerate(manipulation_poses):
        axis.add_patch(
            Polygon(
                _footprint(pose), fill=False, edgecolor="#2ca02c", alpha=0.16, lw=0.7,
                label="successful M" if index == 0 else None,
            )
        )
    axis.add_patch(
        Polygon(_footprint(observation_pose), fill=False, edgecolor="#1f77b4", lw=2.0, label="O")
    )
    yaw = float(observation_pose[2])
    axis.arrow(
        observation_pose[0], observation_pose[1],
        0.35 * math.cos(yaw), 0.35 * math.sin(yaw),
        color="#1f77b4", width=0.015, head_width=0.10, length_includes_head=True,
    )
    points = [[center_x - half_x, center_y - half_y], [center_x + half_x, center_y + half_y]]
    points += [list(observation_pose[:2]), list(cube_pose[:2])]
    points += [list(pose[:2]) for pose in manipulation_poses]
    points = np.asarray(points, dtype=np.float64)
    padding = 0.55
    axis.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
    axis.set_ylim(float(points[:, 1].min() - padding), float(points[:, 1].max() + padding))
    axis.set_title(f"world map: {len(manipulation_poses)} successful M")
    axis.set_xlabel("world X [m]")
    axis.set_ylabel("world Y [m]")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=7)


def _save_single_plot(path: Path, draw) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    draw(axis)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_observation_preview(
    directory: Path,
    *,
    observation_id: str,
    status: str,
    reason: str | None,
    rgb: np.ndarray | None,
    depth: np.ndarray | None,
    cube_mask: np.ndarray | None,
    xyz: np.ndarray | None,
    point_rgb: np.ndarray | None,
    cube_center_base: Sequence[float] | None,
    cube_centroid_base: Sequence[float] | None,
    observation_pose_world: Sequence[float],
    cube_pose_world: Sequence[float],
    manipulation_poses_world: Iterable[Sequence[float]] = (),
    metrics: dict | None = None,
    max_points: int = 40000,
    preview_max_depth: float = 3.0,
) -> None:
    """Write individual inspection images and a six-panel QC overview."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    manipulation_poses_world = list(manipulation_poses_world)

    if rgb is not None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        Image.fromarray(rgb).save(directory / "rgb.png")
    overlay = None
    if rgb is not None:
        overlay = rgb.copy()
        if cube_mask is not None:
            mask = np.asarray(cube_mask, dtype=bool)
            overlay[mask] = np.clip(
                0.35 * overlay[mask].astype(np.float32) + 0.65 * np.array([255, 0, 255]),
                0,
                255,
            ).astype(np.uint8)
        Image.fromarray(overlay).save(directory / "rgb_overlay.png")

    valid_depth = None
    depth_scale = 1.0
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64)
        valid_depth = np.where(
            np.isfinite(depth) & (depth > 0.0) & (depth <= float(preview_max_depth)),
            depth,
            np.nan,
        )
        finite = valid_depth[np.isfinite(valid_depth)]
        depth_scale = float(np.percentile(finite, 95)) if finite.size else 1.0
        plt.imsave(directory / "depth.png", valid_depth, cmap="turbo", vmin=0.0, vmax=depth_scale)
    if cube_mask is not None:
        Image.fromarray(np.asarray(cube_mask, dtype=np.uint8) * 255).save(
            directory / "cube_mask.png"
        )

    cloud_xyz = None if xyz is None else np.asarray(xyz)
    cloud_rgb = None if point_rgb is None else np.asarray(point_rgb, dtype=np.float32) / 255.0
    if cloud_xyz is not None and len(cloud_xyz) > max_points:
        indices = np.random.default_rng(0).choice(len(cloud_xyz), max_points, replace=False)
        cloud_xyz = cloud_xyz[indices]
        cloud_rgb = cloud_rgb[indices]

    _save_single_plot(
        directory / "cloud_top.png",
        lambda axis: _draw_cloud_top(
            axis, cloud_xyz, cloud_rgb, cube_center_base, cube_centroid_base
        ),
    )
    _save_single_plot(
        directory / "cloud_side.png",
        lambda axis: _draw_cloud_side(
            axis, cloud_xyz, cloud_rgb, cube_center_base, cube_centroid_base
        ),
    )
    _save_single_plot(
        directory / "world_map.png",
        lambda axis: _draw_world_map(
            axis, observation_pose_world, cube_pose_world, manipulation_poses_world
        ),
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    metric_text = "  ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
    title = f"{observation_id} | {status}"
    if reason:
        title += f" | {reason}"
    if metric_text:
        title += f"\n{metric_text}"
    figure.suptitle(title, fontsize=11)
    _image_or_message(axes[0, 0], overlay, "RGB + cube mask")
    _image_or_message(
        axes[0, 1], valid_depth, f"depth (95%={depth_scale:.2f}m)",
        cmap="turbo", vmin=0.0, vmax=depth_scale,
    )
    _image_or_message(axes[0, 2], cube_mask, "cube segmentation", cmap="gray", vmin=0, vmax=1)
    _draw_cloud_top(axes[1, 0], cloud_xyz, cloud_rgb, cube_center_base, cube_centroid_base)
    _draw_cloud_side(axes[1, 1], cloud_xyz, cloud_rgb, cube_center_base, cube_centroid_base)
    _draw_world_map(
        axes[1, 2], observation_pose_world, cube_pose_world, manipulation_poses_world
    )
    figure.savefig(directory / "qc_panel.png", dpi=150)
    plt.close(figure)
