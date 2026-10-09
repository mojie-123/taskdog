#!/usr/bin/env python3
"""Run N2M on one saved observation and export relative/world pose candidates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from custom_envs.scripts.N2M.geometry import transform_se2, wrap_angle
from custom_envs.scripts.N2M.manifests import read_jsonl, write_json
from custom_envs.scripts.N2M.vendor_n2m import N2MNet


def _point_tensor(path: Path, count: int, device: torch.device) -> torch.Tensor:
    with np.load(path) as cloud:
        xyz = cloud["xyz"].astype(np.float32)
        rgb = cloud["rgb"].astype(np.float32)
    if rgb.size and rgb.max() > 1.0:
        rgb /= 255.0
    rng = np.random.default_rng(0)
    indices = rng.choice(len(xyz), count, replace=len(xyz) < count)
    points = np.concatenate((xyz[indices], rgb[indices]), axis=1)
    return torch.from_numpy(points).unsqueeze(0).to(device)


def _pose_rows(relative_poses: np.ndarray, observation_pose: list[float]) -> list[dict]:
    rows = []
    for pose in relative_poses:
        relative = [float(pose[0]), float(pose[1]), wrap_angle(float(pose[2]))]
        world = transform_se2(observation_pose, relative)
        rows.append({"pose_observation": relative, "pose_world": world.tolist()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = copy.deepcopy(config["model"])
    # The N2M checkpoint already contains the complete encoder.
    model_config["encoder"]["ckpt"] = None
    model = N2MNet(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset_dir = args.dataset_dir.resolve()
    observations = {
        row["observation_id"]: row
        for row in read_jsonl(dataset_dir / "observations.jsonl")
    }
    observation = observations.get(args.observation_id)
    if observation is None:
        raise KeyError(f"unknown observation id: {args.observation_id}")
    count = int(config["dataset"].get("num_points", 8192))
    points = _point_tensor(dataset_dir / observation["cloud_file"], count, device)
    with torch.no_grad():
        means, covariance, weights = model(points)
        samples = model.sample(points, args.num_samples)
    means_np = means[0].cpu().numpy()
    samples_np = samples[0].cpu().numpy()
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "observation_id": args.observation_id,
        "observation_pose_world": observation["observation_pose_world"],
        "modes": [
            {
                **pose,
                "weight": float(weights[0, index].cpu()),
                "covariance": covariance[0, index].cpu().tolist(),
            }
            for index, pose in enumerate(
                _pose_rows(means_np, observation["observation_pose_world"])
            )
        ],
        "samples": _pose_rows(samples_np, observation["observation_pose_world"]),
    }
    output = args.output or dataset_dir / "predictions" / f"{args.observation_id}.json"
    write_json(output, result)
    print(f"[N2M] prediction saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
