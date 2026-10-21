#!/usr/bin/env python3.10
"""AnyGrasp worker with target-region steering for the standalone MoveIt flow.

INPUT ``/tmp/pointcloud.npz`` contains the full camera-frame point cloud plus
``region_steering``, a boolean vector aligned element-for-element with points.
The full cloud remains available to AnyGrasp collision detection, while grasp
proposal generation is steered toward the SAM2 target region.

This is intentionally separate from ``grasp_worker.py`` so N2M and SysNav keep
their existing worker behaviour unchanged.
"""

import argparse
import os
import sys

# gsnet.so resolves its license directory relative to the working directory.
SDK_DETECTION_DIR = "/home/mojie/anygrasp_sdk/grasp_detection"
sys.path.insert(0, SDK_DETECTION_DIR)
os.chdir(SDK_DETECTION_DIR)

INPUT_PATH = "/tmp/pointcloud.npz"
OUTPUT_PATH = "/tmp/grasp_result.npz"

MAX_GRIPPER_WIDTH = 0.08
GRIPPER_HEIGHT = 0.06


def _save_empty(np):
    np.savez(
        OUTPUT_PATH,
        translations=np.zeros((0, 3), dtype=np.float32),
        rotations=np.zeros((0, 3, 3), dtype=np.float32),
        widths=np.zeros(0, dtype=np.float32),
        scores=np.zeros(0, dtype=np.float32),
        depths=np.zeros(0, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser("grasp_worker_masked")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument(
        "--max_gripper_width", type=float, default=MAX_GRIPPER_WIDTH
    )
    parser.add_argument("--gripper_height", type=float, default=GRIPPER_HEIGHT)
    parser.add_argument("--no_collision", action="store_true")
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    # Import AnyGrasp before numpy.  This preserves the ABI/load order required
    # by the installed gsnet/MinkowskiEngine build.
    from argparse import Namespace
    from gsnet import create_detector
    import numpy as np

    print(f"[grasp_worker_masked] Loading AnyGrasp detector from: {args.checkpoint}")
    detector = create_detector(
        Namespace(
            checkpoint_path=args.checkpoint,
            max_gripper_width=args.max_gripper_width,
            gripper_height=args.gripper_height,
        )
    )
    if detector is None:
        print("[grasp_worker_masked] ERROR: AnyGrasp initialization failed")
        return 1

    if not os.path.exists(INPUT_PATH):
        print(f"[grasp_worker_masked] ERROR: input not found: {INPUT_PATH}")
        return 1

    data = np.load(INPUT_PATH)
    if "points" not in data or "region_steering" not in data:
        print("[grasp_worker_masked] ERROR: points/region_steering missing")
        return 1
    points = np.asarray(data["points"], dtype=np.float32)
    region = np.asarray(data["region_steering"], dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"[grasp_worker_masked] ERROR: invalid points shape {points.shape}")
        return 1
    if len(region) != len(points):
        print(
            "[grasp_worker_masked] ERROR: region length mismatch "
            f"{len(region)} != {len(points)}"
        )
        return 1
    target_count = int(region.sum())
    if len(points) < 50 or target_count < 20:
        print(
            f"[grasp_worker_masked] WARN: insufficient cloud/target points "
            f"full={len(points)} target={target_count}"
        )
        _save_empty(np)
        return 0

    print(
        f"[grasp_worker_masked] Input cloud: full={len(points)} "
        f"target_region={target_count}"
    )
    grasps = detector.get_grasp(
        points,
        {
            "collision_detection": not args.no_collision,
            "dense_grasp": args.dense,
            "region_steering": region,
        },
    )
    if grasps is None or len(grasps) == 0:
        print("[grasp_worker_masked] WARN: no valid target-region grasp")
        _save_empty(np)
        return 0

    print(f"[grasp_worker_masked] Raw grasps: {len(grasps)}")
    if not args.dense:
        grasps = grasps.nms()
    grasps = grasps.sort_by_score()
    topk = min(max(0, int(args.topk)), len(grasps))
    if topk == 0:
        _save_empty(np)
        return 0
    grasps = grasps[:topk]

    np.savez(
        OUTPUT_PATH,
        translations=grasps.translations.astype(np.float32),
        rotations=grasps.rotation_matrices.astype(np.float32),
        widths=grasps.widths.astype(np.float32),
        scores=grasps.scores.astype(np.float32),
        depths=grasps.depths.astype(np.float32),
    )
    print(f"[grasp_worker_masked] Saved {topk} target-region grasps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
