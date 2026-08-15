#!/usr/bin/env python3
"""grasp_worker.py — Subprocess grasp detection worker.

Protocol:
  INPUT  : /tmp/pointcloud.npz  {points: (N,3) float32, colors: (N,3) float32}
  OUTPUT : /tmp/grasp_result.npz {
               translations: (K,3) float32,
               rotations:    (K,3,3) float32,
               widths:       (K,) float32,
               scores:       (K,) float32,
           }
"""

import os
import sys
import argparse
import numpy as np

GRASPNET_ROOT = "/home/mojie/graspnet-baseline"
sys.path.insert(0, GRASPNET_ROOT)
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "models"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "utils"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "pointnet2"))

INPUT_PATH  = "/tmp/pointcloud.npz"
OUTPUT_PATH = "/tmp/grasp_result.npz"

NUM_POINT = 20000
NUM_VIEW  = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01


def load_model(checkpoint_path, device):
    from graspnet import GraspNet
    net = GraspNet(
        input_feature_dim=0, num_view=NUM_VIEW, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    print(f"[grasp_worker] Loaded checkpoint (epoch {checkpoint['epoch']})")
    net.eval()
    return net


def prepare_input(points, colors, device):
    N = len(points)
    if N >= NUM_POINT:
        idxs = np.random.choice(N, NUM_POINT, replace=False)
    else:
        idxs = np.concatenate([
            np.arange(N), np.random.choice(N, NUM_POINT - N, replace=True)
        ])
    pts = points[idxs].astype(np.float32)
    col = colors[idxs].astype(np.float32)
    cloud_tensor = torch.from_numpy(pts[np.newaxis]).to(device)
    return {"point_clouds": cloud_tensor, "cloud_colors": col}, pts


def run_inference(net, end_points):
    from graspnet import pred_decode
    from graspnetAPI.grasp import GraspGroup
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)


def do_collision_filter(gg, scene_points):
    from collision_detector import ModelFreeCollisionDetector
    mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(
        gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]


def main():
    parser = argparse.ArgumentParser("grasp_worker")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--no_collision", action="store_true")
    args = parser.parse_args()

    global torch
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[grasp_worker] Using device: {device}")

    if not os.path.exists(INPUT_PATH):
        print(f"[grasp_worker] ERROR: input not found: {INPUT_PATH}")
        sys.exit(1)

    data = np.load(INPUT_PATH)
    points = data["points"].astype(np.float32)
    colors = data["colors"].astype(np.float32)
    print(f"[grasp_worker] Input cloud: {len(points)} points")

    if len(points) < 100:
        print("[grasp_worker] ERROR: too few points")
        sys.exit(1)

    net = load_model(args.checkpoint, device)
    end_points, pts_sampled = prepare_input(points, colors, device)
    gg = run_inference(net, end_points)
    print(f"[grasp_worker] Raw grasps: {len(gg)}")

    gg.nms()
    gg.sort_by_score()

    if not args.no_collision and len(gg) > 0:
        gg = do_collision_filter(gg, points)
        print(f"[grasp_worker] After collision filter: {len(gg)}")

    topk = min(args.topk, len(gg))
    if topk == 0:
        print("[grasp_worker] WARN: no valid grasps found")
        np.savez(OUTPUT_PATH,
                 translations=np.zeros((0, 3), dtype=np.float32),
                 rotations=np.zeros((0, 3, 3), dtype=np.float32),
                 widths=np.zeros(0, dtype=np.float32),
                 scores=np.zeros(0, dtype=np.float32))
        return

    gg = gg[:topk]
    # GraspGroup array: [score, width, height, depth, r00..r22, tx,ty,tz, obj_id]
    arr = gg.grasp_group_array          # (K, 17)
    scores       = arr[:, 0].astype(np.float32)
    widths       = arr[:, 1].astype(np.float32)
    rotations    = arr[:, 4:13].reshape(-1, 3, 3).astype(np.float32)
    translations = arr[:, 13:16].astype(np.float32)

    np.savez(OUTPUT_PATH,
             translations=translations,
             rotations=rotations,
             widths=widths,
             scores=scores)
    print(f"[grasp_worker] Saved {topk} grasps to {OUTPUT_PATH}")
    for i in range(topk):
        print(f"  [{i}] score={scores[i]:.3f} t={np.round(translations[i],3)} w={widths[i]:.3f}")


if __name__ == "__main__":
    main()
