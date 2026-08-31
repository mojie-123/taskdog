#!/usr/bin/env python
"""3D point-cloud visualisation of a teleop mapping session.

Supports:
  - Raw point cloud:  <name>_cloud.npy    (from teleop_mapping.py M-save)
  - Occupancy grid:   <name>.npz          (converts occupied cells to 3D points)

Usage:
    python scripts/navigation/viz_map.py maps/my_map_cloud.npy
    python scripts/navigation/viz_map.py maps/my_map_cloud.npy --no_ground
    python scripts/navigation/viz_map.py maps/my_map.npz   <---注意建图完后就是要执行这一句
"""

import argparse, os, sys
import numpy as np


def _load_from_npy(path):
    """Load raw point cloud .npy → (N, 3) array."""
    pts = np.load(path)
    return pts


def _load_from_npz(path):
    """Convert occupancy-grid .npz to (N, 3) point cloud of occupied cells."""
    data = np.load(path)
    grid = data["grid"]
    origin = data["origin"]
    res = float(data["resolution"])
    pts_list = []
    rows, cols = np.where(grid > 0.5)  # occupied cells only
    for r, c in zip(rows, cols):
        wx = origin[0] + (c + 0.5) * res
        wy = origin[1] + (r + 0.5) * res
        pts_list.append((wx, wy, grid[r, c] * 0.1))  # fake z from confidence
    pts = np.array(pts_list) if pts_list else np.zeros((0, 3))
    return pts


def main():
    parser = argparse.ArgumentParser("3D Point Cloud Visualisation")
    parser.add_argument("map_path", help="path to .npz or _cloud.npy file")
    parser.add_argument("--downsample", type=int, default=1,
                        help="keep 1/N points (default 5→20%%)")
    parser.add_argument("--no_ground", action="store_true",
                        help="hide ground points (z < 0.15)")
    parser.add_argument("--save", default=None,
                        help="save static 3D PNG instead of opening viewer")
    args = parser.parse_args()

    # Detect format
    if args.map_path.endswith(".npz"):
        print("[INFO] Loading occupancy grid .npz — showing occupied cells as 3D points")
        pts = _load_from_npz(args.map_path)
    elif args.map_path.endswith("_cloud.npy") or args.map_path.endswith(".npy"):
        pts = _load_from_npy(args.map_path)
    else:
        print("[ERROR] Unknown format — expected .npz or _cloud.npy")
        sys.exit(1)

    if len(pts) == 0:
        print("[ERROR] No points loaded.  Did you save the map (press M) during teleop?")
        print("  Re-run teleop_mapping.py, walk around, and press M to save.")
        sys.exit(1)

    print(f"Loaded {len(pts):,} points")

    # Filter ground
    if args.no_ground:
        mask = np.abs(pts[:, 2]) > 0.15
        pts = pts[mask]
        print(f"  after removing ground: {len(pts):,} points")

    # Downsample
    if args.downsample > 1 and len(pts) > 1:
        pts = pts[::args.downsample]
        print(f"  after downsampling {args.downsample}×: {len(pts):,} points")

    print(f"  X: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
    print(f"  Y: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
    print(f"  Z: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")

    # ---- 3D rendering ----
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    z_range = max(z_max - z_min, 1e-6)
    z_norm = (pts[:, 2] - z_min) / z_range
    colours = np.zeros((len(pts), 3))
    colours[:, 0] = z_norm
    colours[:, 2] = 1.0 - z_norm

    if args.save:
        _render_png(pts, colours, args.save)
        return

    # Always save a static PNG (works everywhere)
    png_path = os.path.splitext(args.map_path)[0] + "_3d.png"
    _render_png(pts, colours, png_path)

    # Optionally try interactive viewer
    if not args.save:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            o3d.visualization.draw_geometries(
                [pcd, frame], window_name="LiDAR Point Cloud", width=1200, height=800)
        except Exception:
            pass


def _render_png(pts, colours, path):
    """Render a static 3D point cloud as PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Subsample for speed if needed
    if len(pts) > 50000:
        idx = np.random.choice(len(pts), 50000, replace=False)
        pts, colours = pts[idx], colours[idx]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colours, s=1, alpha=0.6, marker=".")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("LiDAR Point Cloud")
    # Mark origin
    ax.scatter([0], [0], [0.58], c="blue", s=50, marker="*", label="origin (spawn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
