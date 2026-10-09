#!/usr/bin/env python
"""交互式查看建图 ``_cloud.npy`` 的纯诊断工具，不参与导航或抓取。

点坐标直接来自雷达世界系命中，因此它比二维 PGM 更适合判断障碍是否被投射到
正确位置。当前图中只显示 LiDAR hits，不再叠加 banana/table 等人工标记点。

Usage:
    python scripts/navigation/view_3d.py maps/my_map_cloud.npy
    python scripts/navigation/view_3d.py maps/my_map_cloud.npy --no_ground
    python scripts/navigation/view_3d.py maps/my_map_cloud.npy --no_ground --point_size 5
"""

import argparse, sys, webbrowser
import numpy as np


def main():
    parser = argparse.ArgumentParser("3D Point Cloud Viewer (browser)")
    parser.add_argument("cloud_path", help="path to _cloud.npy file")
    parser.add_argument("--no_ground", action="store_true",
                        help="hide ground points (|z| < 0.15)")
    parser.add_argument("--point_size", type=float, default=2.0,
                        help="marker size (default 2)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="keep 1/N points")
    args = parser.parse_args()

    # ---- load ----
    pts = np.load(args.cloud_path)
    if len(pts) == 0:
        print("[ERROR] Empty point cloud.")
        sys.exit(1)

    print(f"Loaded {len(pts):,} points")
    print(f"  X: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
    print(f"  Y: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
    print(f"  Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

    if args.no_ground:
        mask = np.abs(pts[:, 2]) > 0.15
        pts = pts[mask]
        print(f"  after --no_ground: {len(pts):,} points")
    if args.downsample > 1:
        pts = pts[::args.downsample]
        print(f"  after downsampling: {len(pts):,} points")

    # ---- build interactive plot ----
    import plotly.graph_objects as go

    # Main point cloud (colour by height)
    cloud_trace = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=args.point_size,
            color=pts[:, 2],
            colorscale="Viridis",
            opacity=0.8,
            colorbar=dict(title="Z (m)"),
        ),
        name="LiDAR hits",
    )

    fig = go.Figure(data=[cloud_trace])
    fig.update_layout(
        title="LiDAR Point Cloud — drag to rotate · scroll to zoom · right‑drag to pan",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    html_path = args.cloud_path.replace(".npy", "_3d.html")
    fig.write_html(html_path)
    print(f"\nSaved → {html_path}")
    webbrowser.open(html_path)


if __name__ == "__main__":
    main()
