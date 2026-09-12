#!/usr/bin/env python3
"""Convert a custom .npz occupancy grid map to Nav2-compatible PGM + YAML format.

The .npz format stores log-odds values (float32, shape HxW) built by
occupancy_grid.py/teleop_mapping.py.  Nav2's map_server expects:
  - A PGM image  (uint8, 0=occupied/black, 254=free/white, 205=unknown/grey)
  - A YAML sidecar describing resolution, origin, and thresholds.

Conversion rules:
  log-odds >  0.5  ->   0   (occupied -> black pixel)
  log-odds < -0.5  -> 254   (free     -> white pixel)
  -0.5 to 0.5     -> 205   (unknown  -> grey pixel)

Usage:
    python scripts/navigation/convert_map.py \\
        --map maps/my_map.npz \\
        --out maps/my_map_nav2

    Produces:
        maps/my_map_nav2.pgm
        maps/my_map_nav2.yaml
"""

import argparse
import os
import sys

import numpy as np


def log_odds_to_pgm_pixel(log_odds_grid: np.ndarray) -> np.ndarray:
    """Convert log-odds float array to Nav2 PGM pixel values.

    Args:
        log_odds_grid: (H, W) float32 array of log-odds values.

    Returns:
        (H, W) uint8 array with values in {0, 205, 254}.
          0   = occupied (black)
          205 = unknown  (grey)
          254 = free     (white)
    """
    pgm = np.full(log_odds_grid.shape, 205, dtype=np.uint8)  # default: unknown
    pgm[log_odds_grid < -0.5] = 254  # free -> white
    pgm[log_odds_grid > 0.5] = 0     # occupied -> black
    # Nav2 的 PGM 格式 row=0 在图像顶部（世界 Y 最大），
    # 而 npz grid row=0 对应世界 Y 最小（origin_y），需要垂直翻转修正 Y 轴方向。
    return np.flipud(pgm)


def write_pgm(pgm: np.ndarray, path: str) -> None:
    """Write a PGM P5 (binary) file.

    Args:
        pgm: (H, W) uint8 array.
        path: output file path (e.g. 'my_map.pgm').
    """
    H, W = pgm.shape
    header = "P5\n{} {}\n255\n".format(W, H).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(pgm.tobytes())
    print("[convert_map] Written PGM -> {}  ({}x{} px)".format(path, W, H), flush=True)


def write_yaml(yaml_path: str, pgm_filename: str,
               resolution: float, origin: tuple) -> None:
    """Write a Nav2 map YAML sidecar file.

    Args:
        yaml_path:    output .yaml file path.
        pgm_filename: relative path to the PGM file (just the filename).
        resolution:   meters per pixel.
        origin:       (ox, oy) world coordinate of grid[0,0] (top-left corner).
    """
    # Nav2 map_server origin = [x, y, theta] of the lower-left pixel corner.
    # Our OccupancyGrid stores origin = world coord of grid[row=0, col=0] = top-left.
    # With negate=0 and image stored row=0 at top:
    #   nav2 bottom-left Y = origin_y (grid row=0 is top; PGM stores top-down)
    # We pass origin as-is and set negate=0 so nav2 reads the image correctly.
    ox, oy = float(origin[0]), float(origin[1])
    content = (
        "image: {}\n"
        "resolution: {:.6f}\n"
        "origin: [{:.4f}, {:.4f}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
        "mode: trinary\n"
    ).format(pgm_filename, resolution, ox, oy)
    with open(yaml_path, "w") as f:
        f.write(content)
    print("[convert_map] Written YAML -> {}".format(yaml_path), flush=True)
    print("[convert_map]   resolution={} m/px  origin=({:.3f}, {:.3f})".format(
        resolution, ox, oy), flush=True)


def convert(npz_path: str, out_prefix: str) -> None:
    """Run the full conversion pipeline.

    Args:
        npz_path:   path to the input .npz map file.
        out_prefix: output path prefix (without extension).
    """
    if not os.path.exists(npz_path):
        print("[ERROR] Map file not found: {}".format(npz_path))
        sys.exit(1)

    data = np.load(npz_path)
    log_odds = data["grid"].astype(np.float32)  # (H, W)
    resolution = float(data["resolution"])
    origin = tuple(data["origin"])  # (ox, oy)
    H, W = log_odds.shape

    print("[convert_map] Loaded .npz: {}x{} cells, res={} m/px, origin={}".format(
        H, W, resolution, origin), flush=True)

    n_occ = int((log_odds > 0.5).sum())
    n_free = int((log_odds < -0.5).sum())
    n_unk = H * W - n_occ - n_free
    print("[convert_map] Cell stats: occupied={} free={} unknown={}".format(
        n_occ, n_free, n_unk), flush=True)

    pgm = log_odds_to_pgm_pixel(log_odds)

    pgm_path = out_prefix + ".pgm"
    yaml_path = out_prefix + ".yaml"
    pgm_filename = os.path.basename(pgm_path)

    write_pgm(pgm, pgm_path)
    write_yaml(yaml_path, pgm_filename, resolution, origin)

    print("[convert_map] Done! To use with Nav2:", flush=True)
    print("  ros2 run nav2_map_server map_server --ros-args \\", flush=True)
    print("    -p yaml_filename:={} \\".format(os.path.abspath(yaml_path)), flush=True)
    print("    -p use_sim_time:=false", flush=True)


def main():
    parser = argparse.ArgumentParser("Convert .npz map to Nav2 PGM+YAML")
    parser.add_argument("--map", required=True, help="input .npz map file")
    parser.add_argument("--out", default=None,
                        help="output prefix (default: same path without .npz + '_nav2')")
    args = parser.parse_args()

    npz_path = args.map
    if args.out is None:
        out_prefix = os.path.splitext(npz_path)[0] + "_nav2"
    else:
        out_prefix = args.out

    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    convert(npz_path, out_prefix)


if __name__ == "__main__":
    main()
