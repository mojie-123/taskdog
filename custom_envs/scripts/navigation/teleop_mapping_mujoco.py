#!/usr/bin/env python
"""Keyboard teleoperation and occupancy mapping in the MuJoCo scene.

The locomotion policy remains the low-level controller.  Keyboard commands
provide the same vx/vy/wz observations used by the IsaacLab mapper, while a
MuJoCo ``mj_multiRay`` Mid-360 supplies world-frame scan hits.
"""

import argparse
import os
import sys
import threading
import time

# Select X11 before mujoco/glfw is imported.  This mirrors the navigation
# entry points and keeps passive-viewer keyboard input working under Wayland.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("MUJOCO_GL", "glx")
os.environ.setdefault("DISPLAY", ":0")

import glfw
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_CUSTOM_ENVS = os.path.join(_ROOT, "custom_envs")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from custom_envs.mujoco.env import MuJoCoEnv
from custom_envs.mujoco.lidar import MuJoCoMid360
from custom_envs.mujoco.obs_builder import build_policy_obs
from custom_envs.mujoco.policy_loader import load_locomotion_policy
from custom_envs.mujoco.teleop_viewer import MuJoCoTeleopViewer
from custom_envs.scripts.navigation.convert_map import convert
from custom_envs.utils.occupancy_grid import OccupancyGrid


class TeleopController:
    """Thread-safe velocity command that is active only while keys are held."""

    _MOVEMENT_KEYS = {
        glfw.KEY_W, glfw.KEY_S,
        glfw.KEY_A, glfw.KEY_D,
        glfw.KEY_Q, glfw.KEY_E,
    }

    def __init__(self, linear_speed, lateral_speed, yaw_speed):
        self.linear_speed = float(linear_speed)
        self.lateral_speed = float(lateral_speed)
        self.yaw_speed = float(yaw_speed)
        self._pressed = set()
        self._lock = threading.Lock()
        self._save_requested = False
        self._quit_requested = False

    def _command_unlocked(self):
        return np.array([
            self.linear_speed * (
                (glfw.KEY_W in self._pressed)
                - (glfw.KEY_S in self._pressed)),
            self.lateral_speed * (
                (glfw.KEY_A in self._pressed)
                - (glfw.KEY_D in self._pressed)),
            self.yaw_speed * (
                (glfw.KEY_Q in self._pressed)
                - (glfw.KEY_E in self._pressed)),
        ], dtype=np.float32)

    def on_key(self, window, keycode, scancode, action, mods):
        del window, scancode, mods
        with self._lock:
            before = self._command_unlocked()
            if keycode in self._MOVEMENT_KEYS:
                if action == glfw.PRESS:
                    self._pressed.add(keycode)
                elif action == glfw.RELEASE:
                    self._pressed.discard(keycode)
                else:
                    return
            elif keycode in (glfw.KEY_SPACE, glfw.KEY_X) \
                    and action == glfw.PRESS:
                self._pressed.clear()
            elif keycode == glfw.KEY_M and action == glfw.PRESS:
                self._save_requested = True
            elif keycode == glfw.KEY_ESCAPE and action == glfw.PRESS:
                self._pressed.clear()
                self._quit_requested = True
            else:
                return
            after = self._command_unlocked()
            if not np.array_equal(before, after):
                print(
                    "[KEY] command vx/vy/wz="
                    f"{np.round(after, 3).tolist()}",
                    flush=True,
                )

    def snapshot(self):
        with self._lock:
            command = self._command_unlocked()
            save = self._save_requested
            quit_requested = self._quit_requested
            self._save_requested = False
        return command, save, quit_requested


class VoxelPointAccumulator:
    """Bound point-cloud memory by retaining one point per 3-D voxel."""

    def __init__(self, voxel_size=0.05):
        self.voxel_size = float(voxel_size)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        self._points = {}

    def add(self, points):
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        for key, point in zip(keys, points):
            self._points.setdefault(tuple(key.tolist()), point.copy())

    def array(self):
        if not self._points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(list(self._points.values()), dtype=np.float32)

    def __len__(self):
        return len(self._points)


def save_mapping_outputs(grid, cloud, maps_dir, map_name, auto_convert=True):
    os.makedirs(maps_dir, exist_ok=True)
    npz_path = os.path.join(maps_dir, map_name + ".npz")
    cloud_path = os.path.join(maps_dir, map_name + "_cloud.npy")
    grid.save(npz_path)
    np.save(cloud_path, cloud.array())
    print(
        f"[MAP] saved {npz_path} and {cloud_path} "
        f"({len(cloud):,} voxel points)",
        flush=True,
    )
    outputs = {"npz": npz_path, "cloud": cloud_path}
    if auto_convert:
        prefix = os.path.join(maps_dir, map_name + "_nav2")
        pgm_path, yaml_path = convert(npz_path, prefix)
        outputs.update(pgm=pgm_path, yaml=yaml_path)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser("MuJoCo teleop + Mid-360 mapping")
    parser.add_argument("--policy_path", required=True,
                        help="rsl_rl model checkpoint")
    parser.add_argument("--map_name", default="mujoco_map")
    parser.add_argument("--maps_dir", default=os.path.join(_CUSTOM_ENVS, "maps"))
    parser.add_argument("--scene", default=None,
                        help="MJCF scene path (default: custom_envs/mujoco/scene.xml)")
    parser.add_argument("--grid_size", type=int, default=400)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--origin_x", type=float, default=None)
    parser.add_argument("--origin_y", type=float, default=None)
    parser.add_argument("--scan_period", type=float, default=0.1)
    parser.add_argument("--obstacle_min_z", type=float, default=0.15)
    parser.add_argument("--obstacle_max_z", type=float, default=None)
    parser.add_argument("--min_obstacle_hits", type=int, default=1)
    parser.add_argument("--cloud_voxel", type=float, default=0.05)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--linear_speed", type=float, default=0.6)
    parser.add_argument("--lateral_speed", type=float, default=0.5)
    parser.add_argument("--yaw_speed", type=float, default=0.8)
    parser.add_argument("--auto_save_seconds", type=float, default=30.0)
    parser.add_argument("--robot_x", type=float, default=0.0)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="headless smoke-run limit; 0 means unlimited")
    parser.add_argument("--no_realtime", action="store_true")
    parser.add_argument("--no_auto_convert", action="store_true")
    parser.add_argument("--no_save_on_exit", action="store_true")
    args = parser.parse_args()
    if not args.render and args.max_steps <= 0:
        parser.error("use --render for teleoperation or set --max_steps for headless mode")
    if args.grid_size <= 0 or args.resolution <= 0.0:
        parser.error("grid_size and resolution must be positive")
    if args.scan_period <= 0.0:
        parser.error("scan_period must be positive")
    if args.point_stride <= 0:
        parser.error("point_stride must be positive")
    if args.min_obstacle_hits <= 0:
        parser.error("min_obstacle_hits must be positive")
    return args


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    keyboard = TeleopController(
        args.linear_speed, args.lateral_speed, args.yaw_speed)
    env = MuJoCoEnv(
        robot_init_pos=(args.robot_x, args.robot_y, 0.58),
        render=False,
        scene_path=args.scene,
    )
    policy = load_locomotion_policy(args.policy_path, device=device)
    lidar = MuJoCoMid360(env.model)
    viewer = (
        MuJoCoTeleopViewer(
            env.model,
            env.data,
            key_callback=keyboard.on_key,
        )
        if args.render else None
    )

    grid = OccupancyGrid(
        width=args.grid_size,
        height=args.grid_size,
        resolution=args.resolution,
    )
    default_origin = -args.grid_size * args.resolution / 2.0
    grid.set_origin(
        default_origin if args.origin_x is None else args.origin_x,
        default_origin if args.origin_y is None else args.origin_y,
    )
    cloud = VoxelPointAccumulator(args.cloud_voxel)

    last_action = np.zeros(16, dtype=np.float32)
    next_scan_time = 0.0
    next_auto_save_time = (
        args.auto_save_seconds if args.auto_save_seconds > 0.0 else np.inf)
    frame = 0
    last_outputs = None

    print("\n" + "=" * 68)
    print("  MUJOCO TELEOP + MAPPING")
    print("  W/S forward/back  A/D strafe  Q/E turn  Space/X stop")
    print("  M save map        ESC save and quit")
    print("  Movement is active only while a key is held.")
    print(f"  scene: {env.scene_path}")
    print(f"  rays: {lidar.num_rays}, scan period: {args.scan_period:.3f}s")
    print("=" * 68 + "\n")

    try:
        if viewer is not None:
            viewer.render()
        while viewer is None or viewer.is_running():
            loop_started = time.perf_counter()
            if viewer is not None:
                viewer.poll_events()
                if not viewer.is_running():
                    break
            command, save_requested, quit_requested = keyboard.snapshot()
            if quit_requested:
                break

            state = env.get_robot_state()
            obs_np = build_policy_obs(
                state["joint_pos"],
                state["joint_vel"],
                state["ang_vel"],
                state["quat"],
                float(command[0]),
                float(command[1]),
                float(command[2]),
                last_action,
            )
            obs = torch.as_tensor(
                obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.inference_mode():
                action = policy(obs)
            action_np = action.detach().cpu().numpy().reshape(-1)[:16]
            last_action = action_np.astype(np.float32, copy=True)
            env.step(action_np)
            frame += 1
            if viewer is not None:
                viewer.render()

            if env.data.time + 1e-12 >= next_scan_time:
                scan = lidar.scan(env.data)
                finite = grid.integrate_scan(
                    scan.origin_w,
                    scan.hits_w,
                    obstacle_min_z=args.obstacle_min_z,
                    obstacle_max_z=args.obstacle_max_z,
                    min_hits=args.min_obstacle_hits,
                )
                cloud.add(finite[::args.point_stride])
                next_scan_time = env.data.time + args.scan_period
                if frame == 1 or frame % 250 == 0:
                    occupied = int((grid.grid > 0.5).sum())
                    print(
                        f"[SCAN] frame={frame} time={env.data.time:.1f}s "
                        f"hits={len(finite)} occupied={occupied} "
                        f"cloud_voxels={len(cloud):,}",
                        flush=True,
                    )

            if save_requested:
                last_outputs = save_mapping_outputs(
                    grid, cloud, args.maps_dir, args.map_name,
                    auto_convert=not args.no_auto_convert)

            if env.data.time >= next_auto_save_time:
                last_outputs = save_mapping_outputs(
                    grid, cloud, args.maps_dir, args.map_name + "_auto",
                    auto_convert=not args.no_auto_convert)
                next_auto_save_time = env.data.time + args.auto_save_seconds

            if args.max_steps > 0 and frame >= args.max_steps:
                print(f"[MAP] reached --max_steps={args.max_steps}", flush=True)
                break

            if not args.no_realtime:
                elapsed = time.perf_counter() - loop_started
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)

    except KeyboardInterrupt:
        print("\n[MAP] interrupted", flush=True)
    finally:
        try:
            if not args.no_save_on_exit:
                last_outputs = save_mapping_outputs(
                    grid, cloud, args.maps_dir, args.map_name,
                    auto_convert=not args.no_auto_convert)
        finally:
            if viewer is not None:
                viewer.close()
            env.close()
        if last_outputs:
            print(f"[MAP] final outputs: {last_outputs}", flush=True)


if __name__ == "__main__":
    main()
