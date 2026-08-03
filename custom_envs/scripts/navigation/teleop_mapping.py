#!/usr/bin/env python
"""Keyboard-teleop the robot while building an occupancy-grid map from LiDAR.

Usage:
    python scripts/navigation/teleop_mapping.py \\
        --task=Flat-Deeprobotics-M20Pro-Lidar-v0

Controls:
    W / S       forward / backward
    A / D       left / right
    Q / E       rotate left / right
    M           save map to ../maps/
    ESC         quit
"""

import argparse
import os
import sys

import numpy as np
import torch

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "maps")


def _save_pointcloud(maps_dir, name, points_list):
    """Concatenate accumulated points and save as .npy for 3D viz."""
    if not points_list:
        print("[WARN] No elevated points collected — point cloud will be empty. "
              "Did the LiDAR see any above-ground objects?")
        # still save an empty file so the pipeline doesn't break
        np.save(os.path.join(maps_dir, name + "_cloud.npy"), np.zeros((0, 3)))
        return
    pts = np.concatenate(points_list, axis=0)
    path = os.path.join(maps_dir, name + "_cloud.npy")
    np.save(path, pts)
    print(f"[INFO] 3D point cloud saved → {path}  ({len(pts)} pts)")


def main():
    parser = argparse.ArgumentParser("Teleop + Mapping for M20 Pro")
    parser.add_argument("--task", default="Flat-Deeprobotics-M20Pro-Lidar-v0")
    parser.add_argument("--grid_size", type=int, default=400,
                        help="map cells per side (400 × 0.05 m = 20 m)")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--map_name", default="my_map")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--load_run", default=None, help="run dir to load policy from (default: latest)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint filename (default: latest)")
    parser.add_argument("--policy_task", default=None,
                        help="task whose trained model to use (default: same as --task)")
    args, unknown = parser.parse_known_args()

    # ---- Isaac Sim launch ----
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import custom_envs.tasks  # noqa: F401

    from custom_envs.utils.occupancy_grid import OccupancyGrid
    from custom_envs.utils.nav_utils import euler_from_quat

    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

    # ---- build env config ----
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.terminations.time_out = None

    # keyboard: override velocity-command observation
    controller = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.5,
    ))
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.tensor(
            controller.advance(), dtype=torch.float32
        ).unsqueeze(0).to(env.device),
    )
    env_cfg.commands.base_velocity.debug_vis = False

    # ---- load locomotion policy to convert velocity cmd → joint actions ----
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    # Policy task can differ from sim task — e.g. use Flat-Deeprobotics-M20-v0
    # model to drive the LiDAR-enabled Flat-Deeprobotics-M20Pro-Lidar-v0 env.
    policy_task = args.policy_task or args.task
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry as load_cfg
    agent_cfg = load_cfg(policy_task, "rsl_rl_cfg_entry_point")

    # If policy was trained WITHOUT LiDAR, strip LiDAR obs from sim env —
    # otherwise obs dim mismatch crashes on policy load.
    if "Lidar" not in policy_task and "Lidar" in args.task:
        print("[INFO] Policy has no LiDAR — stripping LiDAR observations from env")
        env_cfg.observations.policy.lidar = None
        env_cfg.observations.critic.lidar = None
        if hasattr(env_cfg.scene, "mid360_lidar"):
            env_cfg.scene.mid360_lidar.debug_vis = False

    # Policy logs are under deps/rl_training (where training scripts live)
    _rl_training_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "deps", "rl_training")
    )
    log_root = os.path.join(_rl_training_root, "logs", "rsl_rl", agent_cfg.experiment_name)

    env = gym.make(args.task, cfg=env_cfg)
    obs = env.reset()[0]

    # Resolve checkpoint: explicit args take priority, else auto-latest
    if args.checkpoint:
        if args.load_run:
            agent_cfg.load_run = args.load_run
        resume_path = get_checkpoint_path(
            os.path.abspath(log_root),
            agent_cfg.load_run if args.load_run else ".*",
            args.checkpoint,
        )
    else:
        import glob as _glob
        if args.load_run:
            run_dir = os.path.join(log_root, args.load_run)
        else:
            runs = sorted(_glob.glob(os.path.join(log_root, "*")))
            if not runs:
                print(f"[ERROR] No training runs in {log_root}")
                simulation_app.close(); return
            run_dir = runs[-1]
        agent_cfg.load_run = os.path.basename(run_dir)
        resume_path = get_checkpoint_path(os.path.abspath(log_root), agent_cfg.load_run, "model_.*.pt")

    print(f"[INFO] Loading policy: {resume_path}")
    wrapped = RslRlVecEnvWrapper(env)
    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cuda:0")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print("[INFO] Policy loaded — robot ready for teleop")

    # ---- occupancy grid (2D, for A* planning) ----
    grid = OccupancyGrid(
        width=args.grid_size, height=args.grid_size, resolution=args.resolution
    )
    grid.set_origin(-args.grid_size * args.resolution / 2.0,
                    -args.grid_size * args.resolution / 2.0)

    # ---- raw point cloud (3D, for visualisation) ----
    all_points_3d = []  # list of (x, y, z) arrays, concatenated at save time
    frame = 0

    print("\n" + "=" * 60)
    print("  TELEOP + MAPPING")
    print("  ↑↓ = forward/back   ←→ = strafe   Z/X = turn")
    print("  M = save map   ESC = quit")
    print("  (focus the Isaac Sim window for keyboard input)")
    print("=" * 60 + "\n")

    # ---- key capture via Isaac Sim carb keyboard events ----
    import omni.appwindow
    import carb
    _appwin = omni.appwindow.get_default_app_window()
    _keyboard = _appwin.get_keyboard()
    _input = carb.input.acquire_input_interface()
    _pending_action = None  # "save" | "quit" | None

    def _on_key(event, *args, **kwargs):
        nonlocal _pending_action
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.M:
                _pending_action = "save"
            elif event.input == carb.input.KeyboardInput.ESCAPE:
                _pending_action = "quit"

    _key_sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)

    last_save_frame = 0
    try:
        while simulation_app.is_running():
            try:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs = env.step(actions)[0]
            except Exception as e:
                print(f"[ERROR] env.step failed: {e}")
                import traceback; traceback.print_exc()
                break

            # --- LiDAR → map every 5 frames ---
            frame += 1
            if frame % 5 == 0:
                try:
                    lidar = env.unwrapped.scene["mid360_lidar"]
                    robot = env.unwrapped.scene["robot"]
                    if lidar is not None:
                        hits = lidar.data.ray_hits_w[0].cpu().numpy()
                        # keep only finite hits
                        finite = hits[np.isfinite(hits).all(axis=1)]
                        pos = robot.data.root_pos_w[0].cpu().numpy()
                        quat = robot.data.root_quat_w[0].cpu().numpy()
                        yaw = euler_from_quat(quat)
                        if len(finite) > 0:
                            # 2D grid: use all hits (ground provides free-space reference)
                            grid.update((pos[0], pos[1], yaw), finite)
                            # 3D cloud: keep all finite hits (no z filter).
                            # Banana sits on the ground (z≈0.03), so an absolute
                            # z threshold like >0.15 would reject it.  Instead,
                            # save everything; use viz_map.py --no_ground later.
                            if len(finite) > 0:
                                all_points_3d.append(finite[::2])
                            if frame % 100 == 0 and frame > 0:
                                # Count hits near banana position (5, 5) — verify PhysX
                                banana_pos = np.array([5.0, 5.0])
                                near_banana = finite[
                                    (np.abs(finite[:, 0] - banana_pos[0]) < 1.0)
                                    & (np.abs(finite[:, 1] - banana_pos[1]) < 1.0)
                                ]
                                print(f"[DEBUG] frame {frame}: {len(finite)} hits, "
                                      f"z=[{finite[:,2].min():.3f},{finite[:,2].max():.3f}], "
                                      f"near_banana={len(near_banana)}, "
                                      f"total_cloud={sum(len(p) for p in all_points_3d)}")
                except Exception as e:
                    print(f"[WARN] LiDAR read failed (frame {frame}): {e}")

            # Progress every 200 frames
            if frame % 200 == 0:
                pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(f"[INFO] frame {frame}  robot @ ({pos[0]:.1f}, {pos[1]:.1f})")

            # --- handle key actions ---
            if _pending_action == "save":
                os.makedirs(MAPS_DIR, exist_ok=True)
                path = os.path.join(MAPS_DIR, args.map_name + ".npz")
                grid.save(path)
                _save_pointcloud(MAPS_DIR, args.map_name, all_points_3d)
                print(f"[INFO] Map saved → {path}")
                _pending_action = None
            elif _pending_action == "quit":
                print("[INFO] ESC pressed, exiting")
                break

            # Periodic auto-save every 600 frames (~30 s at 20 Hz)
            if frame - last_save_frame > 600:
                os.makedirs(MAPS_DIR, exist_ok=True)
                path = os.path.join(MAPS_DIR, args.map_name + "_auto.npz")
                grid.save(path)
                _save_pointcloud(MAPS_DIR, args.map_name + "_auto", all_points_3d)
                print(f"[INFO] Auto-saved → {path}  (frame {frame})")
                last_save_frame = frame

        print(f"[INFO] Simulation loop ended at frame {frame}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C")
    except Exception as e:
        print(f"\n[ERROR] Unexpected: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
