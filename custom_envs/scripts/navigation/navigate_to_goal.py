#!/usr/bin/env python
"""Load a saved occupancy-grid map and navigate the robot to a goal.

Usage:
    python scripts/navigation/navigate_to_goal.py \
        --task=Flat-Deeprobotics-M20Pro-Lidar-v0 \
        --policy_task=Flat-Deeprobotics-M20-v0 \
        --load_run=2026-07-18_10-57-32 \
        --checkpoint=model_4999.pt \
        --map=maps/my_map.npz \
        --goal 7 7 \
        --target_speed 0.5
"""

import argparse
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser("Navigate to Goal (M20 Pro)")
    parser.add_argument("--task", default="Flat-Deeprobotics-M20Pro-Lidar-v0")
    parser.add_argument("--map", required=True, help="path to .npz map file")
    parser.add_argument("--goal", nargs=2, type=float, required=True,
                        help="goal world coordinates, e.g. --goal 5.0 2.0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_speed", type=float, default=0.8)
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy_task", default=None)
    args, unknown = parser.parse_known_args()

    # ---- Isaac Sim launch ----
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import custom_envs.tasks  # noqa: F401

    from custom_envs.utils.occupancy_grid import OccupancyGrid
    from custom_envs.utils.astar_planner import astar_plan
    from custom_envs.utils.pure_pursuit import PurePursuitController
    from custom_envs.utils.nav_utils import (
        euler_from_quat, is_goal_reached, smooth_path, world_to_grid
    )

    # ---- load map ----
    grid = OccupancyGrid.load(args.map)
    goal_world = (args.goal[0], args.goal[1])

    # ---- build env config ----
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_reset_base = None   # spawn at origin, facing +x
    env_cfg.events.randomize_reset_joints = None
    env_cfg.terminations.time_out = None

    # ---- load policy (RSL-RL standard pattern) ----
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    policy_task = args.policy_task or args.task
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry as load_cfg
    agent_cfg = load_cfg(policy_task, "rsl_rl_cfg_entry_point")

    if "Lidar" not in policy_task and "Lidar" in args.task:
        print("[INFO] Policy has no LiDAR — stripping LiDAR observations from env")
        env_cfg.observations.policy.lidar = None
        env_cfg.observations.critic.lidar = None
        if hasattr(env_cfg.scene, "mid360_lidar"):
            env_cfg.scene.mid360_lidar.debug_vis = False

    # create env
    env = gym.make(args.task, cfg=env_cfg)
    obs = env.reset()[0]  # raw env returns tensor (policy obs group)

    # load checkpoint
    _rl_training_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "deps", "rl_training")
    )
    log_root = os.path.join(_rl_training_root, "logs", "rsl_rl", agent_cfg.experiment_name)

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
                print(f"[ERROR] No runs in {log_root}"); env.close(); simulation_app.close(); return
            run_dir = runs[-1]
        agent_cfg.load_run = os.path.basename(run_dir)
        resume_path = get_checkpoint_path(os.path.abspath(log_root), agent_cfg.load_run, "model_.*.pt")

    print(f"[INFO] Loading policy: {resume_path}")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print("[INFO] Policy loaded")
    print(f"[DEBUG] obs type: {type(obs)}, keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

    # ---- controller ----
    pp = PurePursuitController(target_speed=args.target_speed)
    path_world = None
    replan_counter = 0
    REPLAN_EVERY = 50

    print(f"\n[INFO] Navigate to goal {goal_world}")
    print(f"[INFO] Map: {args.map}\n")

    import time, traceback
    loop_count = 0
    MAX_STEPS = 10000  # ~200 seconds at 0.02s/step, plenty to reach goal
    try:
        while loop_count < MAX_STEPS:
            loop_count += 1
            # get robot pose from unwrapped env
            robot = env.unwrapped.scene["robot"]
            pos_w = robot.data.root_pos_w[0].cpu().numpy()
            quat_w = robot.data.root_quat_w[0].cpu().numpy()
            yaw = euler_from_quat(quat_w)
            robot_pose = (float(pos_w[0]), float(pos_w[1]), yaw)

            # ---- check arrival ----
            if is_goal_reached(robot_pose, goal_world, threshold=0.5):
                print("[INFO] Goal reached!")
                break

            # ---- replan ----
            replan_counter += 1
            if replan_counter % REPLAN_EVERY == 0 or path_world is None:
                bin_map = grid.get_inflated_binary_map(robot_radius=0.4)
                start_rc = world_to_grid(
                    robot_pose[0], robot_pose[1], grid.origin, grid.resolution
                )
                goal_rc = world_to_grid(
                    goal_world[0], goal_world[1], grid.origin, grid.resolution
                )
                # Clear a patch around start/goal so A* has room to plan.
                # Clearing just the single cell isn't enough when the
                # robot is deep in the inflation zone (all neighbours blocked).
                clear_radius = max(1, int(0.3 / grid.resolution))  # ~6 cells
                for rr in range(start_rc[0]-clear_radius, start_rc[0]+clear_radius+1):
                    for cc in range(start_rc[1]-clear_radius, start_rc[1]+clear_radius+1):
                        if 0 <= rr < bin_map.shape[0] and 0 <= cc < bin_map.shape[1]:
                            bin_map[rr, cc] = 0
                for rr in range(goal_rc[0]-clear_radius, goal_rc[0]+clear_radius+1):
                    for cc in range(goal_rc[1]-clear_radius, goal_rc[1]+clear_radius+1):
                        if 0 <= rr < bin_map.shape[0] and 0 <= cc < bin_map.shape[1]:
                            bin_map[rr, cc] = 0
                raw_path = astar_plan(bin_map, start_rc, goal_rc)
                if raw_path is None:
                    roi = bin_map[max(0,start_rc[0]-3):start_rc[0]+4,
                                  max(0,start_rc[1]-3):start_rc[1]+4]
                    print(f"[WARN] A* failed: start={start_rc} "
                          f"nearby_blocked={roi.sum()} pos=({robot_pose[0]:.1f},{robot_pose[1]:.1f})")
                    path_world = None
                else:
                    path_world = [
                        grid.grid_to_world(r, c) for r, c in raw_path
                    ]
                    path_world = smooth_path(path_world)
                    print(f"[INFO] Path planned: {len(path_world)} waypoints")

            # ---- compute control ----
            if path_world is not None:
                vx, omega = pp.compute_velocity(path_world, robot_pose)
            else:
                vx, omega = 0.0, 0.0

            # Inject velocity command into the policy observation.
            # obs is a dict; "policy" key is tensor shape (1, 57).
            # Indices: base_ang_vel(0:3), projected_gravity(3:6),
            # velocity_commands(6:9), joint_pos(9:25), joint_vel(25:41), actions(41:57)
            # Must clone — the tensor is an "inference tensor" (created inside
            # torch.inference_mode) and cannot be modified inplace.
            p_obs = obs["policy"].clone()
            p_obs[0, 6] = vx
            p_obs[0, 7] = 0.0
            p_obs[0, 8] = omega
            obs["policy"] = p_obs

            # ---- send to locomotion ----
            with torch.inference_mode():
                actions = policy(obs)
                step_result = env.step(actions)
                obs = step_result[0]
                print(f"[TRACE] step done, loop={loop_count}", flush=True)

            if loop_count <= 5:
                pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(f"[DEBUG] step {loop_count}: robot=({pos[0]:.1f},{pos[1]:.1f}) "
                      f"vx={vx:.2f} w={omega:.2f} running={simulation_app.is_running()}")

            # ---- save map PNG periodically ----
            if loop_count % 200 == 0 or loop_count == 1:
                _save_nav_png(grid, path_world, robot_pose, goal_world, loop_count)

            # ---- terminal debug ----
            if loop_count % 500 == 0 or loop_count <= 3:
                dist = np.sqrt((robot_pose[0]-goal_world[0])**2 + (robot_pose[1]-goal_world[1])**2)
                print(f"[NAV] step {loop_count}: robot=({robot_pose[0]:.1f},{robot_pose[1]:.1f}) "
                      f"yaw={np.degrees(robot_pose[2]):.0f}deg "
                      f"vx={vx:.2f} w={omega:.3f} dist={dist:.1f}m")

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()


def _save_nav_png(grid, path_world, robot_pose, goal_world, step):
    """Save a PNG snapshot of the navigation map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz = grid.get_visualization()
    # Convert BGR to RGB for matplotlib
    viz_rgb = viz[:, :, ::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(viz_rgb, origin="upper")

    if path_world is not None:
        px, py = [], []
        for wx, wy in path_world:
            r, c = grid.world_to_grid(wx, wy)
            px.append(c); py.append(r)
        ax.plot(px, py, "g-", linewidth=1.0, alpha=0.8)

    rr, rc = grid.world_to_grid(robot_pose[0], robot_pose[1])
    ax.plot(rc, rr, "bo", markersize=8, label="Robot")

    gr, gc = grid.world_to_grid(goal_world[0], goal_world[1])
    ax.plot(gc, gr, "r*", markersize=12, label="Goal")

    ax.set_title(f"Navigation — step {step}")
    ax.legend()

    path = f"/home/mojie/taskdog/custom_envs/maps/nav_step_{step:04d}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[NAV] snapshot saved → {path}")


if __name__ == "__main__":
    main()
