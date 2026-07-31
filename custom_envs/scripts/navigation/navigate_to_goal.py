#!/usr/bin/env python
"""Load a saved occupancy-grid map and navigate the robot to a goal.

Usage:
    python scripts/navigation/navigate_to_goal.py \\
        --task=Flat-Deeprobotics-M20Pro-Lidar-v0 \\
        --map=maps/my_map.npz \\
        --goal 5.0 2.0
"""

import argparse
import os
import sys

import cv2
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
    parser.add_argument("--load_run", default=None, help="run dir (default: latest)")
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
    from custom_envs.utils.astar_planner import astar_plan
    from custom_envs.utils.pure_pursuit import PurePursuitController
    from custom_envs.utils.nav_utils import (
        euler_from_quat, is_goal_reached, smooth_path, world_to_grid
    )

    # ---- load map ----
    grid = OccupancyGrid.load(args.map)
    goal_world = (args.goal[0], args.goal[1])

    # ---- env ----
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.terminations.time_out = None

    # ---- load locomotion policy ----
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    policy_task = args.policy_task or args.task
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry as load_cfg
    agent_cfg = load_cfg(policy_task, "rsl_rl_cfg_entry_point")

    # If policy was trained WITHOUT LiDAR, strip LiDAR obs from sim env
    if "Lidar" not in policy_task and "Lidar" in args.task:
        print("[INFO] Policy has no LiDAR — stripping LiDAR observations from env")
        env_cfg.observations.policy.lidar = None
        env_cfg.observations.critic.lidar = None
        if hasattr(env_cfg.scene, "mid360_lidar"):
            env_cfg.scene.mid360_lidar.debug_vis = False

    env = gym.make(args.task, cfg=env_cfg)
    obs = env.reset()[0]
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

    # ---- controller ----
    pp = PurePursuitController(target_speed=args.target_speed)
    path_world = None
    replan_counter = 0
    REPLAN_EVERY = 50

    print(f"\n[INFO] Navigate to goal {goal_world}")
    print(f"[INFO] Map: {args.map}\n")

    try:
        while simulation_app.is_running():
            robot = env.unwrapped.scene["robot"]
            pos_w = robot.data.root_pos_w[0].cpu().numpy()
            quat_w = robot.data.root_quat_w[0].cpu().numpy()
            yaw = euler_from_quat(quat_w)
            robot_pose = (float(pos_w[0]), float(pos_w[1]), yaw)

            # ---- check arrival ----
            if is_goal_reached(robot_pose, goal_world, threshold=0.5):
                print("[INFO] Goal reached!")
                cv2.waitKey(2000)
                break

            # ---- replan ----
            replan_counter += 1
            if replan_counter % REPLAN_EVERY == 0 or path_world is None:
                bin_map = grid.get_binary_map()
                start_rc = world_to_grid(
                    robot_pose[0], robot_pose[1], grid.origin, grid.resolution
                )
                goal_rc = world_to_grid(
                    goal_world[0], goal_world[1], grid.origin, grid.resolution
                )
                raw_path = astar_plan(bin_map, start_rc, goal_rc)
                if raw_path is None:
                    print("[WARN] A* returned no path — retrying next cycle")
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

            # ---- send to locomotion (via observation override) ----
            with torch.inference_mode():
                actions = policy(obs)
                obs = env.step(actions)[0]

            # ---- visualisation ----
            viz = grid.get_visualization()
            # draw path
            if path_world is not None:
                for wx, wy in path_world:
                    r, c = grid.world_to_grid(wx, wy)
                    if 0 <= r < grid.height and 0 <= c < grid.width:
                        cv2.circle(viz, (c, r), 1, (0, 255, 0), -1)
            # draw robot
            rr, rc = grid.world_to_grid(robot_pose[0], robot_pose[1])
            cv2.circle(viz, (rc, rr), 4, (255, 0, 0), -1)
            # draw goal
            gr, gc = grid.world_to_grid(goal_world[0], goal_world[1])
            cv2.circle(viz, (gc, gr), 5, (0, 0, 255), -1)

            status = f"vx={vx:.2f} ω={omega:.2f}  path={'OK' if path_world else 'NONE'}"
            cv2.putText(viz, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("Navigation", viz)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
