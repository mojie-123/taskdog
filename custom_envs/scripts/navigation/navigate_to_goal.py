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
    parser.add_argument("--grasp_checkpoint", default=None,
                        help="Path to graspnet checkpoint-rs.tar (enables Phase 2 grasp)")
    parser.add_argument("--grasp_topk", type=int, default=1,
                        help="Number of top grasps to attempt (default 1)")
    args, unknown = parser.parse_known_args()

    # ---- Isaac Sim launch ----
    from isaaclab.app import AppLauncher
    # Enable camera rendering when a grasp checkpoint is provided (wrist camera
    # is required for Phase 2 point cloud acquisition).
    if args.grasp_checkpoint is not None:
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import custom_envs.tasks  # noqa: F401

    # Only the legacy dual-articulation Piper-v0 task needs the sub-step sync
    # callback. Single-v0 embeds Piper inside the robot articulation itself and
    # has no separate 'piper' scene entity — calling setup_piper_sync on it
    # raises a KeyError.
    _piper_mode = "Piper" in args.task and "Piper-Single" not in args.task
    if _piper_mode:
        from custom_envs.tasks.deeprobotics_m20_pro.piper_env_cfg import setup_piper_sync

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

    # ---- register Piper sub-step sync ----
    if _piper_mode:
        setup_piper_sync(env)

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
                print("[INFO] Goal reached! Starting Phase 2 (grasp)...")
                _phase2_grasp(env, args, robot_pose)
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


def _phase2_grasp(env, args, robot_pose):
    """Phase 2: wrist camera → grasp detection → IK → execute arm."""
    import subprocess
    import sys as _sys

    checkpoint = getattr(args, "grasp_checkpoint", None)
    if checkpoint is None:
        print("[Phase2] No --grasp_checkpoint specified, skipping grasp.")
        return
    if not os.path.exists(checkpoint):
        print(f"[Phase2] Checkpoint not found: {checkpoint}")
        return

    print("[Phase2] Reading wrist camera...")
    raw_env = env.unwrapped
    raw_env.sim.step()
    raw_env.sim.render()

    try:
        camera = raw_env.scene["wrist_camera"]
        rgb   = camera.data.output["rgb"][0].cpu().numpy()
        depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
    except (KeyError, AttributeError) as e:
        print(f"[Phase2] Camera not available: {e}. Skipping.")
        return

    H, W = depth.shape
    fx, fy, cx, cy = 616.0, 616.0, W / 2.0, H / 2.0
    u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)
    valid = (z > 0.05) & (z < 2.0)
    z_v = z[valid]
    points = np.stack([
        (u_g[valid] - cx) * z_v / fx,
        (v_g[valid] - cy) * z_v / fy,
        z_v,
    ], axis=-1).astype(np.float32)
    colors = (rgb[:, :, :3][valid] / 255.0).astype(np.float32)
    print(f"[Phase2] Point cloud: {len(points)} pts")

    if len(points) < 200:
        print("[Phase2] Too few points, skipping grasp.")
        return

    np.savez("/tmp/pointcloud.npz", points=points, colors=colors)

    worker = os.path.join(os.path.dirname(__file__), "grasp_worker.py")
    topk = getattr(args, "grasp_topk", 1)
    print("[Phase2] Running grasp_worker subprocess...")
    result = subprocess.run(
        [_sys.executable, worker,
         "--checkpoint", checkpoint,
         "--topk", str(topk)],
        timeout=120,
    )
    if result.returncode != 0:
        print(f"[Phase2] grasp_worker failed (code {result.returncode})")
        return

    if not os.path.exists("/tmp/grasp_result.npz"):
        print("[Phase2] No grasp result file.")
        return
    gr = np.load("/tmp/grasp_result.npz")
    if len(gr["scores"]) == 0:
        print("[Phase2] No valid grasps.")
        return

    t_cam = gr["translations"][0]
    R_cam = gr["rotations"][0]
    score = gr["scores"][0]
    width = gr["widths"][0]
    print(f"[Phase2] Best grasp score={score:.3f} t={np.round(t_cam,3)} w={width:.3f}")

    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
    from arm_ik import solve as ik_solve, world_pos_to_arm_frame

    robot = raw_env.scene["robot"]
    pos_w  = robot.data.root_pos_w[0].cpu().numpy()
    quat_w = robot.data.root_quat_w[0].cpu().numpy()
    t_arm = world_pos_to_arm_frame(t_cam, pos_w, quat_w)

    try:
        joint_angles = ik_solve(t_arm, target_rot=R_cam)
    except Exception as e:
        print(f"[Phase2] IK failed: {e}")
        return
    print(f"[Phase2] IK angles: {np.round(joint_angles, 3)}")

    _execute_arm(raw_env, joint_angles)
    _close_gripper(raw_env)
    print("[Phase2] Grasp complete!")


def _execute_arm(raw_env, target_angles, n_steps=100):
    """Interpolate arm to target_angles over n_steps."""
    import torch
    robot = raw_env.scene["robot"]
    ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    joint_ids = []
    for name in ARM_JOINTS:
        try:
            joint_ids.append(robot.find_joints(name)[0][0])
        except Exception:
            pass
    if len(joint_ids) != 6:
        print(f"[Phase2] Found {len(joint_ids)}/6 arm joints, skipping.")
        return
    current = robot.data.joint_pos[0, joint_ids].cpu().numpy()
    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interp = (1 - alpha) * current + alpha * target_angles
        pos_t = robot.data.joint_pos_target[0].clone()
        for i, jid in enumerate(joint_ids):
            pos_t[jid] = interp[i]
        robot.set_joint_position_target(pos_t.unsqueeze(0))
        raw_env.sim.step()
        raw_env.sim.render()


def _close_gripper(raw_env, n_steps=50):
    """Close Piper gripper."""
    import torch
    robot = raw_env.scene["robot"]
    GRIPPER = {"joint7": 0.035, "joint8": -0.035}
    joint_ids, targets = [], []
    for name, val in GRIPPER.items():
        try:
            joint_ids.append(robot.find_joints(name)[0][0])
            targets.append(val)
        except Exception:
            pass
    if not joint_ids:
        print("[Phase2] Gripper joints not found.")
        return
    for _ in range(n_steps):
        pos_t = robot.data.joint_pos_target[0].clone()
        for jid, val in zip(joint_ids, targets):
            pos_t[jid] = val
        robot.set_joint_position_target(pos_t.unsqueeze(0))
        raw_env.sim.step()
        raw_env.sim.render()


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

    path = f"/home/mojie/taskdog/custom_envs/maps/nav_process/nav_step_{step:04d}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[NAV] snapshot saved → {path}")


if __name__ == "__main__":
    main()
