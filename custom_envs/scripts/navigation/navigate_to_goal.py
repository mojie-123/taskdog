#!/usr/bin/env python
"""Load a saved occupancy-grid map and navigate the robot to a goal.

State-machine pipeline
----------------------
  NAV        -- locomotion policy drives robot toward goal
  ALIGN_YAW  -- rotate in-place to yaw=+90deg (head faces world +Y, arm side faces table)
  ARM_INIT   -- move arm to side-facing home pose, joint1=-pi/2 (150 steps)
  SCAN       -- warm camera, accumulate 30 depth frames, build point cloud
  GRASP_PLAN -- run GraspNet worker subprocess, parse best grasp
  PRE_GRASP  -- move arm to pre-grasp waypoint (250 steps)
  ORIENT     -- rotate wrist to grasp orientation (300 steps)
  REACH      -- advance end-effector to grasp translation (150 steps)
  CLOSE      -- close gripper fingers (40 steps)
  LIFT       -- lift arm 0.15 m (200 steps)
  DONE       -- stop

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
import enum
import math
import os
import sys
import traceback

import numpy as np
import torch


class PipelineState(enum.Enum):
    NAV          = "NAV"          # locomotion: drive to goal
    ALIGN_YAW_1  = "ALIGN_YAW_1"  # rotate in-place to target yaw before PAN
    PAN_VX       = "PAN_VX"       # forward/back only: align world-Y to goal Y
    PAN_VY       = "PAN_VY"       # strafe only: align world-X to goal X
    ALIGN_YAW    = "ALIGN_YAW"    # fine-tune rotate in-place to target yaw (+Y)
    ARM_INIT     = "ARM_INIT"     # retract arm to home
    SCAN       = "SCAN"       # accumulate depth frames
    GRASP_PLAN = "GRASP_PLAN" # run GraspNet
    PRE_GRASP  = "PRE_GRASP"  # arm to pre-grasp
    ORIENT     = "ORIENT"     # wrist orientation
    REACH      = "REACH"      # advance to grasp translation
    CLOSE      = "CLOSE"      # close gripper
    LIFT       = "LIFT"       # lift object
    DONE       = "DONE"       # finished


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
    # If --checkpoint is an existing absolute file path, use it directly
    # (bypasses experiment_name / load_run registry lookup).
    _rl_training_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "deps", "rl_training")
    )
    log_root = os.path.join(_rl_training_root, "logs", "rsl_rl", agent_cfg.experiment_name)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        # Absolute path supplied directly -- skip registry lookup.
        resume_path = args.checkpoint
    elif args.checkpoint:
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

    # ---- pure-pursuit controller (used in NAV state) ----
    pp = PurePursuitController(target_speed=args.target_speed)

    # ---- IK utils path ----
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))

    ARM_JOINT_NAMES   = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_OPEN_POS  = {"joint7":  0.0,   "joint8":  0.0}
    GRIPPER_CLOSE_POS = {"joint7":  0.035, "joint8": -0.035}
    # ARM_HOME_ANGLES: arm tucked facing robot front (+X body), used during walking
    ARM_HOME_ANGLES   = np.array([0.0,     0.5, -1.0, 0.0, 0.5, 0.0], dtype=np.float32)
    # ARM_SIDE_ANGLES: joint1 rotated -π/2 so arm faces robot right-side (-Y body).
    # When dog stops with yaw=+π/2 (head → world +Y), robot -Y = world +X → arm
    # points toward the table.  Used as the home pose during ARM_INIT/grasp phases.
    ARM_SIDE_ANGLES   = np.array([-math.pi/2, 0.5, -1.0, 0.0, 1.0, 0], dtype=np.float32)
    # Target yaw for the dog after reaching the goal: head faces world +Y (+π/2).
    # The table is at world +X from the goal, so robot -Y side (arm side) faces it.
    TARGET_YAW        = math.pi / 2   # +90 degrees

    BUDGET = {
        PipelineState.ARM_INIT:  150,
        PipelineState.PRE_GRASP: 250,
        PipelineState.ORIENT:    300,
        PipelineState.REACH:     150,
        PipelineState.CLOSE:      40,
        PipelineState.LIFT:      200,
    }
    SCAN_WARMUP  = 10
    SCAN_FRAMES  = 30

    state          = PipelineState.NAV
    state_step     = 0
    path_world     = None
    replan_counter = 0
    REPLAN_EVERY   = 50
    grasp_result   = None
    arm_joint_ids  = None
    gripper_ids    = None
    depth_accum    = []
    scan_rgb       = None
    target_angles_arm = None

    print(f"\n[INFO] Navigate to goal {goal_world}")
    print(f"[INFO] Map: {args.map}")
    print(f"[INFO] Grasp checkpoint: {args.grasp_checkpoint}\n")

    def _get_arm_ids(robot):
        nonlocal arm_joint_ids, gripper_ids
        if arm_joint_ids is not None:
            return arm_joint_ids, gripper_ids
        arm_joint_ids = []
        for name in ARM_JOINT_NAMES:
            try:
                arm_joint_ids.append(robot.find_joints(name)[0][0])
            except Exception:
                pass
        gripper_ids = {}
        for name in list(GRIPPER_OPEN_POS) + list(GRIPPER_CLOSE_POS):
            try:
                gripper_ids[name] = robot.find_joints(name)[0][0]
            except Exception:
                pass
        if len(arm_joint_ids) != 6:
            print(f"[WARN] Only {len(arm_joint_ids)}/6 arm joints found")
        return arm_joint_ids, gripper_ids

    def _arm_step(robot, q6):
        ids, _ = _get_arm_ids(robot)
        if len(ids) != 6:
            return
        pos_t = robot.data.joint_pos_target[0].clone()
        for i, jid in enumerate(ids):
            pos_t[jid] = float(q6[i])
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _gripper_step(robot, close=False):
        _, gids = _get_arm_ids(robot)
        targets = GRIPPER_CLOSE_POS if close else GRIPPER_OPEN_POS
        pos_t = robot.data.joint_pos_target[0].clone()
        for name, val in targets.items():
            if name in gids:
                pos_t[gids[name]] = float(val)
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _alpha(s):
        budget = BUDGET.get(s, 1)
        return min(1.0, state_step / max(budget - 1, 1))

    loop_count = 0
    MAX_STEPS  = 20000
    try:
        while loop_count < MAX_STEPS and state != PipelineState.DONE:
            loop_count += 1
            state_step += 1

            raw_env    = env.unwrapped
            robot      = raw_env.scene["robot"]
            pos_w      = robot.data.root_pos_w[0].cpu().numpy()
            quat_w     = robot.data.root_quat_w[0].cpu().numpy()
            yaw        = euler_from_quat(quat_w)
            robot_pose = (float(pos_w[0]), float(pos_w[1]), yaw)

            # ==============================================================
            # STATE MACHINE
            # ==============================================================

            # ---- NAV: locomotion policy drives robot to goal ----
            if state == PipelineState.NAV:
                replan_counter += 1
                if replan_counter % REPLAN_EVERY == 0 or path_world is None:
                    bin_map  = grid.get_inflated_binary_map(robot_radius=0.15)
                    start_rc = world_to_grid(
                        robot_pose[0], robot_pose[1], grid.origin, grid.resolution)
                    goal_rc  = world_to_grid(
                        goal_world[0], goal_world[1], grid.origin, grid.resolution)
                    clear_r  = max(1, int(0.3 / grid.resolution))
                    for rr in range(start_rc[0]-clear_r, start_rc[0]+clear_r+1):
                        for cc in range(start_rc[1]-clear_r, start_rc[1]+clear_r+1):
                            if 0 <= rr < bin_map.shape[0] and 0 <= cc < bin_map.shape[1]:
                                bin_map[rr, cc] = 0
                    for rr in range(goal_rc[0]-clear_r, goal_rc[0]+clear_r+1):
                        for cc in range(goal_rc[1]-clear_r, goal_rc[1]+clear_r+1):
                            if 0 <= rr < bin_map.shape[0] and 0 <= cc < bin_map.shape[1]:
                                bin_map[rr, cc] = 0
                    raw_path = astar_plan(bin_map, start_rc, goal_rc)
                    if raw_path is None:
                        roi = bin_map[max(0,start_rc[0]-3):start_rc[0]+4,
                                      max(0,start_rc[1]-3):start_rc[1]+4]
                        print(f"[NAV] A* failed start={start_rc} nearby={roi.sum()} "
                              f"pos=({robot_pose[0]:.1f},{robot_pose[1]:.1f})", flush=True)
                        path_world = None
                    else:
                        path_world = [grid.grid_to_world(r, c) for r, c in raw_path]
                        path_world = smooth_path(path_world)
                        print(f"[NAV] Path: {len(path_world)} wpts", flush=True)

                dist_to_goal = np.hypot(robot_pose[0]-goal_world[0], robot_pose[1]-goal_world[1])

                if dist_to_goal <= 1.5:
                    # ---- Terminal PD controller (replaces Pure Pursuit within 1.5 m) ----
                    # Compute angle from robot to goal in world frame
                    angle_to_goal = math.atan2(
                        goal_world[1] - robot_pose[1],
                        goal_world[0] - robot_pose[0]
                    )
                    # Angular error: difference between heading-to-goal and current yaw
                    angle_err = (angle_to_goal - yaw + math.pi) % (2 * math.pi) - math.pi
                    # Linear speed: proportional to dist, capped at 0.4 m/s, zero when very close
                    vx    = float(np.clip(1.2 * dist_to_goal, 0.0, 0.4))
                    # Only move forward if roughly facing the goal (|err| < 60°)
                    if abs(angle_err) > math.radians(60):
                        vx = 0.0
                    omega = float(np.clip(2.0 * angle_err, -1.5, 1.5))
                else:
                    # ---- Pure Pursuit for long-range navigation ----
                    if path_world is not None:
                        vx, omega = pp.compute_velocity(path_world, robot_pose)
                    else:
                        vx, omega = 0.0, 0.0
                    # ---- Blend in yaw-alignment heading correction as we near goal ----
                    # Within 3 m, linearly increase weight of heading PD toward TARGET_YAW
                    # so the dog arrives already roughly facing +Y (head toward +Y).
                    if dist_to_goal < 3.0:
                        yaw_err_nav   = (TARGET_YAW - yaw + math.pi) % (2*math.pi) - math.pi
                        heading_w     = float(np.clip(1.0 - dist_to_goal / 3.0, 0.0, 1.0))
                        omega_heading = float(np.clip(0.8 * yaw_err_nav, -1.2, 1.2))
                        omega = (1.0 - heading_w) * omega + heading_w * omega_heading

                p_obs = obs["policy"].clone()
                p_obs[0, 6] = vx
                p_obs[0, 7] = 0.0
                p_obs[0, 8] = omega
                obs["policy"] = p_obs
                with torch.inference_mode():
                    actions  = policy(obs)
                    step_res = env.step(actions)
                    obs      = step_res[0]

                if loop_count % 100 == 0 or loop_count <= 3:
                    print(f"[NAV] step {loop_count}: pos=({robot_pose[0]:.1f},{robot_pose[1]:.1f}) "
                          f"yaw={np.degrees(yaw):.0f}° vx={vx:.2f} w={omega:.3f} "
                          f"dist={dist_to_goal:.1f}m", flush=True)
                if loop_count % 200 == 0 or loop_count == 1:
                    _save_nav_png(grid, path_world, robot_pose, goal_world, loop_count)

                if is_goal_reached(robot_pose, goal_world, threshold=0.6):
                    print(f"[NAV] Within 0.6 m — stopping. "
                          f"pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"yaw={np.degrees(yaw):.0f}° dist={dist_to_goal:.3f}m", flush=True)
                    if args.grasp_checkpoint:
                        print("[SM] NAV -> ALIGN_YAW_1", flush=True)
                        state = PipelineState.ALIGN_YAW_1
                        state_step = 0
                    else:
                        print("[SM] No grasp checkpoint — DONE.", flush=True)
                        state = PipelineState.DONE
                continue  # NAV drives sim via env.step()

            # ---- From ALIGN_YAW_1/PAN/ALIGN_YAW/ARM_INIT onwards: policy keeps robot standing.
            #      ALIGN_YAW_1 and ALIGN_YAW drive omega; PAN drives vx+vy; others zero vel.
            if state != PipelineState.GRASP_PLAN:
                p_obs = obs["policy"].clone()
                if state in (PipelineState.ALIGN_YAW_1, PipelineState.ALIGN_YAW):
                    _yaw_err = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                    p_obs[0, 6] = 0.0
                    p_obs[0, 7] = 0.0
                    p_obs[0, 8] = float(np.clip(100.0 * _yaw_err, -1.2, 1.2))
                elif state in (PipelineState.PAN_VX, PipelineState.PAN_VY):
                    # World-frame position error
                    _dx_w = goal_world[0] - robot_pose[0]
                    _dy_w = goal_world[1] - robot_pose[1]
                    _cy, _sy = math.cos(yaw), math.sin(yaw)
                    if state == PipelineState.PAN_VX:
                        # Only vx: align world-Y. body_vx ≈ dy_world when yaw≈+π/2
                        _bvx  = _dx_w * _sy + _dy_w * _cy
                        _err  = abs(_dy_w)
                        _v_cap = float(np.clip(0.4 * _err, 0.0, 0.3))
                        p_obs[0, 6] = float(np.clip(math.copysign(_v_cap, _bvx), -0.3, 0.3))
                        p_obs[0, 7] = 0.0
                    else:  # PAN_VY
                        # Only vy: align world-X. body_vy ≈ -dx_world when yaw≈+π/2
                        # Note: dx_w>0 means goal is at +X, body_vy should be negative (right)
                        # so negate: vy = -copysign(v_cap, dx_w)
                        _err  = abs(_dx_w)
                        _v_cap = float(np.clip(2.0 * _err, 0.0, 0.3))
                        p_obs[0, 6] = 0.0
                        p_obs[0, 7] = float(np.clip(-math.copysign(_v_cap, _dx_w), -0.3, 0.3))
                    p_obs[0, 8] = 0.0
                else:
                    p_obs[0, 6] = 0.0
                    p_obs[0, 7] = 0.0
                    p_obs[0, 8] = 0.0
                obs["policy"] = p_obs
                with torch.inference_mode():
                    actions  = policy(obs)
                    step_res = env.step(actions)
                    obs      = step_res[0]

            # ---- ALIGN_YAW_1: rotate in-place to TARGET_YAW before PAN ----
            if state == PipelineState.ALIGN_YAW_1:
                _yaw_err_1 = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                if state_step == 1:
                    print(f"[SM] ALIGN_YAW_1 start: target=+90deg "
                          f"cur={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_1):.1f}deg", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] ALIGN_YAW_1 step {state_step}: "
                          f"yaw={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_1):.1f}deg", flush=True)
                if abs(_yaw_err_1) < 0.01 or state_step >= 400:
                    print(f"[SM] ALIGN_YAW_1 done "
                          f"(yaw={math.degrees(yaw):.1f}deg) -> PAN_VX", flush=True)
                    state = PipelineState.PAN_VX
                    state_step = 0

            # ---- PAN_VX: vx only, align world-Y coordinate to goal ----
            elif state == PipelineState.PAN_VX:
                _dy_err = abs(robot_pose[1] - goal_world[1])
                if state_step == 1:
                    print(f"[SM] PAN_VX start: pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"dy_err={_dy_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VX step {state_step}: "
                          f"pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"dy_err={_dy_err:.3f}m", flush=True)
                if _dy_err < 0.05 or state_step >= 400:
                    print(f"[SM] PAN_VX done "
                          f"(dy_err={_dy_err:.3f}m step={state_step}) -> PAN_VY",
                          flush=True)
                    state = PipelineState.PAN_VY
                    state_step = 0

            # ---- PAN_VY: vy only, align world-X coordinate to goal ----
            elif state == PipelineState.PAN_VY:
                _dx_err = abs(robot_pose[0] - goal_world[0])
                if state_step == 1:
                    print(f"[SM] PAN_VY start: pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"dx_err={_dx_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VY step {state_step}: "
                          f"pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"dx_err={_dx_err:.3f}m", flush=True)
                if _dx_err < 0.1 or state_step >= 400:
                    print(f"[SM] PAN_VY done "
                          f"(dx_err={_dx_err:.3f}m step={state_step}) -> ALIGN_YAW",
                          flush=True)
                    print(f"[SM] PAN_VY step {state_step}: "
                          f"pos=({robot_pose[0]:.3f},{robot_pose[1]:.3f}) "
                          f"dx_err={_dx_err:.3f}m", flush=True)
                    state = PipelineState.ALIGN_YAW
                    state_step = 0

            # ---- ALIGN_YAW: fine-tune rotation after PAN ----
            elif state == PipelineState.ALIGN_YAW:
                _yaw_err_align = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                if state_step == 1:
                    print(f"[SM] ALIGN_YAW start: target=+90deg "
                          f"cur={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_align):.1f}deg", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] ALIGN_YAW step {state_step}: "
                          f"yaw={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_align):.1f}deg", flush=True)
                if abs(_yaw_err_align) < 0.01 or state_step >= 400:
                    print(f"[SM] ALIGN_YAW done "
                          f"(yaw={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_align):.1f}deg) -> ARM_INIT",
                          flush=True)
                    state = PipelineState.ARM_INIT
                    state_step = 0

            # ---- ARM_INIT: retract arm to side pose (joint1=-pi/2) ----
            elif state == PipelineState.ARM_INIT:
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                alpha = _alpha(state)
                q6    = (1.0 - alpha) * cur_q + alpha * ARM_SIDE_ANGLES
                _arm_step(robot, q6)
                _gripper_step(robot, close=False)
                if state_step == 1:
                    print("[SM] ARM_INIT: retracting arm...", flush=True)
                    # Print banana actual resting position; sim ran during NAV
                    # so the banana has already settled on the table surface.
                    try:
                        banana = raw_env.scene["banana"]
                        bpos = banana.data.root_pos_w[0].cpu().numpy()
                        print(
                            f"[INFO] Banana resting pos: "
                            f"x={bpos[0]:.4f}  y={bpos[1]:.4f}  z={bpos[2]:.4f}",
                            flush=True,
                        )
                    except Exception as _be:
                        print(f"[INFO] Could not read banana pos: {_be}", flush=True)
                if state_step >= BUDGET[PipelineState.ARM_INIT]:
                    print("[SM] ARM_INIT done -> SCAN", flush=True)
                    state = PipelineState.SCAN
                    state_step = 0
                    depth_accum = []
                    scan_rgb    = None

            # ---- SCAN: warm camera, accumulate depth frames ----
            elif state == PipelineState.SCAN:
                try:
                    camera = raw_env.scene["wrist_camera"]
                    if state_step == 1:
                        print(f"[SM] SCAN: warmup {SCAN_WARMUP} + accumulate "
                              f"{SCAN_FRAMES} frames...", flush=True)
                    if state_step > SCAN_WARMUP:
                        d = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                        if d.ndim == 3:
                            d = d[:, :, 0]
                        depth_accum.append(d.astype(np.float32))
                        if scan_rgb is None:
                            scan_rgb = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                    if state_step >= SCAN_WARMUP + SCAN_FRAMES:
                        depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)
                        valid_pct = np.mean((depth_med > 0.05) & (depth_med < 4.0)) * 100
                        print(f"[SM] SCAN done: {len(depth_accum)} frames, "
                              f"valid depth {valid_pct:.1f}%", flush=True)
                        state = PipelineState.GRASP_PLAN
                        state_step = 0
                except (KeyError, AttributeError) as e:
                    print(f"[SM] SCAN camera unavailable ({e}) — DONE.", flush=True)
                    state = PipelineState.DONE

            # ---- GRASP_PLAN: build point cloud, run GraspNet ----
            elif state == PipelineState.GRASP_PLAN:
                import subprocess, sys as _sys
                print("[SM] GRASP_PLAN: building point cloud...", flush=True)
                depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)
                H, W = depth_med.shape
                fx_c, fy_c, cx_c, cy_c = 616.0, 616.0, W/2.0, H/2.0
                u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
                z    = depth_med
                mask = (z > 0.05) & (z < 4.0)
                z_v  = z[mask]
                pts  = np.stack([
                    (u_g[mask] - cx_c) * z_v / fx_c,
                    (v_g[mask] - cy_c) * z_v / fy_c,
                    z_v,
                ], axis=-1).astype(np.float32)
                rgb_u = scan_rgb if scan_rgb is not None else np.zeros((H,W,3), np.uint8)
                cols  = (rgb_u[mask] / 255.0).astype(np.float32)
                print(f"[SM] Point cloud: {len(pts)} pts "
                      f"(z min={z_v.min():.2f} max={z_v.max():.2f})", flush=True)
                if len(pts) < 200:
                    print("[SM] Too few points — DONE.", flush=True)
                    state = PipelineState.DONE
                else:
                    np.savez("/tmp/pointcloud.npz", points=pts, colors=cols)
                    worker = os.path.join(os.path.dirname(__file__), "grasp_worker.py")
                    topk   = getattr(args, "grasp_topk", 1)
                    print("[SM] Running grasp_worker...", flush=True)
                    res = subprocess.run(
                        [_sys.executable, worker,
                         "--checkpoint", args.grasp_checkpoint,
                         "--topk", str(topk)],
                        timeout=120,
                    )
                    if res.returncode != 0 or not os.path.exists("/tmp/grasp_result.npz"):
                        print("[SM] GraspNet failed — DONE.", flush=True)
                        state = PipelineState.DONE
                    else:
                        gr = np.load("/tmp/grasp_result.npz")
                        if len(gr["scores"]) == 0:
                            print("[SM] No valid grasps — DONE.", flush=True)
                            state = PipelineState.DONE
                        else:
                            grasp_result = {
                                "t_cam": gr["translations"][0],
                                "R_cam": gr["rotations"][0],
                                "score": float(gr["scores"][0]),
                                "width": float(gr["widths"][0]),
                            }
                            print(f"[SM] Best grasp score={grasp_result['score']:.3f} "
                                  f"t={np.round(grasp_result['t_cam'],3)}", flush=True)
                            state = PipelineState.PRE_GRASP
                            state_step = 0

            # ---- PRE_GRASP: move arm to pre-grasp pose (10cm back) ----
            elif state == PipelineState.PRE_GRASP:
                from arm_ik import solve as ik_solve, world_pos_to_arm_frame
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    t_arm = world_pos_to_arm_frame(t_cam, pos_w, quat_w)
                    pre_t = t_arm - np.array([0.0, 0.0, 0.10])
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    target_angles_arm = ik_solve(pre_t, initial_angles=cur_q)
                    print(f"[SM] PRE_GRASP IK: {np.round(target_angles_arm,3)}", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * target_angles_arm
                _arm_step(robot, q6)
                if state_step >= BUDGET[PipelineState.PRE_GRASP]:
                    print("[SM] PRE_GRASP done -> ORIENT", flush=True)
                    state = PipelineState.ORIENT
                    state_step = 0

            # ---- ORIENT: IK with full orientation to align wrist ----
            elif state == PipelineState.ORIENT:
                from arm_ik import solve as ik_solve, world_pos_to_arm_frame
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    R_cam = grasp_result["R_cam"]
                    t_arm = world_pos_to_arm_frame(t_cam, pos_w, quat_w)
                    pre_t = t_arm - np.array([0.0, 0.0, 0.10])
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    target_angles_arm = ik_solve(pre_t, target_rot=R_cam, initial_angles=cur_q)
                    print(f"[SM] ORIENT IK: {np.round(target_angles_arm,3)}", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * target_angles_arm
                _arm_step(robot, q6)
                if state_step >= BUDGET[PipelineState.ORIENT]:
                    print("[SM] ORIENT done -> REACH", flush=True)
                    state = PipelineState.REACH
                    state_step = 0

            # ---- REACH: advance EE to grasp translation ----
            elif state == PipelineState.REACH:
                from arm_ik import solve as ik_solve, world_pos_to_arm_frame
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    R_cam = grasp_result["R_cam"]
                    t_arm = world_pos_to_arm_frame(t_cam, pos_w, quat_w)
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    target_angles_arm = ik_solve(t_arm, target_rot=R_cam, initial_angles=cur_q)
                    print(f"[SM] REACH IK: {np.round(target_angles_arm,3)}", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * target_angles_arm
                _arm_step(robot, q6)
                if state_step >= BUDGET[PipelineState.REACH]:
                    print("[SM] REACH done -> CLOSE", flush=True)
                    state = PipelineState.CLOSE
                    state_step = 0

            # ---- CLOSE: close gripper ----
            elif state == PipelineState.CLOSE:
                _gripper_step(robot, close=True)
                if state_step == 1:
                    print("[SM] CLOSE: closing gripper...", flush=True)
                if state_step >= BUDGET[PipelineState.CLOSE]:
                    print("[SM] CLOSE done -> LIFT", flush=True)
                    state = PipelineState.LIFT
                    state_step = 0

            # ---- LIFT: raise arm 0.15 m ----
            elif state == PipelineState.LIFT:
                from arm_ik import solve as ik_solve, world_pos_to_arm_frame
                if state_step == 1:
                    t_cam  = grasp_result["t_cam"]
                    t_arm  = world_pos_to_arm_frame(t_cam, pos_w, quat_w)
                    lift_t = t_arm + np.array([0.0, 0.0, 0.15])
                    cur_q  = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    target_angles_arm = ik_solve(lift_t, initial_angles=cur_q)
                    print(f"[SM] LIFT IK: {np.round(target_angles_arm,3)}", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * target_angles_arm
                _arm_step(robot, q6)
                _gripper_step(robot, close=True)
                if state_step >= BUDGET[PipelineState.LIFT]:
                    print("[SM] LIFT done -> DONE", flush=True)
                    state = PipelineState.DONE

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        env.close()
        simulation_app.close()


def _phase2_grasp(env, args, robot_pose):
    """[DEPRECATED] Replaced by state-machine SCAN/GRASP_PLAN states.

    Kept for backward-compatibility. The state machine handles all Phase 2
    logic inline; this function is no longer called.
    """
    print("[Phase2] _phase2_grasp() called (deprecated, state machine handles this).")
    return


def _phase2_grasp_legacy(env, args, robot_pose):
    """Legacy Phase 2 implementation (archived for reference)."""
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

    # IsaacLab Camera output shape: (H, W) or (H, W, 1) — squeeze to 2D
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    H, W = depth.shape
    fx, fy, cx, cy = 616.0, 616.0, W / 2.0, H / 2.0
    u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float32)
    valid = (z > 0.05) & (z < 4.0)   # relaxed from 2.0 → 4.0
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
