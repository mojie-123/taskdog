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

机器狗走到合适位置：        
python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 4.5 4.9 --target_speed 1.0 --grasp_checkpoint /home/mojie/graspnet-baseline/logs/checkpoint-rs.tar --enable_cameras True

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
    GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
    GRIPPER_OPEN_POS    = [ 0.035, -0.035]   # fingers extended = fully open
    GRIPPER_CLOSE_POS   = [ 0.0,    0.0  ]   # fingers fully closed (GraspNet width used dynamically)
    # Pre-grasp retreat distance (along GraspNet approach axis, in camera frame).
    # Piper gripper_base → fingertip ≈ 0.19 m; GraspNet t is the ideal gripper root.
    # Retreat 15 cm so fingertips clear the object during ORIENT rotation.
    PRE_GRASP_RETREAT   = 0.1
    # REACH target: stop 5 cm short of GraspNet t so fingertips lightly contact the object.
    REACH_RETREAT       = 0.05
    # ARM_HOME_ANGLES: arm tucked facing robot front (+X body), used during walking
    ARM_HOME_ANGLES   = np.array([0.0,     0.5, -1.0, 0.0, 0.5, 0.0], dtype=np.float32)
    # ARM_SIDE_ANGLES: arm SCAN pose — camera pointing straight down for GraspNet.
    # joint1=-π/2: arm rotates to robot right-side (-Y body / world +X direction).
    # joint4=+1.5 (86°): wrist bends to flip gripper_base +Z from horizontal → downward.
    # This makes wrist_camera optical axis (= gripper_base +Z via Rz(-90) offset) point
    # straight down (-Z world), allowing GraspNet to see the floor/table beneath.
    # FK-verified: optical axis in arm_base = [-0.07, -0.09, -0.994] ≈ straight down.
    # Gripper pos in arm_base = [0, -0.49, 0.47] → world ~0.47m above arm_base
    # (arm_base is ~0.66m above ground → gripper ~1.13m above ground).
    # When dog stops with yaw=+π/2 (head → world +Y), robot -Y = world +X → arm
    # extends toward the table/object area.  Used as the home pose during ARM_INIT/grasp phases.
    ARM_SIDE_ANGLES   = np.array([-math.pi/2, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float32)
    # Target yaw for the dog after reaching the goal: head faces world +Y (+π/2).
    # The table is at world +X from the goal, so robot -Y side (arm side) faces it.
    TARGET_YAW        = math.pi / 2   # +90 degrees

    BUDGET = {
        PipelineState.ARM_INIT:  600,   # hard ceiling; early-exit once max_joint_err < 0.02 rad
        PipelineState.PRE_GRASP: 800,   # was 400; PRE_GRASP showed 33cm err at 400 steps
        PipelineState.ORIENT:    300,
        PipelineState.REACH:     500,   # was 250; REACH showed max_err=1.16 at 250 steps
        PipelineState.CLOSE:      80,   # was 40; gripper only reached 1/3 close at 40 steps
        PipelineState.LIFT:      300,   # was 200; extra time to stabilise with object
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
        for name in GRIPPER_JOINT_NAMES:
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
        for name, val in zip(GRIPPER_JOINT_NAMES, targets):
            if name in gids:
                pos_t[gids[name]] = float(val)
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _gripper_width_step(robot, width):
        """Set gripper opening to match GraspNet width output.

        GraspNet width is the gap between the two fingertips at the grasp point.
        Piper joint7 controls +Z finger (limit [0, 0.035]),
        Piper joint8 controls -Z finger (limit [-0.035, 0]).
        Each finger moves width/2 from centre.
        """
        _, gids = _get_arm_ids(robot)
        half = float(np.clip(width / 2.0, 0.0, 0.035))
        targets = {"joint7": half, "joint8": -half}
        pos_t = robot.data.joint_pos_target[0].clone()
        for name, val in targets.items():
            if name in gids:
                pos_t[gids[name]] = val
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
            # JOINT ANGLE RECORDER: every 10 steps during arm phases,
            # append a CSV row to /tmp/arm_joint_log.csv for full traceability.
            # Format: loop,state,step,j1,j2,j3,j4,j5,j6,g_left,g_right
            # ==============================================================
            _ARM_LOG_STATES = {
                PipelineState.ARM_INIT, PipelineState.PRE_GRASP,
                PipelineState.ORIENT,   PipelineState.REACH,
                PipelineState.CLOSE,    PipelineState.LIFT,
            }
            if state in _ARM_LOG_STATES and state_step % 10 == 0:
                try:
                    _log_arm_ids, _log_grip_ids = _get_arm_ids(robot)
                    _log_q = robot.data.joint_pos[0, list(_log_arm_ids)].cpu().numpy()
                    _log_g = robot.data.joint_pos[0, list(_log_grip_ids.values())].cpu().numpy()
                    import csv as _csv_mod, os as _os_mod
                    _log_path = "/tmp/arm_joint_log.csv"
                    _write_hdr = not _os_mod.path.exists(_log_path)
                    with open(_log_path, "a", newline="") as _lf:
                        _w = _csv_mod.writer(_lf)
                        if _write_hdr:
                            _w.writerow(["loop","state","step",
                                         "j1","j2","j3","j4","j5","j6",
                                         "g_left","g_right"])
                        _w.writerow(
                            [loop_count, state.value, state_step]
                            + [f"{v:.5f}" for v in _log_q]
                            + [f"{v:.5f}" for v in _log_g]
                        )
                except Exception as _log_ex:
                    pass  # never block the sim loop

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
                # Overdrive control: ramp alpha beyond 1.0 so the PD controller
                # command overshoots the target by up to 30%, giving extra torque
                # to overcome gravity/damping on j2/j3.  Once converged (<0.02 rad)
                # the early-exit below fires before any overshoot becomes harmful.
                # alpha ramps 0->1.3 over the first 60% of the budget, then stays
                # clamped at 1.3 until convergence or hard timeout.
                _arm_budget = BUDGET[PipelineState.ARM_INIT]
                _arm_alpha  = min(1.3, state_step / max(_arm_budget * 0.6, 1))
                q6 = cur_q + _arm_alpha * (ARM_SIDE_ANGLES - cur_q)
                _arm_step(robot, q6)
                _gripper_step(robot, close=False)
                if state_step == 1:
                    print(f"[SM] ARM_INIT: retracting arm... base_pos_w={np.round(pos_w,3)}", flush=True)
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
                _arm_err_now = np.abs(cur_q - ARM_SIDE_ANGLES)
                if state_step % 50 == 0:
                    _ai_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] ARM_INIT step {state_step}/{BUDGET[PipelineState.ARM_INIT]}: "
                          f"q={np.round(cur_q,4)} err={np.round(_arm_err_now,4)} max={_arm_err_now.max():.4f} "
                          f"base_pos_w={np.round(_ai_pos_w,3)}",
                          flush=True)
                _arm_converged = (_arm_err_now.max() < 0.02)   # ~1.1 deg threshold
                _arm_timeout   = (state_step >= BUDGET[PipelineState.ARM_INIT])
                if _arm_converged or _arm_timeout:
                    _arm_final_err = _arm_err_now
                    _reason = "converged" if _arm_converged else "timeout"
                    _ai_done_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] ARM_INIT done ({_reason} at step {state_step}): final_q={np.round(cur_q,4)} base_pos_w={np.round(_ai_done_pos_w,3)}", flush=True)
                    print(f"[SM] ARM_INIT target:        {np.round(ARM_SIDE_ANGLES,4)}", flush=True)
                    print(f"[SM] ARM_INIT joint_err:     {np.round(_arm_final_err,4)} max={_arm_final_err.max():.4f} rad ({np.degrees(_arm_final_err.max()):.1f}deg)",
                          flush=True)
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
                        _scan_start_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                        print(f"[SM] SCAN: warmup {SCAN_WARMUP} + accumulate "
                              f"{SCAN_FRAMES} frames... base_pos_w={np.round(_scan_start_pos_w,3)}", flush=True)
                    if state_step == SCAN_WARMUP + 1:
                        # ---- Save RGB snapshot after warmup (first valid frame) ----
                        try:
                            from PIL import Image as _PIL_Image
                            _arm_ids_snap, _ = _get_arm_ids(robot)
                            _cur_q_snap = robot.data.joint_pos[
                                0, list(_arm_ids_snap)].cpu().numpy()
                            # joint indices: 0=j1,1=j2,2=j3,3=j4,4=j5,5=j6
                            def _fmt(v):
                                return f"{v:.2f}".replace("-", "n").replace(".", "p")
                            _snap_name = (
                                f"j2_{_fmt(_cur_q_snap[1])}"
                                f"_j3_{_fmt(_cur_q_snap[2])}"
                                f"_j4_{_fmt(_cur_q_snap[3])}"
                                f"_j5_{_fmt(_cur_q_snap[4])}.png"
                            )
                            _snap_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                            os.makedirs(_snap_dir, exist_ok=True)
                            _rgb_snap = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                            _PIL_Image.fromarray(_rgb_snap).save(
                                os.path.join(_snap_dir, _snap_name))
                            print(f"[SCAN] Snapshot saved: {_snap_dir}/{_snap_name}",
                                  flush=True)
                        except Exception as _snap_e:
                            raise RuntimeError(f"[SCAN] Snapshot failed: {_snap_e}") from _snap_e
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
                        _scan_done_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                        print(f"[SM] SCAN done: {len(depth_accum)} frames, "
                              f"valid depth {valid_pct:.1f}% base_pos_w={np.round(_scan_done_pos_w,3)}", flush=True)
                        # ---- DIAG: 相机外参验证 (修正版) ----
                        # 坐标系约定说明：
                        #   camera.data.pos_w      — 相机在「env-local」坐标系中的位置
                        #                            （Isaac Lab 多环境下各 env 有偏移，env_0 原点通常≠世界原点）
                        #   camera.data.quat_w_ros — ROS convention 四元数：前轴=+Z，上轴=-Y
                        #                            旋转矩阵第3列([:,2]) = 相机光轴方向（+Z=朝前/朝物体）
                        #   camera.data.quat_w_world — World convention：前轴=+X，旋转矩阵[:,0]=光轴，不要用[:,2]
                        # 因此：
                        #   光轴应用 quat_w_ros → R[:,2]
                        #   位置对比需先获取 env_origins 补偿 env-local → world
                        try:
                            from arm_ik import cam_to_world as _c2w_scan
                            _cur_q_scan_diag = robot.data.joint_pos[
                                0, list(_get_arm_ids(robot)[0])].cpu().numpy()

                            # --- A) 代码计算的相机原点（全局世界坐标）---
                            _cam_orig_code = _c2w_scan(
                                np.array([0.0, 0.0, 0.0]),
                                _cur_q_scan_diag, pos_w, quat_w
                            )

                            # --- B) Isaac Lab 真值：pos_w 是 env-local，需加 env_origin 才是世界坐标 ---
                            _cam_pos_envlocal = camera.data.pos_w[0].cpu().numpy()  # env-local (3,)
                            try:
                                # Isaac Lab 场景中每个 env 的世界原点偏移
                                _env_origin = raw_env.scene.env_origins[0].cpu().numpy()  # (3,)
                            except Exception:
                                _env_origin = np.zeros(3)
                            _cam_pos_world_isaac = _cam_pos_envlocal + _env_origin

                            _cam_pos_err = np.linalg.norm(_cam_orig_code - _cam_pos_world_isaac)
                            print(f"[DIAG-CAM] cur_q at SCAN      : {np.round(_cur_q_scan_diag, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] env_origin          : {np.round(_env_origin, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] Isaac cam env-local : {np.round(_cam_pos_envlocal, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] Isaac cam world     : {np.round(_cam_pos_world_isaac, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] Code  cam world     : {np.round(_cam_orig_code, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] Position error (code vs Isaac+origin) = {_cam_pos_err*100:.2f} cm",
                                  flush=True)

                            # --- C) 光轴方向：用 quat_w_ros，ROS convention 下 R[:,2] = 光轴(+Z=朝前) ---
                            _quat_ros = camera.data.quat_w_ros[0].cpu().numpy()  # (w,x,y,z)
                            _wr, _xr, _yr, _zr = _quat_ros
                            _R_ros = np.array([
                                [1-2*(_yr*_yr+_zr*_zr), 2*(_xr*_yr-_wr*_zr), 2*(_xr*_zr+_wr*_yr)],
                                [2*(_xr*_yr+_wr*_zr), 1-2*(_xr*_xr+_zr*_zr), 2*(_yr*_zr-_wr*_xr)],
                                [2*(_xr*_zr-_wr*_yr), 2*(_yr*_zr+_wr*_xr), 1-2*(_xr*_xr+_yr*_yr)],
                            ])
                            # ROS: +Z = forward (光轴), +X = right, -Y = up
                            _optical_axis_ros = _R_ros[:, 2]   # 真正的光轴方向
                            _up_axis_ros      = -_R_ros[:, 1]  # 相机上方向
                            print(f"[DIAG-CAM] quat_w_ros          : {np.round(_quat_ros, 4)}",
                                  flush=True)
                            print(f"[DIAG-CAM] Optical axis (world): {np.round(_optical_axis_ros, 4)}"
                                  f"  z={_optical_axis_ros[2]:.3f}  (负值=朝下=正确)",
                                  flush=True)
                            print(f"[DIAG-CAM] Up axis     (world): {np.round(_up_axis_ros, 4)}",
                                  flush=True)

                            # --- D) 香蕉在相机视野内的角度验证 ---
                            try:
                                _banana_diag = raw_env.scene["banana"]
                                _bp_diag = _banana_diag.data.root_pos_w[0].cpu().numpy()
                                # 香蕉在世界坐标，相机也要用世界坐标
                                _cam_to_banana = _bp_diag - _cam_pos_world_isaac
                                _dist_cam_banana = np.linalg.norm(_cam_to_banana)
                                _dir_cam_banana  = _cam_to_banana / (_dist_cam_banana + 1e-9)
                                _cos_angle = float(np.dot(_optical_axis_ros, _dir_cam_banana))
                                _angle_deg = float(np.degrees(np.arccos(np.clip(_cos_angle, -1, 1))))
                                print(f"[DIAG-CAM] Banana world pos    : {np.round(_bp_diag, 4)}",
                                      flush=True)
                                print(f"[DIAG-CAM] Cam -> Banana dir   : {np.round(_dir_cam_banana, 4)}",
                                      flush=True)
                                print(f"[DIAG-CAM] Cam-Banana distance : {_dist_cam_banana*100:.1f} cm",
                                      flush=True)
                                print(f"[DIAG-CAM] Angle(optical,banana): {_angle_deg:.1f} deg"
                                      f"  (VFOV/2=21.3deg, HFOV/2=27.5deg — 若<21deg则在中心视野内)",
                                      flush=True)
                            except Exception as _be2:
                                print(f"[DIAG-CAM] banana pos unavailable: {_be2}", flush=True)
                        except Exception as _cam_diag_e:
                            print(f"[DIAG-CAM] camera diag failed: {_cam_diag_e}", flush=True)
                        # ---- END DIAG 相机外参验证 ----
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
                            # Capture the joint angles at SCAN time before
                            # anything moves.  Used to compute the desired EE
                            # orientation in arm_base_link frame independently
                            # of subsequent PRE_GRASP motion.
                            _q_scan_saved = robot.data.joint_pos[
                                0, list(_get_arm_ids(robot)[0])
                            ].cpu().numpy().copy()
                            from arm_ik import compute_desired_ee_rot_in_arm as _cder
                            _R_desired = _cder(gr["rotations"][0], _q_scan_saved)
                            grasp_result = {
                                "t_cam": gr["translations"][0],
                                "R_cam": gr["rotations"][0],
                                "score": float(gr["scores"][0]),
                                "width": float(gr["widths"][0]),
                                # Desired gripper_base rotation in arm_base_link
                                # frame, computed using SCAN-time joint angles.
                                # Passed as target_rot in PRE_GRASP IK so that
                                # all 6 joints arrive at the correct orientation.
                                "R_desired_EE_in_arm": _R_desired,
                                "q_scan": _q_scan_saved,
                                # DRIFT FIX: save robot pose at SCAN time.
                                # t_cam is measured in the camera frame at this exact
                                # moment, so cam_to_world MUST use these values
                                # (not the current drifted pos_w) to get the correct
                                # world-frame target.  pos_w and quat_w are updated
                                # at the top of every loop iteration, so they reflect
                                # the robot pose at the time grasp_result is built.
                                "pos_w_scan":  pos_w.copy(),
                                "quat_w_scan": quat_w.copy(),
                            }
                            print(f"[SM] Best grasp score={grasp_result['score']:.3f} "
                                  f"t={np.round(grasp_result['t_cam'],3)}", flush=True)
                            # ---- DIAG: GraspNet physical interpretation + RGB visualisation ----
                            try:
                                import cv2 as _cv2
                                from arm_ik import (
                                    cam_to_world as _c2w, quat_to_rot as _q2r,
                                    _CAM_OFFSET_ROT as _COR, fk as _fk,
                                    fk_gripper as _fkg,
                                )
                                _cur_q_d = robot.data.joint_pos[
                                    0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                                _t_c = grasp_result["t_cam"]
                                _R_c = grasp_result["R_cam"]
                                _R_rob = _q2r(quat_w)
                                # BUG FIX: use q_scan (SCAN-time joints) for cam_to_world
                                _q_scan_d = grasp_result["q_scan"]
                                # FIX: use fk_gripper (gripper_base frame) not fk (joint7 frame)
                                # fk()[:3,:3] is joint7 frame (Rx+90 from gripper_base),
                                # giving wrong camera world orientation (Frobenius diff=2.0)
                                _T_fk  = _fkg(_q_scan_d)
                                _R_cw  = _R_rob @ _T_fk[:3, :3] @ _COR
                                _app_w = _R_cw @ _R_c[:, 0]
                                _clo_w = _R_cw @ _R_c[:, 1]
                                _t_obj_w = _c2w(_t_c, _q_scan_d, pos_w, quat_w)
                                print(f"[DIAG] t_cam={np.round(_t_c,4)}  depth={_t_c[2]:.3f}m", flush=True)
                                print(f"[DIAG] t_obj_world={np.round(_t_obj_w,4)}  z={_t_obj_w[2]:.3f}m"
                                      f" (tabletop expect ~0.70m)", flush=True)
                                print(f"[DIAG] approach cam={np.round(_R_c[:,0],3)}"
                                      f" => world={np.round(_app_w,3)}", flush=True)
                                print(f"[DIAG] closing  cam={np.round(_R_c[:,1],3)}"
                                      f" => world={np.round(_clo_w,3)}", flush=True)
                                print(f"[DIAG] R_cam full=\n{np.round(_R_c,3)}", flush=True)
                                print(f"[DIAG] robot pos_w={np.round(pos_w,3)}"
                                      f" quat_w={np.round(quat_w,4)}", flush=True)
                                print(f"[DIAG] SCAN cur_q={np.round(_cur_q_d,4)}", flush=True)
                                _snap_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                                os.makedirs(_snap_dir, exist_ok=True)
                                _img_bgr = _cv2.cvtColor(
                                    rgb_u if rgb_u is not None
                                    else np.zeros((H, W, 3), np.uint8),
                                    _cv2.COLOR_RGB2BGR)
                                _u = int(_t_c[0] / _t_c[2] * fx_c + cx_c)
                                _v = int(_t_c[1] / _t_c[2] * fy_c + cy_c)
                                _cv2.circle(_img_bgr, (_u, _v), 10, (0, 255, 0), 2)
                                _cv2.putText(_img_bgr,
                                             f"z={_t_c[2]:.2f}m",
                                             (_u + 12, _v),
                                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                _sc = 60.0
                                _du  = int(_R_c[0, 0] * _sc)
                                _dv  = int(_R_c[1, 0] * _sc)
                                _du2 = int(_R_c[0, 1] * _sc)
                                _dv2 = int(_R_c[1, 1] * _sc)
                                _cv2.arrowedLine(_img_bgr, (_u, _v),
                                                 (_u+_du, _v+_dv),
                                                 (0, 0, 255), 2, tipLength=0.3)
                                _cv2.arrowedLine(_img_bgr, (_u, _v),
                                                 (_u+_du2, _v+_dv2),
                                                 (255, 0, 0), 2, tipLength=0.3)
                                _cv2.putText(_img_bgr,
                                             "RED=approach  BLUE=closing",
                                             (10, 20),
                                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                _diag_path = os.path.join(_snap_dir, "diag_grasp.png")
                                _cv2.imwrite(_diag_path, _img_bgr)
                                print(f"[DIAG] Saved -> {_diag_path}", flush=True)
                            except Exception as _de:
                                print(f"[DIAG] vis failed: {_de}", flush=True)
                            # ---- APPROACH DIRECTION SANITY CHECK ----
                            # GraspNet sometimes returns a grasp with approach pointing
                            # upward (approach_world[2] > 0), which means it found a
                            # "from-below" solution on a corrupted/noisy point cloud.
                            # This causes PRE_GRASP to retreat below the object, making
                            # the entire pipeline fail. Strategy: if approach_world z > 0,
                            # flip the grasp frame (negate X and Y columns) so the gripper
                            # comes from above. Note: this preserves the closing direction
                            # (Z column) but reverses approach + binormal.
                            _approach_world_z = float(_app_w[2])
                            if _approach_world_z > 0.0:
                                print(f"[WARN] approach_world[2]={_approach_world_z:.3f} > 0 "
                                      f"(upward approach detected). Flipping grasp frame.",
                                      flush=True)
                                # Flip columns 0 and 1 of R_cam (approach and binormal),
                                # keeping column 2 (closing axis) to form a valid right-hand frame.
                                _R_flipped = grasp_result["R_cam"].copy()
                                _R_flipped[:, 0] = -_R_flipped[:, 0]
                                _R_flipped[:, 1] = -_R_flipped[:, 1]
                                # Recompute R_desired with flipped rotation
                                from arm_ik import compute_desired_ee_rot_in_arm as _cder2
                                grasp_result["R_cam"] = _R_flipped
                                grasp_result["R_desired_EE_in_arm"] = _cder2(
                                    _R_flipped, grasp_result["q_scan"]
                                )
                                # Verify flip resolved the issue
                                _R_rob_flip = _q2r(quat_w)
                                # FIX: use fk_gripper (gripper_base frame) for correct camera orientation
                                _T_fk_flip  = _fkg(grasp_result["q_scan"])
                                _R_cw_flip  = _R_rob_flip @ _T_fk_flip[:3, :3] @ _COR
                                _app_w_flip = _R_cw_flip @ _R_flipped[:, 0]
                                print(f"[WARN] After flip: approach_world={np.round(_app_w_flip,3)}"
                                      f" z={_app_w_flip[2]:.3f} (should be < 0)", flush=True)
                            else:
                                print(f"[INFO] approach_world[2]={_approach_world_z:.3f} < 0 "
                                      f"(downward approach — OK)", flush=True)
                            # ---- END APPROACH SANITY CHECK ----
                            state = PipelineState.PRE_GRASP
                            state_step = 0

            # ---- PRE_GRASP: move arm along GraspNet approach axis, open gripper to width ----
            elif state == PipelineState.PRE_GRASP:
                from arm_ik import solve_for_gripper_base as ik_solve_gb, cam_to_world, world_pos_to_arm_frame, _IK_JOINT_LIMITS
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    R_cam = grasp_result["R_cam"]
                    width = grasp_result["width"]
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    # Retreat PRE_GRASP_RETREAT m along approach axis (R_cam[:,0]) in cam frame
                    # so fingertips clear the object and ORIENT can rotate freely.
                    pre_t_cam = t_cam - PRE_GRASP_RETREAT * R_cam[:, 0]
                    # BUG FIX: use grasp_result["q_scan"] (SCAN-time joint angles) instead of
                    # cur_q (current joints at PRE_GRASP step=1, arm already moving).
                    # t_cam is in the camera frame at SCAN time, so cam_to_world must
                    # use the joint angles at that moment to get the correct camera pose.
                    # DRIFT FIX: use pos_w_scan/quat_w_scan (robot pose at SCAN time) instead of
                    # the current pos_w (which may already be drifted from PRE_GRASP start).
                    # t_cam was measured in camera frame at SCAN time, so cam_to_world must
                    # use the robot pose at that same moment to get the correct world target.
                    t_world = cam_to_world(pre_t_cam, grasp_result["q_scan"],
                                           grasp_result["pos_w_scan"], grasp_result["quat_w_scan"])
                    # DRIFT FIX: save fixed world-frame target so every step can re-project
                    # it into the current robot frame, compensating for body drift during PRE_GRASP.
                    _pg_t_world_fixed = t_world.copy()
                    pre_t = world_pos_to_arm_frame(t_world, pos_w, quat_w)
                    # Include the desired EE orientation (computed at SCAN time)
                    # so that PRE_GRASP IK solves for all 6 joints simultaneously.
                    # This replaces the ORIENT stage, which was mathematically
                    # incapable of achieving the target orientation by adjusting
                    # joint6 alone after PRE_GRASP had moved joint1-5.
                    R_desired_EE = grasp_result["R_desired_EE_in_arm"]
                    # FIX: use solve_for_gripper_base() so that gripper_base (not joint7)
                    # reaches pre_t.  solve() passes target_pos to ikpy as the joint7
                    # target; solve_for_gripper_base() corrects this by adding the
                    # joint7->gripper_base offset (R_gb_desired @ [0,0,0.1358]) to pre_t
                    # before calling solve(), so that joint7 lands at the right place and
                    # gripper_base ends up at pre_t (error ~0 cm instead of ~13.58 cm).
                    target_angles_arm = ik_solve_gb(
                        pre_t,
                        target_rot_j7=R_desired_EE,
                        initial_angles=cur_q,
                    )
                    print(f"[SM] PRE_GRASP t_cam={np.round(t_cam,3)} retreat={PRE_GRASP_RETREAT}m", flush=True)
                    print(f"[SM] PRE_GRASP pre_t_cam={np.round(pre_t_cam,3)} t_world={np.round(t_world,3)}", flush=True)
                    print(f"[SM] PRE_GRASP t_arm={np.round(pre_t,3)} IK={np.round(target_angles_arm,3)}", flush=True)
                    print(f"[SM] PRE_GRASP R_desired_EE_in_arm (from SCAN q):\n{np.round(R_desired_EE,3)}", flush=True)
                    print(f"[SM] PRE_GRASP opening gripper to width={width:.3f}m (half={width/2:.3f})", flush=True)
                    # ---- DIAG: IK quality check ----
                    from arm_ik import fk_gripper as _fk_pg, cam_to_world as _c2w_pg, world_pos_to_arm_frame as _w2a_pg, _IK_JOINT_LIMITS as _jlims_pg
                    # FIX: use fk_gripper to get gripper_base position (pre_t is gripper_base origin)
                    _T_ik = _fk_pg(target_angles_arm)
                    _ee_arm_ik = _T_ik[:3, 3]
                    _ik_pos_err = float(np.linalg.norm(_ee_arm_ik - pre_t))
                    _ik_at_lim = any(
                        abs(float(_qi) - _lo) < 0.01 or abs(float(_qi) - _hi) < 0.01
                        for _qi, (_lo, _hi) in zip(target_angles_arm, _jlims_pg)
                    )
                    _ik_status = "✓ OK" if (_ik_pos_err < 0.03 and not _ik_at_lim) else "✗ FAIL"
                    print(f"[DIAG] PRE_GRASP IK FK check: target_arm={np.round(pre_t,4)}"
                          f" actual_EE_arm={np.round(_ee_arm_ik,4)}"
                          f" pos_err={_ik_pos_err*100:.1f}cm at_limit={_ik_at_lim} {_ik_status}",
                          flush=True)
                    _t_obj_world_pg = _c2w_pg(t_cam, grasp_result["q_scan"], pos_w, quat_w)
                    print(f"[DIAG] PRE_GRASP object world pos={np.round(_t_obj_world_pg,4)}"
                          f" (z={_t_obj_world_pg[2]:.3f}m)", flush=True)
                    print(f"[DIAG] PRE_GRASP target world pos={np.round(t_world,4)}"
                          f" (z={t_world[2]:.3f}m, should be ~0.05-0.15m ABOVE object)", flush=True)
                    # approach axis in world
                    from arm_ik import _CAM_OFFSET_ROT as _COR_pg, quat_to_rot as _q2r_pg
                    _R_rob_pg = _q2r_pg(quat_w)
                    # FIX: use fk_gripper (gripper_base frame) for correct camera world rotation
                    _T_fk_pg  = _fk_pg(grasp_result["q_scan"])
                    _R_cw_pg  = _R_rob_pg @ _T_fk_pg[:3, :3] @ _COR_pg
                    _app_w_pg = _R_cw_pg @ R_cam[:, 0]
                    print(f"[DIAG] PRE_GRASP approach world={np.round(_app_w_pg,3)}"
                          f" (expect to point FROM cam TOWARD object)", flush=True)
                    print(f"[DIAG] PRE_GRASP retreat world dir={np.round(-_app_w_pg,3)}"
                          f" (pre-grasp is in this direction from object)", flush=True)
                    # Record initial joint angles for feed-forward interpolation
                    _pg_q_init = cur_q.copy()
                    # ---- END DIAG PRE_GRASP ----
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                # Rolling tracking for PRE_GRASP:
                # Each step command = cur_q + α*(target - cur_q), where α controls
                # how fast joints move toward the IK target.  Unlike the fixed q_init
                # feed-forward interpolation (which caused oscillation when IK target
                # updated mid-way), this approach always starts from the actual current
                # joint position, so IK target updates are absorbed smoothly.
                _PRE_GRASP_ALPHA = 0.08  # fraction of remaining error to close per step
                # DRIFT FIX: re-project the fixed world-frame target into the current
                # robot arm frame every 5 steps, then re-solve IK.  This compensates
                # for body drift while keeping IK overhead low.
                if state_step % 5 == 1:  # fires on step=1,6,11,... (also first step)
                    _cur_pos_w  = robot.data.root_pos_w[0].cpu().numpy()
                    _cur_quat_w = robot.data.root_quat_w[0].cpu().numpy()
                    _pg_pre_t_cur = world_pos_to_arm_frame(_pg_t_world_fixed, _cur_pos_w, _cur_quat_w)
                    _pg_target_cur = ik_solve_gb(
                        _pg_pre_t_cur,
                        target_rot_j7=R_desired_EE,
                        initial_angles=cur_q,
                    )
                # Rolling tracking command: move α fraction toward current IK target
                q6 = cur_q + _PRE_GRASP_ALPHA * (_pg_target_cur - cur_q)
                q6 = np.clip(q6, [lo for lo, hi in _IK_JOINT_LIMITS],
                                  [hi for lo, hi in _IK_JOINT_LIMITS])
                _arm_step(robot, q6)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - _pg_target_cur)
                    _pg_step_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] PRE_GRASP step {state_step}/{BUDGET[PipelineState.PRE_GRASP]}: "
                          f"max_err={_err.max():.3f} base_pos_w={np.round(_pg_step_pos_w,3)}", flush=True)
                # Early-exit: once all joints are within 0.05 rad of live IK target
                # AND at least 100 steps have passed (give time to settle), skip
                # waiting the full budget and move directly to REACH.
                _pg_cur_q_check = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                _pg_err_check = np.abs(_pg_cur_q_check - _pg_target_cur)
                _pg_early_exit = (state_step > 100 and _pg_err_check.max() < 0.05)
                if _pg_early_exit and state_step % 50 != 0:  # avoid double-print
                    _pg_exit_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] PRE_GRASP early-exit at step {state_step}: "
                          f"max_err={_pg_err_check.max():.4f} rad < 0.05 -> REACH "
                          f"base_pos_w={np.round(_pg_exit_pos_w,3)}",
                          flush=True)
                if state_step >= BUDGET[PipelineState.PRE_GRASP] or _pg_early_exit:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    _pg_done_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] PRE_GRASP done: final={np.round(cur_q_final,3)} -> REACH (ORIENT skipped) "
                          f"base_pos_w={np.round(_pg_done_pos_w,3)}",
                          flush=True)
                    # ---- DIAG: actual EE position after PRE_GRASP ----
                    from arm_ik import fk as _fk_pgd, fk_gripper as _fkg_pgd, quat_to_rot as _q2r_pgd, cam_to_world as _c2w_pgd
                    # FIX: use fk_gripper for position (target pre_t is gripper_base origin)
                    # keep fk() for rotation (R_desired_EE is joint7-frame, so compare in same frame)
                    _T_final_gb = _fkg_pgd(cur_q_final)
                    _T_final    = _fk_pgd(cur_q_final)
                    _ee_final_arm = _T_final_gb[:3, 3]   # gripper_base position
                    _R_final_ee = _T_final[:3, :3]        # joint7 rotation (for rot_err vs R_desired_EE)
                    _R_rob_pgd = _q2r_pgd(quat_w)
                    _ee_final_world = _R_rob_pgd @ _ee_final_arm + pos_w + _R_rob_pgd @ np.array([0.,0.,0.0888])
                    # BUG FIX: use q_scan (SCAN-time joints) for cam_to_world, not cur_q_final
                    _target_world = _c2w_pgd(t_cam - PRE_GRASP_RETREAT * R_cam[:, 0], grasp_result["q_scan"], pos_w, quat_w)
                    _pos_err_final = float(np.linalg.norm(_ee_final_world - _target_world))
                    _joint_err_final = np.abs(cur_q_final - target_angles_arm)
                    _rot_err_final = np.linalg.norm(_R_final_ee - R_desired_EE, ord='fro')
                    print(f"[DIAG] PRE_GRASP actual EE arm ={np.round(_ee_final_arm,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP actual EE world={np.round(_ee_final_world,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP target    world={np.round(_target_world,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP pos err={_pos_err_final*100:.1f}cm"
                          f" (joint err max={_joint_err_final.max()*57.3:.1f}deg)", flush=True)
                    print(f"[DIAG] PRE_GRASP joint err per joint (deg): {np.round(_joint_err_final*57.3,2)}",
                          flush=True)
                    print(f"[DIAG] PRE_GRASP IK target q: {np.round(target_angles_arm,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP actual    q: {np.round(cur_q_final,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP EE rot_err (Frobenius vs R_desired): {_rot_err_final:.4f}",
                          flush=True)
                    print(f"[DIAG] PRE_GRASP actual EE rot:\n{np.round(_R_final_ee,4)}", flush=True)
                    print(f"[DIAG] PRE_GRASP desired EE rot:\n{np.round(R_desired_EE,4)}", flush=True)
                    # ---- END DIAG PRE_GRASP done ----
                    state = PipelineState.REACH
                    state_step = 0

            # ---- ORIENT: [BYPASSED] PRE_GRASP now includes target_rot=R_desired_EE_in_arm ----
            # The ORIENT stage was mathematically incapable of achieving the target orientation:
            # after PRE_GRASP moves joint1-5, the desired gripper direction is no longer reachable
            # by adjusting joint6 alone (FK Frobenius error ≈ 0.50 vs ideal 0.0). ORIENT is kept
            # here but the state machine jumps from PRE_GRASP directly to REACH.
            # ---- ORIENT: rotate wrist (joint6 only) to match GraspNet orientation ----
            # joint1-5 stay fixed at their PRE_GRASP end position.
            # Only joint6 is adjusted to align the gripper closing axis with GraspNet R_cam.
            elif state == PipelineState.ORIENT:
                from arm_ik import cam_rot_to_arm_frame, extract_j6_angle
                if state_step == 1:
                    R_cam = grasp_result["R_cam"]
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    # Convert GraspNet rotation from camera frame to arm_base_link frame
                    R_arm = cam_rot_to_arm_frame(R_cam, cur_q, quat_w)
                    # Compute target joint6 angle; joint1-5 are held fixed.
                    # Pass R_cam so the formula can correctly isolate the Ry(j6) component.
                    j6_target = extract_j6_angle(cur_q, R_cam, R_arm)
                    # Build full target: copy cur_q, replace only joint6
                    target_angles_arm = cur_q.copy()
                    target_angles_arm[5] = j6_target
                    print(f"[SM] ORIENT R_arm=\n{np.round(R_arm,3)}", flush=True)
                    print(f"[SM] ORIENT j6_target={j6_target:.4f} rad ({math.degrees(j6_target):.1f} deg), "
                          f"cur_j6={cur_q[5]:.4f} rad", flush=True)
                    # ---- DIAG: ORIENT physical meaning ----
                    from arm_ik import _CAM_OFFSET_ROT as _COR_or, quat_to_rot as _q2r_or, fk as _fk_or, fk_gripper as _fkg_or
                    _R_rob_or = _q2r_or(quat_w)
                    # FIX: use fk_gripper for correct camera world rotation
                    _R_cw_or  = _R_rob_or @ _fkg_or(cur_q)[:3, :3] @ _COR_or
                    _app_w_or  = _R_cw_or @ R_cam[:, 0]
                    _clo_w_or  = _R_cw_or @ R_cam[:, 1]
                    print(f"[DIAG] ORIENT approach world={np.round(_app_w_or,3)}", flush=True)
                    print(f"[DIAG] ORIENT closing  world={np.round(_clo_w_or,3)}"
                          f" (expect along banana long-axis)", flush=True)
                    print(f"[DIAG] ORIENT cur_q (PRE_GRASP end)={np.round(cur_q,4)}", flush=True)
                    # verify extract_j6_angle by FK
                    _q_j6t = cur_q.copy(); _q_j6t[5] = j6_target
                    _T_j6t = _fk_or(_q_j6t)
                    _ee_rot_after = _T_j6t[:3, :3] @ _COR_or
                    _rot_err = np.linalg.norm(_ee_rot_after - R_arm, ord='fro')
                    print(f"[DIAG] ORIENT j6 extraction rot_err={_rot_err:.4f}"
                          f" (0=perfect, <0.01 is fine)", flush=True)
                    # ---- END DIAG ORIENT ----
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                # Interpolate only joint6; hold joint1-5 at cur_q (PRE_GRASP end position)
                q_cmd = cur_q.copy()
                q_cmd[5] = (1.0 - _alpha(state)) * cur_q[5] + _alpha(state) * target_angles_arm[5]
                _arm_step(robot, q_cmd)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    j6_err = abs(cur_q[5] - target_angles_arm[5])
                    print(f"[SM] ORIENT step {state_step}/{BUDGET[PipelineState.ORIENT]}: "
                          f"j6_err={j6_err:.4f} rad", flush=True)
                if state_step >= BUDGET[PipelineState.ORIENT]:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    print(f"[SM] ORIENT done: j6={cur_q_final[5]:.4f} (target={target_angles_arm[5]:.4f}) -> REACH",
                          flush=True)
                    state = PipelineState.REACH
                    state_step = 0

            # ---- REACH: advance along approach axis to REACH_RETREAT short of GraspNet t ----
            # Start from ORIENT's end joint angles (cur_q) so joint1-5 keep the PRE_GRASP
            # configuration and only small adjustments are needed to advance forward.
            elif state == PipelineState.REACH:
                from arm_ik import solve_for_gripper_base as ik_solve_gb_re, cam_to_world, cam_rot_to_arm_frame, world_pos_to_arm_frame, _IK_JOINT_LIMITS
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    R_cam = grasp_result["R_cam"]
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    # Target: REACH_RETREAT m back from GraspNet t along approach axis.
                    # Approach axis R_cam[:,0] is in camera frame; retreat in cam frame,
                    # then convert to world/arm frame for IK.
                    reach_t_cam = t_cam - REACH_RETREAT * R_cam[:, 0]
                    # BUG FIX: use grasp_result["q_scan"] (SCAN-time joint angles) instead of
                    # cur_q (current joints at REACH step=1, arm already at PRE_GRASP position).
                    # t_cam is in the camera frame at SCAN time, so cam_to_world must
                    # use the joint angles at that moment to get the correct camera pose.
                    # DRIFT FIX: use pos_w_scan/quat_w_scan (robot pose at SCAN time) instead of
                    # the current pos_w (which is drifted after PRE_GRASP 800 steps).
                    # t_cam was measured in camera frame at SCAN time, so cam_to_world must
                    # use the robot pose at that same moment to get the correct world target.
                    t_world = cam_to_world(reach_t_cam, grasp_result["q_scan"],
                                           grasp_result["pos_w_scan"], grasp_result["quat_w_scan"])
                    # DRIFT FIX: save fixed world-frame target for per-step re-projection
                    _re_t_world_fixed = t_world.copy()
                    t_arm = world_pos_to_arm_frame(t_world, pos_w, quat_w)
                    # Keep orientation established by PRE_GRASP (which now includes target_rot).
                    # Use the stored R_desired_EE_in_arm (from SCAN time) as the orientation
                    # constraint so REACH IK warm-starts from the PRE_GRASP end config and only
                    # makes small forward adjustments along the approach axis.
                    R_arm = grasp_result["R_desired_EE_in_arm"]
                    # FIX: use solve_for_gripper_base() same reason as PRE_GRASP:
                    # t_arm is the desired gripper_base position; ikpy needs the joint7 target.
                    target_angles_arm = ik_solve_gb_re(t_arm, target_rot_j7=R_arm, initial_angles=cur_q)
                    print(f"[SM] REACH t_cam={np.round(t_cam,3)} retreat={REACH_RETREAT}m", flush=True)
                    print(f"[SM] REACH reach_t_cam={np.round(reach_t_cam,3)} "
                          f"t_world={np.round(t_world,3)} t_arm={np.round(t_arm,3)}", flush=True)
                    print(f"[SM] REACH IK: {np.round(target_angles_arm,3)}", flush=True)
                    # ---- DIAG: REACH IK quality check ----
                    from arm_ik import fk_gripper as _fk_re, quat_to_rot as _q2r_re, _IK_JOINT_LIMITS as _jlims_re
                    # FIX: use fk_gripper (t_arm is gripper_base position, not joint7)
                    _T_re = _fk_re(target_angles_arm)
                    _ee_re_arm = _T_re[:3, 3]
                    _re_pos_err = float(np.linalg.norm(_ee_re_arm - t_arm))
                    _re_q_delta = np.abs(target_angles_arm - cur_q)
                    _re_at_lim = any(
                        abs(float(_qi) - _lo) < 0.01 or abs(float(_qi) - _hi) < 0.01
                        for _qi, (_lo, _hi) in zip(target_angles_arm, _jlims_re)
                    )
                    _re_status = "✓ OK" if (_re_pos_err < 0.03 and not _re_at_lim) else "✗ FAIL"
                    print(f"[DIAG] REACH IK FK check: target_arm={np.round(t_arm,4)}"
                          f" actual_EE_arm={np.round(_ee_re_arm,4)}"
                          f" pos_err={_re_pos_err*100:.1f}cm at_limit={_re_at_lim} {_re_status}", flush=True)
                    print(f"[DIAG] REACH joint deltas from ORIENT end: {np.round(_re_q_delta*57.3,1)} deg"
                          f" max={_re_q_delta.max()*57.3:.1f}deg", flush=True)
                    _R_rob_re = _q2r_re(quat_w)
                    _re_world = _R_rob_re @ _ee_re_arm + pos_w + _R_rob_re @ np.array([0.,0.,0.0888])
                    print(f"[DIAG] REACH EE world pos={np.round(_re_world,4)}", flush=True)
                    # Record initial joint angles for feed-forward interpolation
                    _re_q_init = cur_q.copy()
                    # ---- END DIAG REACH ----
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                # Rolling tracking for REACH (same logic as PRE_GRASP):
                # Each step command = cur_q + α*(target - cur_q).
                _REACH_ALPHA = 0.08
                # DRIFT FIX: re-project the fixed world-frame target into the current
                # robot arm frame every 5 steps, then re-solve IK.
                if state_step % 5 == 1:  # fires on step=1,6,11,...
                    _re_cur_pos_w  = robot.data.root_pos_w[0].cpu().numpy()
                    _re_cur_quat_w = robot.data.root_quat_w[0].cpu().numpy()
                    _re_t_cur = world_pos_to_arm_frame(_re_t_world_fixed, _re_cur_pos_w, _re_cur_quat_w)
                    _re_target_cur = ik_solve_gb_re(
                        _re_t_cur,
                        target_rot_j7=R_arm,
                        initial_angles=cur_q,
                    )
                # Rolling tracking command: move α fraction toward current IK target
                q6 = cur_q + _REACH_ALPHA * (_re_target_cur - cur_q)
                q6 = np.clip(q6, [lo for lo, hi in _IK_JOINT_LIMITS],
                                  [hi for lo, hi in _IK_JOINT_LIMITS])
                _arm_step(robot, q6)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - _re_target_cur)
                    print(f"[SM] REACH step {state_step}/{BUDGET[PipelineState.REACH]}: "
                          f"max_err={_err.max():.3f}", flush=True)
                # Early-exit: joints converged within 0.05 rad
                _reach_err_check = np.abs(cur_q - _re_target_cur)
                _reach_early = (state_step > 50 and _reach_err_check.max() < 0.05)
                if _reach_early and state_step % 50 != 0:
                    print(f"[SM] REACH early-exit at step {state_step}: "
                          f"max_err={_reach_err_check.max():.4f} rad < 0.05", flush=True)
                if state_step >= BUDGET[PipelineState.REACH] or _reach_early:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    print(f"[SM] REACH done: final={np.round(cur_q_final,3)} -> CLOSE", flush=True)
                    # ---- DIAG: REACH actual EE vs banana position ----
                    from arm_ik import fk as _fk_red, fk_gripper as _fkg_red, quat_to_rot as _q2r_red
                    # FIX: use fk_gripper for position; keep fk() for rotation (joint7 frame = R_desired_EE frame)
                    _T_red_gb = _fkg_red(cur_q_final)
                    _T_red    = _fk_red(cur_q_final)
                    _ee_red_arm = _T_red_gb[:3, 3]   # gripper_base position
                    _R_red_ee   = _T_red[:3, :3]      # joint7 rotation (for rot_err vs R_desired_EE)
                    _R_rob_red  = _q2r_red(quat_w)
                    _ee_red_world = _R_rob_red @ _ee_red_arm + pos_w + _R_rob_red @ np.array([0.,0.,0.0888])
                    _re_joint_err = np.abs(cur_q_final - target_angles_arm)
                    _re_rot_err   = np.linalg.norm(_R_red_ee - grasp_result["R_desired_EE_in_arm"], ord='fro')
                    print(f"[DIAG] REACH actual EE arm  ={np.round(_ee_red_arm,4)}", flush=True)
                    print(f"[DIAG] REACH actual EE world={np.round(_ee_red_world,4)}", flush=True)
                    print(f"[DIAG] REACH IK target q:    {np.round(target_angles_arm,4)}", flush=True)
                    print(f"[DIAG] REACH actual    q:    {np.round(cur_q_final,4)}", flush=True)
                    print(f"[DIAG] REACH joint err (deg):{np.round(_re_joint_err*57.3,2)}", flush=True)
                    print(f"[DIAG] REACH EE rot_err vs R_desired: {_re_rot_err:.4f}", flush=True)
                    try:
                        _banana_re = raw_env.scene["banana"]
                        _bp_re = _banana_re.data.root_pos_w[0].cpu().numpy()
                        _dist_re = float(np.linalg.norm(_ee_red_world - _bp_re))
                        print(f"[DIAG] REACH banana world={np.round(_bp_re,4)}", flush=True)
                        print(f"[DIAG] REACH EE-banana dist={_dist_re*100:.1f}cm"
                              f" (EE should be within ~5cm for successful grasp)", flush=True)
                    except Exception:
                        pass
                    # ---- END DIAG REACH done ----
                    state = PipelineState.CLOSE
                    state_step = 0

            # ---- CLOSE: close gripper to GraspNet width then fully grip ----
            # Use grasp_result["width"] for the first half, then fully close.
            # This avoids slamming into the object while still ensuring a firm grip.
            elif state == PipelineState.CLOSE:
                # First half: approach to grasp width; second half: fully close
                _close_budget = BUDGET[PipelineState.CLOSE]
                if state_step <= _close_budget // 2:
                    _gripper_width_step(robot, grasp_result["width"])
                else:
                    _gripper_step(robot, close=True)   # targets GRIPPER_CLOSE_POS = [0, 0]
                if state_step == 1:
                    _gids = _get_arm_ids(robot)[1]
                    _gj = robot.data.joint_pos[0, list(_gids.values())].cpu().numpy()
                    print(f"[SM] CLOSE: width={grasp_result['width']:.4f}m then fully close"
                          f" (cur={np.round(_gj,4)})...", flush=True)
                if state_step >= _close_budget:
                    _gids = _get_arm_ids(robot)[1]
                    _gj = robot.data.joint_pos[0, list(_gids.values())].cpu().numpy()
                    print(f"[SM] CLOSE done: gripper={np.round(_gj,4)} -> LIFT", flush=True)
                    state = PipelineState.LIFT
                    state_step = 0

            # ---- LIFT: retract arm to ARM_SIDE_ANGLES while keeping gripper closed ----
            elif state == PipelineState.LIFT:
                if state_step == 1:
                    print("[SM] LIFT: retracting arm to side pose (ARM_SIDE_ANGLES) with object...",
                          flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                # Joint-space interpolation to ARM_SIDE_ANGLES — no IK needed,
                # avoids IK failures when carrying a grasped object.
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * ARM_SIDE_ANGLES
                _arm_step(robot, q6)
                _gripper_step(robot, close=True)   # hold grip throughout
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - ARM_SIDE_ANGLES)
                    print(f"[SM] LIFT step {state_step}/{BUDGET[PipelineState.LIFT]}: "
                          f"max_err={_err.max():.3f}", flush=True)
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
