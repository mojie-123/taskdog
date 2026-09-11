#!/usr/bin/env python
"""Navigate to goal using Nav2 (Humble) instead of custom A*/Pure Pursuit.

All post-arrival states (ALIGN_YAW_1->PAN->ALIGN_YAW->ARM_INIT->SCAN->
GRASP_PLAN->PRE_GRASP->ORIENT->REACH->CLOSE->LIFT->DONE) are identical
to navigate_to_goal.py.  Only the NAV state is replaced by Nav2.

Pre-requisites
--------------
1. sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-simple-commander
2. python scripts/navigation/convert_map.py --map maps/my_map.npz \\
       --out maps/my_map_nav2
3. In a separate terminal:
   source /opt/ros/humble/setup.bash
   ros2 launch nav2_bringup navigation_launch.py \\
       params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \\
       map:=$(pwd)/custom_envs/maps/my_map_nav2.yaml
4. source /opt/ros/humble/setup.bash && conda activate env_isaaclab
   python scripts/navigation/navigate_to_goal_nav2.py \\
       --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 \\
       --policy_task Flat-Deeprobotics-M20-v0 \\
       --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt \\
       --map maps/my_map.npz --goal 4.5 4.9 \\
       --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
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
    NAV          = "NAV"
    ALIGN_YAW_1  = "ALIGN_YAW_1"
    PAN_VX       = "PAN_VX"
    PAN_VY       = "PAN_VY"
    ALIGN_YAW    = "ALIGN_YAW"
    ARM_INIT     = "ARM_INIT"
    SCAN         = "SCAN"
    PRE_ADJUST   = "PRE_ADJUST"
    GRASP_PLAN   = "GRASP_PLAN"
    PRE_GRASP    = "PRE_GRASP"
    ORIENT       = "ORIENT"
    REACH        = "REACH"
    CLOSE        = "CLOSE"
    LIFT         = "LIFT"
    DONE         = "DONE"


def main():
    parser = argparse.ArgumentParser("Navigate to Goal (Nav2)")
    parser.add_argument("--task", default="Flat-Deeprobotics-M20Pro-Lidar-v0")
    parser.add_argument("--map", required=True)
    parser.add_argument("--goal", nargs=2, type=float, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_speed", type=float, default=0.8)
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy_task", default=None)
    parser.add_argument("--grasp_checkpoint", default=None)
    parser.add_argument("--grasp_topk", type=int, default=1)
    parser.add_argument("--nav2_arrival_radius", type=float, default=0.55)
    args, unknown = parser.parse_known_args()

    # ---- Isaac Sim launch ----
    from isaaclab.app import AppLauncher
    if args.grasp_checkpoint is not None:
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import custom_envs.tasks  # noqa: F401

    _piper_mode = (
        "Piper" in args.task
        and "Piper-Single" not in args.task
        and "TwoTables" not in args.task
    )
    if _piper_mode:
        from custom_envs.tasks.deeprobotics_m20_pro.piper_env_cfg import setup_piper_sync

    from custom_envs.utils.occupancy_grid import OccupancyGrid
    from custom_envs.utils.nav_utils import euler_from_quat, is_goal_reached

    grid = OccupancyGrid.load(args.map)
    goal_world = (args.goal[0], args.goal[1])

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_reset_base = None
    env_cfg.events.randomize_reset_joints = None
    env_cfg.terminations.time_out = None

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path

    policy_task = args.policy_task or args.task
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry as load_cfg
    agent_cfg = load_cfg(policy_task, "rsl_rl_cfg_entry_point")

    if "Lidar" not in policy_task and "Lidar" in args.task:
        print("[INFO] Policy has no LiDAR -- stripping LiDAR obs from env")
        env_cfg.observations.policy.lidar = None
        env_cfg.observations.critic.lidar = None
        if hasattr(env_cfg.scene, "mid360_lidar"):
            env_cfg.scene.mid360_lidar.debug_vis = False

    env = gym.make(args.task, cfg=env_cfg)
    obs = env.reset()[0]

    if _piper_mode:
        setup_piper_sync(env)

    _rl_training_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "deps", "rl_training")
    )
    log_root = os.path.join(_rl_training_root, "logs", "rsl_rl",
                            agent_cfg.experiment_name)

    if args.checkpoint and os.path.isfile(args.checkpoint):
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
                print(f"[ERROR] No runs in {log_root}")
                env.close(); simulation_app.close(); return
            run_dir = runs[-1]
        agent_cfg.load_run = os.path.basename(run_dir)
        resume_path = get_checkpoint_path(
            os.path.abspath(log_root), agent_cfg.load_run, "model_.*.pt")

    print(f"[INFO] Loading policy: {resume_path}")
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(),
                            log_dir=None, device="cuda:0")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print("[INFO] Policy loaded")

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))

    # ---- Start ROS2 bridge subprocess (publishes /tf, /odom; subscribes /cmd_vel)
    #      The subprocess runs under /usr/bin/python3.10 and owns all rclpy/Nav2 code.
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge
    bridge = IsaacROS2Bridge(cmd_vel_timeout=0.5)
    bridge.start(timeout=30.0)

    # Constants (copied verbatim from navigate_to_goal.py)
    ARM_JOINT_NAMES   = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
    LEG_JOINT_NAMES   = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    GRIPPER_OPEN_POS  = [ 0.035, -0.035]
    GRIPPER_CLOSE_POS = [-0.035,  0.035]
    PRE_GRASP_RETREAT = 0.1
    REACH_RETREAT     = 0.0
    ARM_HOME_ANGLES   = np.array([0.0, 0.5, -1.0, 0.0, 0.5, 0.0], dtype=np.float32)
    ARM_SIDE_ANGLES   = np.array([-math.pi/2, 1.5, -1.5, 0.0, 1.2, 0.0],
                                 dtype=np.float32)
    ARM_PREGRASP_ANGLES = np.array([-math.pi/2, 1.5, -1.5, 0.0, 0.0, 0.0],
                                   dtype=np.float32)
    TARGET_YAW        = math.pi / 2

    BUDGET = {
        PipelineState.ARM_INIT:   600,
        PipelineState.PRE_ADJUST: 0,
        PipelineState.PRE_GRASP:  500,
        PipelineState.ORIENT:     300,
        PipelineState.REACH:      250,
        PipelineState.CLOSE:      200,
        PipelineState.LIFT:       300,
    }
    PRE_GRASP_MAX_WORLD_ERR = 0.10
    PRE_GRASP_MAX_JOINT_ERR = 0.35
    PRE_GRASP_MAX_ROT_ERR   = 2.90
    REACH_MAX_WORLD_ERR     = 0.08
    REACH_MAX_JOINT_ERR     = 0.25
    REACH_MAX_ROT_ERR       = 0.80
    REACH_MAX_OBJECT_DIST   = 0.15
    SCAN_WARMUP  = 10
    SCAN_FRAMES  = 30

    # ---- Helper functions (copied verbatim from navigate_to_goal.py) ----
    def _get_arm_ids(robot):
        arm_ids = []
        for name in ARM_JOINT_NAMES:
            try:
                arm_ids.append(robot.find_joints(name)[0][0])
            except Exception:
                pass
        grip_ids = {}
        for name in GRIPPER_JOINT_NAMES:
            try:
                grip_ids[name] = robot.find_joints(name)[0][0]
            except Exception:
                pass
        return arm_ids, grip_ids

    def _get_leg_ids(robot):
        ids = []
        for name in LEG_JOINT_NAMES:
            try:
                ids.append(robot.find_joints(name)[0][0])
            except Exception:
                pass
        return ids

    def _arm_step(robot, target_angles):
        arm_ids, _ = _get_arm_ids(robot)
        pos_t = robot.data.joint_pos_target[0].clone()
        for i, jid in enumerate(arm_ids):
            pos_t[jid] = float(target_angles[i])
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _gripper_step(robot, close=False):
        _, grip_ids = _get_arm_ids(robot)
        pos_t = robot.data.joint_pos_target[0].clone()
        targets = GRIPPER_CLOSE_POS if close else GRIPPER_OPEN_POS
        for i, name in enumerate(GRIPPER_JOINT_NAMES):
            if name in grip_ids:
                pos_t[grip_ids[name]] = float(targets[i])
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _gripper_width_step(robot, width):
        half = float(np.clip(width / 2.0, 0.0, 0.035))
        _, grip_ids = _get_arm_ids(robot)
        pos_t = robot.data.joint_pos_target[0].clone()
        for name, sign in zip(GRIPPER_JOINT_NAMES, [1.0, -1.0]):
            if name in grip_ids:
                pos_t[grip_ids[name]] = sign * half
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _leg_step(robot, leg_q):
        leg_ids = _get_leg_ids(robot)
        pos_t = robot.data.joint_pos_target[0].clone()
        for i, jid in enumerate(leg_ids):
            pos_t[jid] = float(leg_q[i])
        robot.set_joint_position_target(pos_t.unsqueeze(0))

    def _alpha(s_state):
        b = BUDGET.get(s_state, 1)
        return float(min(state_step / max(b, 1), 1.0))

    def _ransac_remove_plane(pts, n_iter=150, dist_thresh=0.008, min_inlier_ratio=0.15):
        """RANSAC 平面去除。找到内点最多的主平面（桌面），返回 True=保留(非桌面)。
        若最大平面内点数 < 总点数的 min_inlier_ratio，认为场景无显著平面，保留所有点。"""
        N = len(pts)
        if N < 10:
            return np.ones(N, dtype=bool)
        best_n    = 0
        best_mask = np.ones(N, dtype=bool)
        best_info = [None, None]
        rng = np.random.default_rng(42)
        for _ in range(n_iter):
            idx = rng.choice(N, 3, replace=False)
            p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            nv = np.cross(p2 - p1, p3 - p1)
            nn = float(np.linalg.norm(nv))
            if nn < 1e-8:
                continue
            nv = nv / nn
            dv = float(nv @ p1)
            inlier = np.abs(pts @ nv - dv) < dist_thresh
            ni = int(inlier.sum())
            if ni > best_n:
                best_n    = ni
                best_mask = inlier
                best_info = [nv, dv]
        if best_n < int(N * min_inlier_ratio):
            print(f"[SM] RANSAC: no dominant plane (best={best_n}/{N}), keeping all.", flush=True)
            return np.ones(N, dtype=bool)
        keep = ~best_mask
        pct  = best_n * 100 // N
        print(f"[SM] RANSAC plane={best_n} pts ({pct}%), object pts={int(keep.sum())}, "
              f"normal={np.round(best_info[0], 3)} d={best_info[1]:.3f}", flush=True)
        return keep

    # ---- State machine variables ----
    state       = PipelineState.NAV
    state_step  = 0
    loop_count  = 0
    grasp_result = None
    depth_accum  = []
    scan_rgb     = None
    target_angles_arm = None
    _freeze_root_link_pose = None
    _frozen_leg_q   = None
    _frozen_leg_ids = None
    _pg_cmd = None
    _PG_MAX_DELTA = np.array([0.03, 0.05, 0.05, 0.04, 0.04, 0.04])
    _pg_target_cur = None
    _q_scan_at_scan = None
    _pos_w_at_scan  = None
    _quat_w_at_scan = None
    _orient_q_fixed = None
    _orient_j6_start = None
    j6_target = None
    _re_target_cur = None
    target_angles_arm = None  # re-declared for REACH/CLOSE/LIFT scope

    # Nav2-specific state
    _nav2_goal_sent = False
    _nav2_done      = False

    # ---- 预加载 YOLO 分割模型（避免每次 GRASP_PLAN 重复加载）----
    _yolo_seg = None
    try:
        from ultralytics import YOLO as _YOLO_CLS
        _yolo_model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "yolov8m-seg.pt")
        _yolo_seg = _YOLO_CLS(_yolo_model_path)
        print(f"[YOLO] 分割模型已加载: {_yolo_model_path}", flush=True)
    except Exception as _ye:
        print(f"[YOLO] 模型加载失败，将回退到排除桌面法: {_ye}", flush=True)

    raw_env = env.unwrapped

    # ---- Send Nav2 goal via bridge subprocess ----
    print(f"[NAV2] Sending goal: x={goal_world[0]:.2f} y={goal_world[1]:.2f}",
          flush=True)
    bridge.send_goal(float(goal_world[0]), float(goal_world[1]))
    _nav2_goal_sent = True
    print("[NAV2] Goal sent. Waiting for Nav2 to complete...", flush=True)

    os.makedirs("/home/mojie/taskdog/custom_envs/maps/nav_process",
                exist_ok=True)

    try:
        while simulation_app.is_running():
            loop_count += 1
            if state != PipelineState.GRASP_PLAN:
                state_step += 1

            # ---- Read robot pose ----
            robot = raw_env.scene["robot"]
            pos_w   = robot.data.root_pos_w[0].cpu().numpy()
            quat_w  = robot.data.root_quat_w[0].cpu().numpy()  # [w,x,y,z]
            yaw     = euler_from_quat(quat_w)
            robot_pose = (float(pos_w[0]), float(pos_w[1]), float(yaw))

            # ---- Publish ground-truth pose to Nav2 ----
            try:
                lin_vel = robot.data.root_com_vel_w[0, :3].cpu().numpy()
                ang_vel = robot.data.root_com_vel_w[0, 3:].cpu().numpy()
            except Exception:
                lin_vel = np.zeros(3)
                ang_vel = np.zeros(3)
            bridge.update_robot_pose(pos_w, quat_w, lin_vel, ang_vel)

            if state == PipelineState.DONE:
                print("[SM] DONE.", flush=True)
                break

            # ==============================================================
            # STATE MACHINE
            # ==============================================================

            # ---- NAV: Nav2 drives the robot; policy executes cmd_vel ----
            if state == PipelineState.NAV:
                # Poll Nav2 task completion via bridge
                _nav_done, _nav_failed = bridge.get_nav_status()
                if _nav_done or _nav_failed:
                    if _nav_done:
                        print("[NAV2] Task SUCCEEDED by Nav2.", flush=True)
                    else:
                        print("[NAV2] Task FAILED/CANCELLED by Nav2, checking distance.",
                              flush=True)
                    # Regardless of Nav2 result, check arrival distance
                    dist = np.hypot(pos_w[0] - goal_world[0],
                                    pos_w[1] - goal_world[1])
                    if dist <= args.nav2_arrival_radius:
                        print(f"[NAV2] Within {args.nav2_arrival_radius}m of goal "
                              f"(dist={dist:.3f}m). Transitioning.", flush=True)
                        if args.grasp_checkpoint:
                            print("[SM] NAV -> ALIGN_YAW_1", flush=True)
                            state = PipelineState.ALIGN_YAW_1
                            state_step = 0
                        else:
                            state = PipelineState.DONE
                        continue
                    else:
                        # Nav2 gave up but still too far — resend goal
                        print(f"[NAV2] dist={dist:.3f}m > {args.nav2_arrival_radius}m, "
                              f"resending goal.", flush=True)
                        bridge.send_goal(float(goal_world[0]), float(goal_world[1]))

                # Get cmd_vel from Nav2 and inject into locomotion policy
                vx, omega_z = bridge.get_cmd_vel()

                if loop_count % 100 == 0:
                    dist = np.hypot(pos_w[0] - goal_world[0],
                                    pos_w[1] - goal_world[1])
                    print(f"[NAV2] step={loop_count} pos=({pos_w[0]:.1f},{pos_w[1]:.1f}) "
                          f"yaw={np.degrees(yaw):.0f}deg dist={dist:.2f}m "
                          f"vx={vx:.2f} w={omega_z:.3f}", flush=True)

                p_obs = obs["policy"].clone()
                p_obs[0, 6] = vx
                p_obs[0, 7] = 0.0
                p_obs[0, 8] = omega_z
                obs["policy"] = p_obs
                with torch.inference_mode():
                    actions  = policy(obs)
                    step_res = env.step(actions)
                    obs      = step_res[0]
                continue  # back to top

            # ---- From ALIGN_YAW_1 onwards: policy keeps robot standing ----
            if state != PipelineState.GRASP_PLAN:
                p_obs = obs["policy"].clone()
                if state in (PipelineState.ALIGN_YAW_1, PipelineState.ALIGN_YAW):
                    _yaw_err = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                    p_obs[0, 6] = 0.0
                    p_obs[0, 7] = 0.0
                    p_obs[0, 8] = float(np.clip(100.0 * _yaw_err, -1.2, 1.2))
                elif state in (PipelineState.PAN_VX, PipelineState.PAN_VY):
                    _dx_w = goal_world[0] - robot_pose[0]
                    _dy_w = goal_world[1] - robot_pose[1]
                    if state == PipelineState.PAN_VX:
                        _err  = abs(_dy_w)
                        _v_cap = float(np.clip(0.4 * _err, 0.0, 0.3))
                        p_obs[0, 6] = float(np.clip(math.copysign(_v_cap, _dy_w), -0.3, 0.3))
                        p_obs[0, 7] = 0.0
                    else:
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
                    _FREEZE_STATES_ACT = {
                        PipelineState.ARM_INIT, PipelineState.SCAN,
                        PipelineState.PRE_ADJUST, PipelineState.PRE_GRASP,
                        PipelineState.ORIENT, PipelineState.REACH,
                        PipelineState.CLOSE, PipelineState.LIFT,
                    }
                    if state in _FREEZE_STATES_ACT:
                        actions = torch.zeros_like(actions)
                    step_res = env.step(actions)
                    obs      = step_res[0]

                _FREEZE_STATES_LEG = {
                    PipelineState.REACH, PipelineState.CLOSE, PipelineState.LIFT,
                }
                if state in _FREEZE_STATES_LEG and _frozen_leg_q is not None:
                    _leg_step(robot, _frozen_leg_q)
                    with torch.inference_mode(False):
                        robot.data.root_com_vel_w.data.fill_(0.0)
                        robot.data.body_acc_w.data.fill_(0.0)
                        if robot.data._root_com_state_w.data is not None:
                            robot.data._root_com_state_w.data.data[:, 7:].fill_(0.0)
                        if robot.data._root_state_w.data is not None:
                            robot.data._root_state_w.data.data[:, 7:].fill_(0.0)
                        robot.root_physx_view.set_root_velocities(
                            robot.data.root_com_vel_w.clone(),
                            indices=robot._ALL_INDICES)

                _FREEZE_STATES = {
                    PipelineState.ARM_INIT, PipelineState.SCAN,
                    PipelineState.PRE_ADJUST, PipelineState.PRE_GRASP,
                    PipelineState.ORIENT,
                }
                if state in _FREEZE_STATES and _freeze_root_link_pose is not None:
                    with torch.inference_mode(False):
                        _dev = robot.data.root_link_pose_w.device
                        robot.data.root_link_pose_w.data[0:1] = _freeze_root_link_pose
                        if robot.data._root_link_state_w.data is not None:
                            robot.data._root_link_state_w.data.data[0:1, :7] = \
                                _freeze_root_link_pose
                        if robot.data._root_state_w.data is not None:
                            robot.data._root_state_w.data.data[0:1, :7] = \
                                _freeze_root_link_pose
                        robot.data._body_link_pose_w.timestamp = -1.0
                        robot.data._body_com_pose_w.timestamp  = -1.0
                        robot.data._body_state_w.timestamp     = -1.0
                        robot.data._body_link_state_w.timestamp = -1.0
                        robot.data._body_com_state_w.timestamp  = -1.0
                        _pose_xyzw = _freeze_root_link_pose.clone()
                        _pose_xyzw[:, 3:] = torch.cat([
                            _freeze_root_link_pose[:, 4:],
                            _freeze_root_link_pose[:, 3:4],
                        ], dim=-1)
                        robot.root_physx_view.set_root_transforms(
                            _pose_xyzw, indices=robot._ALL_INDICES)
                        robot.data.root_com_vel_w.data.fill_(0.0)
                        robot.data.body_acc_w.data.fill_(0.0)
                        if robot.data._root_com_state_w.data is not None:
                            robot.data._root_com_state_w.data.data[:, 7:].fill_(0.0)
                        if robot.data._root_state_w.data is not None:
                            robot.data._root_state_w.data.data[:, 7:].fill_(0.0)
                        robot.root_physx_view.set_root_velocities(
                            robot.data.root_com_vel_w.clone(),
                            indices=robot._ALL_INDICES)

            # ---- ALIGN_YAW_1 ----
            if state == PipelineState.ALIGN_YAW_1:
                _yaw_err_1 = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                if state_step == 1:
                    print(f"[SM] ALIGN_YAW_1 start: cur={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_1):.1f}deg", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] ALIGN_YAW_1 step {state_step}: "
                          f"yaw={math.degrees(yaw):.1f}deg "
                          f"err={math.degrees(_yaw_err_1):.1f}deg", flush=True)
                if abs(_yaw_err_1) < 0.01 or state_step >= 400:
                    print("[SM] ALIGN_YAW_1 done -> PAN_VX", flush=True)
                    state = PipelineState.PAN_VX
                    state_step = 0

            # ---- PAN_VX ----
            elif state == PipelineState.PAN_VX:
                _dy_err = abs(robot_pose[1] - goal_world[1])
                if state_step == 1:
                    print(f"[SM] PAN_VX start dy_err={_dy_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VX step {state_step}: dy_err={_dy_err:.3f}m",
                          flush=True)
                if _dy_err < 0.05 or state_step >= 400:
                    print("[SM] PAN_VX done -> PAN_VY", flush=True)
                    state = PipelineState.PAN_VY
                    state_step = 0

            # ---- PAN_VY ----
            elif state == PipelineState.PAN_VY:
                _dx_err = abs(robot_pose[0] - goal_world[0])
                if state_step == 1:
                    print(f"[SM] PAN_VY start dx_err={_dx_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VY step {state_step}: dx_err={_dx_err:.3f}m",
                          flush=True)
                if _dx_err < 0.1 or state_step >= 400:
                    print("[SM] PAN_VY done -> ALIGN_YAW", flush=True)
                    state = PipelineState.ALIGN_YAW
                    state_step = 0

            # ---- ALIGN_YAW ----
            elif state == PipelineState.ALIGN_YAW:
                _yaw_err_align = (TARGET_YAW - yaw + math.pi) % (2 * math.pi) - math.pi
                if state_step == 1:
                    print(f"[SM] ALIGN_YAW start: cur={math.degrees(yaw):.1f}deg",
                          flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] ALIGN_YAW step {state_step}: "
                          f"err={math.degrees(_yaw_err_align):.1f}deg", flush=True)
                if abs(_yaw_err_align) < 0.02 or state_step >= 600:
                    print("[SM] ALIGN_YAW done -> ARM_INIT", flush=True)
                    _freeze_root_link_pose = robot.data.root_link_pose_w[0:1].clone()
                    state = PipelineState.ARM_INIT
                    state_step = 0

            # ---- ARM_INIT: retract arm to ARM_SIDE_ANGLES ----
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
                _arm_alpha  = min(1.3, state_step / max(_arm_budget * 0.6, 1))   # 超调系数
                q6 = cur_q + _arm_alpha * (ARM_SIDE_ANGLES - cur_q)   # 目标位置稍微超调
                _arm_step(robot, q6)   # 将计算出的目标角度 q6 发送给机器人，执行移动
                _gripper_step(robot, close=False)   # 夹爪张开
                if state_step == 1:
                    print(f"[SM] ARM_INIT: retracting arm... base_pos_w={np.round(pos_w,3)}", flush=True)
                    # Print banana actual resting position; sim ran during NAV
                    # so the banana has already settled on the table surface.
                    try:
                        banana = raw_env.scene["banana"]
                        bpos = banana.data.root_pos_w[0].cpu().numpy()   # 香蕉位置
                        print(
                            f"[INFO] Banana resting pos: "
                            f"x={bpos[0]:.4f}  y={bpos[1]:.4f}  z={bpos[2]:.4f}",
                            flush=True,
                        )
                    except Exception as _be:
                        print(f"[INFO] Could not read banana pos: {_be}", flush=True)
                _arm_err_now = np.abs(cur_q - ARM_SIDE_ANGLES)   # 当前角度到目标的差距
                if state_step % 50 == 0:
                    _ai_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] ARM_INIT step {state_step}/{BUDGET[PipelineState.ARM_INIT]}: "
                          f"q={np.round(cur_q,4)} err={np.round(_arm_err_now,4)} max={_arm_err_now.max():.4f} "
                          f"base_pos_w={np.round(_ai_pos_w,3)}",
                          flush=True)
                _arm_converged = (_arm_err_now.max() < 0.02)   # ~1.1 deg threshold
                _arm_timeout   = (state_step >= BUDGET[PipelineState.ARM_INIT])
                if _arm_converged or _arm_timeout:   # 阶段结束
                    _arm_final_err = _arm_err_now
                    _reason = "converged" if _arm_converged else "timeout"
                    _ai_done_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] ARM_INIT done ({_reason} at step {state_step}): final_q={np.round(cur_q,4)} base_pos_w={np.round(_ai_done_pos_w,3)}", flush=True)
                    print(f"[SM] ARM_INIT target:        {np.round(ARM_SIDE_ANGLES,4)}", flush=True)
                    print(f"[SM] ARM_INIT joint_err:     {np.round(_arm_final_err,4)} max={_arm_final_err.max():.4f} rad ({np.degrees(_arm_final_err.max()):.1f}deg)",
                          flush=True)
                    print(f"[SM] ARM_INIT freeze anchor (current): "
                          f"pos={np.round(_freeze_root_link_pose[0,:3].cpu().numpy(),4)}"
                          f" quat={np.round(_freeze_root_link_pose[0,3:].cpu().numpy(),4)}",
                          flush=True)
                    print("[SM] ARM_INIT done -> SCAN", flush=True)
                    if args.grasp_checkpoint:
                        state = PipelineState.SCAN
                    else:
                        state = PipelineState.DONE
                    state_step = 0
                    depth_accum = []
                    scan_rgb    = None

            # ---- SCAN: warm camera, accumulate depth frames ----
            elif state == PipelineState.SCAN:
                try:
                    camera = raw_env.scene["wrist_camera"]   # 获取名为 "wrist_camera" 的相机对象（手腕相机）
                    if state_step == 1:
                        _scan_start_pos_w = robot.data.root_pos_w[0].cpu().numpy()   # 记录机器人根坐标
                        print(f"[SM] SCAN: warmup {SCAN_WARMUP} + accumulate "
                              f"{SCAN_FRAMES} frames... base_pos_w={np.round(_scan_start_pos_w,3)}", flush=True)
                    if state_step == SCAN_WARMUP + 1:
                        # ---- Save RGB snapshot after warmup (first valid frame) ----
                        try:
                            from PIL import Image as _PIL_Image
                            _arm_ids_snap, _ = _get_arm_ids(robot)   # 获取手臂关节索引
                            _cur_q_snap = robot.data.joint_pos[
                                0, list(_arm_ids_snap)].cpu().numpy()   # 获取手臂关节角度
                            # joint indices: 0=j1,1=j2,2=j3,3=j4,4=j5,5=j6
                            def _fmt(v):   # 记录SCAN_WARMUP结束时的手臂关节角度
                                return f"{v:.2f}".replace("-", "n").replace(".", "p")
                            _snap_name = (
                                f"j2_{_fmt(_cur_q_snap[1])}"
                                f"_j3_{_fmt(_cur_q_snap[2])}"
                                f"_j4_{_fmt(_cur_q_snap[3])}"
                                f"_j5_{_fmt(_cur_q_snap[4])}.png"
                            )
                            _snap_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                            os.makedirs(_snap_dir, exist_ok=True)
                            _rgb_snap = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]   # 从相机输出中提取 RGB 图像（取第一张图，并保留前三个通道）
                            _PIL_Image.fromarray(_rgb_snap).save(
                                os.path.join(_snap_dir, _snap_name))
                            print(f"[SCAN] Snapshot saved: {_snap_dir}/{_snap_name}",
                                  flush=True)
                        except Exception as _snap_e:
                            raise RuntimeError(f"[SCAN] Snapshot failed: {_snap_e}") from _snap_e
                    if state_step > SCAN_WARMUP:   # 预热完毕
                        d = camera.data.output["distance_to_image_plane"][0].cpu().numpy()   # 获取深度图
                        if d.ndim == 3:   # 如果深度图是三维（宽×高×通道），取第一个通道（因为深度通常单通道）
                            d = d[:, :, 0]
                        depth_accum.append(d.astype(np.float32))   # 将深度图转换为 float32 并添加到累积列表
                        if scan_rgb is None:   # 如果 scan_rgb 尚未赋值，则保存当前的 RGB 图像
                            scan_rgb = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]
                    if state_step >= SCAN_WARMUP + SCAN_FRAMES:
                        depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)   # 取中位数深度图
                        valid_pct = np.mean((depth_med > 0.05) & (depth_med < 4.0)) * 100   # 有效深度像素比例
                        _scan_done_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                        print(f"[SM] SCAN done: {len(depth_accum)} frames, "
                              f"valid depth {valid_pct:.1f}% base_pos_w={np.round(_scan_done_pos_w,3)}", flush=True)
                        _q_scan_at_scan = robot.data.joint_pos[
                            0, list(_get_arm_ids(robot)[0])
                        ].cpu().numpy().copy()
                        _pos_w_at_scan  = robot.data.root_pos_w[0].cpu().numpy().copy()
                        _quat_w_at_scan = robot.data.root_quat_w[0].cpu().numpy().copy()
                        state = PipelineState.PRE_ADJUST
                        state_step = 0
                except (KeyError, AttributeError) as e:
                    print(f"[SM] SCAN camera unavailable ({e}) -- DONE.", flush=True)
                    state = PipelineState.DONE


            # ---- PRE_ADJUST: transition j5 from +1.2 to 0 before GraspNet IK ----
            elif state == PipelineState.PRE_ADJUST:
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                _pa_budget = BUDGET[PipelineState.PRE_ADJUST]
                _pa_delta = np.clip(ARM_PREGRASP_ANGLES - cur_q, -0.015, 0.015)
                q6_pa = cur_q + _pa_delta
                _arm_step(robot, q6_pa)
                _gripper_step(robot, close=False)
                if state_step == 1:
                    print(f"[SM] PRE_ADJUST: j5 {cur_q[4]:.3f} -> 0.0", flush=True)
                _pa_err = np.abs(cur_q - ARM_PREGRASP_ANGLES)
                if state_step % 50 == 0:
                    print(f"[SM] PRE_ADJUST step {state_step}: max_err={_pa_err.max():.4f}",
                          flush=True)
                _pa_converged = (_pa_err.max() < 0.05)
                _pa_timeout   = (state_step >= _pa_budget) if _pa_budget > 0 else True
                if _pa_converged or _pa_timeout:
                    print(f"[SM] PRE_ADJUST done -> GRASP_PLAN", flush=True)
                    state = PipelineState.GRASP_PLAN
                    state_step = 0

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
                print(f"[SM] Point cloud raw: {len(pts)} pts "
                      f"(z min={z_v.min():.2f} max={z_v.max():.2f})", flush=True)

                # ---- 颜色筛选桌面点云并保存（须在 RANSAC 之前，供穿桌检测使用）----
                import cv2 as _cv2_tb
                _cols_u8_tb = (cols * 255).astype(np.uint8).reshape(1, -1, 3)
                _hsv_tb = _cv2_tb.cvtColor(_cols_u8_tb, _cv2_tb.COLOR_RGB2HSV)[0]
                _table_mask_tb = (_hsv_tb[:, 1] < 40) & (_hsv_tb[:, 2] > 40) & (_hsv_tb[:, 2] < 160)
                _pts_table_cam = pts[_table_mask_tb]
                print(f'[SM] 桌面颜色点云: {len(_pts_table_cam)} pts', flush=True)
                if len(_pts_table_cam) > 50:
                    from arm_ik import fk_gripper as _fk_gb_tc, quat_to_rot as _q2r_tc, _CAM_OFFSET_POS as _cop_tc, _CAM_OFFSET_ROT as _cor_tc
                    _R_rob_tc    = _q2r_tc(_quat_w_at_scan.astype(np.float64))
                    _arm_base_tc = _pos_w_at_scan + _R_rob_tc @ np.array([0., 0., 0.0888])
                    _T_gb_tc     = _fk_gb_tc(_q_scan_at_scan.astype(np.float64))
                    _R_c2w_tc    = _R_rob_tc @ _T_gb_tc[:3, :3] @ _cor_tc
                    _t_cam_tc    = _R_rob_tc @ (_T_gb_tc[:3, :3] @ _cop_tc + _T_gb_tc[:3, 3]) + _arm_base_tc
                    _pts_table_world = (_pts_table_cam.astype(np.float64) @ _R_c2w_tc.T) + _t_cam_tc
                    np.savez('/tmp/table_cloud.npz', points=_pts_table_world.astype(np.float32))
                    print(f'[SM] 桌面点云已保存 /tmp/table_cloud.npz（{len(_pts_table_world)} pts）', flush=True)
                else:
                    print(f'[SM] WARN: 桌面颜色点云过少({len(_pts_table_cam)} pts)，跳过保存', flush=True)

                _keep_mask = _ransac_remove_plane(pts)
                pts  = pts[_keep_mask]
                cols = cols[_keep_mask]
                print(f"[SM] After plane removal: {len(pts)} pts sent to GraspNet", flush=True)

                if len(pts) < 200:
                    print("[SM] Too few points — DONE.", flush=True)
                    state = PipelineState.DONE
                else:
                    np.savez("/tmp/pointcloud.npz", points=pts, colors=cols)
                    worker = os.path.join(os.path.dirname(__file__), "grasp_worker.py")
                    _topk_worker = 50
                    print(f"[SM] Running grasp_worker (topk={_topk_worker})...", flush=True)
                    res = subprocess.run(
                        [_sys.executable, worker,
                         "--checkpoint", args.grasp_checkpoint,
                         "--topk", str(_topk_worker)],
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
                            # ---- YOLO 实例分割过滤（回退到排除桌面法）----
                            _best_grasp_idx = 0
                            _bmask = None
                            try:
                                import cv2 as _cv2_f
                                if scan_rgb is not None:
                                    if _yolo_seg is not None:
                                        # ---- YOLO 分割路径 ----
                                        _bgr_f = _cv2_f.cvtColor(scan_rgb, _cv2_f.COLOR_RGB2BGR)
                                        _yolo_results = _yolo_seg(_bgr_f, verbose=False)
                                        _banana_seg_mask = np.zeros(
                                            (scan_rgb.shape[0], scan_rgb.shape[1]), dtype=bool)
                                        for _r in _yolo_results:
                                            if _r.masks is None:
                                                continue
                                            _cls_ids = _r.boxes.cls.cpu().numpy().astype(int)
                                            for _mi, _cls in enumerate(_cls_ids):
                                                if _cls == 46:  # COCO banana
                                                    _seg_f = _r.masks.data[_mi].cpu().numpy()
                                                    _seg_u8 = (_seg_f * 255).astype(np.uint8)
                                                    _seg_bin = _cv2_f.resize(
                                                        _seg_u8,
                                                        (_banana_seg_mask.shape[1],
                                                         _banana_seg_mask.shape[0]),
                                                        interpolation=_cv2_f.INTER_NEAREST,
                                                    ) > 127
                                                    _banana_seg_mask |= _seg_bin
                                        _n_bpx = int(_banana_seg_mask.sum())
                                        print(f"[SM] YOLO 香蕉像素: {_n_bpx} px", flush=True)
                                        # 保存带识别框的调试图
                                        try:
                                            _annotated_bgr = _yolo_results[0].plot()
                                            _detect_path = "/home/mojie/taskdog/custom_envs/tmp_pictures/detect.png"
                                            _cv2_f.imwrite(_detect_path, _annotated_bgr)
                                            print(f"[SM] YOLO 识别图已保存: {_detect_path}", flush=True)
                                        except Exception as _de:
                                            print(f"[SM] detect.png 保存失败: {_de}", flush=True)
                                        # YOLO路径：2D mask → 点云索引 → KDTree 过滤
                                        _depth_valid_mask = (depth_med > 0.05) & (depth_med < 4.0)
                                        _banana_in_cloud = _banana_seg_mask[_depth_valid_mask][_keep_mask]
                                        _banana_pts_cam = pts[_banana_in_cloud]
                                        print(f"[SM] YOLO 香蕉点云: {len(_banana_pts_cam)} pts", flush=True)
                                        if len(_banana_pts_cam) >= 20:
                                            from scipy.spatial import cKDTree as _KDTree
                                            _kd = _KDTree(_banana_pts_cam)
                                            _trans_all = gr["translations"]
                                            _dists_all, _ = _kd.query(_trans_all, k=1)
                                            _bmask = _dists_all < 0.05
                                            _n_ok = int(_bmask.sum())
                                            print(f"[SM] YOLO过滤: {_n_ok}/{len(_trans_all)} 落在香蕉上", flush=True)
                                            if _n_ok > 0:
                                                _best_grasp_idx = int(np.where(_bmask)[0][0])
                                            else:
                                                print("[SM] 无抓取落在香蕉区域，使用最高分 [0]", flush=True)
                                        else:
                                            print(f"[SM] YOLO香蕉点云不足({len(_banana_pts_cam)}pts)，使用最高分 [0]", flush=True)
                                    else:
                                        # ---- 排除桌面法（YOLO 不可用时回退）----
                                        # 对 post-RANSAC 点云颜色做 HSV，找灰白色桌面点，排除落在桌面上的候选
                                        _cols_u8_fb = (cols * 255).astype(np.uint8).reshape(1, -1, 3)
                                        _hsv_pts_fb = _cv2_f.cvtColor(_cols_u8_fb, _cv2_f.COLOR_RGB2HSV)[0]
                                        _table_mask_fb = (
                                            (_hsv_pts_fb[:, 1] < 40) &
                                            (_hsv_pts_fb[:, 2] > 40) &
                                            (_hsv_pts_fb[:, 2] < 160)
                                        )
                                        _table_pts_fb = pts[_table_mask_fb]
                                        print(f"[SM] 排桌面回退: 桌面点 {len(_table_pts_fb)} pts", flush=True)
                                        if len(_table_pts_fb) > 20:
                                            from scipy.spatial import cKDTree as _KDTree_fb
                                            _kd_fb = _KDTree_fb(_table_pts_fb)
                                            _dists_tb, _ = _kd_fb.query(gr["translations"], k=1)
                                            _bmask = _dists_tb >= 0.03  # 排除桌面上的候选，保留其余
                                            _n_ok_fb = int(_bmask.sum())
                                            print(f"[SM] 排桌面: 保留 {_n_ok_fb}/{len(gr['translations'])} 非桌面候选", flush=True)
                                            if _n_ok_fb > 0:
                                                _best_grasp_idx = int(np.where(_bmask)[0][0])
                                            else:
                                                print("[SM] 无非桌面候选，使用最高分 [0]", flush=True)
                                        else:
                                            print("[SM] 桌面点云不足，使用最高分 [0]", flush=True)
                            except Exception as _cfe:
                                print(f"[SM] 目标过滤异常: {_cfe}", flush=True)
                            # ---- IK pre-screening + ranking ----
                            _q_scan_saved   = _q_scan_at_scan
                            _pos_w_scan_pre  = _pos_w_at_scan.copy()
                            _quat_w_scan_pre = _quat_w_at_scan.copy()
                            from arm_ik import (
                                solve as _ik_solve_pos,
                                cam_to_world as _ctw_pre,
                                world_pos_to_arm_frame as _w2a_pre,
                                compute_desired_ee_rot_in_arm as _cder,
                                quat_to_rot as _q2r_scan,
                                fk_gripper  as _fkg_scan,
                                _CAM_OFFSET_ROT as _COR_scan,
                            )
                            _R_rob_scan  = _q2r_scan(_quat_w_scan_pre)
                            _R_gb_scan   = _fkg_scan(_q_scan_saved)[:3, :3]
                            _R_cam2world = _R_rob_scan @ _R_gb_scan @ _COR_scan
                            _banana_idxs_pre = list(np.where(_bmask)[0]) if _bmask is not None else [_best_grasp_idx]
                            if not _banana_idxs_pre:
                                _banana_idxs_pre = [_best_grasp_idx]
                            _ranked_pre = []
                            for _ci_pre in _banana_idxs_pre:
                                _t_ci  = gr["translations"][_ci_pre]
                                _tw_ci = _ctw_pre(_t_ci, _q_scan_saved, _pos_w_scan_pre, _quat_w_scan_pre)
                                _ta_ci = _w2a_pre(_tw_ci, _pos_w_scan_pre, _quat_w_scan_pre)
                                try:
                                    _q_pre = _ik_solve_pos(_ta_ci, target_rot=None, initial_angles=_q_scan_saved)
                                    _ik_ok_pre = (float(_q_pre[1]) < 2.8)
                                except Exception:
                                    _ik_ok_pre = False
                                _R_cam_ci = gr["rotations"][_ci_pre]
                                _approach_world_ci = _R_cam2world @ _R_cam_ci[:, 0]
                                _approach_z_ci = float(_approach_world_ci[2])
                                _ranked_pre.append((_ik_ok_pre, float(abs(_approach_z_ci)), float(gr["scores"][_ci_pre]), _ci_pre, _approach_z_ci))
                            _ranked_pre.sort(key=lambda x: (-int(x[0]), x[4], -x[2]))
                            _ranked_idxs = [x[3] for x in _ranked_pre]
                            print(f"[SM] 候选排序({len(_ranked_idxs)}个): "
                                  + " ".join(f"[{r[3]}]{'✓' if r[0] else '✗'}s={r[2]:.2f}az={r[4]:.2f}"
                                             for r in _ranked_pre[:6]), flush=True)
                            _best_grasp_idx = _ranked_idxs[0]
                            _R_desired = _cder(gr["rotations"][_best_grasp_idx], _q_scan_saved)
                            grasp_result = {
                                "t_cam":    gr["translations"][_best_grasp_idx],
                                "R_cam":    gr["rotations"][_best_grasp_idx],
                                "score":    float(gr["scores"][_best_grasp_idx]),
                                "width":    float(gr["widths"][_best_grasp_idx]),
                                "R_desired_EE_in_arm": _R_desired,
                                "q_scan":   _q_scan_saved,
                                "pos_w_scan":  _pos_w_scan_pre,
                                "quat_w_scan": _quat_w_scan_pre,
                                "ranked_candidate_idxs": _ranked_idxs,
                                "gr_translations": gr["translations"],
                                "gr_rotations":    gr["rotations"],
                                "gr_scores":       gr["scores"],
                                "gr_widths":       gr["widths"],
                                "tried_set":       {_best_grasp_idx},
                            }
                            print(f"[SM] Best grasp score={grasp_result['score']:.3f}", flush=True)
                            # ---- DIAG + APPROACH SANITY CHECK ----
                            try:
                                from arm_ik import (
                                    cam_to_world as _c2w, quat_to_rot as _q2r,
                                    _CAM_OFFSET_ROT as _COR, fk_gripper as _fkg,
                                )
                                _t_c = grasp_result["t_cam"]
                                _R_c = grasp_result["R_cam"]
                                _T_fk  = _fkg(grasp_result["q_scan"])
                                _R_cw  = _q2r(quat_w) @ _T_fk[:3, :3] @ _COR
                                _app_w = _R_cw @ _R_c[:, 0]
                                _t_obj_w = _c2w(_t_c, grasp_result["q_scan"], pos_w, quat_w)
                                print(f"[DIAG] t_cam={np.round(_t_c,4)} depth={_t_c[2]:.3f}m", flush=True)
                                print(f"[DIAG] t_obj_world={np.round(_t_obj_w,4)} z={_t_obj_w[2]:.3f}m", flush=True)
                                print(f"[DIAG] approach world={np.round(_app_w,3)}", flush=True)
                                _approach_world_z = float(_app_w[2])
                                if _approach_world_z > 0.0:
                                    print(f"[WARN] approach z={_approach_world_z:.3f}>0, flipping.", flush=True)
                                    _R_flipped = grasp_result["R_cam"].copy()
                                    _R_flipped[:, 0] = -_R_flipped[:, 0]
                                    _R_flipped[:, 1] = -_R_flipped[:, 1]
                                    from arm_ik import compute_desired_ee_rot_in_arm as _cder2
                                    grasp_result["R_cam"] = _R_flipped
                                    grasp_result["R_desired_EE_in_arm"] = _cder2(
                                        _R_flipped, grasp_result["q_scan"])
                                    print(f"[WARN] After flip z={float((_R_cw@_R_flipped[:,0])[2]):.3f}", flush=True)
                                else:
                                    print(f"[INFO] approach z={_approach_world_z:.3f}<0 (OK)", flush=True)
                            except Exception as _de:
                                print(f"[DIAG] failed: {_de}", flush=True)
                            state = PipelineState.PRE_GRASP
                            state_step = 0


            # ---- PRE_GRASP: move arm to GraspNet target ----
            elif state == PipelineState.PRE_GRASP:
                from arm_ik import (solve_for_gripper_base as ik_solve_gb,
                                    cam_to_world, world_pos_to_arm_frame,
                                    _IK_JOINT_LIMITS, fk_gripper as _fk_pg,
                                    quat_to_rot as _q2r_pg,
                                    cam_to_world as _c2w_pg)
                if state_step == 1:
                    t_cam = grasp_result["t_cam"]
                    R_cam = grasp_result["R_cam"]
                    width = grasp_result["width"]
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    t_world = cam_to_world(
                        t_cam, grasp_result["q_scan"],
                        grasp_result["pos_w_scan"], grasp_result["quat_w_scan"])
                    _pg_t_world_fixed = t_world.copy()
                    pre_t = world_pos_to_arm_frame(t_world, pos_w, quat_w)
                    _R_rob_pg = _q2r_pg(quat_w)
                    _T_fk_pg  = _fk_pg(grasp_result["q_scan"])
                    from arm_ik import _CAM_OFFSET_ROT as _COR_pg
                    _R_cw_pg  = _R_rob_pg @ _T_fk_pg[:3, :3] @ _COR_pg
                    _approach_arm_pg = (_R_rob_pg.T @ (_R_cw_pg @ R_cam[:, 0]))
                    from arm_ik import _CAM_OFFSET_POS as _COP_pg
                    _arm_base_pg = pos_w + _R_rob_pg @ np.array([0., 0., 0.0888])
                    _J7_OFFSET = 0.16
                    pre_t_gb = pre_t - _J7_OFFSET * _approach_arm_pg
                    R_desired_EE = grasp_result["R_desired_EE_in_arm"]
                    target_angles_arm = ik_solve_gb(
                        pre_t_gb, target_rot_j7=R_desired_EE,
                        initial_angles=grasp_result["q_scan"])
                    # IK quality check
                    from arm_ik import fk_gripper as _fkg_pgq, _IK_JOINT_LIMITS as _jlims_pgq
                    _ik_fail = target_angles_arm is None
                    if not _ik_fail:
                        _T_ik_pgq = _fkg_pgq(target_angles_arm)
                        _ik_pos_err = float(np.linalg.norm(_T_ik_pgq[:3, 3] - pre_t_gb))
                        _ik_at_lim = any(
                            abs(float(_qi) - _lo) < 0.01 or abs(float(_qi) - _hi) < 0.01
                            for _qi, (_lo, _hi) in zip(target_angles_arm, _jlims_pgq)
                        )
                        _ik_fail = (_ik_pos_err >= 0.03 or _ik_at_lim)
                        print(f"[DIAG] PRE_GRASP IK pos_err={_ik_pos_err*100:.1f}cm at_lim={_ik_at_lim} {'✗ FAIL' if _ik_fail else '✓ OK'}", flush=True)
                    if _ik_fail:
                        _cand_list = grasp_result.get("ranked_candidate_idxs", [])
                        _tried     = grasp_result.get("tried_set", set())
                        _next_ci   = next((i for i in _cand_list if i not in _tried), None)
                        if _next_ci is None:
                            print("[WARN] PRE_GRASP: all candidates IK-failed -> ARM_INIT", flush=True)
                            grasp_result = None; depth_accum.clear(); scan_rgb = None
                            target_angles_arm = None
                            state = PipelineState.ARM_INIT; state_step = 0; continue
                        _tried.add(_next_ci)
                        grasp_result["tried_set"]   = _tried
                        grasp_result["t_cam"]        = grasp_result["gr_translations"][_next_ci]
                        grasp_result["R_cam"]        = grasp_result["gr_rotations"][_next_ci]
                        grasp_result["width"]        = float(grasp_result["gr_widths"][_next_ci])
                        grasp_result["score"]        = float(grasp_result["gr_scores"][_next_ci])
                        from arm_ik import compute_desired_ee_rot_in_arm as _cder_pg
                        grasp_result["R_desired_EE_in_arm"] = _cder_pg(
                            grasp_result["gr_rotations"][_next_ci], grasp_result["q_scan"])
                        print(f"[SM] PRE_GRASP IK FAIL -> switch to candidate [{_next_ci}]", flush=True)
                        state_step = 0; continue
                    grasp_result["target_angles_pre"] = target_angles_arm.copy()
                    grasp_result["pre_t_gb_arm"]      = pre_t_gb.copy()
                    grasp_result["approach_arm"]      = _approach_arm_pg.copy()
                    _arm_base_w_pg = pos_w + _R_rob_pg @ np.array([0., 0., 0.0888])
                    _pg_t_gb_world_fixed = _R_rob_pg @ pre_t_gb + _arm_base_w_pg
                    print(f"[SM] PRE_GRASP IK={np.round(target_angles_arm,3)}", flush=True)
                    print(f"[DIAG] PRE_GRASP gripper_base IK target world={np.round(_pg_t_gb_world_fixed,4)}", flush=True)
                    _pg_cmd = cur_q.copy()
                    _pg_target_cur = target_angles_arm.copy()
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                for _ji in range(6):
                    _pg_cmd[_ji] += float(np.clip(
                        _pg_target_cur[_ji] - _pg_cmd[_ji],
                        -_PG_MAX_DELTA[_ji], +_PG_MAX_DELTA[_ji]))
                _pg_cmd = np.clip(_pg_cmd,
                                  [lo for lo, hi in _IK_JOINT_LIMITS],
                                  [hi for lo, hi in _IK_JOINT_LIMITS])
                _arm_step(robot, _pg_cmd)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - _pg_target_cur)
                    _pg_step_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] PRE_GRASP step {state_step}/{BUDGET[PipelineState.PRE_GRASP]}: "
                          f"max_err={_err.max():.3f} base_pos_w={np.round(_pg_step_pos_w,3)}", flush=True)
                _pg_cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                _pg_err_check = np.abs(_pg_cur_q - _pg_target_cur)
                _pg_early_exit = (state_step > 150 and _pg_err_check.max() < 0.05)
                if _pg_early_exit and state_step % 50 != 0:
                    _pg_exit_pos_w = robot.data.root_pos_w[0].cpu().numpy()
                    print(f"[SM] PRE_GRASP early-exit at step {state_step}: "
                          f"max_err={_pg_err_check.max():.4f} rad < 0.05 base_pos_w={np.round(_pg_exit_pos_w,3)}",
                          flush=True)
                if state_step >= BUDGET[PipelineState.PRE_GRASP] or _pg_early_exit:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    # ---- DIAG: convergence check (three-gate) ----
                    from arm_ik import fk as _fk_pgd, fk_gripper as _fkg_pgd, quat_to_rot as _q2r_pgd
                    from arm_ik import _RX_NEG90 as _RX_NEG90_pgd
                    _T_final_gb = _fkg_pgd(cur_q_final)
                    _T_final    = _fk_pgd(cur_q_final)
                    _ee_final_arm = _T_final_gb[:3, 3]
                    _R_final_j7  = _T_final[:3, :3]
                    _R_final_gb  = _R_final_j7  @ _RX_NEG90_pgd
                    _R_desired_gb = R_desired_EE @ _RX_NEG90_pgd
                    _R_rob_pgd = _q2r_pgd(quat_w)
                    _arm_base_offset_pgd = _R_rob_pgd @ np.array([0., 0., 0.0888])
                    _ee_final_world = _R_rob_pgd @ _ee_final_arm + pos_w + _arm_base_offset_pgd
                    _pos_err_final = float(np.linalg.norm(_ee_final_world - _pg_t_gb_world_fixed))
                    _joint_err_final = np.abs(cur_q_final - target_angles_arm)
                    _rot_err_final = np.linalg.norm(_R_final_gb - _R_desired_gb, ord='fro')
                    print(f"[DIAG] PRE_GRASP pos_err={_pos_err_final*100:.1f}cm "
                          f"joint_err_max={_joint_err_final.max()*57.3:.1f}deg "
                          f"rot_err={_rot_err_final:.4f}", flush=True)
                    _pg_can_advance = (
                        _pos_err_final <= PRE_GRASP_MAX_WORLD_ERR
                        and _joint_err_final.max() <= PRE_GRASP_MAX_JOINT_ERR
                        and _rot_err_final <= PRE_GRASP_MAX_ROT_ERR
                    )
                    if not _pg_can_advance:
                        print(f"[WARN] PRE_GRASP reject -> ARM_INIT: "
                              f"pos_err={_pos_err_final*100:.1f}cm "
                              f"joint_err={_joint_err_final.max()*57.3:.1f}deg "
                              f"rot_err={_rot_err_final:.4f}", flush=True)
                        grasp_result = None; depth_accum.clear(); scan_rgb = None
                        target_angles_arm = None
                        state = PipelineState.ARM_INIT; state_step = 0; continue
                    print(f"[SM] PRE_GRASP done -> ORIENT", flush=True)
                    state = PipelineState.ORIENT
                    state_step = 0


            # ---- ORIENT: rotate joint6 only to align closing axis with AnyGrasp direction ----
            elif state == PipelineState.ORIENT:
                from arm_ik import extract_j6_angle, _RX_NEG90 as _RXN90_or, fk_gripper as _fkg_or
                if state_step == 1:
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    R_gb_desired_or = grasp_result["R_desired_EE_in_arm"] @ _RXN90_or
                    j6_target = extract_j6_angle(cur_q, R_gb_desired_or)
                    _orient_q_fixed = grasp_result["target_angles_pre"].copy()
                    _orient_j6_start = float(cur_q[5])
                    print(f"[SM] ORIENT j6_target={j6_target:.4f} rad ({math.degrees(j6_target):.1f} deg), "
                          f"j6_start={_orient_j6_start:.4f} rad", flush=True)
                    # DIAG: verify by FK
                    _q_j6t_or = _orient_q_fixed.copy(); _q_j6t_or[5] = j6_target
                    _R_gb_after_or = _fkg_or(_q_j6t_or)[:3, :3]
                    _rot_err_or = np.linalg.norm(_R_gb_after_or - R_gb_desired_or, ord='fro')
                    print(f"[DIAG] ORIENT FK verify rot_err={_rot_err_or:.6f} (expect <0.001)", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q_cmd = _orient_q_fixed.copy()
                q_cmd[5] = j6_target
                _arm_step(robot, q_cmd)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    j6_err = abs(cur_q[5] - j6_target)
                    print(f"[SM] ORIENT step {state_step}/{BUDGET[PipelineState.ORIENT]}: "
                          f"j6_err={j6_err:.4f} rad", flush=True)
                if state_step >= BUDGET[PipelineState.ORIENT]:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    print(f"[SM] ORIENT done: j6={cur_q_final[5]:.4f} (target={j6_target:.4f}) -> REACH",
                          flush=True)
                    _frozen_leg_ids = _get_leg_ids(robot)
                    _frozen_leg_q = robot.data.joint_pos[
                        0, _frozen_leg_ids].cpu().numpy().copy()
                    print(f"[SM] ORIENT done: frozen leg_q={np.round(_frozen_leg_q, 3)}", flush=True)
                    state = PipelineState.REACH
                    state_step = 0

            # ---- REACH: advance along approach axis toward object ----
            elif state == PipelineState.REACH:
                from arm_ik import solve_for_gripper_base as ik_solve_gb_re, _IK_JOINT_LIMITS
                if state_step == 1:
                    cur_q = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])
                    ].cpu().numpy()
                    REACH_ADVANCE = 0.06
                    _re_pre_t_gb = grasp_result["pre_t_gb_arm"]
                    _re_approach = grasp_result["approach_arm"]
                    _re_t_gb_arm = _re_pre_t_gb + REACH_ADVANCE * _re_approach
                    R_arm = grasp_result["R_desired_EE_in_arm"]
                    target_angles_arm = ik_solve_gb_re(
                        _re_t_gb_arm, target_rot_j7=R_arm,
                        initial_angles=grasp_result["target_angles_pre"])
                    _re_q_jump = np.abs(target_angles_arm - grasp_result["target_angles_pre"])
                    print(f"[DIAG] REACH jump check: max={_re_q_jump.max()*57.3:.1f}deg threshold=28.6deg", flush=True)
                    if _re_q_jump.max() > 0.5:
                        print(f"[WARN] REACH IK jump too large ({_re_q_jump.max()*57.3:.1f}deg) -> fallback", flush=True)
                        target_angles_arm = grasp_result["target_angles_pre"].copy()
                        _re_t_gb_arm = _re_pre_t_gb.copy()
                    _re_target_cur = target_angles_arm.copy()
                    from arm_ik import quat_to_rot as _q2r_re, fk_gripper as _fkg_re
                    _R_rob_re = _q2r_re(quat_w)
                    _arm_base_re = pos_w + _R_rob_re @ np.array([0., 0., 0.0888])
                    _re_t_world_fixed = _R_rob_re @ _re_t_gb_arm + _arm_base_re
                    _re_orient_end_world = _R_rob_re @ _fkg_re(cur_q)[:3, 3] + _arm_base_re
                    print(f"[SM] REACH: advance {REACH_ADVANCE*100:.0f}cm along approach", flush=True)
                    print(f"[SM] REACH target world={np.round(_re_t_world_fixed,3)}", flush=True)
                    print(f"[SM] REACH IK={np.round(target_angles_arm,3)}", flush=True)
                cur_q = robot.data.joint_pos[
                    0, list(_get_arm_ids(robot)[0])
                ].cpu().numpy()
                q6 = np.clip(_re_target_cur.copy(), [lo for lo, hi in _IK_JOINT_LIMITS],
                                                    [hi for lo, hi in _IK_JOINT_LIMITS])
                _arm_step(robot, q6)
                _gripper_width_step(robot, grasp_result["width"])
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - _re_target_cur)
                    _ee_now_arm = _fkg_re(cur_q)[:3, 3]
                    _ee_now_world = _R_rob_re @ _ee_now_arm + _arm_base_re
                    _disp = _ee_now_world - _re_orient_end_world
                    print(f"[SM] REACH step {state_step}/{BUDGET[PipelineState.REACH]}: "
                          f"max_err={_err.max():.3f} |disp|={float(np.linalg.norm(_disp))*100:.1f}cm", flush=True)
                if state_step >= BUDGET[PipelineState.REACH]:
                    cur_q_final = robot.data.joint_pos[
                        0, list(_get_arm_ids(robot)[0])].cpu().numpy()
                    print(f"[SM] REACH done: final={np.round(cur_q_final,3)} -> CLOSE", flush=True)
                    # DIAG: distance to banana
                    try:
                        _banana_re = raw_env.scene["banana"]
                        _bp_re = _banana_re.data.root_pos_w[0].cpu().numpy()
                        _ee_re_arm = _fkg_re(cur_q_final)[:3, 3]
                        _ee_re_world = _R_rob_re @ _ee_re_arm + _arm_base_re
                        _dist_re = float(np.linalg.norm(_ee_re_world - _bp_re))
                        print(f"[DIAG] REACH banana world={np.round(_bp_re,4)}", flush=True)
                        print(f"[DIAG] REACH EE-banana dist={_dist_re*100:.1f}cm", flush=True)
                    except Exception:
                        pass
                    state = PipelineState.CLOSE
                    state_step = 0

            # ---- CLOSE: close gripper fully while holding arm at REACH pose ----
            elif state == PipelineState.CLOSE:
                _close_budget = BUDGET[PipelineState.CLOSE]
                _arm_step(robot, target_angles_arm)   # j1~j6的target不变
                _gripper_step(robot, close=True)   # 关闭夹爪
                if state_step == 1:
                    _gids = _get_arm_ids(robot)[1]
                    _gj = robot.data.joint_pos[0, list(_gids.values())].cpu().numpy()
                    print(f"[SM] CLOSE: full close throughout budget={_close_budget}"
                          f" (cur_gripper={np.round(_gj,4)})...", flush=True)
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
                q6 = (1.0 - _alpha(state)) * cur_q + _alpha(state) * ARM_SIDE_ANGLES
                _arm_step(robot, q6)
                _gripper_step(robot, close=True)   # hold grip throughout
                if state_step % 50 == 0:
                    _err = np.abs(cur_q - ARM_SIDE_ANGLES)
                    print(f"[SM] LIFT step {state_step}/{BUDGET[PipelineState.LIFT]}: "
                          f"max_err={_err.max():.3f}", flush=True)
                if state_step >= BUDGET[PipelineState.LIFT]:
                    try:
                        _banana_pos = raw_env.scene["banana"].data.root_pos_w[0].cpu().numpy()
                        _banana_z = float(_banana_pos[2])
                    except Exception:
                        _banana_z = 0.0
                    if _banana_z > 0.75:
                        print(f"[SM] LIFT done: banana z={_banana_z:.3f}m > 0.75 -> DONE (SUCCESS)",
                              flush=True)
                        state = PipelineState.DONE
                    else:
                        print(f"[SM] LIFT done: banana z={_banana_z:.3f}m <= 0.75 -> GRASP FAILED, retry PRE_GRASP",
                              flush=True)
                        grasp_result = None
                        depth_accum.clear()
                        scan_rgb = None
                        target_angles_arm = None
                        state = PipelineState.ARM_INIT
                        state_step = 0
                        continue


    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        try:
            bridge.stop()
        except Exception:
            pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

