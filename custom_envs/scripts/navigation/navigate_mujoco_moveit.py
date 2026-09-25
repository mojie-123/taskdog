#!/usr/bin/env python
"""navigate_mujoco_moveit.py — MuJoCo + Nav2 + AnyGrasp + MoveIt 全流程脚本。

对应 navigate_to_goal_nav2_whole.py，替换所有 Isaac Lab API，
复用现有 ROS2 Nav2 bridge、grasp_worker 与 MuJoCo 控制层；抓取规划改由 MoveIt 负责。

用法示例：
  python custom_envs/scripts/navigation/navigate_mujoco_moveit.py \\
      --map maps/my_map.npz \\
      --goal 4.5 5.0 \\
      --policy_path /path/to/model_4999.pt \\
      --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \\
      --destination 8.0 6.0 \\
      --object banana
"""

import argparse
import enum
from dataclasses import dataclass, field
from typing import Any
import math
import os
import sys
import time

# 在任何 mujoco/glfw 被 import 之前强制指定 X11 后端
# 修复 Wayland 会话下 MuJoCo passive viewer 鼠标/键盘完全无响应的问题
# PYGLFW_LIBRARY_VARIANT=x11 强制加载 glfw/x11/libglfw.so（而非 wayland 版本）
os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')
os.environ.setdefault('MUJOCO_GL', 'glx')
os.environ.setdefault('DISPLAY', ':0')

import numpy as np
import torch

from navigate_mujoco_diagnostics import (
    capture_body_pose as _diag_capture_body_pose,
    finger_contact_info as _diag_finger_contact_info,
    print_close_step_diag as _diag_print_close_step,
    print_finger_deep_diag as _diag_print_finger_deep,
    print_reach_done_diag as _diag_print_reach_done,
    save_choice_image as _diag_save_choice_image,
    save_filter_image as _diag_save_filter_image,
    save_scan_raw as _diag_save_scan_raw,
)

# ── 路径设置 ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "utils"))


# ── 当前实际状态机 ──
class PipelineState(enum.Enum):
    NAV          = "NAV"
    ALIGN_YAW_1  = "ALIGN_YAW_1"
    PAN_XY       = "PAN_XY"
    ARM_INIT     = "ARM_INIT"
    SCAN         = "SCAN"
    GRASP_PLAN   = "GRASP_PLAN"
    MOVEIT_PRE_GRASP = "MOVEIT_PRE_GRASP"
    MOVEIT_APPROACH  = "MOVEIT_APPROACH"
    LOCAL_REACH      = "LOCAL_REACH"
    CLOSE        = "CLOSE"
    LIFT         = "LIFT"
    PAN_NEG_X    = "PAN_NEG_X"
    NAV2_DEST    = "NAV2_DEST"
    ALIGN_YAW_2  = "ALIGN_YAW_2"
    PAN_DES_X    = "PAN_DES_X"
    ROTATE       = "ROTATE"
    PUT_DOWN     = "PUT_DOWN"
    DONE         = "DONE"


# ── 机械臂常量（和 Isaac Lab 版一致）──
ARM_JOINT_NAMES     = ["joint1","joint2","joint3","joint4","joint5","joint6"]
GRIPPER_OPEN_POS    = np.array([ 0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE_POS   = np.array([ 0.000,  0.000], dtype=np.float32)
ARM_INIT_ANGLES     = np.array([0.0,  0.5, -1.0, 0.0,  0.5, 0.0], dtype=np.float32)  # = ARM_HOME_ANGLES in navigate_to_goal_nav2_whole.py
ARM_SIDE_ANGLES     = np.array([-math.pi/2, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float32)  # matches navigate_to_goal_nav2_whole.py

# BUDGET（每个状态最大步数）
BUDGET = {
    PipelineState.ARM_INIT:   300,
    PipelineState.MOVEIT_PRE_GRASP:  250,
    PipelineState.MOVEIT_APPROACH:     100,
    PipelineState.LOCAL_REACH:      300,
    PipelineState.CLOSE:      150,
    PipelineState.LIFT:       1000,
}

SCAN_WARMUP  = 10
SCAN_FRAMES  = 30

# 状态机中原本散落的固定步数限制；只集中命名，不改变数值/行为。
ALIGN_YAW_MAX_STEPS = 600
PAN_MAX_STEPS       = 500
PAN_NEG_X_MAX_STEPS = 450
NAV2_DEST_MAX_STEPS = 6000
ROTATE_STEPS        = 600
PUT_DOWN_STEPS      = 100


@dataclass
class CloseState:
    """CLOSE 的夹持目标、接触历史和跨 env.step 诊断状态。"""
    arm_q_hold: Any = None
    finger_targets: list = field(default_factory=lambda: [None, None])
    force_hold_started: bool = False
    diag_obj_pose0: Any = (None, None)
    diag_prev_contact: Any = None
    deep_post_pending: Any = None
    contact_hist: list = field(default_factory=list)
    obj_pos_hist: list = field(default_factory=list)

    def reset(self):
        self.arm_q_hold = None
        self.finger_targets[:] = [None, None]
        self.force_hold_started = False
        self.diag_obj_pose0 = (None, None)
        self.diag_prev_contact = None
        self.deep_post_pending = None
        self.contact_hist.clear()
        self.obj_pos_hist.clear()


@dataclass
class LocalReachState:
    """LOCAL_REACH 单侧首触后的安全继续深入状态。"""
    first_contact_tcp_world: Any = None
    first_contact_side: Any = None
    first_contact_pos_err_m: Any = None
    single_contact_steps: int = 0
    overforce_steps: int = 0

    def reset(self):
        self.first_contact_tcp_world = None
        self.first_contact_side = None
        self.first_contact_pos_err_m = None
        self.single_contact_steps = 0
        self.overforce_steps = 0


@dataclass
class LiftState:
    """LIFT 两阶段轨迹及 CLOSE 后 hold 检验的跨帧状态。"""
    stage: int = 0
    q_stage1_start: Any = None
    q_stage1_cmd: Any = None
    stage1_step: int = 0
    ee_p_start: Any = None
    fk_dxyz: Any = None
    q_retract0: Any = None
    stage2_step: int = 0
    obj_z_start: Any = None
    hold_remaining: int = 0
    hold_seen: int = 0
    hold_contacts: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.int32))
    hold_obj_p0: Any = None

    def reset(self):
        self.stage = 0
        self.q_stage1_start = None
        self.q_stage1_cmd = None
        self.stage1_step = 0
        self.ee_p_start = None
        self.fk_dxyz = None
        self.q_retract0 = None
        self.stage2_step = 0
        self.obj_z_start = None
        self.hold_remaining = 0
        self.hold_seen = 0
        self.hold_contacts[:] = 0
        self.hold_obj_p0 = None

    def begin(self, cur_q, ee_p_start, hold_steps):
        self.reset()
        self.stage = 1
        self.q_stage1_start = np.asarray(cur_q, dtype=np.float64).copy()
        self.q_stage1_cmd = np.asarray(cur_q, dtype=np.float64).copy()
        self.ee_p_start = np.asarray(ee_p_start, dtype=np.float64).copy()
        self.hold_remaining = int(hold_steps)


def load_policy(policy_path: str, device: str = "cuda"):
    """用 OnPolicyRunner + MockVecEnv 加载 rsl_rl checkpoint。

    从与 policy_path 同目录的 params/agent.yaml 读取网络配置，
    自动从 checkpoint 的 state_dict 推断 obs/action 维度，
    通过 OnPolicyRunner 官方 API 构建网络并加载权重。
    以后更换网络只需换 checkpoint 和 params/agent.yaml，无需改此函数。

    返回 act_inference callable：obs_tensor -> action_tensor
    """
    import copy

    import yaml

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        raise ImportError("请安装 rsl_rl")

    # 1. 读取与 checkpoint 同目录的 params/agent.yaml
    run_dir = os.path.dirname(os.path.abspath(policy_path))
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_yaml):
        raise FileNotFoundError(
            f"找不到 agent.yaml: {agent_yaml}\n"
            f"请确保 params/agent.yaml 与 checkpoint 在同一 run 目录下。"
        )
    with open(agent_yaml) as _f:
        train_cfg = yaml.safe_load(_f)

    # 2. 从 checkpoint state_dict 自动推断网络维度
    _ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
    _sd   = _ckpt["model_state_dict"]
    # actor 第一层输入 = num_actor_obs；最大序号层输出 = num_actions
    _num_actor_obs  = int(_sd["actor.0.weight"].shape[1])
    _num_critic_obs = int(_sd["critic.0.weight"].shape[1])
    _actor_last_key = max(
        (k for k in _sd if k.startswith("actor.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1])
    )
    _num_actions = int(_sd[_actor_last_key].shape[0])

    # 3. 构建最小 Mock VecEnv（OnPolicyRunner.__init__ 只调用 get_observations()）
    _dev = device
    _na  = _num_actions
    _nao = _num_actor_obs
    _nco = _num_critic_obs

    class _MockEnv:
        num_envs           = 1
        num_actions        = _na
        max_episode_length = 1000
        episode_length_buf = torch.zeros(1, dtype=torch.long)
        device             = _dev
        cfg                = {}

        def get_observations(self):
            return {
                "policy": torch.zeros(1, _nao),
                "critic": torch.zeros(1, _nco),
            }

        def step(self, actions):
            obs = self.get_observations()
            return obs, torch.zeros(1), torch.zeros(1), {}

    # 4. OnPolicyRunner 构建网络（会 pop class_name，需深拷贝）
    runner = OnPolicyRunner(
        _MockEnv(),
        copy.deepcopy(train_cfg),
        log_dir=None,
        device=device,
    )

    # 5. 加载权重（不加载 optimizer，推理用）
    runner.load(policy_path, load_optimizer=False)
    _raw_policy = runner.get_inference_policy(device=device)

    # 6. 包装适配器：新版 act_inference 期望 obs 是 dict{"policy": tensor}
    #    navigate_mujoco.py 调用处传的是普通 tensor，此处自动转换
    def policy(obs_tensor):
        return _raw_policy({"policy": obs_tensor})

    print(
        f"[MJ] Policy loaded from {policy_path} "
        f"(actor_obs={_num_actor_obs}, critic_obs={_num_critic_obs}, "
        f"actions={_num_actions})"
    )
    return policy


def _run_grasp_plan(
    env, args, depth_accum, scan_rgb,
    q_scan_at_scan, pos_w_at_scan, quat_w_at_scan,
    pos_w, quat_w, ransac_fit_plane, ransac_remove_plane,
    scan_transforms,
):
    """Run AnyGrasp and geometry/color filtering, but do *no* legacy IK.

    MoveIt owns kinematic reachability, goal collision checking and full-path
    collision checking in the next state.  This function deliberately keeps
    AnyGrasp outputs in camera optical coordinates until that hand-off.
    """
    from mujoco_moveit_frames import transform_points
    import subprocess as _subproc

    state = PipelineState.GRASP_PLAN
    grasp_result = None

    depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)
    H, W = depth_med.shape
    fx_c = fy_c = 616.0
    cx_c, cy_c = W / 2.0, H / 2.0
    u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_med
    valid = (z > 0.05) & (z < 4.0)
    z_v = z[valid]
    if len(z_v) < 200:
        print(f"[SM] GRASP_PLAN: too few valid depth points ({len(z_v)})", flush=True)
        return PipelineState.DONE, None

    pts = np.stack([
        (u_g[valid] - cx_c) * z_v / fx_c,
        (v_g[valid] - cy_c) * z_v / fy_c,
        z_v,
    ], axis=-1).astype(np.float32)
    rgb_u = scan_rgb if scan_rgb is not None else np.zeros((H, W, 3), np.uint8)
    cols = (rgb_u[valid] / 255.0).astype(np.float32)

    # Remove the dominant plane only for target filtering.  AnyGrasp itself sees
    # the full cloud so table geometry remains visible to its collision filter.
    keep_obj = ransac_remove_plane(pts)
    pts_obj = pts[keep_obj]
    cols_obj = cols[keep_obj]
    obj_filter_ok = len(pts_obj) >= 200
    print(f"[SM] GRASP_PLAN: full={len(pts)} object={len(pts_obj)} pts", flush=True)

    # Keep a world-frame table cloud for diagnostics only.  Unlike the old code,
    # this conversion uses the SCAN-time MuJoCo transform directly and never
    # passes through IKPy/joint7.
    table_cloud_world = None
    try:
        import cv2 as _cv2_tb
        cols_u8 = (cols * 255).astype(np.uint8).reshape(1, -1, 3)
        hsv = _cv2_tb.cvtColor(cols_u8, _cv2_tb.COLOR_RGB2HSV)[0]
        table_mask = (hsv[:, 1] < 40) & (hsv[:, 2] > 40) & (hsv[:, 2] < 160)
        table_cam = pts[table_mask]
        if len(table_cam) > 50:
            T_w_c = scan_transforms["T_world_base"] @ scan_transforms["T_base_camera_optical"]
            table_world = transform_points(T_w_c, table_cam)
            nv, dv, _ = ransac_fit_plane(table_world)
            if nv is not None and abs(float(nv[2])) >= 0.9:
                keep = np.abs(table_world @ nv - dv) < 0.010
                if int(keep.sum()) >= 50:
                    table_world = table_world[keep]
            table_cloud_world = table_world.astype(np.float32)
    except Exception as exc:
        print(f"[SM] table-cloud diagnostic skipped: {exc}", flush=True)

    np.savez("/tmp/pointcloud.npz", points=pts, colors=cols)
    worker = os.path.join(_HERE, "grasp_worker.py")
    requested_topk = max(50, int(args.moveit_candidate_limit), int(args.grasp_topk))
    res = _subproc.run([
        sys.executable, worker,
        "--checkpoint", args.grasp_checkpoint,
        "--topk", str(requested_topk),
    ], timeout=120)
    if res.returncode != 0 or not os.path.exists("/tmp/grasp_result.npz"):
        print("[SM] AnyGrasp worker failed", flush=True)
        return PipelineState.DONE, None

    gr = np.load("/tmp/grasp_result.npz", allow_pickle=True)
    scores = np.asarray(gr["scores"], dtype=np.float64)
    if len(scores) == 0:
        print("[SM] AnyGrasp returned no candidates", flush=True)
        return PipelineState.DONE, None

    translations = np.asarray(gr["translations"], dtype=np.float64)
    rotations = np.asarray(gr["rotations"], dtype=np.float64)
    depths = (np.asarray(gr["depths"], dtype=np.float64)
              if "depths" in gr else np.full(len(scores), 0.0658, dtype=np.float64))
    widths = (np.asarray(gr["widths"], dtype=np.float64)
              if "widths" in gr else np.zeros(len(scores), dtype=np.float64))

    candidate_mask = np.ones(len(scores), dtype=bool)

    # Preserve the current project's vivid-object preference, but only use it if
    # it leaves at least one candidate.
    try:
        import cv2 as _cv2
        from scipy.spatial import cKDTree
        cols_u8_obj = (cols_obj * 255).astype(np.uint8).reshape(1, -1, 3)
        hsv_obj = _cv2.cvtColor(cols_u8_obj, _cv2.COLOR_RGB2HSV)[0]
        vivid = pts_obj[hsv_obj[:, 1] >= 80]
        _diag_save_filter_image(scan_rgb, valid, keep_obj, hsv_obj[:, 1] >= 80)
        if len(vivid) >= 20:
            dist, _ = cKDTree(vivid).query(translations, k=1)
            vivid_mask = dist < 0.05
            if int(vivid_mask.sum()) > 0:
                candidate_mask &= vivid_mask
    except Exception as exc:
        print(f"[SM] vivid filter skipped: {exc}", flush=True)

    # Preserve the old actual threshold (8 cm), but expose it as a CLI argument
    # instead of hiding the number behind a misleading '<1cm' comment.
    if obj_filter_ok:
        try:
            from scipy.spatial import cKDTree
            kd = cKDTree(pts_obj.astype(np.float64))
            grasp_point_cam = translations + depths[:, None] * rotations[:, :, 0]
            dist, _ = kd.query(grasp_point_cam, k=1)
            on_obj = dist < float(args.moveit_object_point_dist)
            combined = candidate_mask & on_obj
            if int(combined.sum()) > 0:
                candidate_mask = combined
            elif int(on_obj.sum()) > 0:
                # Do not let the optional color preference erase all geometric candidates.
                candidate_mask = on_obj
        except Exception as exc:
            print(f"[SM] object-point filter skipped: {exc}", flush=True)

    idxs = np.where(candidate_mask)[0].tolist()
    if not idxs:
        idxs = list(range(len(scores)))
    idxs.sort(key=lambda i: -float(scores[i]))
    idxs = idxs[:max(1, int(args.moveit_candidate_limit))]

    grasp_result = {
        "gr_translations": translations,
        "gr_rotations": rotations,
        "gr_scores": scores,
        "gr_depths": depths,
        "gr_widths": widths,
        "candidate_idxs": idxs,
        "q_scan": np.asarray(q_scan_at_scan, dtype=np.float64).copy(),
        "pos_w_scan": np.asarray(pos_w_at_scan, dtype=np.float64).copy(),
        "quat_w_scan": np.asarray(quat_w_at_scan, dtype=np.float64).copy(),
        "scan_transforms": {k: np.asarray(v, dtype=np.float64).copy()
                            for k, v in scan_transforms.items()},
        "table_cloud_world": table_cloud_world,
    }
    print("[SM] AnyGrasp candidates for MoveIt: " + " ".join(
        f"[{i}]s={scores[i]:.3f}" for i in idxs[:8]), flush=True)
    return PipelineState.MOVEIT_PRE_GRASP, grasp_result

def main():
    parser = argparse.ArgumentParser("Navigate MuJoCo")
    parser.add_argument("--map",            required=True)
    parser.add_argument("--goal",           nargs=2, type=float, required=True)
    parser.add_argument("--policy_path",    required=True, help="model_4999.pt 路径")
    parser.add_argument("--grasp_checkpoint", default=None)
    parser.add_argument("--grasp_topk",     type=int, default=1)
    parser.add_argument("--moveit_candidate_limit", type=int, default=12,
                        help="最多交给 MoveIt 检查的 AnyGrasp 候选数")
    parser.add_argument("--moveit_pregrasp_distance", type=float, default=0.115,
                        help="grasp_tcp contact pose 沿 -approach 后退的 pre-grasp 距离(m)")
    parser.add_argument("--moveit_local_reach_distance", type=float, default=0.015,
                        help="MoveIt Cartesian approach 停在 contact 前的距离(m)，剩余由 MuJoCo 局部伺服完成")
    parser.add_argument("--moveit_insertion", type=float, default=0.005,
                        help="在 AnyGrasp palm+depth*approach 基础上的额外 TCP 插入量(m)")
    parser.add_argument("--moveit_object_point_dist", type=float, default=0.08,
                        help="AnyGrasp 指尖落点到去桌面物体点云的最大距离(m)，默认保持现有代码实际 8cm")
    parser.add_argument("--moveit_position_tolerance", type=float, default=0.008,
                        help="MoveIt pre-grasp 位置目标容差(m)")
    parser.add_argument("--moveit_orientation_tolerance", type=float, default=0.12,
                        help="MoveIt pre-grasp 每轴姿态容差(rad)")
    parser.add_argument("--moveit_plan_time", type=float, default=2.5,
                        help="每个候选 MoveIt OMPL 规划时间(s)")
    parser.add_argument("--moveit_cart_step", type=float, default=0.005,
                        help="MoveIt Cartesian approach 最大笛卡尔步长(m)")
    parser.add_argument("--moveit_cart_fraction", type=float, default=0.95,
                        help="Cartesian approach 最低可接受完成比例")
    parser.add_argument("--moveit_exec_joint_tolerance", type=float, default=0.05,
                        help="MoveIt PRE轨迹播放结束后，实际关节到最终轨迹点的最大允许误差(rad)")
    parser.add_argument("--moveit_approach_joint_tolerance", type=float, default=0.025,
                        help="MoveIt APPROACH交给LOCAL_REACH前的最大关节误差(rad)")
    parser.add_argument("--moveit_exec_tcp_position_tolerance", type=float, default=0.015,
                        help="MoveIt轨迹执行后的TCP位置异常保护阈值(m)")
    parser.add_argument("--moveit_exec_tcp_orientation_tolerance", type=float, default=0.1745329252,
                        help="MoveIt轨迹执行后的TCP姿态异常保护阈值(rad，默认10deg)")
    parser.add_argument("--local_single_contact_speed_scale", type=float, default=0.25,
                        help="LOCAL_REACH单侧首触后局部伺服速度倍率")
    parser.add_argument("--local_single_contact_max_advance", type=float, default=0.008,
                        help="LOCAL_REACH单侧首触后最多允许继续沿approach深入的距离(m)")
    parser.add_argument("--local_single_contact_force_limit", type=float, default=1.5,
                        help="LOCAL_REACH单侧接触法向力安全上限(N)；连续超限后中止当前抓取并重规划")
    parser.add_argument("--local_single_contact_force_confirm_steps", type=int, default=3,
                        help="LOCAL_REACH单侧力连续超限多少个控制步才判定为异常")
    parser.add_argument("--moveit_max_approach_world_z", type=float, default=0.0,
                        help="拒绝 approach_world.z 大于该值的候选；默认不允许向上抓")
    parser.add_argument("--object",          default="banana",
                        choices=["banana", "apple", "bowl", "cube"],
                        help="抓取目标物体（default: banana）")
    parser.add_argument("--target_speed",   type=float, default=0.8)
    parser.add_argument("--nav2_arrival_radius", type=float, default=0.55)
    parser.add_argument("--destination",    nargs=2, type=float, default=None,
                        metavar=("X","Y"))
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--render",         action="store_true",
                        help="启动 MuJoCo viewer（需要 display）")
    parser.add_argument("--robot_x",        type=float, default=0.0,
                        help="机器人初始世界 X（对应 Isaac Lab 场景坐标）")
    parser.add_argument("--robot_y",        type=float, default=0.0,
                        help="机器人初始世界 Y")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── 加载 MuJoCo 环境 ──
    from custom_envs.mujoco.env import MuJoCoEnv
    from custom_envs.mujoco.obs_builder import build_policy_obs

    env = MuJoCoEnv(
        robot_init_pos=(args.robot_x, args.robot_y, 0.58),
        render=args.render,
    )
    print("[MJ] MuJoCo env loaded, dt={:.4f}s/step".format(env.dt))

    # ── 加载策略网络 ──
    policy = load_policy(args.policy_path, device=device)

    # ── 启动 ROS2 bridge ──
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge
    bridge = IsaacROS2Bridge(cmd_vel_timeout=0.5)
    bridge.start(timeout=30.0)
    print("[MJ] ROS2 bridge started")

    # ── 启动独立 MoveIt planning bridge（只规划，不接管 MuJoCo 执行）──
    from custom_envs.utils.moveit_bridge import MoveItROS2Bridge
    moveit_bridge = MoveItROS2Bridge(request_timeout=10.0)
    moveit_bridge.start(timeout=35.0)
    print("[MJ] MoveIt planning bridge started")

    from mujoco_moveit_frames import (
        Pose as _MoveItPose,
        anygrasp_candidate_to_tcp as _ag_to_tcp,
        capture_scan_transforms as _capture_scan_tf,
        mujoco_body_pose_world as _mj_body_pose_world,
        pose_to_ros_dict as _pose_to_ros,
        tcp_pose_to_gripper_base_pose as _tcp_to_gb_pose,
        validate_camera_optical_transform as _validate_camera_tf,
        T_gripper_base_grasp_tcp as _T_gb_tcp,
        inv_T as _inv_T,
        pose_error as _pose_error,
    )
    from mujoco_local_servo import (
        MuJoCoGripperBaseServo as _MuJoCoGBServo,
        finger_object_contact as _local_finger_contact,
        robot_table_contact as _robot_table_contact,
    )

    _cam_tf_diag = _validate_camera_tf(env)
    print(
        f"[FRAME] wrist_cam optical transform verified against MuJoCo: "
        f"pos_err={_cam_tf_diag['position_error_m']*1e6:.2f}um "
        f"rot_err={math.degrees(_cam_tf_diag['rotation_error_rad']):.6f}deg",
        flush=True,
    )

    # ── 发送导航目标 ──
    bridge.send_goal(args.goal[0], args.goal[1])

    # ── 状态机变量初始化 ──
    state          = PipelineState.NAV
    state_step     = 0
    last_action    = np.zeros(16, dtype=np.float32)
    grasp_result   = None   # grasp_worker 返回的抓取结果 dict
    depth_accum    = []     # SCAN 阶段累积的 depth 帧列表
    scan_rgb       = None   # SCAN 阶段保存的 RGB 帧
    _q_scan_at_scan  = None  # SCAN 完成时的机械臂关节角（供 IK 用）
    _pos_w_at_scan   = None  # SCAN 完成时的机器人世界位置
    _quat_w_at_scan  = None  # SCAN 完成时的机器人四元数
    _scan_transforms  = None  # SCAN 时 MuJoCo 真值 base/camera 变换快照
    _moveit_scene_applied = False
    _moveit_selected = None
    _moveit_pre_cmds = None
    _moveit_pre_i = 0
    _moveit_approach_cmds = None
    _moveit_approach_i = 0
    _local_servo = None
    _pan_neg_x_goal  = None  # PAN_NEG_X 目标坐标
    # PAN 平移统一平滑控制状态：用加速度限幅 + 停车滞回，避免 0↔最小步速来回跳。
    _pan_ctrl_state   = None
    _pan_cmd_vx       = 0.0
    _pan_cmd_vy       = 0.0
    _pan_cmd_wz       = 0.0
    _pan_settling     = False
    _pan_settle_count = 0
    # ALIGN_YAW 专用控制状态：小角速度对 locomotion policy 几乎不起作用，因此
    # 使用最小有效转速，并要求“角度 + 实际角速度”连续稳定后才进入 PAN。
    _align_ctrl_state   = None
    _align_cmd_wz       = 0.0
    _align_settle_count = 0
    _dest_goal_sent  = False # NAV2_DEST 是否已发送 goal
    _rotate_j_target = None  # ROTATE 目标关节角
    _rotate_j1_start = None
    _rotate_j2_start = None
    _rotate_j3_start = None

    # REACH / CLOSE / LIFT 的跨帧状态集中管理。
    # 只收纳真正需要跨 policy step 保存的量；单帧几何/IK 临时量仍留在对应状态分支内。
    close_ctx = CloseState()
    local_reach_ctx = LocalReachState()
    lift_ctx = LiftState()

    # ── 抓取阶段底盘物理锁定（MuJoCo equality weld） ──
    # 从 ARM_INIT 开始到 LIFT 结束，全程共用同一个世界锚点。weld 在每个 physics
    # substep 内由 MuJoCo 约束求解器施加约束反力，不再使用 set_root_pose()
    # 对根节点做离散“瞬移”式冻结。ROTATE 阶段仍沿用同一物理锁机制。
    import mujoco as _mj_lock
    _base_lock_bid = _mj_lock.mj_name2id(
        env._model, _mj_lock.mjtObj.mjOBJ_BODY, "base_link")
    _base_lock_anchor_bid = _mj_lock.mj_name2id(
        env._model, _mj_lock.mjtObj.mjOBJ_BODY, "base_lock_anchor")
    _base_lock_eqid = _mj_lock.mj_name2id(
        env._model, _mj_lock.mjtObj.mjOBJ_EQUALITY, "base_lock_weld")

    if _base_lock_bid < 0 or _base_lock_anchor_bid < 0 or _base_lock_eqid < 0:
        raise RuntimeError(
            "scene.xml 缺少 base_lock_site/base_lock_anchor/base_lock_weld；"
            "请使用配套的 weld 版 scene.xml")

    _base_lock_mocapid = int(env._model.body_mocapid[_base_lock_anchor_bid])
    if _base_lock_mocapid < 0:
        raise RuntimeError("base_lock_anchor 必须是 mocap=true body")

    _base_lock = dict(
        active=False,
        ref_pos=None,
        ref_quat=None,
    )

    # 抓取阶段从机械臂开始动作起就固定底盘，并在整个抓取链路中保持同一个 anchor。
    # 失败后回到 ARM_INIT 仍保持锁定；只有真正离开抓取阶段时才释放。
    _GRASP_BASE_LOCK_STATES = {
        PipelineState.ARM_INIT,
        PipelineState.SCAN,
        PipelineState.GRASP_PLAN,
        PipelineState.MOVEIT_PRE_GRASP,
        PipelineState.MOVEIT_APPROACH,
        PipelineState.LOCAL_REACH,
        PipelineState.CLOSE,
        PipelineState.LIFT,
    }
    _BASE_LOCK_STATES = _GRASP_BASE_LOCK_STATES | {PipelineState.ROTATE}

    # 每帧固定不变的状态集合只构造一次，避免主循环内重复创建。
    _ARM_IDLE_STATES = {
        PipelineState.ALIGN_YAW_1,
        PipelineState.PAN_XY,
    }
    _LOCOMOTION_DISABLED_STATES = {
        PipelineState.ARM_INIT, PipelineState.SCAN,
        PipelineState.GRASP_PLAN, PipelineState.MOVEIT_PRE_GRASP,
        PipelineState.MOVEIT_APPROACH, PipelineState.LOCAL_REACH,
        PipelineState.CLOSE, PipelineState.LIFT,
    }

    def _set_base_physical_lock(enable):
        """底盘物理锁：mocap anchor + site weld；仅在锁定状态集合边界动作一次。"""
        _enable = bool(enable)
        if _enable == bool(_base_lock["active"]):
            return

        if _enable:
            # 把世界锚点放到此刻 base_link 的实际世界位姿；两个 site 初始完全重合，
            # 因此启用 weld 不会把机器人拉回 XML 初始位姿。
            _p = env._data.xpos[_base_lock_bid].astype(np.float64).copy()
            _q = env._data.xquat[_base_lock_bid].astype(np.float64).copy()
            env._data.mocap_pos[_base_lock_mocapid] = _p
            env._data.mocap_quat[_base_lock_mocapid] = _q
            env._data.eq_active[_base_lock_eqid] = 1
            _mj_lock.mj_forward(env._model, env._data)

            _base_lock["active"] = True
            _base_lock["ref_pos"] = _p
            _base_lock["ref_quat"] = _q
            print(
                f"[BASE_LOCK] ON physical weld at pos={np.round(_p,6)} "
                f"quat={np.round(_q,6)}",
                flush=True)
        else:
            env._data.eq_active[_base_lock_eqid] = 0
            _mj_lock.mj_forward(env._model, env._data)
            _base_lock["active"] = False
            print("[BASE_LOCK] OFF physical weld", flush=True)

    def _print_base_lock_diag(tag="BASE_LOCK"):
        """只读诊断：显示 weld active 时 base_link 相对锁定瞬间的残余漂移。"""
        if not _base_lock["active"] or _base_lock["ref_pos"] is None:
            return
        _p = env._data.xpos[_base_lock_bid].astype(np.float64).copy()
        _q = env._data.xquat[_base_lock_bid].astype(np.float64).copy()
        _dp_mm = (_p - _base_lock["ref_pos"]) * 1000.0
        _q0 = np.asarray(_base_lock["ref_quat"], dtype=np.float64)
        _dot = float(np.clip(abs(np.dot(_q0, _q)), 0.0, 1.0))
        _dang_deg = math.degrees(2.0 * math.acos(_dot))
        print(
            f"[{tag}] drift_mm={np.round(_dp_mm,4)} "
            f"drift_rot={_dang_deg:.5f}deg",
            flush=True)

    # ── arm / gripper 在 joint_pos(ALL_JOINT_NAMES) 中的索引 ──
    # 不再硬编码下标，直接复用 env 按关节名建立的映射，避免 obs_builder 顺序变化。
    _ARM_IDX = np.asarray(env._arm_all_idx, dtype=np.int32).copy()
    _GRIP_IDX = np.asarray(env._grip_all_idx, dtype=np.int32).copy()
    _MOVEIT_STATE_NAMES = ARM_JOINT_NAMES + ["joint7", "joint8"]

    def _get_moveit_joint_state(_jpos):
        _arm = _jpos[_ARM_IDX].astype(np.float64)
        _grip = _jpos[_GRIP_IDX].astype(np.float64)
        return _MOVEIT_STATE_NAMES, np.concatenate([_arm, _grip])

    def _dense_moveit_commands(_reply):
        _traj = _reply.get("trajectory", {})
        _path = MoveItROS2Bridge.extract_arm_path(_traj, ARM_JOINT_NAMES)
        if len(_path) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        return MoveItROS2Bridge.densify_joint_path(
            _path, np.array([0.012, 0.018, 0.018, 0.015, 0.015, 0.018], dtype=np.float64))

    def _current_tcp_pose_world():
        """MuJoCo真值 grasp_tcp 世界位姿。TCP固定在gripper_base +Z 0.1358m。"""
        _T_w_gb = _mj_body_pose_world(env, "gripper_base").matrix()
        _T_w_tcp = _T_w_gb @ _T_gb_tcp()
        return _MoveItPose(_T_w_tcp[:3, 3].copy(), _T_w_tcp[:3, :3].copy())

    def _current_tcp_pose_base():
        """MuJoCo真值 grasp_tcp 在当前 base_link 坐标系下的位姿。"""
        _T_w_b = _mj_body_pose_world(env, "base_link").matrix()
        _T_w_tcp = _current_tcp_pose_world().matrix()
        _T_b_tcp = _inv_T(_T_w_b) @ _T_w_tcp
        return _MoveItPose(_T_b_tcp[:3, 3].copy(), _T_b_tcp[:3, :3].copy())

    def _tcp_target_error_base(_target_pose):
        """返回当前真实TCP到base_link目标TCP的(位置误差m, 姿态误差rad)。"""
        _cur = _current_tcp_pose_base()
        _ep, _er = _pose_error(_cur, _target_pose)
        return float(np.linalg.norm(_ep)), float(np.linalg.norm(_er))

    def _finger_object_normal_forces(_obj_name):
        """读取当前两指与目标物体的总法向接触力；只用于LOCAL_REACH安全限幅。"""
        import mujoco as _mj_force
        _out = {"link7": 0.0, "link8": 0.0}
        _m, _d = env._model, env._data
        _obj = _mj_force.mj_name2id(_m, _mj_force.mjtObj.mjOBJ_BODY, _obj_name)
        if _obj < 0:
            return _out
        _fids = {
            _name: _mj_force.mj_name2id(_m, _mj_force.mjtObj.mjOBJ_BODY, _name)
            for _name in _out
        }
        for _ci in range(int(_d.ncon)):
            _ct = _d.contact[_ci]
            _b1 = int(_m.geom_bodyid[int(_ct.geom1)])
            _b2 = int(_m.geom_bodyid[int(_ct.geom2)])
            if _obj not in (_b1, _b2):
                continue
            for _name, _bid in _fids.items():
                if _bid >= 0 and _bid in (_b1, _b2):
                    _force6 = np.zeros(6, dtype=np.float64)
                    _mj_force.mj_contactForce(_m, _d, _ci, _force6)
                    _out[_name] += max(0.0, float(_force6[0]))
        return _out


    # PRE_GRASP 每步最大增量（与原版 navigate_to_goal_nav2_whole.py 一致）
    _PG_MAX_DELTA = np.array([0.03, 0.05, 0.05, 0.04, 0.04, 0.04], dtype=np.float64)

    _LIFT_HOLD_STEPS      = 20                     # CLOSE 后先静置约 0.4s，让接触/夹持力稳定
    # 新LIFT Stage1：不再强制末端走世界坐标竖直直线。先固定j1/j4~j6，
    # 仅让j2/j3朝ARM_SIDE方向缓慢移动；末端实际升高20mm后进入Stage2。
    _LIFT_STAGE1_EE_RISE = 0.020                   # m；当前调轨迹阶段只看末端高度
    _LIFT_STAGE1_MAX_DQ   = 0.003                   # rad/policy-step；j2/j3命令最大增量
    _LIFT_STAGE1_TEST_DQ  = 0.010                   # rad；初始化时一次性FK方向试探步
    _LIFT_STAGE1_FK_MIN_DZ= 1e-5                   # m；预测dz必须为正（留数值余量）
    _LIFT_RETRACT_STEPS   = 400                     # Stage2全关节收回ARM_SIDE_ANGLES

    # ── 辅助函数 ──
    def _wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _yaw_err_to(target_yaw, current_yaw):
        return _wrap_angle(target_yaw - current_yaw)

    # ── PAN 统一平滑速度控制 ──
    # locomotion policy 在很小的非零速度附近容易在“站立/迈步”之间反复切换。
    # PAN 精调因此避免长期落在 0.1~0.2m/s 的低速区：真正移动时至少给 0.25m/s，
    # 同时把速度斜坡稍微加快，缩短起停时穿过低速区间的时间。
    _PAN_MIN_MOVE_SPEED = 0.25       # m/s；进一步避开 0.1~0.2m/s 的低速抖动区
    _PAN_DV_PER_STEP    = 0.030      # m/s per policy step；更快穿过低速起停区
    _PAN_DW_PER_STEP    = 0.060      # rad/s per policy step；yaw 命令也快速进入有效区
    _PAN_YAW_KP         = 3.0
    _PAN_YAW_MIN        = 0.30       # rad/s；避免长期停留在小角速度低响应区
    _PAN_YAW_MAX        = 0.60
    _PAN_YAW_DEADBAND   = math.radians(1.5)
    _PAN_SETTLE_SPEED   = 0.06       # m/s；只检查当前受控世界轴的实际速度
    _PAN_SETTLE_STEPS   = 8          # 连续稳定约0.16s后才切状态

    _ALIGN_YAW_KP          = 4.0
    _ALIGN_YAW_MIN         = 0.40    # rad/s；专门避开“给了 wz 但 policy 几乎不转”的区间
    _ALIGN_YAW_MAX         = 0.90
    _ALIGN_YAW_DW_PER_STEP = 0.08
    _ALIGN_YAW_TOL         = math.radians(1.5)
    _ALIGN_YAW_RATE_TOL    = 0.08    # rad/s，实际机体角速度也必须基本停住
    _ALIGN_SETTLE_STEPS    = 8

    def _slew(cur, target, max_delta):
        return float(cur + np.clip(float(target) - float(cur), -max_delta, max_delta))

    def _yaw_target_with_min(err, kp, min_abs, max_abs, deadband):
        """把 yaw 误差变成具有最小有效幅值的角速度命令。"""
        ae = abs(float(err))
        if ae <= float(deadband):
            return 0.0
        mag = min(float(max_abs), max(float(min_abs), float(kp) * ae))
        return math.copysign(mag, float(err))

    def _align_yaw_control(align_state, target_yaw, yaw_now, ang_vel_z):
        """ALIGN_YAW 专用：有效转速转向 + 停稳判定，避免一碰阈值就立刻切状态。"""
        nonlocal _align_ctrl_state, _align_cmd_wz, _align_settle_count
        if _align_ctrl_state != align_state:
            _align_ctrl_state = align_state
            _align_cmd_wz = 0.0
            _align_settle_count = 0

        err = _yaw_err_to(float(target_yaw), float(yaw_now))
        wz_target = _yaw_target_with_min(
            err, _ALIGN_YAW_KP, _ALIGN_YAW_MIN, _ALIGN_YAW_MAX, _ALIGN_YAW_TOL)
        _align_cmd_wz = _slew(_align_cmd_wz, wz_target, _ALIGN_YAW_DW_PER_STEP)

        if (abs(err) <= _ALIGN_YAW_TOL
                and abs(float(ang_vel_z)) <= _ALIGN_YAW_RATE_TOL
                and abs(_align_cmd_wz) <= _ALIGN_YAW_DW_PER_STEP):
            _align_settle_count += 1
        else:
            _align_settle_count = 0
        done = (_align_settle_count >= _ALIGN_SETTLE_STEPS)
        return _align_cmd_wz, done, dict(
            err=err, wz_actual=float(ang_vel_z),
            settle_count=_align_settle_count)

    def _pan_world_axis_control(
        pan_state, world_err, world_axis, target_yaw,
        pos_tol, resume_tol, kp, vmax, yaw_now, base_lin_vel,
        settle_use_vxy=False,
    ):
        """按世界坐标轴做 PAN 精调，再用实时 yaw 转成机体系 vx/vy。

        这样即使机器人没有精确停在 target_yaw，世界系运动方向也不会跟着机体偏航。
        world_axis 目前支持 'x' / 'y'；world_err = 目标世界坐标 - 当前世界坐标。
        settle_use_vxy=True 时沿用旧 PAN_VY/PAN_NEG_X 的停车判据：要求整机水平
        速度范数 vxy 足够小；False 时只要求当前受控世界轴速度足够小。
        """
        nonlocal _pan_ctrl_state, _pan_cmd_vx, _pan_cmd_vy, _pan_cmd_wz
        nonlocal _pan_settling, _pan_settle_count

        if _pan_ctrl_state != pan_state:
            _pan_ctrl_state = pan_state
            _pan_cmd_vx = 0.0
            _pan_cmd_vy = 0.0
            _pan_cmd_wz = 0.0
            _pan_settling = False
            _pan_settle_count = 0

        ae = abs(float(world_err))
        if _pan_settling:
            if ae > float(resume_tol):
                _pan_settling = False
                _pan_settle_count = 0
        elif ae <= float(pos_tol):
            _pan_settling = True
            _pan_settle_count = 0

        if _pan_settling:
            world_vx_target = 0.0
            world_vy_target = 0.0
        else:
            mag = min(float(vmax), max(_PAN_MIN_MOVE_SPEED, float(kp) * ae))
            world_v = math.copysign(mag, float(world_err))
            if world_axis == 'x':
                world_vx_target, world_vy_target = world_v, 0.0
            elif world_axis == 'y':
                world_vx_target, world_vy_target = 0.0, world_v
            else:
                raise ValueError(f"unsupported world_axis={world_axis!r}")

        # world -> body: Rz(yaw)^T @ v_world
        cy = math.cos(float(yaw_now))
        sy = math.sin(float(yaw_now))
        vx_target = cy * world_vx_target + sy * world_vy_target
        vy_target = -sy * world_vx_target + cy * world_vy_target
        _pan_cmd_vx = _slew(_pan_cmd_vx, vx_target, _PAN_DV_PER_STEP)
        _pan_cmd_vy = _slew(_pan_cmd_vy, vy_target, _PAN_DV_PER_STEP)

        yaw_err = _yaw_err_to(float(target_yaw), float(yaw_now))
        wz_target = _yaw_target_with_min(
            yaw_err, _PAN_YAW_KP, _PAN_YAW_MIN, _PAN_YAW_MAX, _PAN_YAW_DEADBAND)
        _pan_cmd_wz = _slew(_pan_cmd_wz, wz_target, _PAN_DW_PER_STEP)

        _vworld = np.asarray(base_lin_vel, dtype=np.float64)[:2]
        _vxy = float(np.linalg.norm(_vworld))
        _axis_vel = float(_vworld[0] if world_axis == 'x' else _vworld[1])
        _settle_speed = _vxy if settle_use_vxy else abs(_axis_vel)
        if (_pan_settling and _settle_speed <= _PAN_SETTLE_SPEED
                and abs(yaw_err) <= math.radians(3.0)):
            _pan_settle_count += 1
        else:
            _pan_settle_count = 0
        done = (_pan_settle_count >= _PAN_SETTLE_STEPS)

        return (_pan_cmd_vx, _pan_cmd_vy, _pan_cmd_wz, done,
                dict(err=ae, signed_err=float(world_err), vxy=_vxy,
                     axis_vel=_axis_vel, yaw_err=yaw_err,
                     settling=_pan_settling, settle_count=_pan_settle_count,
                     settle_speed=_settle_speed,
                     world_vx_target=world_vx_target,
                     world_vy_target=world_vy_target))

    def _pan_world_xy_control(
        pan_state, err_x, err_y, target_yaw,
        pos_tol=0.05, resume_tol=0.10,
        kp_x=1.0, kp_y=0.8,
        vmax_x=0.25, vmax_y=0.35,
        yaw_now=0.0, base_lin_vel=None,
    ):
        """世界系 XY 同时闭环精定位。

        完成条件不是欧式距离，而是两个轴分别满足：
            |err_x| <= pos_tol 且 |err_y| <= pos_tol
        并且水平速度、yaw 都已经稳定。

        某一轴已经进入容差时，该轴目标速度置零；若随后被另一轴运动带出容差，
        下一控制周期会自动重新纠正该轴。
        """
        nonlocal _pan_ctrl_state, _pan_cmd_vx, _pan_cmd_vy, _pan_cmd_wz
        nonlocal _pan_settling, _pan_settle_count

        if _pan_ctrl_state != pan_state:
            _pan_ctrl_state = pan_state
            _pan_cmd_vx = 0.0
            _pan_cmd_vy = 0.0
            _pan_cmd_wz = 0.0
            _pan_settling = False
            _pan_settle_count = 0

        ex = float(err_x)
        ey = float(err_y)
        ax = abs(ex)
        ay = abs(ey)

        in_x = ax <= float(pos_tol)
        in_y = ay <= float(pos_tol)
        both_in = in_x and in_y

        # 只有两个轴都进入 ±5 cm 才进入停车/稳定阶段。
        if _pan_settling:
            if ax > float(resume_tol) or ay > float(resume_tol):
                _pan_settling = False
                _pan_settle_count = 0
        elif both_in:
            _pan_settling = True
            _pan_settle_count = 0

        if _pan_settling:
            world_vx_target = 0.0
            world_vy_target = 0.0
        else:
            if in_x:
                world_vx_target = 0.0
            else:
                mag_x = min(
                    float(vmax_x),
                    max(_PAN_MIN_MOVE_SPEED, float(kp_x) * ax),
                )
                world_vx_target = math.copysign(mag_x, ex)

            if in_y:
                world_vy_target = 0.0
            else:
                mag_y = min(
                    float(vmax_y),
                    max(_PAN_MIN_MOVE_SPEED, float(kp_y) * ay),
                )
                world_vy_target = math.copysign(mag_y, ey)

        # world -> body: Rz(yaw)^T @ v_world
        cy = math.cos(float(yaw_now))
        sy = math.sin(float(yaw_now))
        vx_target = cy * world_vx_target + sy * world_vy_target
        vy_target = -sy * world_vx_target + cy * world_vy_target

        _pan_cmd_vx = _slew(_pan_cmd_vx, vx_target, _PAN_DV_PER_STEP)
        _pan_cmd_vy = _slew(_pan_cmd_vy, vy_target, _PAN_DV_PER_STEP)

        yaw_err = _yaw_err_to(float(target_yaw), float(yaw_now))
        wz_target = _yaw_target_with_min(
            yaw_err, _PAN_YAW_KP, _PAN_YAW_MIN,
            _PAN_YAW_MAX, _PAN_YAW_DEADBAND)
        _pan_cmd_wz = _slew(_pan_cmd_wz, wz_target, _PAN_DW_PER_STEP)

        if base_lin_vel is None:
            _vworld = np.zeros(2, dtype=np.float64)
        else:
            _vworld = np.asarray(base_lin_vel, dtype=np.float64)[:2]
        _vxy = float(np.linalg.norm(_vworld))

        if (_pan_settling
                and ax <= float(pos_tol)
                and ay <= float(pos_tol)
                and _vxy <= _PAN_SETTLE_SPEED
                and abs(yaw_err) <= math.radians(3.0)):
            _pan_settle_count += 1
        else:
            _pan_settle_count = 0

        done = (_pan_settle_count >= _PAN_SETTLE_STEPS)

        return (
            _pan_cmd_vx, _pan_cmd_vy, _pan_cmd_wz, done,
            dict(
                err_x=ex, err_y=ey,
                abs_err_x=ax, abs_err_y=ay,
                in_x=in_x, in_y=in_y,
                vxy=_vxy, yaw_err=yaw_err,
                settling=_pan_settling,
                settle_count=_pan_settle_count,
                world_vx_target=world_vx_target,
                world_vy_target=world_vy_target,
            )
        )

    def _arm_step(q6):
        """设置机械臂目标角（6维）。"""
        env.set_arm_target(np.asarray(q6, dtype=np.float64))

    def _gripper_step(close=False):
        tgt = GRIPPER_CLOSE_POS if close else GRIPPER_OPEN_POS
        env.set_gripper_target(tgt.astype(np.float64))

    def _get_arm_q(jpos):
        """从 24 维 joint_pos 中取出 arm 6 关节角。"""
        return jpos[_ARM_IDX].copy()

    # ── CLOSE 逐指接触检测 + 定力冻结（第四轮，从碰撞/力角度修"抓不起来"）──
    # 旧行为：CLOSE 盲发 [0,0] 100 步。物体的存在会把 PD 误差压得很小
    # （碗实测只压到 ~7.9mm 误差 → 1.58N/指），单侧摩擦容量 15.8N ≈ 碗重 15.7N
    # → 边缘打滑，用户看到"抓不稳"。
    # 新行为：分别检测 link7/link8 与物体的接触。
    # 第一段 — 某指首次接触后，不再“零力轻贴”，而给该指约 0.25N 的轻预压力；同时把
    #   夹爪命令速率从 1mm/policy-step 临时降到 0.4mm/policy-step，让另一指慢慢靠近，
    #   避免后接触的一侧把 cube 在桌面上撞/推走。
    # 第二段 — **两指都已接触**后，切换为对称恒力夹持，
    # 不再继续追逐物体内部的位置目标。
    _CLOSE_PRELOAD_S = 0.003125     # m；首触侧轻预压，静态比例项约 0.25N
    _CLOSE_SEARCH_MAX_DELTA = 0.0004 # m/policy-step；单侧已接触时另一侧慢速靠近
    _GRIP_MAX_DELTA_NORMAL = float(env._grip_max_delta)
    _LIFT_HOLD_CONTACT_RATIO = 0.40  # hold窗口内每侧至少40%的采样帧有接触
    _LIFT_HOLD_MAX_OBJ_MOVE = 0.002  # m；hold期间cube位移超过2mm视为不稳定
    _CLOSE_CHECK_WINDOW = 20          # CLOSE 完成前检查最近20个 policy step
    _CLOSE_CONTACT_RATE_MIN = 0.40    # 每侧最近窗口接触率至少40%
    _CLOSE_STABLE_MOVE_MAX = 0.002    # m；最近窗口cube净位移不超过2mm
    _GRI_Q7_I, _GRI_Q8_I = map(int, np.asarray(env._grip_all_idx).tolist())

    # ── 指板最低点 FK（第九轮，REACH 纠偏地板用）──
    # 纠偏会把命令往桌子方向压，判"指板会不会进桌面"必须按**真实指板几何**：第一轮
    # 实测指板最低点 0.6525 < 桌面 0.6613，就是旧地板公式（指尖 + 35.1mm×sin(倾角)
    # 的估计）漏掉的。这里直接取 link7/link8 的**碰撞网格顶点**（MuJoCo mesh 按凸包
    # 碰撞，极值点必在顶点上）做 FK 后取世界 z 最小 —— 与探针实测一致：approach 竖直
    # 时指尖就是板的最低点，approach 有倾角时板角更低，FK 自动包含，不需要再估倾角。
    _pad_v_cache = None

    def _pad_min_z_fk(q6):
        """按候选臂角 q6 做 FK，返回 link7/link8 碰撞网格的最低点世界 z（m）。"""
        import mujoco as _mj_pz
        nonlocal _pad_v_cache
        _m = env._model
        if _pad_v_cache is None:
            _cache = []
            for _bn in ("link7", "link8"):
                _bb = _mj_pz.mj_name2id(_m, _mj_pz.mjtObj.mjOBJ_BODY, _bn)
                for _g in range(_m.ngeom):
                    if int(_m.geom_bodyid[_g]) != _bb:
                        continue
                    if int(_m.geom_type[_g]) != int(_mj_pz.mjtGeom.mjGEOM_MESH):
                        continue
                    if int(_m.geom_contype[_g]) == 0 and int(_m.geom_conaffinity[_g]) == 0:
                        continue                      # 同网格的视觉副本，不参与碰撞
                    _mid = int(_m.geom_dataid[_g])
                    _a0  = int(_m.mesh_vertadr[_mid])
                    _n0  = int(_m.mesh_vertnum[_mid])
                    _cache.append((_g, _m.mesh_vert[_a0:_a0 + _n0]
                                   .reshape(-1, 3).astype(np.float64).copy()))
            _pad_v_cache = _cache
        _sd = _mj_pz.MjData(_m)
        _sd.qpos[:] = env._data.qpos                # 其余关节（含 joint7/8 张开位）保持当前值
        _sd.qpos[np.asarray(env._all_qpos_idx)[np.asarray(env._arm_all_idx)]] = \
            np.asarray(q6, dtype=np.float64)
        _mj_pz.mj_forward(_m, _sd)
        _z = np.inf
        for _g, _V in _pad_v_cache:
            _W = _sd.geom_xpos[_g] + _V @ _sd.geom_xmat[_g].reshape(3, 3).T
            _z = min(_z, float(_W[:, 2].min()))
        return _z

    def _fingers_contact(obj_name):
        """{link7: bool, link8: bool}: 各指板是否与 obj_name 物体接触。"""
        out = {"link7": False, "link8": False}
        try:
            import mujoco as _mj_fc
            _m, _d = env._model, env._data
            _fing = {_n: _mj_fc.mj_name2id(_m, _mj_fc.mjtObj.mjOBJ_BODY, _n)
                     for _n in ("link7", "link8")}
            _obj = _mj_fc.mj_name2id(_m, _mj_fc.mjtObj.mjOBJ_BODY, obj_name)
            for _ci in range(int(_d.ncon)):
                _ct = _d.contact[_ci]
                _b1 = int(_m.geom_bodyid[int(_ct.geom1)])
                _b2 = int(_m.geom_bodyid[int(_ct.geom2)])
                if _obj not in (_b1, _b2):
                    continue
                for _n, _bid in _fing.items():
                    if _bid >= 0 and _bid in (_b1, _b2):
                        out[_n] = True
        except Exception:
            pass
        return out



    def _ransac_fit_plane(pts, n_iter=150, dist_thresh=0.008):
        """RANSAC 拟合内点最多的主平面。返回 (nv, dv, inlier_mask)；点数过少时 nv=None。
        不做内点比率判定，由调用方决定用途（去除平面 / 保留平面内点）。"""
        N = len(pts)
        if N < 10:
            return None, None, np.zeros(N, dtype=bool)
        best_n    = 0
        best_mask = np.zeros(N, dtype=bool)
        best_nv, best_dv = None, None
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
                best_nv, best_dv = nv, dv
        return best_nv, best_dv, best_mask

    def _ransac_remove_plane(pts, n_iter=150, dist_thresh=0.008, min_inlier_ratio=0.15):
        """RANSAC 平面去除。找到内点最多的主平面（桌面），返回 True=保留(非桌面)。
        若最大平面内点数 < 总点数的 min_inlier_ratio，认为场景无显著平面，保留所有点。"""
        N = len(pts)
        nv, dv, mask = _ransac_fit_plane(pts, n_iter, dist_thresh)
        best_n = int(mask.sum())
        if nv is None or best_n < int(N * min_inlier_ratio):
            print(f"[SM] RANSAC: no dominant plane (best={best_n}/{N}), keeping all.", flush=True)
            return np.ones(N, dtype=bool)
        keep = ~mask
        pct  = best_n * 100 // N
        print(f"[SM] RANSAC plane={best_n} pts ({pct}%), object pts={int(keep.sum())}, "
              f"normal={np.round(nv, 3)} d={dv:.3f}", flush=True)
        return keep


    # ── MuJoCo gripper_base 真值位姿 / 临时 FK 辅助 ──
    import mujoco as _mj_servo
    _GB_BODY_ID = _mj_servo.mj_name2id(
        env._model, _mj_servo.mjtObj.mjOBJ_BODY, "gripper_base")
    if _GB_BODY_ID < 0:
        raise RuntimeError("scene.xml 中找不到 gripper_base body")

    _ARM_JOINT_IDS = np.array([
        _mj_servo.mj_name2id(env._model, _mj_servo.mjtObj.mjOBJ_JOINT, n)
        for n in ARM_JOINT_NAMES
    ], dtype=np.int32)
    _ARM_LO = np.array([env._model.jnt_range[j, 0] for j in _ARM_JOINT_IDS],
                       dtype=np.float64)
    _ARM_HI = np.array([env._model.jnt_range[j, 1] for j in _ARM_JOINT_IDS],
                       dtype=np.float64)
    _ARM_QPOS_ADRS = np.array(
        [env._jnt_qposadr[n] for n in ARM_JOINT_NAMES], dtype=np.int32)

    def _gb_pose_world():
        """MuJoCo 真值：gripper_base 在 world 下的位置和旋转。"""
        p = env._data.xpos[_GB_BODY_ID].astype(np.float64).copy()
        R = env._data.xmat[_GB_BODY_ID].reshape(3, 3).astype(np.float64).copy()
        return p, R

    def _gb_pos_for_arm_q(q6):
        """只做一次/少量离线FK诊断：在临时MjData中替换arm q，不改真实仿真状态。"""
        q6 = np.asarray(q6, dtype=np.float64)
        _tmp = _mj_servo.MjData(env._model)
        _tmp.qpos[:] = env._data.qpos
        if env._model.nmocap:
            _tmp.mocap_pos[:] = env._data.mocap_pos
            _tmp.mocap_quat[:] = env._data.mocap_quat
        _tmp.qpos[_ARM_QPOS_ADRS] = np.clip(q6, _ARM_LO, _ARM_HI)
        _mj_servo.mj_forward(env._model, _tmp)
        return _tmp.xpos[_GB_BODY_ID].astype(np.float64).copy()



    # ── 主循环 ──
    print("[MJ] Starting main loop, state=", state)
    try:
        while state != PipelineState.DONE:
            # 本轮 physics step 的状态固定为 current_state。状态处理器只能修改
            # next_state；所有 action / weld / env.step 都完整按 current_state 执行，
            # physics step 完成后才提交状态切换，避免“一帧内混用两个状态”。
            current_state = state
            next_state = current_state

            # 读取机器人状态
            rs = env.get_robot_state()
            pos_w   = rs["pos"]       # (3,) 世界坐标
            quat_w  = rs["quat"]      # (w,x,y,z)
            yaw     = rs["yaw"]
            jpos    = rs["joint_pos"] # (24,)
            jvel    = rs["joint_vel"] # (24,)
            ang_vel = rs["ang_vel"]   # (3,) 机体系

            # ── 更新 ROS2 bridge 位姿（供 Nav2 定位） ──
            bridge.update_robot_pose(
                pos_w, quat_w,
                rs["lin_vel"], ang_vel,
            )

            # MoveIt current state: arm + gripper joint states from the same MuJoCo frame.
            _mi_names_now, _mi_pos_now = _get_moveit_joint_state(jpos)
            moveit_bridge.update_joint_state(_mi_names_now, _mi_pos_now)

            # ── 从 bridge 获取 cmd_vel ──
            cmd_vx, cmd_wz = bridge.get_cmd_vel()
            cmd_vy = 0.0

            # ── 状态机：速度指令覆盖 ──
            if current_state == PipelineState.NAV:
                # 导航阶段保持 ARM_HOME_ANGLES（收起但不到侧面）
                _arm_step(ARM_INIT_ANGLES)
                nav_done, _ = bridge.get_nav_status()
                if nav_done:
                    next_state = PipelineState.ALIGN_YAW_1
                    state_step = 0
                    print("[MJ] NAV done -> ALIGN_YAW_1")

            elif current_state == PipelineState.ALIGN_YAW_1:
                # 对齐到 +Y。角速度命令设最小有效值，并等实际 yaw rate 停稳后再进入 PAN。
                cmd_vx = 0.0; cmd_vy = 0.0
                cmd_wz, _align_done, _ad = _align_yaw_control(
                    PipelineState.ALIGN_YAW_1, math.pi / 2, yaw, ang_vel[2])
                state_step += 1
                if state_step == 1 or state_step % 20 == 0:
                    print(f"[SM] ALIGN_YAW_1 step {state_step}: "
                          f"yaw={math.degrees(yaw):+.2f}deg "
                          f"err={math.degrees(_ad['err']):+.2f}deg "
                          f"cmd_wz={cmd_wz:+.3f} actual_wz={_ad['wz_actual']:+.3f} "
                          f"settle={_ad['settle_count']}/{_ALIGN_SETTLE_STEPS}", flush=True)
                if _align_done or state_step >= ALIGN_YAW_MAX_STEPS:
                    _why = "settled" if _align_done else "timeout"
                    print(f"[MJ] ALIGN_YAW_1 done ({_why}) -> PAN_XY", flush=True)
                    next_state = PipelineState.PAN_XY
                    state_step = 0

            elif current_state == PipelineState.PAN_XY:
                # 最终二维精定位：
                #   x_target = goal_x + 0.5m（保持原 PAN_VY 的目标）
                #   y_target = goal_y        （保持原 PAN_VX 的目标）
                # 每一步同时看 x/y 两个误差，避免修 x 时把已经调好的 y 再次带偏。
                _x_target = float(args.goal[0]) + 0.5
                _y_target = float(args.goal[1])
                _ex_w = _x_target - float(pos_w[0])
                _ey_w = _y_target - float(pos_w[1])

                cmd_vx, cmd_vy, cmd_wz, _pan_done, _pd = _pan_world_xy_control(
                    PipelineState.PAN_XY,
                    _ex_w, _ey_w,
                    target_yaw=math.pi / 2,
                    pos_tol=0.05,       # x/y 分别都要求进入 ±5 cm
                    resume_tol=0.10,    # 被耦合运动带出 ±10 cm 时明确退出停车状态
                    kp_x=1.0, kp_y=0.8,
                    vmax_x=0.25, vmax_y=0.35,
                    yaw_now=yaw,
                    base_lin_vel=rs["lin_vel"],
                )

                if state_step == 0:
                    print(
                        f"[SM] PAN_XY start target=({_x_target:.3f},{_y_target:.3f}) "
                        f"err=({_pd['err_x']:+.3f},{_pd['err_y']:+.3f})m",
                        flush=True)

                if state_step % 25 == 0:
                    print(
                        f"[SM] PAN_XY step {state_step}: "
                        f"ex={_pd['err_x']:+.3f}m ey={_pd['err_y']:+.3f}m "
                        f"in_tol={int(_pd['in_x'])}/{int(_pd['in_y'])} "
                        f"world_cmd=({_pd['world_vx_target']:+.3f},"
                        f"{_pd['world_vy_target']:+.3f}) "
                        f"body_cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                        f"vxy={_pd['vxy']:.3f} "
                        f"yaw_err={math.degrees(_pd['yaw_err']):+.2f}deg "
                        f"settling={int(_pd['settling'])} "
                        f"settle={_pd['settle_count']}/{_PAN_SETTLE_STEPS}",
                        flush=True)

                state_step += 1
                if _pan_done or state_step >= PAN_MAX_STEPS:
                    _why = "settled" if _pan_done else "timeout"
                    _ex_final = _x_target - float(pos_w[0])
                    _ey_final = _y_target - float(pos_w[1])
                    print(
                        f"[SM] PAN_XY done ({_why}): "
                        f"final_pos=({float(pos_w[0]):.3f},{float(pos_w[1]):.3f}) "
                        f"err=({_ex_final:+.3f},{_ey_final:+.3f})m "
                        f"-> ARM_INIT",
                        flush=True)
                    next_state = PipelineState.ARM_INIT
                    state_step = 0

            elif current_state == PipelineState.ARM_INIT:
                # 机械臂插值到 ARM_SIDE_ANGLES；底盘由抓取阶段统一 weld 固定。
                # 每次重试/初始化都先退出恒力夹持，允许夹爪正常张开。
                if state_step == 0:
                    env.disable_gripper_force_hold()
                cur_q = _get_arm_q(jpos)
                _arm_budget = BUDGET[PipelineState.ARM_INIT]
                _arm_alpha = min(1.3, state_step / max(_arm_budget * 0.6, 1))
                q6 = cur_q + _arm_alpha * (ARM_SIDE_ANGLES - cur_q)
                _arm_step(q6)
                _gripper_step(close=False)
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if state_step == 1:
                    print(f"[SM] ARM_INIT: retracting arm... base_pos_w={np.round(pos_w,3)}",
                          flush=True)
                _arm_err_now = np.abs(cur_q - ARM_SIDE_ANGLES)
                if state_step % 50 == 0:
                    print(f"[SM] ARM_INIT step {state_step}/{_arm_budget}: "
                          f"err_max={_arm_err_now.max():.4f}", flush=True)
                state_step += 1
                _arm_converged = (_arm_err_now.max() < 0.02)
                _arm_timeout   = (state_step >= _arm_budget)
                if _arm_converged or _arm_timeout:
                    _reason = "converged" if _arm_converged else "timeout"
                    print(f"[SM] ARM_INIT done ({_reason} at step {state_step-1})",
                          flush=True)
                    if args.grasp_checkpoint:
                        next_state = PipelineState.SCAN
                    else:
                        next_state = PipelineState.DONE
                    state_step = 0
                    depth_accum = []
                    scan_rgb    = None

            elif current_state == PipelineState.SCAN:
                # 收集 wrist_cam 的 RGB + depth 帧
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)
                try:
                    if state_step == 1:
                        print(f"[SM] SCAN: warmup {SCAN_WARMUP} + accumulate "
                              f"{SCAN_FRAMES} frames... base_pos_w={np.round(pos_w,3)}",
                              flush=True)
                    _scan_rgb_now, _scan_dep_now = env.render_camera("wrist_cam")
                    if state_step > SCAN_WARMUP:
                        d = _scan_dep_now
                        if d.ndim == 3:
                            d = d[:, :, 0]
                        depth_accum.append(d.astype(np.float32))
                        if scan_rgb is None:
                            scan_rgb = _scan_rgb_now
                    state_step += 1
                    if state_step >= SCAN_WARMUP + SCAN_FRAMES:
                        depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)
                        valid_pct = np.mean((depth_med > 0.05) & (depth_med < 4.0)) * 100
                        print(f"[SM] SCAN done: {len(depth_accum)} frames, "
                              f"valid depth {valid_pct:.1f}%", flush=True)
                        _q_scan_at_scan  = _get_arm_q(jpos).copy()
                        _pos_w_at_scan   = pos_w.copy()
                        _quat_w_at_scan  = quat_w.copy()
                        _scan_transforms = _capture_scan_tf(env)
                        _Tbc = _scan_transforms["T_base_camera_optical"]
                        print(f"[FRAME] SCAN camera optical in base: p={np.round(_Tbc[:3,3],4)} "
                              f"z_forward={np.round(_Tbc[:3,2],4)}", flush=True)
                        # 调试图输出独立到 diagnostics 模块。
                        _diag_save_scan_raw(scan_rgb)
                        next_state = PipelineState.GRASP_PLAN
                        state_step = 0
                except ValueError as _scan_e:
                    print(f"[SM] SCAN camera unavailable ({_scan_e}) -- DONE.",
                          flush=True)
                    next_state = PipelineState.DONE
                    state_step = 0

            elif current_state == PipelineState.MOVEIT_PRE_GRASP:
                # AnyGrasp candidate -> explicit grasp_tcp target -> MoveIt IK/collision/path plan.
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)

                if grasp_result is None:
                    next_state = PipelineState.ARM_INIT
                    state_step = 0
                else:
                    if _moveit_pre_cmds is None:
                        # Apply table/floor geometry only here, after Nav bridge has already
                        # published map->odom->base_link TF for several seconds.
                        if not _moveit_scene_applied:
                            _scene_reply = moveit_bridge.apply_default_scene(timeout=6.0)
                            if not _scene_reply.get("ok", False):
                                print(f"[MOVEIT] planning scene failed: {_scene_reply}", flush=True)
                                grasp_result = None
                                next_state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                _moveit_scene_applied = True
                                print("[MOVEIT] PlanningScene: table1 + table2 + floor applied", flush=True)

                        if next_state == current_state:
                            _moveit_selected = None
                            _names, _positions = _get_moveit_joint_state(jpos)
                            _T_b_c_scan = grasp_result["scan_transforms"]["T_base_camera_optical"]
                            _R_w_b_scan = grasp_result["scan_transforms"]["T_world_base"][:3, :3]

                            for _ci in grasp_result.get("candidate_idxs", []):
                                try:
                                    _cand = _ag_to_tcp(
                                        _T_b_c_scan,
                                        grasp_result["gr_translations"][_ci],
                                        grasp_result["gr_rotations"][_ci],
                                        float(grasp_result["gr_depths"][_ci]),
                                        insertion=float(args.moveit_insertion),
                                        pregrasp_distance=float(args.moveit_pregrasp_distance),
                                        local_reach_distance=float(args.moveit_local_reach_distance),
                                    )
                                    _approach_b = np.asarray(_cand["approach_base"], dtype=np.float64)
                                    _approach_w = _R_w_b_scan @ _approach_b
                                    _score = float(grasp_result["gr_scores"][_ci])
                                    print(
                                        f"[MOVEIT] candidate[{_ci}] score={_score:.3f} "
                                        f"approach_base={np.round(_approach_b,3)} "
                                        f"approach_world={np.round(_approach_w,3)}",
                                        flush=True)

                                    # Do not hide a frame/sign bug by auto-flipping the grasp.
                                    if float(_approach_w[2]) > float(args.moveit_max_approach_world_z):
                                        print(f"[MOVEIT]   reject: upward approach z={_approach_w[2]:+.3f}", flush=True)
                                        continue

                                    _pre_pose = _pose_to_ros(_cand["pregrasp_tcp_pose_base"])
                                    _ik = moveit_bridge.compute_ik(
                                        _pre_pose, _names, _positions,
                                        avoid_collisions=True,
                                    )
                                    if not _ik.get("ok", False):
                                        print(f"[MOVEIT]   reject: IK/collision error={_ik.get('error_code')}", flush=True)
                                        continue

                                    _plan = moveit_bridge.plan_to_pose(
                                        _pre_pose, _names, _positions,
                                        position_tolerance=float(args.moveit_position_tolerance),
                                        orientation_tolerance=float(args.moveit_orientation_tolerance),
                                        allowed_planning_time=float(args.moveit_plan_time),
                                    )
                                    if not _plan.get("ok", False):
                                        print(f"[MOVEIT]   reject: path planning error={_plan.get('error_code')} "
                                              f"{_plan.get('error','')}", flush=True)
                                        continue

                                    _dense = _dense_moveit_commands(_plan)
                                    if len(_dense) == 0:
                                        print("[MOVEIT]   reject: empty trajectory", flush=True)
                                        continue

                                    _moveit_selected = {
                                        "candidate_idx": int(_ci),
                                        "geometry": _cand,
                                        "pre_plan": _plan,
                                        "approach_world_scan": _approach_w.copy(),
                                    }
                                    _moveit_pre_cmds = _dense
                                    _moveit_pre_i = 0
                                    grasp_result["selected_idx"] = int(_ci)
                                    print(
                                        f"[MOVEIT] SELECT candidate[{_ci}] score={_score:.3f} "
                                        f"pre_tcp_base={np.round(_cand['pregrasp_tcp_pose_base'].position,4)} "
                                        f"contact_tcp_base={np.round(_cand['contact_tcp_pose_base'].position,4)} "
                                        f"plan_points={len(_dense)}",
                                        flush=True)
                                    _diag_save_choice_image(
                                        scan_rgb,
                                        grasp_result["gr_translations"][_ci],
                                        grasp_result["gr_rotations"][_ci],
                                    )
                                    break
                                except Exception as _ce:
                                    print(f"[MOVEIT] candidate[{_ci}] exception: {_ce}", flush=True)

                            if _moveit_selected is None and next_state == current_state:
                                print("[MOVEIT] no AnyGrasp candidate passed IK+collision+path planning -> ARM_INIT", flush=True)
                                grasp_result = None
                                _moveit_pre_cmds = None
                                next_state = PipelineState.ARM_INIT
                                state_step = 0

                    if _moveit_pre_cmds is not None and next_state == current_state:
                        _idx = min(_moveit_pre_i, len(_moveit_pre_cmds) - 1)
                        _arm_step(_moveit_pre_cmds[_idx])
                        if _moveit_pre_i < len(_moveit_pre_cmds) - 1:
                            _moveit_pre_i += 1
                        state_step += 1

                        _final_q = _moveit_pre_cmds[-1]
                        _err = float(np.max(np.abs(_get_arm_q(jpos) - _final_q)))
                        _pre_target = _moveit_selected["geometry"]["pregrasp_tcp_pose_base"]
                        _tcp_pos_err, _tcp_rot_err = _tcp_target_error_base(_pre_target)
                        if state_step % 25 == 0:
                            print(
                                f"[MOVEIT] PRE execute {min(_moveit_pre_i+1,len(_moveit_pre_cmds))}/"
                                f"{len(_moveit_pre_cmds)} qerr={_err:.3f}rad "
                                f"tcp_pos_err={_tcp_pos_err*1000:.1f}mm "
                                f"tcp_rot_err={math.degrees(_tcp_rot_err):.2f}deg",
                                flush=True)

                        _commands_done = (_moveit_pre_i >= len(_moveit_pre_cmds) - 1)
                        # qerr 是真正的执行收敛判据；TCP残差只作为异常保护。
                        _settled = (
                            _commands_done
                            and _err <= float(args.moveit_exec_joint_tolerance)
                            and _tcp_pos_err <= float(args.moveit_exec_tcp_position_tolerance)
                            and _tcp_rot_err <= float(args.moveit_exec_tcp_orientation_tolerance)
                        )
                        if _settled:
                            print(
                                f"[MOVEIT] pre-grasp physically settled: "
                                f"qerr={_err:.3f}rad pos={_tcp_pos_err*1000:.1f}mm "
                                f"rot={math.degrees(_tcp_rot_err):.2f}deg -> MOVEIT_APPROACH",
                                flush=True)
                            _moveit_approach_cmds = None
                            _moveit_approach_i = 0
                            next_state = PipelineState.MOVEIT_APPROACH
                            state_step = 0
                        elif state_step >= max(400, len(_moveit_pre_cmds) + 220):
                            print(
                                f"[MOVEIT] pre-grasp execution timeout "
                                f"qerr={_err:.3f}rad pos={_tcp_pos_err*1000:.1f}mm "
                                f"rot={math.degrees(_tcp_rot_err):.2f}deg -> ARM_INIT",
                                flush=True)
                            grasp_result = None
                            _moveit_pre_cmds = None
                            next_state = PipelineState.ARM_INIT
                            state_step = 0

            elif current_state == PipelineState.MOVEIT_APPROACH:
                # Straight Cartesian approach in MoveIt, stopping short of contact.
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)
                if grasp_result is None or _moveit_selected is None:
                    next_state = PipelineState.ARM_INIT
                    state_step = 0
                else:
                    if _moveit_approach_cmds is None:
                        _names, _positions = _get_moveit_joint_state(jpos)
                        _near_pose = _pose_to_ros(_moveit_selected["geometry"]["near_tcp_pose_base"])
                        try:
                            _cart = moveit_bridge.compute_cartesian_path(
                                [_near_pose], _names, _positions,
                                max_step=float(args.moveit_cart_step),
                                jump_threshold=0.0,
                                avoid_collisions=True,
                            )
                        except Exception as _e:
                            _cart = {"ok": False, "error": repr(_e), "fraction": 0.0}

                        _fraction = float(_cart.get("fraction", 0.0))
                        if (not _cart.get("ok", False) or
                                _fraction < float(args.moveit_cart_fraction)):
                            print(f"[MOVEIT] Cartesian approach failed fraction={_fraction:.3f} "
                                  f"error={_cart.get('error','')} -> ARM_INIT", flush=True)
                            grasp_result = None
                            _moveit_pre_cmds = None
                            _moveit_approach_cmds = None
                            next_state = PipelineState.ARM_INIT
                            state_step = 0
                        else:
                            _dense = _dense_moveit_commands(_cart)
                            if len(_dense) == 0:
                                print("[MOVEIT] Cartesian path returned empty trajectory -> ARM_INIT", flush=True)
                                grasp_result = None
                                next_state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                _moveit_approach_cmds = _dense
                                _moveit_approach_i = 0
                                print(f"[MOVEIT] Cartesian approach accepted fraction={_fraction:.3f} "
                                      f"commands={len(_dense)} stop_short={args.moveit_local_reach_distance*1000:.1f}mm",
                                      flush=True)

                    if _moveit_approach_cmds is not None and next_state == current_state:
                        _idx = min(_moveit_approach_i, len(_moveit_approach_cmds) - 1)
                        _arm_step(_moveit_approach_cmds[_idx])
                        if _moveit_approach_i < len(_moveit_approach_cmds) - 1:
                            _moveit_approach_i += 1
                        state_step += 1
                        _final_q = _moveit_approach_cmds[-1]
                        _err = float(np.max(np.abs(_get_arm_q(jpos) - _final_q)))
                        _near_target = _moveit_selected["geometry"]["near_tcp_pose_base"]
                        _tcp_pos_err, _tcp_rot_err = _tcp_target_error_base(_near_target)
                        if state_step % 20 == 0:
                            print(
                                f"[MOVEIT] APPROACH execute {min(_moveit_approach_i+1,len(_moveit_approach_cmds))}/"
                                f"{len(_moveit_approach_cmds)} qerr={_err:.3f}rad "
                                f"tcp_pos_err={_tcp_pos_err*1000:.1f}mm "
                                f"tcp_rot_err={math.degrees(_tcp_rot_err):.2f}deg",
                                flush=True)

                        _commands_done = (_moveit_approach_i >= len(_moveit_approach_cmds) - 1)
                        # APPROACH 比 PRE 更严格：尽量让 LOCAL_REACH 真正从 stop_short
                        # 附近接手；TCP残差仍只作为异常保护。
                        _settled = (
                            _commands_done
                            and _err <= float(args.moveit_approach_joint_tolerance)
                            and _tcp_pos_err <= float(args.moveit_exec_tcp_position_tolerance)
                            and _tcp_rot_err <= float(args.moveit_exec_tcp_orientation_tolerance)
                        )
                        if _settled:
                            _local_servo = _MuJoCoGBServo(env)
                            local_reach_ctx.reset()
                            print(
                                f"[MOVEIT] near-grasp physically settled: "
                                f"qerr={_err:.3f}rad pos={_tcp_pos_err*1000:.1f}mm "
                                f"rot={math.degrees(_tcp_rot_err):.2f}deg -> LOCAL_REACH",
                                flush=True)
                            next_state = PipelineState.LOCAL_REACH
                            state_step = 0
                        elif state_step >= max(280, len(_moveit_approach_cmds) + 180):
                            print(
                                f"[MOVEIT] approach execution timeout "
                                f"qerr={_err:.3f}rad pos={_tcp_pos_err*1000:.1f}mm "
                                f"rot={math.degrees(_tcp_rot_err):.2f}deg -> ARM_INIT",
                                flush=True)
                            grasp_result = None
                            next_state = PipelineState.ARM_INIT
                            state_step = 0

            elif current_state == PipelineState.LOCAL_REACH:
                # Final short approach: use MuJoCo actual Jacobian/body pose.
                #
                # Important hand-off rule:
                #   * no contact: normal local Cartesian servo
                #   * both fingers touch: CLOSE immediately
                #   * only one finger touches: DO NOT immediately CLOSE.
                #     Slow the servo, allow a small additional advance, and stop when
                #     the other finger also touches / the extra-depth budget is used /
                #     the single-side normal force becomes too large.
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)

                if grasp_result is None or _moveit_selected is None:
                    local_reach_ctx.reset()
                    next_state = PipelineState.ARM_INIT
                    state_step = 0
                else:
                    if _local_servo is None:
                        _local_servo = _MuJoCoGBServo(env)
                        local_reach_ctx.reset()

                    _hit_table, _hit_desc = _robot_table_contact(env)
                    if _hit_table:
                        print(f"[LOCAL_REACH] ABORT table contact: {_hit_desc} -> ARM_INIT", flush=True)
                        grasp_result = None
                        local_reach_ctx.reset()
                        next_state = PipelineState.ARM_INIT
                        state_step = 0
                    else:
                        # Build the final gripper-base target from the selected contact TCP.
                        _tcp_contact_b = _moveit_selected["geometry"]["contact_tcp_pose_base"]
                        _gb_target_b = _tcp_to_gb_pose(_tcp_contact_b)
                        _T_w_b_now = _mj_body_pose_world(env, "base_link").matrix()
                        _T_b_gbt = _gb_target_b.matrix()
                        _T_w_gbt = _T_w_b_now @ _T_b_gbt
                        _gb_target_w = _MoveItPose(_T_w_gbt[:3, 3], _T_w_gbt[:3, :3])

                        # Current/final TCP geometry in world frame, used for contact-depth logic.
                        _cur_tcp_w = _current_tcp_pose_world()
                        _T_w_tcpt = _T_w_b_now @ _tcp_contact_b.matrix()
                        _target_tcp_w = _MoveItPose(
                            _T_w_tcpt[:3, 3].copy(), _T_w_tcpt[:3, :3].copy())
                        _approach_w = _T_w_b_now[:3, :3] @ _tcp_contact_b.rotation[:, 2]
                        _approach_w = _approach_w / max(float(np.linalg.norm(_approach_w)), 1e-12)
                        _tcp_pos_err_now = float(
                            np.linalg.norm(_target_tcp_w.position - _cur_tcp_w.position))

                        _finger_hit = _local_finger_contact(env, args.object)
                        _n_hit = int(bool(_finger_hit["link7"])) + int(bool(_finger_hit["link8"]))

                        if _n_hit == 2:
                            print(
                                f"[LOCAL_REACH] bilateral target contact "
                                f"pos_err={_tcp_pos_err_now*1000:.1f}mm -> CLOSE",
                                flush=True)
                            local_reach_ctx.reset()
                            next_state = PipelineState.CLOSE
                            state_step = 0
                            close_ctx.reset()
                        else:
                            _speed_scale = 1.0

                            if _n_hit == 1:
                                _side = "link7" if _finger_hit["link7"] else "link8"
                                _forces = _finger_object_normal_forces(args.object)
                                _normal_force = float(_forces.get(_side, 0.0))

                                if local_reach_ctx.first_contact_tcp_world is None:
                                    local_reach_ctx.first_contact_tcp_world = (
                                        _cur_tcp_w.position.astype(np.float64).copy())
                                    local_reach_ctx.first_contact_side = _side
                                    local_reach_ctx.first_contact_pos_err_m = _tcp_pos_err_now
                                    local_reach_ctx.single_contact_steps = 0
                                    print(
                                        f"[LOCAL_REACH] first unilateral contact {_side}: "
                                        f"remaining={_tcp_pos_err_now*1000:.1f}mm; "
                                        f"continue slowly up to "
                                        f"{float(args.local_single_contact_max_advance)*1000:.1f}mm",
                                        flush=True)

                                local_reach_ctx.single_contact_steps += 1
                                _advance = float(np.dot(
                                    _cur_tcp_w.position - local_reach_ctx.first_contact_tcp_world,
                                    _approach_w))
                                _advance = max(0.0, _advance)

                                # 单侧力只作为“异常/卡住”判据，不再作为 CLOSE 条件。
                                # 为避免 MuJoCo 单步接触冲击峰值误杀，要求连续多步超限。
                                if _normal_force >= float(args.local_single_contact_force_limit):
                                    local_reach_ctx.overforce_steps += 1
                                else:
                                    local_reach_ctx.overforce_steps = 0

                                _force_abnormal = (
                                    local_reach_ctx.overforce_steps
                                    >= max(1, int(args.local_single_contact_force_confirm_steps))
                                )
                                _depth_done = (
                                    _advance >= float(args.local_single_contact_max_advance)
                                )

                                if _force_abnormal:
                                    print(
                                        f"[LOCAL_REACH] ABORT unilateral {_side}: "
                                        f"Fn={_normal_force:.2f}N >= "
                                        f"{float(args.local_single_contact_force_limit):.2f}N "
                                        f"for {local_reach_ctx.overforce_steps} consecutive steps; "
                                        f"advance={_advance*1000:.1f}mm "
                                        f"remaining={_tcp_pos_err_now*1000:.1f}mm "
                                        f"-> ARM_INIT / replan",
                                        flush=True)
                                    grasp_result = None
                                    _moveit_selected = None
                                    _local_servo = None
                                    local_reach_ctx.reset()
                                    next_state = PipelineState.ARM_INIT
                                    state_step = 0
                                elif _depth_done:
                                    print(
                                        f"[LOCAL_REACH] unilateral {_side} extra advance "
                                        f"{_advance*1000:.1f}mm >= "
                                        f"{float(args.local_single_contact_max_advance)*1000:.1f}mm; "
                                        f"remaining={_tcp_pos_err_now*1000:.1f}mm -> CLOSE",
                                        flush=True)
                                    local_reach_ctx.reset()
                                    next_state = PipelineState.CLOSE
                                    state_step = 0
                                    close_ctx.reset()
                                else:
                                    _speed_scale = float(args.local_single_contact_speed_scale)
                                    if (local_reach_ctx.single_contact_steps == 1 or
                                            local_reach_ctx.single_contact_steps % 5 == 0):
                                        print(
                                            f"[LOCAL_REACH] unilateral {_side}: "
                                            f"advance={_advance*1000:.1f}mm "
                                            f"remaining={_tcp_pos_err_now*1000:.1f}mm "
                                            f"Fn={_normal_force:.2f}N "
                                            f"overforce={local_reach_ctx.overforce_steps}/"
                                            f"{max(1, int(args.local_single_contact_force_confirm_steps))} "
                                            f"speed_scale={_speed_scale:.2f}",
                                            flush=True)
                            else:
                                # Contact disappeared before CLOSE: resume normal servo and
                                # forget the old first-contact reference. A later contact
                                # starts a new bounded slow-advance window.
                                if local_reach_ctx.first_contact_tcp_world is not None:
                                    print(
                                        "[LOCAL_REACH] unilateral contact released; "
                                        "resume normal local servo",
                                        flush=True)
                                local_reach_ctx.reset()

                            if next_state == current_state:
                                _q_raw, _sd = _local_servo.step_world(_gb_target_w)
                                _q_now = _get_arm_q(jpos)
                                if _speed_scale < 1.0:
                                    _q_cmd = _q_now + _speed_scale * (_q_raw - _q_now)
                                else:
                                    _q_cmd = _q_raw
                                _arm_step(_q_cmd)

                                state_step += 1
                                if state_step == 1 or state_step % 10 == 0:
                                    print(
                                        f"[LOCAL_REACH] step={state_step} "
                                        f"pos_err={_sd.pos_error_m*1000:.1f}mm "
                                        f"rot_err={math.degrees(_sd.rot_error_rad):.1f}deg "
                                        f"dqmax={math.degrees(_sd.dq_max_rad):.2f}deg "
                                        f"speed_scale={_speed_scale:.2f}",
                                        flush=True)

                                if _sd.pos_error_m < 0.004 and _sd.rot_error_rad < 0.10:
                                    print(
                                        "[LOCAL_REACH] contact pose reached geometrically -> CLOSE",
                                        flush=True)
                                    local_reach_ctx.reset()
                                    next_state = PipelineState.CLOSE
                                    state_step = 0
                                    close_ctx.reset()
                                elif state_step >= BUDGET[PipelineState.LOCAL_REACH]:
                                    print("[LOCAL_REACH] timeout -> ARM_INIT", flush=True)
                                    grasp_result = None
                                    local_reach_ctx.reset()
                                    next_state = PipelineState.ARM_INIT
                                    state_step = 0

            elif current_state == PipelineState.CLOSE:
                # CLOSE 只允许两个 finger 动；机械臂六关节固定为“进入 CLOSE 时的实际 q”。
                # 这样不会再追逐 REACH 的旧理论 target，接触建立过程中末端姿态保持不变。
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                cur_q = _get_arm_q(jpos)
                if state_step == 0:
                    env.disable_gripper_force_hold()
                    close_ctx.reset()
                    close_ctx.arm_q_hold = cur_q.copy()
                    env._grip_max_delta = _GRIP_MAX_DELTA_NORMAL
                    # 记录 CLOSE 起点物体位姿，后续只用于诊断相对平移/转角。
                    close_ctx.diag_obj_pose0 = _diag_capture_body_pose(env, args.object)
                    print(f"[SM] CLOSE start: arm q 固定；首触约0.25N预压，另一指降速到 "
                          f"{_CLOSE_SEARCH_MAX_DELTA*1000:.1f}mm/step；"
                          f"双指都曾接触后切换恒力夹持 ±2.0N", flush=True)

                _arm_step(close_ctx.arm_q_hold)
                tc = _fingers_contact(args.object)
                q7c, q8c = float(jpos[_GRI_Q7_I]), float(jpos[_GRI_Q8_I])

                # CLOSE 成功判定历史：只记录，不参与当前帧控制。
                close_ctx.contact_hist.append((bool(tc["link7"]), bool(tc["link8"])))
                if len(close_ctx.contact_hist) > _CLOSE_CHECK_WINDOW:
                    close_ctx.contact_hist.pop(0)
                try:
                    _obj_p_hist = np.asarray(
                        env.get_object_pos(args.object), dtype=np.float64
                    ).copy()
                    close_ctx.obj_pos_hist.append(_obj_p_hist)
                    if len(close_ctx.obj_pos_hist) > _CLOSE_CHECK_WINDOW:
                        close_ctx.obj_pos_hist.pop(0)
                except Exception:
                    pass

                if close_ctx.finger_targets[0] is None and tc["link7"]:
                    close_ctx.finger_targets[0] = q7c - _CLOSE_PRELOAD_S
                    print(f"[SM] CLOSE: link7 contact @q7={q7c:.4f} -> "
                          f"preload target={close_ctx.finger_targets[0]:.4f} (~0.25N)", flush=True)
                    ci = _diag_finger_contact_info(env, args.object)["link7"]
                    if ci is not None:
                        p, l = ci["world"], ci["local"]
                        print(f"[DIAG] CLOSE link7↔{ci['obj_geom_name']}: "
                              f"world={np.round(p,4)} local_mm={np.round(l*1000,2)} "
                              f"dist={ci['dist']*1000:+.2f}mm", flush=True)

                if close_ctx.finger_targets[1] is None and tc["link8"]:
                    close_ctx.finger_targets[1] = q8c + _CLOSE_PRELOAD_S
                    print(f"[SM] CLOSE: link8 contact @q8={q8c:.4f} -> "
                          f"preload target={close_ctx.finger_targets[1]:.4f} (~0.25N)", flush=True)
                    ci = _diag_finger_contact_info(env, args.object)["link8"]
                    if ci is not None:
                        p, l = ci["world"], ci["local"]
                        print(f"[DIAG] CLOSE link8↔{ci['obj_geom_name']}: "
                              f"world={np.round(p,4)} local_mm={np.round(l*1000,2)} "
                              f"dist={ci['dist']*1000:+.2f}mm", flush=True)

                # 只有一侧已经首触时，两指都减速：首触侧缓慢建立预压力，另一侧慢速搜索。
                _n_contacted_once = int(close_ctx.finger_targets[0] is not None) + int(close_ctx.finger_targets[1] is not None)
                if (not close_ctx.force_hold_started) and _n_contacted_once == 1:
                    env._grip_max_delta = _CLOSE_SEARCH_MAX_DELTA
                else:
                    env._grip_max_delta = _GRIP_MAX_DELTA_NORMAL

                # 两指都曾经接触后，不再继续追逐“物体内部的位置目标”。
                # 直接切到对称恒力夹持：joint7=-1N、joint8=+1N。
                # 不锁中心、不做中心自适应，让物体自行在双指之间找到稳定位置。
                if (not close_ctx.force_hold_started) and close_ctx.finger_targets[0] is not None and close_ctx.finger_targets[1] is not None:
                    close_ctx.force_hold_started = True
                    env._grip_max_delta = _GRIP_MAX_DELTA_NORMAL
                    env.set_gripper_force_hold(2.0)
                    print(
                        "[SM] CLOSE: both fingers touched -> SYMMETRIC FORCE HOLD "
                        "(joint7=-2.00N, joint8=+2.00N; no center lock)",
                        flush=True)

                # 不再使用旧的“80% budget 单侧加深”兜底。
                # 单侧接触时继续慢速搜索另一侧；只有双指都曾接触才允许进入恒力模式。

                env.set_gripper_target(np.array([
                    close_ctx.finger_targets[0] if close_ctx.finger_targets[0] is not None else 0.0,
                    close_ctx.finger_targets[1] if close_ctx.finger_targets[1] is not None else 0.0,
                ], dtype=np.float64))

                # ── CLOSE 诊断：每 10 step，或接触状态发生变化时立即打印。
                # 只读取当前 MuJoCo 状态，不改变任何控制逻辑。
                _cl_contact_tuple = (bool(tc["link7"]), bool(tc["link8"]))
                _cl_diag_due = (state_step % 10 == 0 or
                                close_ctx.diag_prev_contact is None or
                                _cl_contact_tuple != close_ctx.diag_prev_contact)
                if _cl_diag_due:
                    _diag_print_close_step(
                        env, args.object, state_step, tc, q7c, q8c,
                        close_ctx.diag_obj_pose0,
                    )

                    # 深度 PRE 诊断：与当前 CDIAG 同一时刻；随后在 env.step() 后打印 POST。
                    # PRE 的 ctrl 是上一物理步留下的值，因此只用于和 POST 对照，不单独作因果判断。
                    try:
                        _diag_print_finger_deep(env, "CPRE", state_step, args.object, force_contacts=False)
                        close_ctx.deep_post_pending = dict(
                            step=int(state_step),
                            pre_contact=_cl_contact_tuple,
                        )
                    except Exception as _de_pre:
                        print(f"[CPRE] diagnostic error: {_de_pre}", flush=True)
                        close_ctx.deep_post_pending = None
                close_ctx.diag_prev_contact = _cl_contact_tuple

                state_step += 1
                if state_step % 20 == 0:
                    print(f"[SM] CLOSE step {state_step}/{BUDGET[PipelineState.CLOSE]} "
                          f"q7={q7c:.4f} q8={q8c:.4f} "
                          f"gap={(q7c-q8c)*1000:.2f}mm "
                          f"contact={int(tc['link7'])}/{int(tc['link8'])}", flush=True)

                if state_step >= BUDGET[PipelineState.CLOSE]:
                    frz_s = [None if x is None else round(float(x), 4) for x in close_ctx.finger_targets]

                    # 不再要求最后一帧恰好 1/1：使用最近窗口的接触率判断。
                    if len(close_ctx.contact_hist) > 0:
                        r7 = sum(int(x[0]) for x in close_ctx.contact_hist) / len(close_ctx.contact_hist)
                        r8 = sum(int(x[1]) for x in close_ctx.contact_hist) / len(close_ctx.contact_hist)
                    else:
                        r7 = r8 = 0.0

                    # 最近窗口内 cube 的净位移，用于排除“虽然反复接触但物体仍在滑”的情况。
                    if len(close_ctx.obj_pos_hist) >= 2:
                        close_recent_move = float(np.linalg.norm(
                            close_ctx.obj_pos_hist[-1] - close_ctx.obj_pos_hist[0]
                        ))
                    else:
                        close_recent_move = float("inf")

                    close_stable = (
                        r7 >= _CLOSE_CONTACT_RATE_MIN
                        and r8 >= _CLOSE_CONTACT_RATE_MIN
                        and close_recent_move <= _CLOSE_STABLE_MOVE_MAX
                    )

                    print(
                        f"[DIAG] CLOSE done: q7={q7c:.4f} q8={q8c:.4f} "
                        f"current={int(tc['link7'])}/{int(tc['link8'])} "
                        f"recent_contact_rate={r7:.2f}/{r8:.2f} "
                        f"recent_obj_move={close_recent_move*1000:.2f}mm "
                        f"freeze={frz_s}",
                        flush=True
                    )

                    env._grip_max_delta = _GRIP_MAX_DELTA_NORMAL

                    # CLOSE 稳定性指标当前仅用于诊断；完成 budget 后始终进入 LIFT，
                    # 不用接触率/物体位移在这里拒绝抓取。
                    if close_stable:
                        print(
                            f"[SM] CLOSE stable -> LIFT: contact_rate={r7:.2f}/{r8:.2f}, "
                            f"obj_move={close_recent_move*1000:.2f}mm",
                            flush=True
                        )
                    else:
                        print(
                            f"[DIAG] CLOSE not stable by diagnostic metric -> LIFT: "
                            f"contact_rate={r7:.2f}/{r8:.2f}, "
                            f"obj_move={close_recent_move*1000:.2f}mm",
                            flush=True
                        )

                    # IMPORTANT: keep FORCE HOLD active into LIFT exactly as before on a
                    # successful CLOSE.  This test is intended to reveal whether the
                    # visually plausible grasp can actually support the object.
                    next_state = PipelineState.LIFT
                    state_step = 0

            elif current_state == PipelineState.LIFT:
                # 新LIFT：先用j2/j3把夹爪抬离桌面，再全关节收回ARM_SIDE。
                # Stage1不再要求world x/y不变，也不做DLS/姿态保持。
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                cur_q = _get_arm_q(jpos)
                p_cur_w, _ = _gb_pose_world()

                env.set_gripper_target(np.array([
                    close_ctx.finger_targets[0] if close_ctx.finger_targets[0] is not None else 0.0,
                    close_ctx.finger_targets[1] if close_ctx.finger_targets[1] is not None else 0.0,
                ], dtype=np.float64))

                if lift_ctx.stage == 0:
                    lift_ctx.begin(cur_q, p_cur_w, _LIFT_HOLD_STEPS)

                    try:
                        _lift_obj_p0 = np.asarray(
                            env.get_object_pos(args.object), dtype=np.float64).copy()
                        lift_ctx.obj_z_start = float(_lift_obj_p0[2])
                        lift_ctx.hold_obj_p0 = _lift_obj_p0.copy()
                    except Exception:
                        lift_ctx.obj_z_start = None
                        lift_ctx.hold_obj_p0 = None

                    # 只在LIFT开始时做一次MuJoCo FK方向检查：
                    # 沿j2/j3 -> ARM_SIDE各试探最多0.01rad，确认末端z会增加。
                    _q_test = lift_ctx.q_stage1_start.copy()
                    for _ji in (1, 2):
                        _dq_to_side = float(ARM_SIDE_ANGLES[_ji] - _q_test[_ji])
                        _q_test[_ji] += float(np.clip(
                            _dq_to_side, -_LIFT_STAGE1_TEST_DQ, _LIFT_STAGE1_TEST_DQ))
                    try:
                        _p_fk0 = _gb_pos_for_arm_q(lift_ctx.q_stage1_start)
                        _p_fk1 = _gb_pos_for_arm_q(_q_test)
                        lift_ctx.fk_dxyz = _p_fk1 - _p_fk0
                    except Exception as _e:
                        lift_ctx.fk_dxyz = np.array([np.nan, np.nan, np.nan])
                        print(f"[WARN] LIFT FK direction check failed: {_e} -> ARM_INIT retry",
                              flush=True)
                        grasp_result = None
                        lift_ctx.reset()
                        next_state = PipelineState.ARM_INIT
                        state_step = 0

                    if next_state == current_state:
                        _fk_dz = float(lift_ctx.fk_dxyz[2])
                        print(
                            f"[SM] LIFT stage1 joint-lift init: "
                            f"ee_world={np.round(lift_ctx.ee_p_start,4)}, "
                            f"q_start={np.round(lift_ctx.q_stage1_start,3)}, "
                            f"FK_test_dxyz_mm={np.round(lift_ctx.fk_dxyz*1000.0,3)}",
                            flush=True)
                        if (not np.isfinite(_fk_dz)) or _fk_dz <= _LIFT_STAGE1_FK_MIN_DZ:
                            print(
                                f"[WARN] LIFT j2/j3 -> ARM_SIDE direction does not raise EE: "
                                f"pred_dz={_fk_dz*1000.0:.3f}mm -> ARM_INIT retry",
                                flush=True)
                            grasp_result = None
                            lift_ctx.reset()
                            next_state = PipelineState.ARM_INIT
                            state_step = 0

                if lift_ctx.stage == 1 and next_state == current_state:
                    p_cur_w, _ = _gb_pose_world()
                    _ee_dp = p_cur_w - lift_ctx.ee_p_start
                    _ee_rise = float(_ee_dp[2])
                    try:
                        _obj_z_now = float(env.get_object_pos(args.object)[2])
                    except Exception:
                        _obj_z_now = float('nan')
                    _obj_rise = (
                        _obj_z_now - lift_ctx.obj_z_start
                        if lift_ctx.obj_z_start is not None and np.isfinite(_obj_z_now)
                        else float('nan'))

                    # CLOSE后先短暂静置，沿用已经验证过的窗口接触稳定性检查。
                    if lift_ctx.hold_remaining > 0:
                        _arm_step(lift_ctx.q_stage1_start.copy())

                        _tc_hold = _fingers_contact(args.object)
                        lift_ctx.hold_seen += 1
                        lift_ctx.hold_contacts[0] += int(_tc_hold["link7"])
                        lift_ctx.hold_contacts[1] += int(_tc_hold["link8"])
                        lift_ctx.hold_remaining -= 1

                        if lift_ctx.hold_remaining == 0:
                            _den = max(lift_ctx.hold_seen, 1)
                            _r7 = float(lift_ctx.hold_contacts[0]) / _den
                            _r8 = float(lift_ctx.hold_contacts[1]) / _den
                            try:
                                _obj_now = np.asarray(
                                    env.get_object_pos(args.object), dtype=np.float64)
                                _hold_move = (
                                    float(np.linalg.norm(_obj_now - lift_ctx.hold_obj_p0))
                                    if lift_ctx.hold_obj_p0 is not None else float('nan'))
                            except Exception:
                                _hold_move = float('nan')

                            _contact_ok = (
                                _r7 >= _LIFT_HOLD_CONTACT_RATIO and
                                _r8 >= _LIFT_HOLD_CONTACT_RATIO)
                            _motion_ok = (
                                not np.isfinite(_hold_move) or
                                _hold_move <= _LIFT_HOLD_MAX_OBJ_MOVE)
                            if not (_contact_ok and _motion_ok):
                                _move_msg = (
                                    f"{_hold_move*1000:.2f}mm"
                                    if np.isfinite(_hold_move) else "unavailable")
                                print(
                                    f"[WARN] LIFT hold unstable: "
                                    f"contact_rate={_r7:.2f}/{_r8:.2f}, "
                                    f"obj_move={_move_msg} -> ARM_INIT retry",
                                    flush=True)
                                grasp_result = None
                                lift_ctx.reset()
                                next_state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                _move_msg = (
                                    f"{_hold_move*1000:.2f}mm"
                                    if np.isfinite(_hold_move) else "unavailable")
                                print(
                                    f"[SM] LIFT hold done: "
                                    f"contact_rate={_r7:.2f}/{_r8:.2f}, "
                                    f"obj_move={_move_msg} -> j2/j3 lift",
                                    flush=True)
                        if next_state == current_state:
                            state_step += 1

                    # 当前调轨迹阶段：Stage1 -> Stage2只看末端实际z上升20mm。
                    elif _ee_rise >= _LIFT_STAGE1_EE_RISE:
                        lift_ctx.stage = 2
                        lift_ctx.q_retract0 = cur_q.copy()
                        lift_ctx.stage2_step = 0
                        state_step = 0
                        _obj_msg = (
                            f"{_obj_rise*1000.0:+.1f}mm"
                            if np.isfinite(_obj_rise) else "unavailable")
                        print(
                            f"[SM] LIFT stage1 done: "
                            f"ee_dxyz_mm={np.round(_ee_dp*1000.0,2)}, "
                            f"obj_rise={_obj_msg} -> stage2 all-joint retract",
                            flush=True)

                    elif state_step >= BUDGET[PipelineState.LIFT]:
                        _arm_step(cur_q.copy())
                        print(
                            f"[WARN] LIFT stage1 timeout: "
                            f"ee_dxyz_mm={np.round(_ee_dp*1000.0,2)} "
                            f"-> ARM_INIT retry", flush=True)
                        grasp_result = None
                        lift_ctx.reset()
                        next_state = PipelineState.ARM_INIT
                        state_step = 0

                    else:
                        # Stage1：j1、j4~j6保持LIFT开始时角度；
                        # 仅j2/j3沿固定方向缓慢逼近ARM_SIDE。
                        q_cmd = lift_ctx.q_stage1_start.copy()
                        for _ji in (1, 2):
                            _remain = float(
                                ARM_SIDE_ANGLES[_ji] - lift_ctx.q_stage1_cmd[_ji])
                            _step = float(np.clip(
                                _remain, -_LIFT_STAGE1_MAX_DQ, _LIFT_STAGE1_MAX_DQ))
                            lift_ctx.q_stage1_cmd[_ji] += _step
                            q_cmd[_ji] = lift_ctx.q_stage1_cmd[_ji]
                        q_cmd = np.clip(q_cmd, _ARM_LO, _ARM_HI)
                        _arm_step(q_cmd)
                        lift_ctx.stage1_step += 1

                        if lift_ctx.stage1_step == 1 or lift_ctx.stage1_step % 25 == 0:
                            _obj_msg = (
                                f"{_obj_rise*1000.0:+.1f}mm"
                                if np.isfinite(_obj_rise) else "unavailable")
                            print(
                                f"[DIAG] LIFT stage1 joint step {lift_ctx.stage1_step}: "
                                f"ee_dxyz_mm={np.round(_ee_dp*1000.0,2)} "
                                f"q2/q3=[{cur_q[1]:+.3f},{cur_q[2]:+.3f}] "
                                f"cmd=[{q_cmd[1]:+.3f},{q_cmd[2]:+.3f}] "
                                f"obj_rise={_obj_msg}",
                                flush=True)
                        state_step += 1

                elif lift_ctx.stage == 2 and next_state == current_state:
                    lift_ctx.stage2_step += 1
                    a = min(1.0, lift_ctx.stage2_step / max(_LIFT_RETRACT_STEPS, 1))
                    a_s = a * a * (3.0 - 2.0 * a)
                    q_cmd = lift_ctx.q_retract0 + a_s * (ARM_SIDE_ANGLES - lift_ctx.q_retract0)
                    _arm_step(q_cmd)
                    state_step += 1

                    if lift_ctx.stage2_step % 50 == 0:
                        p_now, _ = _gb_pose_world()
                        _ee_dp2 = p_now - lift_ctx.ee_p_start
                        try:
                            obj_z = float(env.get_object_pos(args.object)[2])
                        except Exception:
                            obj_z = float('nan')
                        obj_rise = (
                            obj_z - lift_ctx.obj_z_start
                            if lift_ctx.obj_z_start is not None and np.isfinite(obj_z)
                            else float('nan'))
                        _obj_msg = (
                            f"{obj_rise*1000.0:+.1f}mm"
                            if np.isfinite(obj_rise) else "unavailable")
                        print(
                            f"[DIAG] LIFT stage2 {lift_ctx.stage2_step}/{_LIFT_RETRACT_STEPS}: "
                            f"ee_dxyz_mm={np.round(_ee_dp2*1000.0,2)} "
                            f"obj_rise={_obj_msg}", flush=True)

                    if a >= 1.0 or state_step >= BUDGET[PipelineState.LIFT]:
                        try:
                            obj_z = float(env.get_object_pos(args.object)[2])
                        except Exception:
                            obj_z = 0.0
                        obj_rise = (obj_z - lift_ctx.obj_z_start
                                    if lift_ctx.obj_z_start is not None else 0.0)
                        p_now, _ = _gb_pose_world()
                        _ee_dp2 = p_now - lift_ctx.ee_p_start
                        print(
                            f"[SM] LIFT trajectory done: "
                            f"ee_dxyz_mm={np.round(_ee_dp2*1000.0,2)}, "
                            f"obj_z={obj_z:.3f}m rise={obj_rise*1000:.1f}mm",
                            flush=True)

                        # 最终抓取成功判据暂时沿用旧逻辑；它发生在完整Stage2之后，
                        # 不会阻止我们观察完整轨迹。Stage1->2已经只看EE高度。
                        if lift_ctx.obj_z_start is not None:
                            lift_success = (obj_rise > 0.04)
                        else:
                            lift_success = (obj_z > 0.75)
                        if lift_success:
                            next_state = PipelineState.PAN_NEG_X
                        else:
                            print(
                                "[SM] Object not lifted after full LIFT trajectory! "
                                "Retry ARM_INIT", flush=True)
                            env.disable_gripper_force_hold()
                            grasp_result = None
                            next_state = PipelineState.ARM_INIT
                        lift_ctx.reset()
                        state_step = 0

            elif current_state == PipelineState.PAN_NEG_X:
                # LIFT 后沿世界 -X 平移约 0.5m，再交给 Nav2。与其他 PAN 一样直接
                # 在世界系闭环，再根据实时 yaw 转成 body vx/vy。
                if _pan_neg_x_goal is None:
                    _pan_neg_x_goal = float(pos_w[0]) - 0.5
                    print(f"[SM] PAN_NEG_X start: x={pos_w[0]:.3f} -> "
                          f"goal={_pan_neg_x_goal:.3f} (-0.5m)", flush=True)
                _px_w = float(_pan_neg_x_goal) - float(pos_w[0])
                cmd_vx, cmd_vy, cmd_wz, _pan_done, _pd = _pan_world_axis_control(
                    PipelineState.PAN_NEG_X, _px_w, 'x', math.pi / 2,
                    pos_tol=0.03, resume_tol=0.08, kp=1.0, vmax=0.25,
                    yaw_now=yaw, base_lin_vel=rs["lin_vel"],
                    settle_use_vxy=True)
                _gripper_step(close=True)
                if state_step % 25 == 0:
                    print(f"[SM] PAN_NEG_X step {state_step}: px_err={_pd['signed_err']:+.3f}m "
                          f"cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                          f"vx_world={_pd['axis_vel']:+.3f} vxy={_pd['vxy']:.3f} "
                          f"yaw_err={math.degrees(_pd['yaw_err']):+.2f}deg "
                          f"settling={int(_pd['settling'])} "
                          f"settle={_pd['settle_count']}/{_PAN_SETTLE_STEPS}", flush=True)
                state_step += 1
                if _pan_done or state_step >= PAN_NEG_X_MAX_STEPS:
                    _why = "settled" if _pan_done else "timeout"
                    print(f"[SM] PAN_NEG_X done ({_why}) -> NAV2_DEST", flush=True)
                    _pan_neg_x_goal = None
                    next_state = PipelineState.NAV2_DEST
                    state_step = 0

            elif current_state == PipelineState.NAV2_DEST:
                # 发送 Nav2 目标到 destination，等待完成。
                # 不覆盖 cmd：与 NAV 阶段一致，主循环开头 get_cmd_vel() 读到的
                # Nav2 速度指令直接进 policy，机器人沿 Nav2 路径驶向 destination。
                # （曾经这里清零 cmd → policy 收到 (0,0,0)，狗站着不动、Nav2
                #  永远到不了 goal，只能等 6000 步超时。）
                _gripper_step(close=True)
                if not _dest_goal_sent:
                    bridge.send_goal(args.destination[0], args.destination[1])
                    _dest_goal_sent = True
                    print(f"[SM] NAV2_DEST goal sent: {args.destination}",
                          flush=True)
                _nd, _ = bridge.get_nav_status()
                state_step += 1
                if _nd or state_step >= NAV2_DEST_MAX_STEPS:
                    _reason2 = "done" if _nd else "timeout"
                    print(f"[SM] NAV2_DEST {_reason2} -> ALIGN_YAW_2", flush=True)
                    _dest_goal_sent = False
                    next_state = PipelineState.ALIGN_YAW_2
                    state_step = 0

            elif current_state == PipelineState.ALIGN_YAW_2:
                # 对齐到 -Y；与 ALIGN_YAW_1 使用同一套有效转速 + 停稳判定。
                cmd_vx = 0.0; cmd_vy = 0.0
                cmd_wz, _align_done, _ad = _align_yaw_control(
                    PipelineState.ALIGN_YAW_2, -math.pi / 2, yaw, ang_vel[2])
                _gripper_step(close=True)
                state_step += 1
                if state_step == 1 or state_step % 20 == 0:
                    print(f"[SM] ALIGN_YAW_2 step {state_step}: "
                          f"yaw={math.degrees(yaw):+.2f}deg "
                          f"err={math.degrees(_ad['err']):+.2f}deg "
                          f"cmd_wz={cmd_wz:+.3f} actual_wz={_ad['wz_actual']:+.3f} "
                          f"settle={_ad['settle_count']}/{_ALIGN_SETTLE_STEPS}", flush=True)
                if _align_done or state_step >= ALIGN_YAW_MAX_STEPS:
                    _why = "settled" if _align_done else "timeout"
                    print(f"[SM] ALIGN_YAW_2 done ({_why}) -> PAN_DES_X", flush=True)
                    next_state = PipelineState.PAN_DES_X
                    state_step = 0

            elif current_state == PipelineState.PAN_DES_X:
                # 第二张桌前精调世界 X。先在世界系生成纯 X 方向速度，再按实时 yaw
                # 转到机体系 vx/vy；因此即使 yaw 与 -π/2 有小误差，实际轨迹仍沿 world X。
                _pdx_w = float(args.destination[0]) - float(pos_w[0])
                cmd_vx, cmd_vy, cmd_wz, _pan_done, _pd = _pan_world_axis_control(
                    PipelineState.PAN_DES_X, _pdx_w, 'x', -math.pi / 2,
                    pos_tol=0.08, resume_tol=0.16, kp=1.0, vmax=0.25,
                    yaw_now=yaw, base_lin_vel=rs["lin_vel"])
                _gripper_step(close=True)
                if state_step == 1:
                    print(f"[SM] PAN_DES_X start dx_err={_pd['signed_err']:+.3f}m", flush=True)
                if state_step % 25 == 0:
                    print(f"[SM] PAN_DES_X step {state_step}: dx_err={_pd['signed_err']:+.3f}m "
                          f"cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                          f"vx_world={_pd['axis_vel']:+.3f} vxy={_pd['vxy']:.3f} "
                          f"yaw_err={math.degrees(_pd['yaw_err']):+.2f}deg "
                          f"settling={int(_pd['settling'])} "
                          f"settle={_pd['settle_count']}/{_PAN_SETTLE_STEPS}", flush=True)
                state_step += 1
                if _pan_done or state_step >= PAN_MAX_STEPS:
                    _why = "settled" if _pan_done else "timeout"
                    print(f"[SM] PAN_DES_X done ({_why}) -> ROTATE", flush=True)
                    _rotate_j1_start = _get_arm_q(jpos)[0]
                    _rotate_j2_start = _get_arm_q(jpos)[1]
                    _rotate_j3_start = _get_arm_q(jpos)[2]
                    _rotate_j_target = np.array([
                        math.pi / 2, 1.8, -1.8,
                        ARM_SIDE_ANGLES[3], ARM_SIDE_ANGLES[4], ARM_SIDE_ANGLES[5]
                    ], dtype=np.float64)
                    next_state = PipelineState.ROTATE
                    state_step = 0

            elif current_state == PipelineState.ROTATE:
                # 插值 j1→+π/2, j2→1.8, j3→-1.8 over 600 steps。
                # 用余弦剖面（起步/收尾速度为零）而非线性：线性插值是恒角速度的
                # 速度阶跃，臂 PD 跟踪快速斜坡产生的力矩反作用到未锁的躯干，
                # 与 policy 腿平衡环路耦合 → 狗和臂一起剧烈抖动。
                # 600 步（≈12s）把 j1 的峰值速度也从 0.0157 rad/步降到约一半以下；
                # 同时 ROTATE 期间底座已加物理锁（见 _set_base_physical_lock 调用处）。
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=True)
                _a_r = 0.5 * (1.0 - math.cos(math.pi * min(1.0, state_step / float(ROTATE_STEPS))))
                _qr  = _get_arm_q(jpos).copy()
                _qr[0] = _rotate_j1_start + _a_r * (_rotate_j_target[0] - _rotate_j1_start)
                _qr[1] = _rotate_j2_start + _a_r * (_rotate_j_target[1] - _rotate_j2_start)
                _qr[2] = _rotate_j3_start + _a_r * (_rotate_j_target[2] - _rotate_j3_start)
                _arm_step(_qr)
                state_step += 1
                if state_step >= ROTATE_STEPS:
                    print("[SM] ROTATE done -> PUT_DOWN", flush=True)
                    next_state = PipelineState.PUT_DOWN
                    state_step = 0

            elif current_state == PipelineState.PUT_DOWN:
                # 松开夹爪 100 步。进入放置阶段时退出恒力夹持，恢复位置PD张开。
                if state_step == 0:
                    env.disable_gripper_force_hold()
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)
                state_step += 1
                if state_step % 20 == 0:
                    print(f"[SM] PUT_DOWN step {state_step}/{PUT_DOWN_STEPS}", flush=True)
                if state_step >= PUT_DOWN_STEPS:
                    print("[SM] PUT_DOWN done -> DONE", flush=True)
                    next_state = PipelineState.DONE

            # ── 无机械臂控制的过渡阶段：每帧保持 ARM_INIT_ANGLES，防止机械臂下垂引起抖动 ──
            if current_state in _ARM_IDLE_STATES:
                _arm_step(ARM_INIT_ANGLES)

            # ── 构造 obs，运行 policy ──
            obs_np = build_policy_obs(
                jpos, jvel, ang_vel, quat_w,
                cmd_vx, cmd_vy, cmd_wz, last_action,
            )
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

            # 机械臂/抓取阶段不需要 locomotion policy 驱动底盘：将 locomotion action 清零。
            # 底盘位置本身由下方 MuJoCo equality weld 约束；这里清零只是避免腿/轮继续
            # 执行上一阶段的运动动作，不承担“冻结位置”的职责。
            with torch.inference_mode():
                action_t = policy(obs_t)
            action_np = action_t.cpu().numpy().flatten()[:16]
            if current_state in _LOCOMOTION_DISABLED_STATES:
                action_np = np.zeros(16, dtype=np.float32)
            last_action = action_np.copy()

            # ── 抓取阶段：统一用 MuJoCo 物理 weld 固定底盘 ──
            # 在 env.step 前同步状态，保证约束参与本轮每一个 physics substep。
            # ARM_INIT 首次进入时捕获一次 anchor；SCAN/GRASP_PLAN/PRE_GRASP/ORIENT/
            # REACH/CLOSE/LIFT 全程保持同一个 anchor，不在状态切换时重新捕获。
            # 离开 LIFT 进入 PAN_NEG_X 时解除。ROTATE 仍独立使用相同物理锁机制。
            _set_base_physical_lock(current_state in _BASE_LOCK_STATES)

            # ── 执行 env.step ──
            env.step(action_np)

            # CLOSE 深度 POST 诊断：真正执行完本轮 policy step（含全部 physics substeps）后读取。
            # 即使 CLOSE 在本轮末尾已经请求切到 LIFT，也仍先打印这最后一个 CLOSE step 的结果。
            if close_ctx.deep_post_pending is not None:
                try:
                    _pd = close_ctx.deep_post_pending
                    _post_tc = _fingers_contact(args.object)
                    _post_tuple = (bool(_post_tc["link7"]), bool(_post_tc["link8"]))
                    _changed_during_step = (_post_tuple != tuple(_pd["pre_contact"]))
                    _diag_print_finger_deep(
                        env, "CPOST", int(_pd["step"]), args.object,
                        force_contacts=bool(_changed_during_step))
                    if _changed_during_step:
                        print(
                            f"[CPOST]   contact_transition within policy step: "
                            f"{int(_pd['pre_contact'][0])}/{int(_pd['pre_contact'][1])} -> "
                            f"{int(_post_tuple[0])}/{int(_post_tuple[1])}",
                            flush=True)
                except Exception as _de_post:
                    print(f"[CPOST] diagnostic error: {_de_post}", flush=True)
                close_ctx.deep_post_pending = None

            # 物理锁定残余漂移：抓取链路与 ROTATE 中每 25 个状态步打印一次。
            if _base_lock["active"] and current_state in _BASE_LOCK_STATES:
                if state_step % 25 == 0:
                    _print_base_lock_diag("BASE_LOCK")

            # 本轮物理执行已经结束，现在才正式提交状态切换。
            state = next_state

            time.sleep(0.005)  # 实时 1×：让 MuJoCo passive viewer 有时间处理鼠标/键盘事件

            # ── GRASP_PLAN：在本轮 physics step 完成后同步执行一次抓取规划 ──
            if state == PipelineState.GRASP_PLAN and state_step == 0:
                state, grasp_result = _run_grasp_plan(
                    env=env,
                    args=args,
                    depth_accum=depth_accum,
                    scan_rgb=scan_rgb,
                    q_scan_at_scan=_q_scan_at_scan,
                    pos_w_at_scan=_pos_w_at_scan,
                    quat_w_at_scan=_quat_w_at_scan,
                    pos_w=pos_w,
                    quat_w=quat_w,
                    ransac_fit_plane=_ransac_fit_plane,
                    ransac_remove_plane=_ransac_remove_plane,
                    scan_transforms=_scan_transforms,
                )
                state_step = 0
                _moveit_pre_cmds = None
                _moveit_pre_i = 0
                _moveit_approach_cmds = None
                _moveit_approach_i = 0
                _moveit_selected = None
                _local_servo = None

    except KeyboardInterrupt:
        print("[MJ] KeyboardInterrupt")
    finally:
        try:
            moveit_bridge.stop()
        except Exception:
            pass
        bridge.stop()
        env.close()
        print("[MJ] Done")


if __name__ == "__main__":
    main()