#!/usr/bin/env python
"""navigate_mujoco.py — MuJoCo 版全流程导航抓取脚本。

对应 navigate_to_goal_nav2_whole.py，替换所有 Isaac Lab API，
复用 ROS2 bridge、grasp_worker、arm_ik 等纯 Python 模块。

用法示例：
  python custom_envs/scripts/navigation/navigate_mujoco.py \\
      --map maps/my_map.npz \\
      --goal 4.5 5.0 \\
      --policy_path /path/to/model_4999.pt \\
      --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \\
      --destination 8.0 6.0 \\
      --object banana
"""

import argparse
import enum
import math
import os
import sys
import time
import traceback

# 在任何 mujoco/glfw 被 import 之前强制指定 X11 后端
# 修复 Wayland 会话下 MuJoCo passive viewer 鼠标/键盘完全无响应的问题
# PYGLFW_LIBRARY_VARIANT=x11 强制加载 glfw/x11/libglfw.so（而非 wayland 版本）
os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')
os.environ.setdefault('MUJOCO_GL', 'glx')
os.environ.setdefault('DISPLAY', ':0')

import numpy as np
import torch

# ── 路径设置 ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "utils"))


# ── 状态机定义（和 Isaac Lab 版完全一致）──
class PipelineState(enum.Enum):
    NAV          = "NAV"
    ALIGN_YAW_1  = "ALIGN_YAW_1"
    PAN_VX       = "PAN_VX"
    PAN_VY       = "PAN_VY"
    PAN_VX_FINAL = "PAN_VX_FINAL"
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
    PAN_NEG_X    = "PAN_NEG_X"
    NAV2_DEST    = "NAV2_DEST"
    ALIGN_YAW_2  = "ALIGN_YAW_2"
    PAN_DES_X    = "PAN_DES_X"
    ROTATE       = "ROTATE"
    PUT_DOWN     = "PUT_DOWN"
    DONE         = "DONE"


# ── 机械臂常量（和 Isaac Lab 版一致）──
ARM_JOINT_NAMES     = ["joint1","joint2","joint3","joint4","joint5","joint6"]
GRIPPER_JOINT_NAMES = ["joint7","joint8"]
GRIPPER_OPEN_POS    = np.array([ 0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE_POS   = np.array([ 0.000,  0.000], dtype=np.float32)
ARM_INIT_ANGLES     = np.array([0.0,  0.5, -1.0, 0.0,  0.5, 0.0], dtype=np.float32)  # = ARM_HOME_ANGLES in navigate_to_goal_nav2_whole.py
ARM_SIDE_ANGLES     = np.array([-math.pi/2, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float32)  # matches navigate_to_goal_nav2_whole.py
# PRE_ADJUST 目标姿态：j1=-π/2, j5 从 1.2->0（为 GraspNet IK 做准备）
ARM_PREGRASP_ANGLES = np.array([-math.pi/2, 1.5, -1.5, 0.0, 0.0, 0.0], dtype=np.float32)

# BUDGET（每个状态最大步数）
BUDGET = {
    PipelineState.ARM_INIT:   300,
    PipelineState.PRE_GRASP:  250,
    PipelineState.ORIENT:     100,
    PipelineState.REACH:      300,
    PipelineState.CLOSE:      150,
    PipelineState.LIFT:       200,
    PipelineState.ALIGN_YAW_2: 200,
    PipelineState.ROTATE:     200,
    PipelineState.PUT_DOWN:   100,
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


def load_policy(policy_path: str, device: str = "cuda"):
    """用 OnPolicyRunner + MockVecEnv 加载 rsl_rl checkpoint。

    从与 policy_path 同目录的 params/agent.yaml 读取网络配置，
    自动从 checkpoint 的 state_dict 推断 obs/action 维度，
    通过 OnPolicyRunner 官方 API 构建网络并加载权重。
    以后更换网络只需换 checkpoint 和 params/agent.yaml，无需改此函数。

    返回 act_inference callable：obs_tensor -> action_tensor
    """
    import copy
    import os

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


def main():
    parser = argparse.ArgumentParser("Navigate MuJoCo")
    parser.add_argument("--map",            required=True)
    parser.add_argument("--goal",           nargs=2, type=float, required=True)
    parser.add_argument("--policy_path",    required=True, help="model_4999.pt 路径")
    parser.add_argument("--grasp_checkpoint", default=None)
    parser.add_argument("--grasp_topk",     type=int, default=1)
    parser.add_argument("--object",          default="banana",
                        choices=["banana", "apple", "bowl"],
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

    # ── IK 求解器：arm_ik 为纯函数模块，无需实例化 ──
    # solve_for_gripper_base / compute_desired_ee_rot_in_arm 等在状态机内懒导入

    # ── 启动 ROS2 bridge ──
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge
    bridge = IsaacROS2Bridge(cmd_vel_timeout=0.5)
    bridge.start(timeout=30.0)
    print("[MJ] ROS2 bridge started")

    # ── 发送导航目标 ──
    bridge.send_goal(args.goal[0], args.goal[1])

    # ── grasp worker 改为 GRASP_PLAN 状态时同步调用，此处仅保留 proc 占位 ──
    grasp_proc = None  # 不再预启动；GRASP_PLAN 状态用 subprocess.run()
    grasp_result_q = None  # 保留变量兼容 finally 块

    # ── 状态机变量初始化 ──
    state          = PipelineState.NAV
    state_step     = 0
    last_action    = np.zeros(16, dtype=np.float32)
    _frozen_pos    = None   # FREEZE 时保存的根节点位置
    _frozen_quat   = None   # FREEZE 时保存的根节点四元数
    grasp_result   = None   # grasp_worker 返回的抓取结果 dict
    depth_accum    = []     # SCAN 阶段累积的 depth 帧列表
    scan_rgb       = None   # SCAN 阶段保存的 RGB 帧
    target_angles_arm = None  # PRE_GRASP/REACH/CLOSE/LIFT 阶段的机械臂目标角
    j6_target      = None   # ORIENT 阶段的 j6 目标角
    _orient_q_fixed = None  # ORIENT 阶段固定的 j1~j5 角度（来自 IK）
    _orient_j6_start = None
    _q_scan_at_scan  = None  # SCAN 完成时的机械臂关节角（供 IK 用）
    _pos_w_at_scan   = None  # SCAN 完成时的机器人世界位置
    _quat_w_at_scan  = None  # SCAN 完成时的机器人四元数
    _pan_neg_x_goal  = None  # PAN_NEG_X 目标坐标
    _dest_goal_sent  = False # NAV2_DEST 是否已发送 goal
    _rotate_j_target = None  # ROTATE 目标关节角
    _rotate_j1_start = None
    _rotate_j2_start = None
    _rotate_j3_start = None
    _pg_cmd          = None  # PRE_GRASP 斜坡指令（按 step 累积）
    _pg_target_cur   = None  # PRE_GRASP IK 目标关节角（供斜坡使用）
    _re_target_cur   = None  # REACH 目标关节角（固定不变）

    # ── arm 关节在 jpos(24维) 中的索引（ALL_JOINT_NAMES 顺序）──
    # joint1=4, joint2=9, joint3=14, joint4=19, joint5=20, joint6=21
    _ARM_IDX = np.array([4, 9, 14, 19, 20, 21], dtype=np.int32)

    # PRE_GRASP 每步最大增量（与原版 navigate_to_goal_nav2_whole.py 一致）
    _PG_MAX_DELTA = np.array([0.03, 0.05, 0.05, 0.04, 0.04, 0.04], dtype=np.float64)

    # ── 辅助函数 ──
    def _wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _yaw_err_to(target_yaw, current_yaw):
        return _wrap_angle(target_yaw - current_yaw)

    def _arm_step(q6):
        """设置机械臂目标角（6维）。"""
        env.set_arm_target(np.asarray(q6, dtype=np.float64))

    def _gripper_step(close=False):
        tgt = GRIPPER_CLOSE_POS if close else GRIPPER_OPEN_POS
        env.set_gripper_target(tgt.astype(np.float64))

    def _get_arm_q(jpos):
        """从 24 维 joint_pos 中取出 arm 6 关节角。"""
        return jpos[_ARM_IDX].copy()

    def _alpha(s_state, step):
        b = BUDGET.get(s_state, 1)
        return min(step / max(b, 1), 1.0)

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

    def _dls_gb_step(q_start, gb_cmd_arm, n_iter=6, lam=1e-4):
        """局部关节增量：数值雅可比 + 阻尼最小二乘(DLS)，把 gripper_base 从
        fk(q_start) 挪到 gb_cmd_arm。纯平移、不重建朝向 → 与 IK 种子/滚动角无关。

        为什么不用全局 IK：带朝向的 solve_for_gripper_base 在 REACH 位形附近对
        目标极敏感（同一个位置目标，实机那次能解出、离线却几乎全被拒绝——实测
        含管线同款种子在内 23 个种子 0 成功，见 /tmp/probe28.py、probe30.py）；
        而"当前位置附近走一个小平移"只需各关节动 2° 量级（同 probe28 [3] 实测），
        用数值雅可比 DLS 即可，不依赖种子也不会跳到另一分支。

        j6(索引 5)：绕 gripper_base 自身 +Z 旋转、不改变其位置（见 REACH j6 钉死
        注释），故不进雅可比、保持 q_start 原值不动。
        返回 (q, 位置残差 m, 最大单关节增量 rad)。"""
        from arm_ik_mujoco import fk_gripper as _fk_dls, _IK_JOINT_LIMITS as _lim_dls
        _lo = np.array([lo for lo, _ in _lim_dls], dtype=np.float64)
        _hi = np.array([hi for _, hi in _lim_dls], dtype=np.float64)
        q   = np.clip(np.asarray(q_start, dtype=np.float64).copy(), _lo, _hi)
        q0  = q.copy()
        tgt = np.asarray(gb_cmd_arm, dtype=np.float64)
        eps = 1e-6
        for _ in range(n_iter):
            p   = _fk_dls(q)[:3, 3]
            err = tgt - p
            if float(np.linalg.norm(err)) < 5e-4:      # 0.5mm 以内即收敛
                break
            J = np.zeros((3, 5))
            for i in range(5):                          # j1..j5；j6 不影响 gb 位置
                qq = q.copy()
                qq[i] += eps
                J[:, i] = (_fk_dls(qq)[:3, 3] - p) / eps
            q[:5] = np.clip(q[:5] + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err),
                            _lo[:5], _hi[:5])
        return (q,
                float(np.linalg.norm(_fk_dls(q)[:3, 3] - tgt)),
                float(np.abs(q - q0).max()))

    # ── 主循环 ──
    print("[MJ] Starting main loop, state=", state)
    try:
        while state != PipelineState.DONE:
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

            # ── 从 bridge 获取 cmd_vel ──
            cmd_vx, cmd_wz = bridge.get_cmd_vel()
            cmd_vy = 0.0

            # ── 状态机：速度指令覆盖 ──
            if state == PipelineState.NAV:
                # 导航阶段保持 ARM_HOME_ANGLES（收起但不到侧面）
                _arm_step(ARM_INIT_ANGLES)
                nav_done, nav_failed = bridge.get_nav_status()
                if nav_done:
                    state = PipelineState.ALIGN_YAW_1
                    state_step = 0
                    print("[MJ] NAV done -> ALIGN_YAW_1")

            elif state == PipelineState.ALIGN_YAW_1:
                # 对齐到 +Y 方向（yaw = +π/2），面向桌子
                target_yaw = math.pi / 2
                err = _yaw_err_to(target_yaw, yaw)
                cmd_vx = 0.0; cmd_vy = 0.0
                cmd_wz = float(np.clip(2.0 * err, -1.2, 1.2))
                if abs(err) < 0.05:
                    state = PipelineState.PAN_VY
                    state_step = 0
                    print("[MJ] ALIGN_YAW_1 done -> PAN_VX")

            elif state == PipelineState.PAN_VX:
                # PD 控制：精调机器人 Y 方向位置（朝向 -Y 时 vx 对应世界 -Y 运动，需取反）
                _dy_w = float(args.goal[1]) - float(pos_w[1])
                _dy_err = abs(_dy_w)
                if state_step == 1:
                    print(f"[SM] PAN_VX start dy_err={_dy_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VX step {state_step}: dy_err={_dy_err:.3f}m", flush=True)
                _v_cap = float(np.clip(0.4 * _dy_err, 0.0, 0.3))
                if _v_cap < 0.10:  # 死区：速度过小时停止，避免 policy 步态不稳
                    _v_cap = 0.0
                # 面向+Y时 cmd_vx>0 => 世界y增大，同向
                cmd_vx = float(np.clip(math.copysign(_v_cap, _dy_w), -0.3, 0.15))
                cmd_vy = 0.0; cmd_wz = 0.0
                state_step += 1
                if _dy_err < 0.10 or state_step >= 400:
                    print("[SM] PAN_VX done -> PAN_VY", flush=True)
                    state = PipelineState.PAN_VY
                    state_step = 0

            elif state == PipelineState.PAN_VY:
                # PD 控制：精调机器人 X 方向位置（朝向 -Y 时 vy>0 => 世界+X，故同向）
                # X 目标在 goal_x 基础上偏移 +0.5m（让机器人站在桌子侧面合适位置）
                _dx_w = float(args.goal[0]) + 0.5 - float(pos_w[0])
                _dx_err = abs(_dx_w)
                if state_step == 1:
                    print(f"[SM] PAN_VY start dx_err={_dx_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VY step {state_step}: dx_err={_dx_err:.3f}m", flush=True)
                _v_cap = float(np.clip(2.0 * _dx_err, 0.0, 0.3))
                if _v_cap < 0.10:  # 死区：速度过小时停止，避免 policy 步态不稳
                    _v_cap = 0.0
                cmd_vx = 0.0
                # 面向+Y时 cmd_vy>0 => 机器人左移 => 世界x减小，故取反
                cmd_vy = float(np.clip(-math.copysign(_v_cap, _dx_w), -0.3, 0.15))
                cmd_wz = 0.0
                state_step += 1
                if _dx_err < 0.12 or state_step >= 400:
                    # ── 临时方案（2026-09-16）：跳过 PAN_VX_FINAL / ALIGN_YAW ──
                    # 运控小速度指令抖动问题解决前，不再用速度指令做末段精调：
                    #   x 保持 PAN_VY 结束时的值（被桌子挡住，已达物理可达位置）；
                    #   y 直接瞬移到目标 goal_y；朝向一步对齐到 yaw=+π/2（直立）。
                    # 注意：瞬移不符合物理实际，仅为绕过运控问题；修复运控后删除本块，
                    # 恢复原流程 "PAN_VY -> PAN_VX_FINAL -> ALIGN_YAW -> ARM_INIT"。
                    cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                    _snap_pos = pos_w.copy()
                    _snap_pos[1] = float(args.goal[1])        # y -> 目标位置（x/z 保持不变）
                    _snap_quat = np.array(
                        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
                        dtype=np.float64)                     # yaw=+π/2 直立四元数 (w,x,y,z)
                    env.set_root_pose(_snap_pos, _snap_quat)  # 瞬移 base + 清零速度
                    _frozen_pos  = _snap_pos.copy()           # 补上 ALIGN_YAW 原本保存的 FREEZE 锚点
                    _frozen_quat = _snap_quat.copy()
                    print(f"[SM] PAN_VY done -> SNAP {np.round(pos_w,3)} -> "
                          f"{np.round(_snap_pos,3)} -> ARM_INIT", flush=True)
                    state = PipelineState.ARM_INIT
                    state_step = 0

            elif state == PipelineState.PAN_VX_FINAL:
                # PAN_VY 之后的 Y 方向补正，消除侧移耦合引入的 Y 误差
                _dy_w = float(args.goal[1]) - float(pos_w[1])
                _dy_err = abs(_dy_w)
                if state_step == 1:
                    print(f"[SM] PAN_VX_FINAL start dy_err={_dy_err:.3f}m", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_VX_FINAL step {state_step}: dy_err={_dy_err:.3f}m", flush=True)
                _v_cap = float(np.clip(0.4 * _dy_err, 0.0, 0.3))
                if _v_cap < 0.10:  # 死区：速度过小时停止，避免 policy 步态不稳
                    _v_cap = 0.1
                cmd_vx = float(np.clip(math.copysign(_v_cap, _dy_w), -0.3, 0.15))
                cmd_vy = 0.0; cmd_wz = 0.0
                state_step += 1
                if _dy_err < 0.10 or state_step >= 200:
                    print("[SM] PAN_VX_FINAL done -> ALIGN_YAW", flush=True)
                    state = PipelineState.ALIGN_YAW
                    state_step = 0

            elif state == PipelineState.ALIGN_YAW:
                # 对齐到 +Y 方向（yaw = +π/2），面向桌子，然后进入 ARM_INIT
                _yaw_err_align = _yaw_err_to(math.pi / 2, yaw)
                if state_step == 1:
                    print(f"[SM] ALIGN_YAW start: cur={math.degrees(yaw):.1f}deg", flush=True)
                if state_step % 50 == 0:
                    print(f"[SM] ALIGN_YAW step {state_step}: "
                          f"err={math.degrees(_yaw_err_align):.1f}deg", flush=True)
                cmd_vx = 0.0; cmd_vy = 0.0
                cmd_wz = float(np.clip(100.0 * _yaw_err_align, -1.2, 1.2))
                state_step += 1
                if abs(_yaw_err_align) < 0.02 or state_step >= 600:
                    print("[SM] ALIGN_YAW done -> ARM_INIT", flush=True)
                    # 保存 FREEZE 锚点
                    _frozen_pos  = pos_w.copy()
                    _frozen_quat = quat_w.copy()
                    state = PipelineState.ARM_INIT
                    state_step = 0

            elif state == PipelineState.ARM_INIT:
                # 机械臂插值到 ARM_SIDE_ANGLES，同时 FREEZE 根节点位姿
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
                        state = PipelineState.SCAN
                    else:
                        state = PipelineState.DONE
                    state_step = 0
                    depth_accum = []
                    scan_rgb    = None

            elif state == PipelineState.SCAN:
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
                        # ── 保存 scan_raw.png：SCAN 阶段采集到的原始 RGB 帧 ──
                        try:
                            import cv2 as _cv2_sr
                            _sr_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                            os.makedirs(_sr_dir, exist_ok=True)
                            if scan_rgb is not None:
                                _sr_bgr = _cv2_sr.cvtColor(scan_rgb, _cv2_sr.COLOR_RGB2BGR)
                                _sr_path = os.path.join(_sr_dir, "scan_raw.png")
                                _cv2_sr.imwrite(_sr_path, _sr_bgr)
                                print(f"[SM] scan_raw.png 已保存: {_sr_path}", flush=True)
                        except Exception as _sr_e:
                            print(f"[SM] scan_raw.png 保存失败: {_sr_e}", flush=True)
                        state = PipelineState.GRASP_PLAN
                        state_step = 0
                except ValueError as _scan_e:
                    print(f"[SM] SCAN camera unavailable ({_scan_e}) -- DONE.",
                          flush=True)
                    state = PipelineState.DONE
                    state_step = 0

            elif state == PipelineState.PRE_GRASP:
                # IK 求解 + 斜坡移动机械臂到 pre-grasp 姿态（对齐原版逻辑）
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None:
                    state = PipelineState.DONE
                else:
                    if _pg_cmd is None:
                        # state_step==0 时一次性求解 IK（对应原版 state_step==1 块）
                        try:
                            from arm_ik_mujoco import (
                                cam_to_world,
                                world_pos_to_arm_frame,
                                solve_for_gripper_base as _sfgb,
                                compute_desired_ee_rot_in_arm as _cder,
                                quat_to_rot as _q2r_pg,
                                fk_gripper as _fk_pg,
                                _CAM_OFFSET_ROT as _COR_pg,
                            )
                            # ── 世界坐标 & arm 系目标点 ──
                            _t_w = cam_to_world(
                                grasp_result["t_cam"], grasp_result["q_scan"],
                                grasp_result["pos_w_scan"], grasp_result["quat_w_scan"])
                            _pre_t_a = world_pos_to_arm_frame(
                                _t_w,
                                grasp_result["pos_w_scan"],
                                grasp_result["quat_w_scan"],
                            )  # 用 SCAN 时的位姿，与 cam_to_world 完全自洽
                            # ── 目标旋转（arm 系，供 IK & ORIENT 使用）──
                            _R_desired_EE = _cder(grasp_result["R_cam"],
                                                  grasp_result["q_scan"])
                            grasp_result["R_desired_EE_in_arm"] = _R_desired_EE
                            # ── approach 向量（R_cam[:,0] 转到 arm 系）──
                            # 对应原版: _R_rob @ _T_fk[:3,:3] @ _CAM_OFFSET_ROT @ R_cam[:,0]
                            # 再转到 arm 系: _R_rob.T @ (...)  = _T_fk[:3,:3] @ _CAM_OFFSET_ROT @ R_cam[:,0]
                            _R_rob_pg  = _q2r_pg(quat_w)
                            _T_fk_pg   = _fk_pg(grasp_result["q_scan"])
                            _R_cw_pg   = _R_rob_pg @ _T_fk_pg[:3, :3] @ _COR_pg
                            _approach_arm_pg = _R_rob_pg.T @ (_R_cw_pg @ grasp_result["R_cam"][:, 0])
                            # ── pre-grasp 目标：沿 approach 退到指尖落在抓取点 ──
                            # AnyGrasp 约定（USAGE.md Note 1）：translation 是**掌心**，
                            #   指尖 = 掌心 + depth·a（depth = 该抓取的手指长度，逐候选不同）。
                            # 目标链: 指尖 = 掌心 + d·a + s（s = REACH_ADVANCE − 0.1358
                            #   = 显式下探量，见 REACH 块；指尖越过抓取中心 s 才夹得深）
                            #   REACH gb = 指尖 − 0.1358·a = 掌心 − (0.1358 − d + s)·a
                            #   PRE   gb = REACH gb − REACH_ADVANCE·a = 掌心 − (0.2458 − d)·a
                            # 旧代码用固定 0.18 = 0.2458 − 0.0658（隐含 d=0.0658），
                            # 对苹果侥幸成立，对香蕉/碗导致指尖停在物体上方 → 抓空。
                            _d_g = float(grasp_result.get("depth", 0.0658))
                            if not (0.0 <= _d_g <= 0.09):
                                print(f"[WARN] PRE_GRASP: grasp depth={_d_g:.4f} 越界, clamp 到 [0,0.09]",
                                      flush=True)
                                _d_g = float(np.clip(_d_g, 0.0, 0.09))
                            _pre_t_gb  = _pre_t_a - (0.2458 - _d_g) * _approach_arm_pg
                            print(f"[DIAG] _pre_t_a(arm)={np.round(_pre_t_a,4)} _pre_t_gb(arm)={np.round(_pre_t_gb,4)} approach_arm={np.round(_approach_arm_pg,4)} d={_d_g:.4f}", flush=True)
                            # ── IK 求解 ──
                            _pgq = _sfgb(_pre_t_gb,
                                         target_rot_j7=_R_desired_EE,
                                         initial_angles=grasp_result["q_scan"])
                            _pgok = (_pgq is not None and
                                     not np.any(np.isnan(_pgq)))
                            if not _pgok:
                                # ── IK 失败：尝试下一个候选（对齐原版逻辑）──
                                _cand_list = grasp_result.get("ranked_candidate_idxs", [])
                                _tried     = grasp_result.get("tried_set", set())
                                _next_ci   = next(
                                    (i for i in _cand_list if i not in _tried), None)
                                if _next_ci is None:
                                    print("[WARN] PRE_GRASP: 全部候选 IK 失败 -> ARM_INIT.",
                                          flush=True)
                                    grasp_result = None
                                    depth_accum.clear()
                                    scan_rgb = None
                                    _pg_cmd = None
                                    state = PipelineState.ARM_INIT
                                    state_step = 0
                                else:
                                    _tried.add(_next_ci)
                                    grasp_result["tried_set"] = _tried
                                    grasp_result["t_cam"]     = grasp_result["gr_translations"][_next_ci]
                                    # GraspNet 原始输出即光学系（x右 y下 z前），直接使用
                                    grasp_result["R_cam"]     = grasp_result["gr_rotations"][_next_ci].copy()
                                    grasp_result["score"]     = float(grasp_result["gr_scores"][_next_ci])
                                    grasp_result["width"]     = float(grasp_result["gr_widths"][_next_ci])
                                    grasp_result["depth"]     = float(grasp_result["gr_depths"][_next_ci])
                                    # 重新计算 R_desired_EE_in_arm（对齐原版逻辑）
                                    try:
                                        from arm_ik_mujoco import compute_desired_ee_rot_in_arm as _cder_pg
                                        grasp_result["R_desired_EE_in_arm"] = _cder_pg(
                                            grasp_result["gr_rotations"][_next_ci],
                                            grasp_result["q_scan"])
                                    except Exception as _cde:
                                        print(f"[SM] R_desired_EE_in_arm 重计算失败: {_cde}", flush=True)
                                    print(f"[SM] PRE_GRASP IK failed -> 候选[{_next_ci}] "
                                          f"score={grasp_result['score']:.3f}", flush=True)
                                    state_step = 0
                                    # _pg_cmd 保持 None，下次循环重新求 IK
                            else:
                                _pg_target_cur = np.asarray(_pgq, dtype=np.float64).copy()
                                # _pg_cmd 从当前实际关节角出发（原版 L1248: _pg_cmd = cur_q.copy()）
                                _pg_cmd = _get_arm_q(jpos).copy()
                                # 保存供 ORIENT / REACH 使用
                                grasp_result["target_angles_pre"] = _pg_target_cur.copy()
                                grasp_result["pre_t_gb_arm"]      = _pre_t_gb.copy()
                                grasp_result["approach_arm"]       = _approach_arm_pg.copy()
                                # 计算 gripper_base 目标的世界坐标（供三门检验用）
                                try:
                                    from arm_ik_mujoco import quat_to_rot as _q2r_pgf
                                    _R_rob_pgf = _q2r_pgf(quat_w)
                                    _arm_base_w_pgf = pos_w + _R_rob_pgf @ np.array([0., 0., 0.0888])
                                    _pg_t_gb_world_fixed = _R_rob_pgf @ _pre_t_gb + _arm_base_w_pgf
                                    grasp_result["_pg_t_gb_world_fixed"] = _pg_t_gb_world_fixed.copy()
                                    _apw_pg    = _R_rob_pgf @ _approach_arm_pg
                                    _ctr_w_pg  = _R_rob_pgf @ _pre_t_a + _arm_base_w_pgf
                                    _j7_w_pg   = _pg_t_gb_world_fixed + 0.1358 * _apw_pg
                                    # AnyGrasp 的 translation 是掌心（指尖=掌心+depth·a）：
                                    # 真实抓取中心 = 掌心 + depth·a，d=0 时两者重合
                                    _gc_w_pg   = _ctr_w_pg + _d_g * _apw_pg
                                    print(f"[DIAG] PRE palm_world(掌心)    = {np.round(_ctr_w_pg,4)}", flush=True)
                                    print(f"[DIAG] PRE grasp_center_world  = {np.round(_gc_w_pg,4)}  (掌心+{_d_g:.4f}·a)", flush=True)
                                    print(f"[DIAG] PRE approach_world      = {np.round(_apw_pg,4)}  ← 应指向物体", flush=True)
                                    print(f"[DIAG] PRE gb_target_world     = {np.round(_pg_t_gb_world_fixed,4)}", flush=True)
                                    print(f"[DIAG] PRE j7_target_world     = {np.round(_j7_w_pg,4)}", flush=True)
                                    try:
                                        _real_obj_pg = env.get_object_pos(args.object)
                                        print(f"[DIAG] PRE object_world        = {np.round(_real_obj_pg,4)}", flush=True)
                                        _dj7 = _j7_w_pg - _real_obj_pg
                                        print(f"[DIAG] PRE j7-obj diff         = {np.round(_dj7,4)}  (j7 vs 物体中心)", flush=True)
                                        print(f"[DIAG] PRE_GRASP pre-grasp vs object: "
                                              f"dX={_pg_t_gb_world_fixed[0]-_real_obj_pg[0]:+.4f}m "
                                              f"dY={_pg_t_gb_world_fixed[1]-_real_obj_pg[1]:+.4f}m "
                                              f"dZ={_pg_t_gb_world_fixed[2]-_real_obj_pg[2]:+.4f}m",
                                              flush=True)
                                    except Exception:
                                        pass
                                except Exception as _pgfe:
                                    print(f"[SM] _pg_t_gb_world_fixed 计算失败: {_pgfe}", flush=True)
                                print(f"[SM] PRE_GRASP IK={np.round(_pg_target_cur,3)}",
                                      flush=True)
                                # ── 保存 choice.png：在 scan_rgb 上标注最终抓取点和 closing 轴 ──
                                try:
                                    import cv2 as _cv2_ch
                                    if scan_rgb is not None:
                                        _ch_img = _cv2_ch.cvtColor(scan_rgb, _cv2_ch.COLOR_RGB2BGR).copy()
                                    else:
                                        _ch_img = np.zeros((480, 640, 3), dtype=np.uint8)
                                    _H_ch, _W_ch = _ch_img.shape[:2]
                                    _fx_ch, _fy_ch = 616.0, 616.0
                                    _cx_ch, _cy_ch = _W_ch / 2.0, _H_ch / 2.0
                                    _t_sel = grasp_result["t_cam"]   # 相机系抓取点 [X,Y,Z]
                                    _R_sel = grasp_result["R_cam"]   # 3×3，[:,1]=closing轴
                                    # 抓取点投影到像素
                                    _u0 = int(_t_sel[0] / _t_sel[2] * _fx_ch + _cx_ch)
                                    _v0 = int(_t_sel[1] / _t_sel[2] * _fy_ch + _cy_ch)
                                    # closing 轴端点（沿 closing 轴延伸 0.05m）
                                    _closing_end = _t_sel + _R_sel[:, 1] * 0.05
                                    _u1 = int(_closing_end[0] / _closing_end[2] * _fx_ch + _cx_ch)
                                    _v1 = int(_closing_end[1] / _closing_end[2] * _fy_ch + _cy_ch)
                                    # 红点（抓取点）+ 绿色箭头（closing 轴方向）
                                    _cv2_ch.circle(_ch_img, (_u0, _v0), 6, (0, 0, 255), -1)
                                    _cv2_ch.arrowedLine(_ch_img, (_u0, _v0), (_u1, _v1),
                                                        (0, 255, 0), 2, tipLength=0.3)
                                    _ch_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                                    os.makedirs(_ch_dir, exist_ok=True)
                                    _choice_path = os.path.join(_ch_dir, "choice.png")
                                    _cv2_ch.imwrite(_choice_path, _ch_img)
                                    print(f"[SM] choice.png 已保存: {_choice_path}  "
                                          f"grasp_pt=({_u0},{_v0}) closing=({_u1},{_v1})",
                                          flush=True)
                                except Exception as _che:
                                    print(f"[SM] choice.png 保存失败: {_che}", flush=True)
                        except Exception as _ike:
                            print(f"[SM] PRE_GRASP IK err: {_ike} — DONE.", flush=True)
                            state = PipelineState.DONE
                    if _pg_cmd is not None and state == PipelineState.PRE_GRASP:
                        cur_q = _get_arm_q(jpos)
                        # 原版: _pg_cmd 滚动 += clip(target - cmd)，每步小步积分
                        for _ji in range(6):
                            _pg_cmd[_ji] += float(np.clip(
                                _pg_target_cur[_ji] - _pg_cmd[_ji],
                                -_PG_MAX_DELTA[_ji], +_PG_MAX_DELTA[_ji]))
                        # 关节限位保护（原版 L1256）
                        from arm_ik_mujoco import _IK_JOINT_LIMITS as _pg_lims
                        _pg_cmd = np.clip(_pg_cmd,
                                          [lo for lo, hi in _pg_lims],
                                          [hi for lo, hi in _pg_lims])
                        _arm_step(_pg_cmd)
                        _gripper_step(close=False)
                        _pg_err_check = np.abs(cur_q - _pg_target_cur)
                        if state_step % 50 == 0:
                            print(f"[SM] PRE_GRASP step {state_step}: "
                                  f"max_err={_pg_err_check.max():.4f}", flush=True)
                        state_step += 1
                        _pg_early_exit = (state_step > 150 and
                                         _pg_err_check.max() < 0.05)
                        if _pg_early_exit or state_step >= BUDGET[PipelineState.PRE_GRASP]:
                            # ── 三门收敛检验（对齐原版 L1278-1311）──
                            _PRE_GRASP_MAX_WORLD_ERR = 0.10   # 10 cm
                            _PRE_GRASP_MAX_JOINT_ERR = 0.35   # rad
                            _PRE_GRASP_MAX_ROT_ERR   = 2.90   # Frobenius
                            _pg_can_advance = True
                            try:
                                from arm_ik_mujoco import (
                                    fk as _fk_pgd, fk_gripper as _fkg_pgd,
                                    quat_to_rot as _q2r_pgd, _RX_NEG90 as _RX_NEG90_pgd,
                                )
                                _cur_q_final = _get_arm_q(jpos)
                                _T_final_gb  = _fkg_pgd(_cur_q_final)
                                _T_final_j7  = _fk_pgd(_cur_q_final)
                                _ee_final_arm = _T_final_gb[:3, 3]
                                _R_final_gb   = _T_final_j7[:3, :3] @ _RX_NEG90_pgd
                                _R_rob_pgd = _q2r_pgd(quat_w)
                                _arm_base_off_pgd = _R_rob_pgd @ np.array([0., 0., 0.0888])
                                _ee_final_world = _R_rob_pgd @ _ee_final_arm + pos_w + _arm_base_off_pgd
                                _pg_tgbw = grasp_result.get("_pg_t_gb_world_fixed", None)
                                _pos_err_final = float(np.linalg.norm(
                                    _ee_final_world - _pg_tgbw)) if _pg_tgbw is not None else 0.0
                                _joint_err_final = np.abs(_cur_q_final - _pg_target_cur)
                                _R_desired_gb = grasp_result.get("R_desired_EE_in_arm", None)
                                if _R_desired_gb is not None:
                                    _R_des_gb_rx = _R_desired_gb @ _RX_NEG90_pgd
                                    _rot_err_final = float(np.linalg.norm(
                                        _R_final_gb - _R_des_gb_rx, ord='fro'))
                                else:
                                    _rot_err_final = 0.0
                                print(f"[DIAG] PRE_GRASP pos_err={_pos_err_final*100:.1f}cm "
                                      f"joint_err_max={_joint_err_final.max()*57.3:.1f}deg "
                                      f"rot_err={_rot_err_final:.4f}", flush=True)
                                if _pg_tgbw is not None:
                                    _pg_can_advance = (
                                        _pos_err_final <= _PRE_GRASP_MAX_WORLD_ERR
                                        and _joint_err_final.max() <= _PRE_GRASP_MAX_JOINT_ERR
                                        and _rot_err_final <= _PRE_GRASP_MAX_ROT_ERR
                                    )
                            except Exception as _pgde:
                                print(f"[SM] 三门检验异常: {_pgde}", flush=True)
                            if not _pg_can_advance:
                                print(f"[WARN] PRE_GRASP 三门检验未通过 -> ARM_INIT", flush=True)
                                grasp_result = None
                                depth_accum.clear()
                                scan_rgb = None
                                _pg_cmd = None
                                state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                print(f"[SM] PRE_GRASP done (step={state_step}) -> ORIENT",
                                      flush=True)
                                state = PipelineState.ORIENT
                                state_step = 0

            elif state == PipelineState.ORIENT:
                # 仅旋转 joint6 对准 GraspNet closing 轴（对齐原版逻辑）
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None or "R_desired_EE_in_arm" not in grasp_result:
                    state = PipelineState.DONE
                else:
                    if state_step == 1:
                        # 对应原版 state_step==1 块：一次性计算 j6 目标
                        from arm_ik_mujoco import (
                            extract_j6_angle,
                            _RX_NEG90 as _RXN90_or,
                            fk_gripper as _fkg_or,
                        )
                        cur_q = _get_arm_q(jpos)
                        # gripper_base 目标旋转 = R_desired_EE_in_arm @ _RX_NEG90
                        _R_gb_desired_or = (grasp_result["R_desired_EE_in_arm"]
                                            @ _RXN90_or)
                        j6_target = extract_j6_angle(cur_q, _R_gb_desired_or)
                        # _orient_q_fixed = PRE_GRASP IK 结果角（非当前实际角）
                        _orient_q_fixed  = grasp_result["target_angles_pre"].copy()
                        _orient_j6_start = float(cur_q[5])
                        print(f"[SM] ORIENT j6_target={j6_target:.4f} rad "
                              f"({math.degrees(j6_target):.1f} deg), "
                              f"j6_start={_orient_j6_start:.4f} rad",
                              flush=True)
                    # step==0: _orient_q_fixed 尚未赋值，先保持 PRE_GRASP 末尾姿态不动
                    cur_q = _get_arm_q(jpos)
                    if _orient_q_fixed is None:
                        # 第0步：发当前角，等待 step==1 时初始化
                        _arm_step(cur_q.copy())
                    else:
                        # step>=1: 固定 j1~j5，直接发 j6_target
                        _q_cmd_or = _orient_q_fixed.copy()
                        _q_cmd_or[5] = j6_target
                        _arm_step(_q_cmd_or)
                    _gripper_step(close=False)
                    if state_step % 50 == 0:
                        j6_err = abs(cur_q[5] - j6_target) if j6_target is not None else float('nan')
                        print(f"[SM] ORIENT step {state_step}/{BUDGET[PipelineState.ORIENT]}: "
                              f"j6_err={j6_err:.4f} rad", flush=True)
                    state_step += 1
                    if state_step >= BUDGET[PipelineState.ORIENT]:
                        _cur_q_orient_done = _get_arm_q(jpos).copy()
                        grasp_result["target_angles_orient"] = _cur_q_orient_done
                        print(f"[SM] ORIENT done: j6={_cur_q_orient_done[5]:.4f} "
                              f"(target={j6_target:.4f}) -> REACH", flush=True)
                        state = PipelineState.REACH
                        state_step = 0
                        target_angles_arm = None  # 强制 REACH 重算 IK

            elif state == PipelineState.REACH:
                # 沿 approach 轴前进 0.06m 到接触点（对齐原版逻辑）
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None or "pre_t_gb_arm" not in grasp_result:
                    state = PipelineState.DONE
                else:
                    if target_angles_arm is None:
                        # state_step==0 时一次性求解 IK（对应原版 state_step==1 块）
                        try:
                            from arm_ik_mujoco import (
                                solve_for_gripper_base as _sfgb_re,
                                _IK_JOINT_LIMITS as _re_lims,
                            )
                            # advance 0.11 → 0.125：0.11 时指尖恰好停在抓取中心 C 上
                            # （0.1358 − 0.1358 = 0），而指板只从指尖往掌心方向长 76.5mm，
                            # 抓取深度 = C 上方物体的高度 → 三物体系统性偏浅（苹果只碰到
                            # 顶、碗只到沿下 14mm）。多出的 0.015 是显式下探量 s：
                            # 指尖 = C + 0.015·a，向下多夹 15mm 材料；桌面 clip 照旧兜底。
                            REACH_ADVANCE   = 0.125
                            TABLE_CLEARANCE = 0.008   # 指尖/指板最低点距桌面的最小高度(m)
                            _re_pre_t_gb = grasp_result["pre_t_gb_arm"]
                            _re_approach = grasp_result["approach_arm"]
                            _re_t_gb_arm = (_re_pre_t_gb
                                            + REACH_ADVANCE * _re_approach)
                            # ── 闭环纠偏状态（REACH 结束时用 gb 实测误差反推命令，见 done 分支）──
                            _re_t_gb_arm0 = _re_t_gb_arm.copy()  # 原始(clip 后)命令目标，纠偏基准
                            _re_corr_iter = 0                    # 已纠偏次数
                            _re_corr_w    = np.zeros(3)          # 世界系累积修正量
                            _re_prev_e    = None                 # 上一次纠偏时的 |实测误差|（判断是否跟随）
                            _re_prev_q    = None                 # 上一次纠偏前的目标关节角（不跟随时的回退点）
                            _tc_h    = None                      # 桌面高度（世界 z）
                            _tc_need = None                      # 指尖最低允许高度（clip 量）
                            # ── 桌面 clip: 手不得插进桌面（沿 approach 自动收缩前进量）──
                            # 指尖目标 = 掌心 + (depth + s)·a：depth < 0.1358 时指尖会落到
                            # 掌心之下较浅处，若物体矮（苹果）则该目标可能低于桌面 → 手指压桌、
                            # 机器狗对抗、抓空。这里按桌面高度收缩 advance 兜底。
                            # 桌面高度取 GRASP_PLAN 阶段保存的 /tmp/table_cloud.npz（世界系，
                            # 白/灰颜色筛出的桌面点），用指尖 XY 邻域点 z 的**中位数**估计
                            # （邻域里可能混有物体表面点，max 会被抬高，中位数对物体/离群点鲁棒）；
                            # 邻域点不足时逐级放大半径；仍无数据或 approach 非向下则不 clip（fail-open）。
                            # 指尖不是最低点：指板沿 approach 长 76.5mm、内侧半宽 35.1mm，
                            # approach 倾斜时指板外角比指尖低 35.1mm×sin(倾角)，一并计入。
                            try:
                                from arm_ik_mujoco import quat_to_rot as _q2r_tc
                                _tc_R_rob  = _q2r_tc(grasp_result["quat_w_scan"])
                                _tc_apw    = _tc_R_rob @ _re_approach
                                _tc_base_w = (grasp_result["pos_w_scan"]
                                              + _tc_R_rob @ np.array([0., 0., 0.0888]))
                                # advance=0 时指尖（j7 原点）的世界位置
                                _tc_tip_w = (_tc_R_rob @ (_re_pre_t_gb + 0.1358 * _re_approach)
                                             + _tc_base_w)
                                if _tc_apw[2] < -0.05 and os.path.exists('/tmp/table_cloud.npz'):
                                    _tc_pts  = np.load('/tmp/table_cloud.npz')['points'].astype(np.float64)
                                    _tc_dxy  = np.linalg.norm(_tc_pts[:, :2] - _tc_tip_w[:2], axis=1)
                                    for _tc_r in (0.03, 0.05, 0.10):
                                        if int((_tc_dxy < _tc_r).sum()) >= 20:
                                            _tc_h = float(np.median(_tc_pts[_tc_dxy < _tc_r, 2]))
                                            break
                                if _tc_h is None:
                                    print("[WARN] 桌面 clip: 桌面点云不足或 approach 非向下 -> 不 clip",
                                          flush=True)
                                else:
                                    # 指尖需停在 need_eff 以上；指板外角再低 35.1mm×sin(倾角)
                                    _tc_need = (_tc_h + TABLE_CLEARANCE
                                                + 0.0351 * float(np.hypot(_tc_apw[0], _tc_apw[1])))
                                    _tc_dmax = ((_tc_tip_w[2] - _tc_need) / (-_tc_apw[2]))
                                    if 0.0 < _tc_dmax < REACH_ADVANCE:
                                        print(f"[SM] REACH clip: 桌面z={_tc_h:.4f} "
                                              f"指尖停在z>={_tc_need:.4f} advance "
                                              f"{REACH_ADVANCE:.3f}->{_tc_dmax:.3f}m", flush=True)
                                        _re_t_gb_arm = _re_pre_t_gb + _tc_dmax * _re_approach
                                    elif _tc_dmax <= 0.0:
                                        print(f"[WARN] REACH clip: PRE 指尖已在桌面下 "
                                              f"(tip_z={_tc_tip_w[2]:.4f} < {_tc_need:.4f})", flush=True)
                            except Exception as _tc_e:
                                print(f"[WARN] 桌面 clip 计算失败: {_tc_e}", flush=True)
                            _R_arm_re = grasp_result["R_desired_EE_in_arm"]
                            # IK 种子用 PRE_GRASP 角度（与原版一致）；j6 会在解完后
                            # 被钉回 ORIENT 结束值，不受求解器影响
                            _re_init_q = grasp_result["target_angles_pre"]
                            _rq = _sfgb_re(
                                _re_t_gb_arm,
                                target_rot_j7=_R_arm_re,
                                initial_angles=_re_init_q)
                            if _rq is None or np.any(np.isnan(_rq)):
                                print("[WARN] REACH IK None -> fallback to orient angles",
                                      flush=True)
                                _rq = _re_init_q.copy()
                            else:
                                # jump check 只检 j1-j5（j6 由下面的钉死逻辑统一处理）
                                _re_jump = np.abs(np.asarray(_rq) - _re_init_q)
                                if _re_jump[:5].max() > 0.5:
                                    print(f"[WARN] REACH IK jump too large "
                                          f"({np.degrees(_re_jump[:5].max()):.1f} deg) "
                                          f"-> fallback", flush=True)
                                    _rq = _re_init_q.copy()
                            # REACH 只沿 approach 平移，j6（绕 gripper_base 自身 +Z 的
                            # 旋转）必须保持 ORIENT 结束时的值：IK 的 roll 搜索会按
                            # "离种子的关节距离"把 j6 滚回 PRE_GRASP 的旧值，fallback
                            # 解则完全不看 j6。这里对以上所有路径统一钉死 j6。
                            # 依据：j6 绕 gripper_base 自身 +Z 旋转、不改变其位置
                            # （extract_j6_angle 注释中已数值验证），钉 j6 不影响位置解；
                            # 夹爪对称使 j6 与 j6±π 物理等价，取最近且限位内的等效分支。
                            _rq = np.asarray(_rq, dtype=np.float64).copy()
                            _j6_ik   = float(_rq[5])
                            _j6_hold = float(grasp_result.get("target_angles_orient",
                                                              _re_init_q)[5])
                            _j6_cands = [c for c in (_j6_hold + k * math.pi
                                                     for k in (-1, 0, 1))
                                         if _re_lims[5][0] <= c <= _re_lims[5][1]]
                            _rq[5] = min(_j6_cands, key=lambda c: abs(c - _j6_ik))
                            if abs(_rq[5] - _j6_ik) > 0.01:
                                print(f"[INFO] REACH j6 pinned: IK={_j6_ik:+.4f} "
                                      f"-> {_rq[5]:+.4f} (ORIENT end)", flush=True)
                            target_angles_arm = np.asarray(_rq,
                                                           dtype=np.float64).copy()
                            _re_target_cur    = target_angles_arm.copy()
                            print(f"[SM] REACH IK={np.round(target_angles_arm,3)}",
                                  flush=True)
                            # 打印 REACH 目标点（gripper_base 原点）的世界坐标
                            try:
                                from arm_ik_mujoco import fk_gripper as _fk_re_diag, quat_to_rot as _q2r_re
                                _re_R_rob   = _q2r_re(grasp_result["quat_w_scan"])
                                _re_pos_rob = grasp_result["pos_w_scan"]
                                _re_arm_base_w = _re_pos_rob + _re_R_rob @ np.array([0.0, 0.0, 0.0888])
                                _re_T_arm = _fk_re_diag(target_angles_arm)
                                _re_gb_world = _re_R_rob @ _re_T_arm[:3, 3] + _re_arm_base_w
                                _apw_re    = _re_R_rob @ _re_approach
                                _re_j7_world = _re_gb_world + 0.1358 * _apw_re
                                # 解析目标（直接从 _re_t_gb_arm 推算，不经过 IK 往返）
                                _re_gb_analytic = _re_R_rob @ _re_t_gb_arm + _re_arm_base_w
                                _re_j7_analytic = _re_gb_analytic + 0.1358 * _apw_re
                                print(f"[DIAG] REACH approach_world    = {np.round(_apw_re,4)}  ← 应指向物体", flush=True)
                                print(f"[DIAG] REACH gb_analytic_world = {np.round(_re_gb_analytic,4)}  (IK 输入)", flush=True)
                                print(f"[DIAG] REACH j7_analytic_world = {np.round(_re_j7_analytic,4)}  (j7 解析目标)", flush=True)
                                print(f"[DIAG] REACH anygrasp depth    = {float(grasp_result.get('depth', 0.0658)):.4f} m"
                                      f"  (掌心->指尖; 0.1358-d={0.1358-float(grasp_result.get('depth', 0.0658)):.4f})", flush=True)
                                print(f"[DIAG] REACH gb_fk_world       = {np.round(_re_gb_world,4)}  (FK验算IK解)", flush=True)
                                print(f"[DIAG] REACH j7_fk_world       = {np.round(_re_j7_world,4)}  (j7 FK验算)", flush=True)
                                try:
                                    _real_obj_re = env.get_object_pos(args.object)
                                    _dj7_re = _re_j7_analytic - _real_obj_re
                                    print(f"[DIAG] REACH object_world      = {np.round(_real_obj_re,4)}", flush=True)
                                    print(f"[DIAG] REACH j7-obj diff       = {np.round(_dj7_re,4)}  (理想时接近0)", flush=True)
                                    print(f"[DIAG] REACH object real world={np.round(_real_obj_re,4)} "
                                          f"dX={_re_gb_world[0]-_real_obj_re[0]:+.4f}m "
                                          f"dY={_re_gb_world[1]-_real_obj_re[1]:+.4f}m "
                                          f"dZ={_re_gb_world[2]-_real_obj_re[2]:+.4f}m",
                                          flush=True)
                                except Exception:
                                    pass
                            except Exception as _re_diag_e:
                                print(f"[DIAG] REACH world calc err: {_re_diag_e}", flush=True)
                        except Exception as _re:
                            print(f"[SM] REACH IK err: {_re} — DONE.", flush=True)
                            state = PipelineState.DONE
                    if target_angles_arm is not None \
                            and state == PipelineState.REACH:
                        from arm_ik_mujoco import _IK_JOINT_LIMITS as _re_lims2
                        cur_q = _get_arm_q(jpos)
                        # 速度限制斜坡：每步最多移动 _PG_MAX_DELTA，避免瞬移穿模
                        _re_delta = np.clip(
                            _re_target_cur - cur_q,
                            -_PG_MAX_DELTA, +_PG_MAX_DELTA)
                        _q6_re = np.clip(
                            cur_q + _re_delta,
                            [lo for lo, hi in _re_lims2],
                            [hi for lo, hi in _re_lims2])
                        _arm_step(_q6_re)
                        _gripper_step(close=False)
                        if state_step % 50 == 0:
                            _re_err = np.abs(cur_q - _re_target_cur)
                            print(f"[SM] REACH step {state_step}/{BUDGET[PipelineState.REACH]}: "
                                  f"max_err={_re_err.max():.4f}", flush=True)
                        state_step += 1
                        if state_step >= BUDGET[PipelineState.REACH]:
                            # ── 闭环纠偏：臂部静差（PD 平衡误差）让实际落点比命令偏高/偏侧 ──
                            # 静差近姿态不变 → 命令' = 命令 − (实际 − 解析)，1 次迭代即收敛
                            # （累积式：每次按"实测 − 原始解析目标"修正，避免把已补偿量又减回去）。
                            # 补偿量用局部 DLS 关节增量施加（_dls_gb_step），不重跑全局 IK。
                            # 只改目标位置（不碰 PD/gravcomp/夹爪参数）。
                            # 两道保护：误差不在 5~50mm 区间视为异常/已够好，不追；
                            # 纠偏后误差没实质减小（实际没跟随 → 硬顶物体/到限位）则回退并停止。
                            _re_corr_done = False
                            if _re_corr_iter < 3:
                                try:
                                    import mujoco as _mj_cc
                                    from arm_ik_mujoco import quat_to_rot as _q2r_cc
                                    _cc_bid   = _mj_cc.mj_name2id(env._model, _mj_cc.mjtObj.mjOBJ_BODY,
                                                                  "gripper_base")
                                    _cc_fing  = {_mj_cc.mj_name2id(env._model, _mj_cc.mjtObj.mjOBJ_BODY,
                                                                   _n)
                                                 for _n in ("link7", "link8")}
                                    _cc_obj   = _mj_cc.mj_name2id(env._model, _mj_cc.mjtObj.mjOBJ_BODY,
                                                                  args.object)
                                    _gb_act_c = env._data.xpos[_cc_bid].astype(np.float64).copy()
                                    _cc_R     = _q2r_cc(np.asarray(grasp_result["quat_w_scan"],
                                                                   dtype=np.float64))
                                    _cc_base  = (np.asarray(grasp_result["pos_w_scan"],
                                                            dtype=np.float64)
                                                 + _cc_R @ np.array([0., 0., 0.0888]))
                                    _cc_gb_ana  = _cc_R @ _re_t_gb_arm0 + _cc_base
                                    _cc_tip_ana = _cc_gb_ana + 0.1358 * (_cc_R @ _re_approach)
                                    _cc_e_w     = _gb_act_c - _cc_gb_ana   # 实测 − 解析(原始基准)
                                    _cc_en      = float(np.linalg.norm(_cc_e_w))
                                    # ── 接触急停：指板已碰到物体 → 停止纠偏（苹果骑肩
                                    # 楔入会把物体推走；碗深下探可能压壁）。检测到接触
                                    # 就直接进 CLOSE，不再往下压。
                                    _cc_touch = False
                                    try:
                                        for _cc_ci in range(int(env._data.ncon)):
                                            _cc_ct  = env._data.contact[_cc_ci]
                                            _cc_b1  = int(env._model.geom_bodyid[int(_cc_ct.geom1)])
                                            _cc_b2  = int(env._model.geom_bodyid[int(_cc_ct.geom2)])
                                            _cc_has = ((_cc_b1 in _cc_fing) ^
                                                       (_cc_b2 in _cc_fing))
                                            if _cc_has and (_cc_obj in (_cc_b1, _cc_b2)):
                                                _cc_touch = True
                                                break
                                    except Exception:
                                        pass
                                    if _cc_touch:
                                        print(f"[SM] REACH corr 接触急停: 指板已接触 "
                                              f"{args.object} (|e|={_cc_en * 1000:.1f}mm) -> "
                                              f"停止纠偏, 直接 CLOSE", flush=True)
                                        _re_corr_iter = 3   # 之后不再纠偏
                                    elif _re_prev_e is not None and _cc_en > 0.7 * _re_prev_e:
                                        # 实际没跟随命令（硬顶物体/到限位）：再纠只会把
                                        # 纠正量越积越大、持续加压 → 回退到纠偏前的目标并停止
                                        print(f"[WARN] REACH corr: 实际未跟随命令 "
                                              f"(|e| {_re_prev_e * 1000:.1f} -> {_cc_en * 1000:.1f}mm)"
                                              f" -> 回退并停止纠偏", flush=True)
                                        _re_corr_iter = 3
                                        if _re_prev_q is not None:
                                            target_angles_arm = _re_prev_q
                                            _re_target_cur    = _re_prev_q.copy()
                                            state_step        = 0
                                            _re_corr_done     = True
                                    elif 0.005 < _cc_en < 0.05:
                                        _cc_corr    = _re_corr_w - _cc_e_w
                                        _cc_tip_cmd = _cc_tip_ana + _cc_R @ _cc_corr
                                        _cc_floor   = float(_cc_tip_ana[2]) - 0.03
                                        if _tc_need is not None:
                                            # 静差近姿态不变 → 命令后的实际指尖 ≈ 命令 + e_z，
                                            # 地板须约束"预测实际"而非命令本身：否则 tip_ana
                                            # 恰落在 need 上时（第二轮香蕉 0.6769 == need）
                                            # 向下纠偏被整体钳回原值，静差永远纠不掉。
                                            _cc_floor = max(_cc_floor,
                                                            float(_tc_need) - float(_cc_e_w[2]))
                                        if _cc_tip_cmd[2] < _cc_floor:   # z 保护：抬回 floor
                                            _cc_corr = _cc_corr + _cc_R.T @ np.array(
                                                [0., 0., _cc_floor - _cc_tip_cmd[2]])
                                        _cc_gb_cmd_arm = _re_t_gb_arm0 + _cc_R.T @ _cc_corr
                                        _cc_cur_q = _get_arm_q(jpos).copy()
                                        # 局部增量（纯平移，j6 不动）：不重建朝向，
                                        # 因此不依赖 IK 种子/滚动角，也不会跳到另一分支
                                        _cc_q, _cc_res, _cc_dq = _dls_gb_step(_cc_cur_q,
                                                                              _cc_gb_cmd_arm)
                                        if _cc_res > 0.010 or _cc_dq > 0.35:
                                            print(f"[WARN] REACH corr: 局部增量不可用 "
                                                  f"(残差 {_cc_res * 1000:.1f}mm, 单关节 "
                                                  f"{np.degrees(_cc_dq):.1f}deg) -> 放弃纠偏",
                                                  flush=True)
                                        else:
                                            _re_prev_q        = _re_target_cur.copy()  # 回退点
                                            _re_corr_iter    += 1
                                            _re_prev_e        = _cc_en
                                            _re_corr_w        = _cc_corr
                                            target_angles_arm = _cc_q
                                            _re_target_cur    = _cc_q.copy()
                                            state_step        = 0
                                            _re_corr_done     = True
                                            print(f"[SM] REACH corr iter={_re_corr_iter} "
                                                  f"err={np.round(_cc_e_w * 1000, 1)}mm "
                                                  f"|err|={_cc_en * 1000:.1f}mm "
                                                  f"-> 新命令 gb={np.round(_cc_gb_cmd_arm, 4)} (arm)"
                                                  f" DLS 残差={_cc_res * 1000:.1f}mm "
                                                  f"单关节Δ={np.degrees(_cc_dq):.1f}deg", flush=True)
                                except Exception as _cc_e:
                                    print(f"[WARN] REACH corr 失败: {_cc_e}", flush=True)
                            if not _re_corr_done:
                                print(f"[SM] REACH done (step={state_step}) -> CLOSE",
                                      flush=True)
                                try:
                                    import mujoco as _mj_rd
                                    _gb_bid_rd = _mj_rd.mj_name2id(env._model, _mj_rd.mjtObj.mjOBJ_BODY, "gripper_base")
                                    _gb_actual_rd = env._data.xpos[_gb_bid_rd].copy()
                                    _j7_bid_rd = _mj_rd.mj_name2id(env._model, _mj_rd.mjtObj.mjOBJ_BODY, "link7")
                                    _j7_actual_rd = env._data.xpos[_j7_bid_rd].copy() if _j7_bid_rd >= 0 else None
                                    print(f"[DIAG] REACH done gb_actual_world  = {np.round(_gb_actual_rd,4)}  ← MuJoCo实际", flush=True)
                                    if _j7_actual_rd is not None:
                                        print(f"[DIAG] REACH done j7_actual_world  = {np.round(_j7_actual_rd,4)}  ← MuJoCo实际", flush=True)
                                except Exception as _rd_e:
                                    print(f"[DIAG] REACH done xpos read err: {_rd_e}", flush=True)
                                try:
                                    _real_obj_reach_done = env.get_object_pos(args.object)
                                    print(f"[DIAG] REACH done object real world={np.round(_real_obj_reach_done,4)}",
                                          flush=True)
                                except Exception:
                                    pass
                                # 打印各关节实际角度 vs IK目标，诊断未收敛原因
                                try:
                                    _cur_q_rd = _get_arm_q(jpos)
                                    _tgt_q_rd = _re_target_cur
                                    _err_q_rd  = _cur_q_rd - _tgt_q_rd
                                    print(f"[DIAG] REACH done joint_actual = {np.round(_cur_q_rd,4)}", flush=True)
                                    print(f"[DIAG] REACH done joint_target = {np.round(_tgt_q_rd,4)}", flush=True)
                                    print(f"[DIAG] REACH done joint_err    = {np.round(_err_q_rd,4)}  (deg={np.round(np.degrees(_err_q_rd),2)})", flush=True)
                                except Exception as _qe_rd:
                                    print(f"[DIAG] REACH done joint read err: {_qe_rd}", flush=True)
                                state = PipelineState.CLOSE
                                state_step = 0

            elif state == PipelineState.CLOSE:
                # 夹爪闭合 100 步
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=True)
                if target_angles_arm is not None:
                    _arm_step(target_angles_arm)
                state_step += 1
                if state_step % 20 == 0:
                    print(f"[SM] CLOSE step {state_step}/100", flush=True)
                if state_step >= 100:
                    print("[SM] CLOSE done -> LIFT", flush=True)
                    state = PipelineState.LIFT
                    state_step = 0

            elif state == PipelineState.LIFT:
                # 斜坡回到 ARM_SIDE_ANGLES 同时抬起物体
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                cur_q = _get_arm_q(jpos)
                _lf_alpha = min(1.3, state_step / max(150.0, 1))
                _arm_step(cur_q + _lf_alpha * (ARM_SIDE_ANGLES - cur_q))
                _gripper_step(close=True)  # 保持夹持
                state_step += 1
                if state_step >= 200:
                    try:
                        _obj_z = env.get_object_pos(args.object)[2]
                    except Exception:
                        _obj_z = 0.0
                    print(f"[SM] LIFT done: obj_z={_obj_z:.3f}m", flush=True)
                    if _obj_z > 0.75:  # 与 Isaac 原版一致（原版 0.65 判定恒真、失效）
                        state = PipelineState.PAN_NEG_X
                    else:
                        print("[SM] Object not lifted! Retry ARM_INIT", flush=True)
                        state = PipelineState.ARM_INIT
                    state_step = 0

            elif state == PipelineState.PAN_NEG_X:
                # PD 移动到 (pos[0]-0.5, pos[1]) 准备导航回放置点
                _px_goal = pos_w[0] - 0.5
                if _pan_neg_x_goal is None:
                    _pan_neg_x_goal = _px_goal
                _px_err = abs(pos_w[0] - _pan_neg_x_goal)
                _pxv = float(np.clip(-2.0 * (pos_w[0] - _pan_neg_x_goal), -0.3, 0.3))
                cmd_vx = 0.0; cmd_vy = _pxv; cmd_wz = 0.0
                _gripper_step(close=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_NEG_X step {state_step}: "
                          f"px_err={_px_err:.3f}m", flush=True)
                state_step += 1
                if _px_err < 0.2 or state_step >= 300:
                    print("[SM] PAN_NEG_X done -> NAV2_DEST", flush=True)
                    _pan_neg_x_goal = None
                    state = PipelineState.NAV2_DEST
                    state_step = 0

            elif state == PipelineState.NAV2_DEST:
                # 发送 Nav2 目标到 destination，等待完成
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=True)
                if not _dest_goal_sent:
                    bridge.send_goal(args.destination[0], args.destination[1])
                    _dest_goal_sent = True
                    print(f"[SM] NAV2_DEST goal sent: {args.destination}",
                          flush=True)
                _nd, _nf = bridge.get_nav_status()
                state_step += 1
                if _nd or state_step >= 6000:
                    _reason2 = "done" if _nd else "timeout"
                    print(f"[SM] NAV2_DEST {_reason2} -> ALIGN_YAW_2", flush=True)
                    _dest_goal_sent = False
                    state = PipelineState.ALIGN_YAW_2
                    state_step = 0

            elif state == PipelineState.ALIGN_YAW_2:
                # 对齐到 -Y 方向（yaw = -π/2），与 ALIGN_YAW 相同
                _yaw_err2 = _yaw_err_to(-math.pi / 2, yaw)
                cmd_vx = 0.0; cmd_vy = 0.0
                cmd_wz = float(np.clip(100.0 * _yaw_err2, -1.2, 1.2))
                _gripper_step(close=True)
                state_step += 1
                if abs(_yaw_err2) < 0.02 or state_step >= 600:
                    print("[SM] ALIGN_YAW_2 done -> PAN_DES_X", flush=True)
                    state = PipelineState.PAN_DES_X
                    state_step = 0

            elif state == PipelineState.PAN_DES_X:
                # 精调 X 位置到目标放置坐标的 X 方向
                _pdx_err = float(args.destination[0]) - float(pos_w[0])
                _pdx_abs = abs(_pdx_err)
                _pdxv = float(np.clip(-2.0 * _pdx_err, -0.3, 0.3))
                cmd_vx = 0.0; cmd_vy = _pdxv; cmd_wz = 0.0
                _gripper_step(close=True)
                if state_step % 50 == 0:
                    print(f"[SM] PAN_DES_X step {state_step}: "
                          f"dx_err={_pdx_abs:.3f}m", flush=True)
                state_step += 1
                if _pdx_abs < 0.1 or state_step >= 400:
                    print("[SM] PAN_DES_X done -> ROTATE", flush=True)
                    _rotate_j1_start = _get_arm_q(jpos)[0]
                    _rotate_j2_start = _get_arm_q(jpos)[1]
                    _rotate_j3_start = _get_arm_q(jpos)[2]
                    _rotate_j_target = np.array([
                        math.pi / 2, 1.8, -1.8,
                        ARM_SIDE_ANGLES[3], ARM_SIDE_ANGLES[4], ARM_SIDE_ANGLES[5]
                    ], dtype=np.float64)
                    state = PipelineState.ROTATE
                    state_step = 0

            elif state == PipelineState.ROTATE:
                # 插值 j1→+π/2, j2→1.8, j3→-1.8 over 200 steps
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=True)
                _a_r = min(1.0, state_step / 200.0)
                _qr  = _get_arm_q(jpos).copy()
                _qr[0] = _rotate_j1_start + _a_r * (_rotate_j_target[0] - _rotate_j1_start)
                _qr[1] = _rotate_j2_start + _a_r * (_rotate_j_target[1] - _rotate_j2_start)
                _qr[2] = _rotate_j3_start + _a_r * (_rotate_j_target[2] - _rotate_j3_start)
                _arm_step(_qr)
                state_step += 1
                if state_step >= 200:
                    print("[SM] ROTATE done -> PUT_DOWN", flush=True)
                    state = PipelineState.PUT_DOWN
                    state_step = 0

            elif state == PipelineState.PUT_DOWN:
                # 松开夹爪 100 步
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                _gripper_step(close=False)
                state_step += 1
                if state_step % 20 == 0:
                    print(f"[SM] PUT_DOWN step {state_step}/100", flush=True)
                if state_step >= 100:
                    print("[SM] PUT_DOWN done -> DONE", flush=True)
                    state = PipelineState.DONE

            # ── 无机械臂控制的过渡阶段：每帧保持 ARM_INIT_ANGLES，防止机械臂下垂引起抖动 ──
            _ARM_IDLE_STATES = {
                PipelineState.ALIGN_YAW_1,
                PipelineState.PAN_VX,
                PipelineState.PAN_VY,
                PipelineState.PAN_VX_FINAL,
                PipelineState.ALIGN_YAW,
            }
            if state in _ARM_IDLE_STATES:
                _arm_step(ARM_INIT_ANGLES)

            # ── 构造 obs，运行 policy ──
            obs_np = build_policy_obs(
                jpos, jvel, ang_vel, quat_w,
                cmd_vx, cmd_vy, cmd_wz, last_action,
            )
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

            # FREEZE_STATES_ACT：policy action 清零（与原版 navigate_to_goal_nav2_whole.py 完全一致）
            # ALIGN_YAW_1/PAN_VX/PAN_VY/ALIGN_YAW 不在此集合中：
            # 这些阶段 policy 完整输出 action，靠 obs 中的 cmd_vel 驱动旋转/平移。
            _FREEZE_STATES_ACT = {
                PipelineState.ARM_INIT, PipelineState.SCAN,
                PipelineState.PRE_GRASP,
                PipelineState.ORIENT, PipelineState.REACH,
                PipelineState.CLOSE, PipelineState.LIFT,
            }
            with torch.inference_mode():
                action_t = policy(obs_t)
            action_np = action_t.cpu().numpy().flatten()[:16]
            if state in _FREEZE_STATES_ACT:
                action_np = np.zeros(16, dtype=np.float32)
            last_action = action_np.copy()

            # ── 执行 env.step ──
            env.step(action_np)
            time.sleep(0.005)  # 实时 1×：让 MuJoCo passive viewer 有时间处理鼠标/键盘事件

            # FREEZE_STATES：强制维持根节点位姿
            _FREEZE_STATES = {
                PipelineState.ARM_INIT, PipelineState.SCAN,
                PipelineState.GRASP_PLAN,   # AnyGrasp 推理期间锁住底盘，防止漂移影响 IK
                PipelineState.PRE_GRASP,
                PipelineState.ORIENT,
                PipelineState.REACH,        # 抓取推进阶段锁住底盘，防止漂移导致夹爪偏离
                PipelineState.CLOSE,        # 夹爪闭合阶段锁住底盘
                PipelineState.LIFT,         # 抬臂阶段锁住底盘，防止香蕉脱手
            }
            if state in _FREEZE_STATES and _frozen_pos is not None:
                env.set_root_pose(_frozen_pos, _frozen_quat)

            # ── 【非阻塞状态块】GRASP_PLAN 异步化为一次性阻塞（在 step 之外执行）──
            if state == PipelineState.GRASP_PLAN and state_step == 0:
                import subprocess as _subproc
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                depth_med = np.median(np.stack(depth_accum, axis=0), axis=0)
                H, W = depth_med.shape
                fx_c, fy_c = 616.0, 616.0
                cx_c, cy_c = W / 2.0, H / 2.0
                u_g, v_g = np.meshgrid(np.arange(W), np.arange(H))
                z = depth_med
                mask = (z > 0.05) & (z < 4.0)
                z_v  = z[mask]
                pts  = np.stack([
                    (u_g[mask] - cx_c) * z_v / fx_c,
                    (v_g[mask] - cy_c) * z_v / fy_c,
                    z_v,    # 点云 Z 正值 = 物体方向，与 AnyGrasp 约定一致（AnyGrasp 期望 Z 正 = 前方）
                ], axis=-1).astype(np.float32)
                rgb_u = scan_rgb if scan_rgb is not None \
                    else np.zeros((H, W, 3), np.uint8)
                cols  = (rgb_u[mask] / 255.0).astype(np.float32)
                print(f"[SM] GRASP_PLAN: 点云原始 {len(pts)} pts "
                      f"(z min={z_v.min():.2f} max={z_v.max():.2f})", flush=True)

                # ── 桌面点云保存（RANSAC 之前，供穿桌检测使用）──
                try:
                    import cv2 as _cv2_tb
                    _cols_u8_tb = (cols * 255).astype(np.uint8).reshape(1, -1, 3)
                    _hsv_tb = _cv2_tb.cvtColor(_cols_u8_tb, _cv2_tb.COLOR_RGB2HSV)[0]
                    _table_mask_tb = (_hsv_tb[:, 1] < 40) & (_hsv_tb[:, 2] > 40) & (_hsv_tb[:, 2] < 160)
                    _pts_table_cam = pts[_table_mask_tb]
                    print(f'[SM] 桌面颜色点云: {len(_pts_table_cam)} pts', flush=True)
                    if len(_pts_table_cam) > 50:
                        from arm_ik_mujoco import fk_gripper as _fk_gb_tc, quat_to_rot as _q2r_tc, \
                            _CAM_OFFSET_POS as _cop_tc, _CAM_OFFSET_ROT as _cor_tc
                        _R_rob_tc    = _q2r_tc(_quat_w_at_scan.astype(np.float64))
                        _arm_base_tc = _pos_w_at_scan + _R_rob_tc @ np.array([0., 0., 0.0888])
                        _T_gb_tc     = _fk_gb_tc(_q_scan_at_scan.astype(np.float64))
                        _R_c2w_tc    = _R_rob_tc @ _T_gb_tc[:3, :3] @ _cor_tc
                        _t_cam_tc    = _R_rob_tc @ (_T_gb_tc[:3, :3] @ _cop_tc + _T_gb_tc[:3, 3]) + _arm_base_tc
                        # 点云是光学系（x右 y下 z前），_R_c2w_tc 已是光学系->世界，无需翻转
                        _pts_table_cam_c = _pts_table_cam.astype(np.float64).copy()
                        _pts_table_world = (_pts_table_cam_c @ _R_c2w_tc.T) + _t_cam_tc
                        # ── 平面过滤：颜色滤波会把白/灰色物体表面也筛进来（如白碗壁、
                        # 苹果顶高光），它们高于桌面 → clip 取邻域 z 中位数时会误判桌面高度
                        # （碗场景实测 h=0.7143=碗壁，导致下探提前截断、抓空）。
                        # 这里只保留主平面（桌面）附近的点；主平面接近水平才采信，否则保留原样。
                        try:
                            _tb_nv, _tb_dv, _tb_inl = _ransac_fit_plane(
                                _pts_table_world.astype(np.float64))
                            if _tb_nv is None:
                                print(f'[SM] 桌面点云平面拟合: 点太少({len(_pts_table_world)})，保留原样',
                                      flush=True)
                            elif abs(float(_tb_nv[2])) < 0.9:
                                print(f'[SM] WARN: 桌面点云主平面非水平 '
                                      f'(normal={np.round(_tb_nv, 3)})，保留原样', flush=True)
                            else:
                                _tb_keep = np.abs(_pts_table_world @ _tb_nv - _tb_dv) < 0.010
                                if int(_tb_keep.sum()) < 50:
                                    print(f'[SM] WARN: 桌面点云平面过滤后仅 '
                                          f'{int(_tb_keep.sum())} pts，保留原样', flush=True)
                                else:
                                    print(f'[SM] 桌面点云平面过滤: {len(_pts_table_world)} -> '
                                          f'{int(_tb_keep.sum())} pts '
                                          f'(z中位={np.median(_pts_table_world[_tb_keep, 2]):.4f})，'
                                          f'剔除物体表面点', flush=True)
                                    _pts_table_world = _pts_table_world[_tb_keep]
                        except Exception as _tb_pf_e:
                            print(f'[SM] 桌面点云平面过滤失败: {_tb_pf_e}', flush=True)
                        np.savez('/tmp/table_cloud.npz', points=_pts_table_world.astype(np.float32))
                        print(f'[SM] 桌面点云已保存 /tmp/table_cloud.npz（{len(_pts_table_world)} pts）', flush=True)
                    else:
                        print(f'[SM] WARN: 桌面颜色点云过少({len(_pts_table_cam)} pts)，跳过保存', flush=True)
                except Exception as _tb_e:
                    print(f'[SM] 桌面点云保存失败: {_tb_e}', flush=True)

                _keep = _ransac_remove_plane(pts)
                pts   = pts[_keep]
                cols  = cols[_keep]
                print(f"[SM] GRASP_PLAN: {len(pts)} pts after plane removal",
                      flush=True)
                if len(pts) < 200:
                    print("[SM] Too few points — DONE.", flush=True)
                    state = PipelineState.DONE
                else:
                    np.savez("/tmp/pointcloud.npz", points=pts, colors=cols)
                    _worker = os.path.join(_HERE, "grasp_worker.py")
                    _res = _subproc.run(
                        [sys.executable, _worker,
                         "--checkpoint", args.grasp_checkpoint,
                         "--topk", "50"],
                        timeout=120,
                    )
                    if _res.returncode != 0 or \
                            not os.path.exists("/tmp/grasp_result.npz"):
                        print("[SM] GraspNet failed — DONE.", flush=True)
                        state = PipelineState.DONE
                    else:
                        _gr = np.load("/tmp/grasp_result.npz",
                                      allow_pickle=True)
                        if len(_gr["scores"]) == 0:
                            print("[SM] No valid grasps — DONE.", flush=True)
                            state = PipelineState.DONE
                        else:
                            # ── 诊断：任取第0个候选，打印各坐标系下的approach方向 ──
                            try:
                                from arm_ik_mujoco import (
                                    quat_to_rot as _q2r_diag,
                                    fk_gripper as _fkg_diag,
                                    _CAM_OFFSET_ROT as _COR_diag,
                                )
                                _R_diag = _gr["rotations"][0]   # 相机系
                                _app_cam = _R_diag[:, 0]
                                _R_gb_diag = _fkg_diag(_q_scan_at_scan.astype(np.float64))[:3, :3]
                                _app_gb    = _COR_diag @ _app_cam
                                _R_rob_diag = _q2r_diag(_quat_w_at_scan.astype(np.float64))
                                _app_base  = _R_rob_diag @ _R_gb_diag @ _app_gb
                                _app_world = _app_base   # world z = base z（平地）
                                print(f"[DIAG-APPROACH] cam  ={np.round(_app_cam,4)}", flush=True)
                                print(f"[DIAG-APPROACH] gb   ={np.round(_app_gb,4)}", flush=True)
                                print(f"[DIAG-APPROACH] world={np.round(_app_world,4)}", flush=True)
                            except Exception as _diag_e:
                                print(f"[DIAG-APPROACH] failed: {_diag_e}", flush=True)
                            # ── 目标过滤：高饱和度过滤 S>=80（对齐原版 fallback 逻辑）──
                            _bmask = None
                            _q_scan_saved    = _q_scan_at_scan.copy()
                            _pos_w_scan_pre  = _pos_w_at_scan.copy()
                            _quat_w_scan_pre = _quat_w_at_scan.copy()
                            try:
                                import cv2 as _cv2_f
                                from scipy.spatial import cKDTree as _KDTree_f
                                _cols_u8_ng = (cols * 255).astype(np.uint8).reshape(1, -1, 3)
                                _hsv_ng = _cv2_f.cvtColor(_cols_u8_ng, _cv2_f.COLOR_RGB2HSV)[0]
                                _vivid_mask = (_hsv_ng[:, 1] >= 80)
                                _vivid_pts = pts[_vivid_mask]
                                print(f"[SM] 高饱和过滤(S>=80): {len(_vivid_pts)} 鲜艳点", flush=True)
                                # ── 保存 filter.png：三类过滤可视化 ──
                                try:
                                    if scan_rgb is not None:
                                        _mask_flat_f = mask.ravel()                          # 深度有效 bool (H*W,)
                                        _valid_idx_flat_f = np.where(_mask_flat_f)[0]         # 有效像素 flat 索引
                                        _filter_img_f = scan_rgb.copy()
                                        _filter_flat_f = _filter_img_f.reshape(-1, 3)
                                        _filter_flat_f[~_mask_flat_f] = 0                    # 类A: 无效深度变黑
                                        _filter_flat_f[_valid_idx_flat_f[~_keep]] = 0        # 类B: RANSAC 桌面点变黑
                                        _filter_flat_f[_valid_idx_flat_f[_keep][~_vivid_mask]] = 0  # 类C: S<80 变黑
                                        _filter_bgr_f = _cv2_f.cvtColor(
                                            _filter_flat_f.reshape(H, W, 3), _cv2_f.COLOR_RGB2BGR)
                                        _filter_dir = "/home/mojie/taskdog/custom_envs/tmp_pictures"
                                        os.makedirs(_filter_dir, exist_ok=True)
                                        _filter_path = os.path.join(_filter_dir, "filter.png")
                                        _cv2_f.imwrite(_filter_path, _filter_bgr_f)
                                        print(f"[SM] filter.png 已保存: {_filter_path}", flush=True)
                                except Exception as _fe:
                                    print(f"[SM] filter.png 保存失败: {_fe}", flush=True)
                                if len(_vivid_pts) >= 20:
                                    _kd_ng = _KDTree_f(_vivid_pts)
                                    _dists_ng, _ = _kd_ng.query(_gr["translations"], k=1)
                                    _bmask_ng = _dists_ng < 0.05
                                    _n_ok_ng = int(_bmask_ng.sum())
                                    print(f"[SM] 高饱和过滤: {_n_ok_ng}/{len(_gr['translations'])} 候选近鲜艳点", flush=True)
                                    if _n_ok_ng > 0:
                                        _bmask = _bmask_ng
                                    else:
                                        print("[SM] 高饱和过滤无候选 -> 全部候选进排序", flush=True)
                                else:
                                    print("[SM] 高饱和点不足 -> 全部候选进排序", flush=True)
                            except Exception as _cfe:
                                print(f"[SM] 目标过滤异常: {_cfe}", flush=True)

                            # ── IK 预筛选 + 排序（对齐原版逻辑）──
                            try:
                                from arm_ik_mujoco import (
                                    solve as _ik_solve_pos_gp,
                                    cam_to_world as _ctw_gp,
                                    world_pos_to_arm_frame as _w2a_gp,
                                    compute_desired_ee_rot_in_arm as _cder_gp,
                                    quat_to_rot as _q2r_gp,
                                    fk_gripper as _fkg_gp,
                                    _CAM_OFFSET_ROT as _COR_gp,
                                )
                                _R_rob_gp    = _q2r_gp(_quat_w_scan_pre)
                                _R_gb_gp     = _fkg_gp(_q_scan_saved)[:3, :3]
                                _R_cam2world = _R_rob_gp @ _R_gb_gp @ _COR_gp
                                # _bmask=None 时全部候选进排序，否则只排序颜色过滤后的候选
                                _banana_idxs_pre = list(np.where(_bmask)[0]) \
                                    if _bmask is not None else list(range(len(_gr["scores"])))
                                if not _banana_idxs_pre:
                                    _banana_idxs_pre = list(range(len(_gr["scores"])))
                                _ranked_pre = []
                                for _ci in _banana_idxs_pre:
                                    _t_ci  = _gr["translations"][_ci]
                                    _tw_ci = _ctw_gp(_t_ci, _q_scan_saved,
                                                     _pos_w_scan_pre, _quat_w_scan_pre)
                                    _ta_ci = _w2a_gp(_tw_ci, _pos_w_scan_pre,
                                                     _quat_w_scan_pre)
                                    try:
                                        _q_pre = _ik_solve_pos_gp(
                                            _ta_ci, target_rot=None,
                                            initial_angles=_q_scan_saved)
                                        _ik_ok_pre = (float(_q_pre[1]) < 2.8)
                                    except Exception:
                                        _ik_ok_pre = False
                                    _R_cam_ci      = _gr["rotations"][_ci]
                                    _app_world_ci  = _R_cam2world @ _R_cam_ci[:, 0]
                                    _approach_z_ci = float(_app_world_ci[2])
                                    _ranked_pre.append((
                                        _ik_ok_pre,
                                        float(abs(_approach_z_ci)),
                                        float(_gr["scores"][_ci]),
                                        _ci,
                                        _approach_z_ci,
                                    ))
                                # 排序：IK可解优先，approach_z 越负越好，score 越高越好
                                _ranked_pre.sort(
                                    key=lambda x: (-int(x[0]), x[4], -x[2]))
                                _ranked_idxs = [x[3] for x in _ranked_pre]
                                print(
                                    f"[SM] 候选排序({len(_ranked_idxs)}个): "
                                    + " ".join(
                                        f"[{r[3]}]{'✓' if r[0] else '✗'}"
                                        f"s={r[2]:.2f}az={r[4]:.2f}"
                                        for r in _ranked_pre[:6]
                                    ),
                                    flush=True,
                                )
                            except Exception as _rank_e:
                                print(f"[SM] IK预筛选异常({_rank_e}),使用原始顺序",
                                      flush=True)
                                _ranked_idxs = list(range(len(_gr["scores"])))
                            _best_gi = _ranked_idxs[0]
                            try:
                                _R_des_best = _cder_gp(
                                    _gr["rotations"][_best_gi], _q_scan_saved)
                            except Exception:
                                _R_des_best = None
                            # ── 每抓取 depth（AnyGrasp 约定：指尖 = 掌心 + depth·a）──
                            # 旧版 npz 无 depths key → 回退 0.0658（= 旧行为），不崩
                            _gr_depths = _gr["depths"] if "depths" in _gr else \
                                np.full(len(_gr["scores"]), 0.0658, dtype=np.float32)
                            print(f"[DIAG] best grasp R_cam approach(col0)={np.round(_gr['rotations'][_best_gi][:,0],4)} depth={float(_gr_depths[_best_gi]):.4f}", flush=True)
                            # GraspNet 原始输出即光学系（x右 y下 z前），直接使用
                            _R_cam_best = _gr["rotations"][_best_gi].copy()
                            grasp_result = {
                                "t_cam":   _gr["translations"][_best_gi],
                                "R_cam":   _R_cam_best,
                                "width":   float(_gr["widths"][_best_gi]),
                                "score":   float(_gr["scores"][_best_gi]),
                                "depth":   float(_gr_depths[_best_gi]),
                                "R_desired_EE_in_arm": _R_des_best,
                                "gr_translations": _gr["translations"],
                                "gr_rotations":    _gr["rotations"],
                                "gr_widths":       _gr["widths"],
                                "gr_scores":       _gr["scores"],
                                "gr_depths":       _gr_depths,
                                "q_scan":      _q_scan_saved,
                                "pos_w_scan":  _pos_w_scan_pre,
                                "quat_w_scan": _quat_w_scan_pre,
                                "ranked_candidate_idxs": _ranked_idxs,
                                "tried_set":             {_best_gi},
                            }
                            print(f"[SM] GRASP_PLAN done: "
                                  f"{len(_gr['scores'])} grasps, "
                                  f"best=[{_best_gi}] score={grasp_result['score']:.3f}",
                                  flush=True)
                            # ── DIAG + approach sanity check（approach_z>0 时自动 flip）──
                            try:
                                from arm_ik_mujoco import (
                                    cam_to_world as _c2w_d, quat_to_rot as _q2r_d,
                                    _CAM_OFFSET_ROT as _COR_d, fk_gripper as _fkg_d,
                                    compute_desired_ee_rot_in_arm as _cder_d,
                                )
                                _t_c_d = grasp_result["t_cam"]
                                _R_c_d = grasp_result["R_cam"]
                                _T_fk_d  = _fkg_d(grasp_result["q_scan"])
                                _R_cw_d  = _q2r_d(quat_w) @ _T_fk_d[:3, :3] @ _COR_d
                                _app_w_d = _R_cw_d @ _R_c_d[:, 0]
                                _t_obj_w_d = _c2w_d(_t_c_d, grasp_result["q_scan"], pos_w, quat_w)
                                print(f"[DIAG] t_cam={np.round(_t_c_d,4)} depth={_t_c_d[2]:.3f}m", flush=True)
                                print(f"[DIAG] t_obj_world={np.round(_t_obj_w_d,4)} z={_t_obj_w_d[2]:.3f}m", flush=True)
                                print(f"[DIAG] approach world={np.round(_app_w_d,3)}", flush=True)
                                try:
                                    _real_obj_pos = env.get_object_pos(args.object)
                                    print(f"[DIAG] object real world={np.round(_real_obj_pos,4)} "
                                          f"(delta from t_obj: {np.round(_real_obj_pos - _t_obj_w_d,4)})",
                                          flush=True)
                                except Exception:
                                    pass
                                _approach_world_z = float(_app_w_d[2])
                                if _approach_world_z > 0.0:
                                    print(f"[WARN] approach z={_approach_world_z:.3f}>0, flipping.", flush=True)
                                    _R_flipped = grasp_result["R_cam"].copy()
                                    _R_flipped[:, 0] = -_R_flipped[:, 0]
                                    _R_flipped[:, 1] = -_R_flipped[:, 1]
                                    grasp_result["R_cam"] = _R_flipped
                                    grasp_result["R_desired_EE_in_arm"] = _cder_d(
                                        _R_flipped, grasp_result["q_scan"])
                                    print(f"[WARN] After flip z={float((_R_cw_d@_R_flipped[:,0])[2]):.3f}", flush=True)
                                else:
                                    print(f"[INFO] approach z={_approach_world_z:.3f}<0 (OK)", flush=True)
                            except Exception as _de:
                                print(f"[DIAG] failed: {_de}", flush=True)
                            state = PipelineState.PRE_GRASP
                            state_step = 0
                            _pg_cmd = None

    except KeyboardInterrupt:
        print("[MJ] KeyboardInterrupt")
    finally:
        bridge.stop()
        if grasp_proc:
            grasp_proc.terminate()
        env.close()
        print("[MJ] Done")


def _handle_state(
    state, state_step, pos_w, yaw, jpos, jvel,
    grasp_pose, args, bridge, ik_solver,
    grasp_result_q, env,
):
    """处理 PAN_VX 之后的所有状态，返回 (cmd_vx, cmd_vy, cmd_wz)。

    注意：此函数通过 nonlocal 修改外部变量（state、state_step、grasp_pose 等）
    不可行，故改为返回值 + env 直接操作的方式。
    状态转移由调用方根据返回的 next_state 处理。

    TODO: 此函数是占位实现，具体逻辑需按照 navigate_to_goal_nav2_whole.py
    中对应状态的逻辑逐一移植。主要工作包括：
      - PAN_VX/PAN_VY：PD 控制精调机器人 X/Y 坐标
      - ALIGN_YAW：对齐偏航角
      - ARM_INIT：机械臂插值到初始扫描姿态，同时 FREEZE 根节点
      - SCAN：收集 RGB+depth 帧，发送给 grasp_worker
      - GRASP_PLAN：等待 grasp_worker 返回抓取位姿
      - PRE_GRASP/ORIENT/REACH：IK 求解 + 机械臂运动
      - CLOSE：夹爪闭合
      - LIFT：抬起物体
      - PAN_NEG_X/NAV2_DEST/ALIGN_YAW_2/PAN_DES_X：放置前导航
      - ROTATE/PUT_DOWN：旋转机械臂到放置位，松夹爪
    """
    # 默认：停止运动
    return 0.0, 0.0, 0.0


if __name__ == "__main__":
    main()
