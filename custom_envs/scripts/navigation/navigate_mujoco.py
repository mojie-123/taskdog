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
    PAN_VX       = "PAN_VX"
    PAN_VY       = "PAN_VY"
    ARM_INIT     = "ARM_INIT"
    SCAN         = "SCAN"
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
GRIPPER_OPEN_POS    = np.array([ 0.035, -0.035], dtype=np.float32)
GRIPPER_CLOSE_POS   = np.array([ 0.000,  0.000], dtype=np.float32)
ARM_INIT_ANGLES     = np.array([0.0,  0.5, -1.0, 0.0,  0.5, 0.0], dtype=np.float32)  # = ARM_HOME_ANGLES in navigate_to_goal_nav2_whole.py
ARM_SIDE_ANGLES     = np.array([-math.pi/2, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float32)  # matches navigate_to_goal_nav2_whole.py

# BUDGET（每个状态最大步数）
BUDGET = {
    PipelineState.ARM_INIT:   300,
    PipelineState.PRE_GRASP:  250,
    PipelineState.ORIENT:     100,
    PipelineState.REACH:      300,
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
class ReachState:
    """REACH 跨 policy step 持久化的闭环纠偏状态。"""
    target_cur: Any = None
    target_gb_arm0: Any = None
    corr_iter: int = 0
    corr_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    prev_actual_gb_world: Any = None
    prev_cmd_move_world: Any = None
    prev_target_q: Any = None
    table_height: Any = None

    def reset(self):
        self.target_cur = None
        self.target_gb_arm0 = None
        self.corr_iter = 0
        self.corr_world[:] = 0.0
        self.prev_actual_gb_world = None
        self.prev_cmd_move_world = None
        self.prev_target_q = None
        self.table_height = None


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
):
    """Run the synchronous GRASP_PLAN computation after the final SCAN physics step.

    This is intentionally a pure structural extraction of the former inline block:
    it keeps the same point-cloud construction, AnyGrasp worker IPC, filtering,
    ranking, diagnostics, and PRE_GRASP/DONE transition decisions.
    """
    state = PipelineState.GRASP_PLAN
    grasp_result = None
    table_cloud_world = None

    import subprocess as _subproc
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
            _R_rob_tc    = _q2r_tc(quat_w_at_scan.astype(np.float64))
            _arm_base_tc = pos_w_at_scan + _R_rob_tc @ np.array([0., 0., 0.0888])
            _T_gb_tc     = _fk_gb_tc(q_scan_at_scan.astype(np.float64))
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
                _tb_nv, _tb_dv, _ = ransac_fit_plane(
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
            table_cloud_world = _pts_table_world.astype(np.float32).copy()
            print(f'[SM] 桌面点云已缓存到内存（{len(table_cloud_world)} pts）', flush=True)
        else:
            print(f'[SM] WARN: 桌面颜色点云过少({len(_pts_table_cam)} pts)，跳过缓存', flush=True)
    except Exception as _tb_e:
        print(f'[SM] 桌面点云构建失败: {_tb_e}', flush=True)

    _keep = ransac_remove_plane(pts)
    # 物体点云（RANSAC 去桌面后）：不再喂 AnyGrasp，只用于"抓取点必须落在
    # 物体上"的几何筛选（见下方 GRASP_PLAN 的落物体筛选）。
    # AnyGrasp 现在吃**完整点云（含桌面）** —— 让它知道桌子存在，避免产生
    # 指尖穿桌之类的不可行候选。
    _pts_obj  = pts[_keep]
    _cols_obj = cols[_keep]
    _obj_filter_ok = (len(_pts_obj) >= 200)
    print(f"[SM] GRASP_PLAN: 完整点云 {len(pts)} pts (喂 AnyGrasp, 含桌面), "
          f"物体点云 {len(_pts_obj)} pts (去桌面后)", flush=True)
    if not _obj_filter_ok:
        print(f"[WARN] 物体点云不足 200 pts ({len(_pts_obj)}) "
              f"—— 跳过落物体筛选, 全部候选放行", flush=True)
    if len(pts) < 200:
        print("[SM] Too few points — DONE.", flush=True)
        state = PipelineState.DONE
    else:
        # 完整点云（含桌面）通过 /tmp/pointcloud.npz 传给独立 grasp_worker；
        # 物体点云只在本进程内用于候选筛选，不再额外写磁盘。
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
                    _R_gb_diag = _fkg_diag(q_scan_at_scan.astype(np.float64))[:3, :3]
                    _app_gb    = _COR_diag @ _app_cam
                    _R_rob_diag = _q2r_diag(quat_w_at_scan.astype(np.float64))
                    _app_base  = _R_rob_diag @ _R_gb_diag @ _app_gb
                    _app_world = _app_base   # world z = base z（平地）
                    print(f"[DIAG-APPROACH] cam  ={np.round(_app_cam,4)}", flush=True)
                    print(f"[DIAG-APPROACH] gb   ={np.round(_app_gb,4)}", flush=True)
                    print(f"[DIAG-APPROACH] world={np.round(_app_world,4)}", flush=True)
                except Exception as _diag_e:
                    print(f"[DIAG-APPROACH] failed: {_diag_e}", flush=True)
                # ── 目标过滤：高饱和度过滤 S>=80（对齐原版 fallback 逻辑）──
                # 只作用在**物体点云**上（_pts_obj/_cols_obj）；AnyGrasp 吃的是
                # 完整点云，桌面候选主要由下方"落物体筛选"拦截，这里降级为
                # 颜色偏好叠加。
                _bmask = None
                _q_scan_saved    = q_scan_at_scan.copy()
                _pos_w_scan_pre  = pos_w_at_scan.copy()
                _quat_w_scan_pre = quat_w_at_scan.copy()
                try:
                    import cv2 as _cv2_f
                    from scipy.spatial import cKDTree as _KDTree_f
                    _cols_u8_ng = (_cols_obj * 255).astype(np.uint8).reshape(1, -1, 3)
                    _hsv_ng = _cv2_f.cvtColor(_cols_u8_ng, _cv2_f.COLOR_RGB2HSV)[0]
                    _vivid_mask = (_hsv_ng[:, 1] >= 80)
                    _vivid_pts = _pts_obj[_vivid_mask]
                    print(f"[SM] 高饱和过滤(S>=80): {len(_vivid_pts)} 鲜艳点", flush=True)
                    # 调试图输出独立到 diagnostics 模块；不参与候选筛选逻辑。
                    _diag_save_filter_image(
                        scan_rgb, mask, _keep, _vivid_mask)
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
                    # ── 落物体筛选（主判据）──
                    # C = 掌心 + d·a 是 AnyGrasp 认为的**指尖落点**；要求 C 距
                    # 物体点云（已去桌面，光学系，与 AnyGrasp 输出同系）最近距离
                    # < 1cm 才算"抓在物体上"。必须判 C 而不是掌心：掌心在物体外
                    # d(2~6cm) 处，用掌心会误杀全部有效抓取。
                    # 物体点云不可用/无候选/异常 -> 警告 + 全部放行（不阻断）。
                    _GS_DIST_MAX = 0.08
                    _gr_depths_pre = _gr["depths"] if "depths" in _gr else \
                        np.full(len(_gr["scores"]), 0.0658, dtype=np.float32)
                    _gs_kd = None
                    if _obj_filter_ok:
                        try:
                            from scipy.spatial import cKDTree as _KDTree_gs
                            _gs_kd = _KDTree_gs(_pts_obj.astype(np.float64))
                            print(f"[SM] 落物体筛选: KD树 {len(_pts_obj)} pts, "
                                  f"阈值 {_GS_DIST_MAX * 100:.0f}cm", flush=True)
                        except Exception as _gs_e:
                            print(f"[WARN] 落物体筛选 KD树构建失败: {_gs_e} "
                                  f"—— 全部候选放行", flush=True)
                            _gs_kd = None
                    else:
                        print("[WARN] 落物体筛选未启用（物体点云不足）"
                              "—— 全部候选放行", flush=True)
                    _gs_ok = {}
                    if _gs_kd is not None:
                        for _ci in _banana_idxs_pre:
                            try:
                                _C_ci = (
                                    np.asarray(_gr["translations"][_ci],
                                               dtype=np.float64)
                                    + float(_gr_depths_pre[_ci])
                                    * np.asarray(_gr["rotations"][_ci][:, 0],
                                                 dtype=np.float64))
                                _gs_ok[_ci] = bool(
                                    _gs_kd.query(_C_ci, k=1)[0] < _GS_DIST_MAX)
                            except Exception:
                                _gs_ok[_ci] = True   # 异常 -> 该候选放行
                        _n_gs_pass = int(sum(_gs_ok.values()))
                        print(f"[SM] 落物体筛选: {_n_gs_pass}/"
                              f"{len(_banana_idxs_pre)} 候选的抓取点距物体 "
                              f"<{_GS_DIST_MAX * 100:.0f}cm", flush=True)
                        if _n_gs_pass == 0:
                            print("[WARN] 落物体筛选后无候选 —— 全部候选放行",
                                  flush=True)
                            _gs_ok = {}
                    _banana_idxs_pre = [i for i in _banana_idxs_pre
                                        if _gs_ok.get(i, True)]
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
                            float(_gr["scores"][_ci]),
                            _ci,
                            _approach_z_ci,
                        ))
                    # 排序：IK可解优先，score 越高越好，approach_z 越负越好
                    _ranked_pre.sort(
                        key=lambda x: (-int(x[0]), -x[1], x[3]))
                    _ranked_idxs = [x[2] for x in _ranked_pre]
                    print(
                        f"[SM] 候选排序({len(_ranked_idxs)}个): "
                        + " ".join(
                            f"[{r[2]}]{'✓' if r[0] else '✗'}"
                            f"s={r[1]:.2f}az={r[3]:.2f}"
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
                    "score":   float(_gr["scores"][_best_gi]),
                    "depth":   float(_gr_depths[_best_gi]),
                    "R_desired_EE_in_arm": _R_des_best,
                    "gr_translations": _gr["translations"],
                    "gr_rotations":    _gr["rotations"],
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

    if grasp_result is not None:
        # 与 REACH 同进程共享，避免 /tmp/table_cloud.npz 写盘再读盘。
        grasp_result["table_cloud_world"] = table_cloud_world

    return state, grasp_result

def main():
    parser = argparse.ArgumentParser("Navigate MuJoCo")
    parser.add_argument("--map",            required=True)
    parser.add_argument("--goal",           nargs=2, type=float, required=True)
    parser.add_argument("--policy_path",    required=True, help="model_4999.pt 路径")
    parser.add_argument("--grasp_checkpoint", default=None)
    parser.add_argument("--grasp_topk",     type=int, default=1)
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

    # ── IK 求解器：arm_ik 为纯函数模块，无需实例化 ──
    # solve_for_gripper_base / compute_desired_ee_rot_in_arm 等在状态机内懒导入

    # ── 启动 ROS2 bridge ──
    from custom_envs.utils.ros2_bridge import IsaacROS2Bridge
    bridge = IsaacROS2Bridge(cmd_vel_timeout=0.5)
    bridge.start(timeout=30.0)
    print("[MJ] ROS2 bridge started")

    # ── 发送导航目标 ──
    bridge.send_goal(args.goal[0], args.goal[1])

    # ── 状态机变量初始化 ──
    state          = PipelineState.NAV
    state_step     = 0
    last_action    = np.zeros(16, dtype=np.float32)
    grasp_result   = None   # grasp_worker 返回的抓取结果 dict
    depth_accum    = []     # SCAN 阶段累积的 depth 帧列表
    scan_rgb       = None   # SCAN 阶段保存的 RGB 帧
    j6_target      = None   # ORIENT 阶段的 j6 目标角
    _orient_q_fixed = None  # ORIENT 阶段固定的 j1~j5 实际角度
    _q_scan_at_scan  = None  # SCAN 完成时的机械臂关节角（供 IK 用）
    _pos_w_at_scan   = None  # SCAN 完成时的机器人世界位置
    _quat_w_at_scan  = None  # SCAN 完成时的机器人四元数
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
    _pg_cmd          = None  # PRE_GRASP 斜坡指令（按 step 累积）
    _pg_target_cur   = None  # PRE_GRASP IK 目标关节角（供斜坡使用）

    # REACH / CLOSE / LIFT 的跨帧状态集中管理。
    # 只收纳真正需要跨 policy step 保存的量；单帧几何/IK 临时量仍留在对应状态分支内。
    reach_ctx = ReachState()
    close_ctx = CloseState()
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
        PipelineState.PRE_GRASP,
        PipelineState.ORIENT,
        PipelineState.REACH,
        PipelineState.CLOSE,
        PipelineState.LIFT,
    }
    _BASE_LOCK_STATES = _GRASP_BASE_LOCK_STATES | {PipelineState.ROTATE}

    # 每帧固定不变的状态集合只构造一次，避免主循环内重复创建。
    _ARM_IDLE_STATES = {
        PipelineState.ALIGN_YAW_1,
        PipelineState.PAN_VX,
        PipelineState.PAN_VY,
    }
    _LOCOMOTION_DISABLED_STATES = {
        PipelineState.ARM_INIT, PipelineState.SCAN,
        PipelineState.GRASP_PLAN, PipelineState.PRE_GRASP,
        PipelineState.ORIENT, PipelineState.REACH,
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
                    print(f"[MJ] ALIGN_YAW_1 done ({_why}) -> PAN_VX", flush=True)
                    next_state = PipelineState.PAN_VX
                    state_step = 0

            elif current_state == PipelineState.PAN_VX:
                # 世界 Y 精调：和 PAN_DES_X 一样直接在世界系闭环，实时 yaw 只负责坐标变换。
                _dy_w = float(args.goal[1]) - float(pos_w[1])
                cmd_vx, cmd_vy, cmd_wz, _pan_done, _pd = _pan_world_axis_control(
                    PipelineState.PAN_VX, _dy_w, 'y', math.pi / 2,
                    pos_tol=0.08, resume_tol=0.16, kp=0.8, vmax=0.35,
                    yaw_now=yaw, base_lin_vel=rs["lin_vel"])
                if state_step == 1:
                    print(f"[SM] PAN_VX start dy_err={_pd['signed_err']:+.3f}m", flush=True)
                if state_step % 25 == 0:
                    print(f"[SM] PAN_VX step {state_step}: dy_err={_pd['signed_err']:+.3f}m "
                          f"cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                          f"vy_world={_pd['axis_vel']:+.3f} vxy={_pd['vxy']:.3f} "
                          f"yaw_err={math.degrees(_pd['yaw_err']):+.2f}deg "
                          f"settling={int(_pd['settling'])} "
                          f"settle={_pd['settle_count']}/{_PAN_SETTLE_STEPS}", flush=True)
                state_step += 1
                if _pan_done or state_step >= PAN_MAX_STEPS:
                    _why = "settled" if _pan_done else "timeout"
                    print(f"[SM] PAN_VX done ({_why}) -> PAN_VY", flush=True)
                    next_state = PipelineState.PAN_VY
                    state_step = 0

            elif current_state == PipelineState.PAN_VY:
                # 世界 X 精调：目标为 goal_x+0.5m。直接按 world X 闭环，再根据实时 yaw
                # 转成 body vx/vy，避免依赖“yaw 必须恰好为 +pi/2”的固定符号假设。
                _dx_w = float(args.goal[0]) + 0.5 - float(pos_w[0])
                cmd_vx, cmd_vy, cmd_wz, _pan_done, _pd = _pan_world_axis_control(
                    PipelineState.PAN_VY, _dx_w, 'x', math.pi / 2,
                    pos_tol=0.10, resume_tol=0.20, kp=1.0, vmax=0.25,
                    yaw_now=yaw, base_lin_vel=rs["lin_vel"],
                    settle_use_vxy=True)
                if state_step == 1:
                    print(f"[SM] PAN_VY start dx_err={_pd['signed_err']:+.3f}m", flush=True)
                if state_step % 25 == 0:
                    print(f"[SM] PAN_VY step {state_step}: dx_err={_pd['signed_err']:+.3f}m "
                          f"cmd=({cmd_vx:+.3f},{cmd_vy:+.3f},{cmd_wz:+.3f}) "
                          f"vx_world={_pd['axis_vel']:+.3f} vxy={_pd['vxy']:.3f} "
                          f"yaw_err={math.degrees(_pd['yaw_err']):+.2f}deg "
                          f"settling={int(_pd['settling'])} "
                          f"settle={_pd['settle_count']}/{_PAN_SETTLE_STEPS}", flush=True)
                state_step += 1
                if _pan_done or state_step >= PAN_MAX_STEPS:
                    _why = "settled" if _pan_done else "timeout"
                    print(f"[SM] PAN_VY done ({_why}) -> ARM_INIT", flush=True)
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
                        # 调试图输出独立到 diagnostics 模块。
                        _diag_save_scan_raw(scan_rgb)
                        next_state = PipelineState.GRASP_PLAN
                        state_step = 0
                except ValueError as _scan_e:
                    print(f"[SM] SCAN camera unavailable ({_scan_e}) -- DONE.",
                          flush=True)
                    next_state = PipelineState.DONE
                    state_step = 0

            elif current_state == PipelineState.PRE_GRASP:
                # IK 求解 + 斜坡移动机械臂到 pre-grasp 姿态（对齐原版逻辑）
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None:
                    next_state = PipelineState.DONE
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
                                    next_state = PipelineState.ARM_INIT
                                    state_step = 0
                                else:
                                    _tried.add(_next_ci)
                                    grasp_result["tried_set"] = _tried
                                    grasp_result["t_cam"]     = grasp_result["gr_translations"][_next_ci]
                                    # GraspNet 原始输出即光学系（x右 y下 z前），直接使用
                                    grasp_result["R_cam"]     = grasp_result["gr_rotations"][_next_ci].copy()
                                    grasp_result["score"]     = float(grasp_result["gr_scores"][_next_ci])
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
                                # 最终抓取选择可视化移到 diagnostics 模块。
                                _diag_save_choice_image(
                                    scan_rgb,
                                    grasp_result["t_cam"],
                                    grasp_result["R_cam"],
                                )
                        except Exception as _ike:
                            print(f"[SM] PRE_GRASP IK err: {_ike} — DONE.", flush=True)
                            next_state = PipelineState.DONE
                    if _pg_cmd is not None and next_state == current_state:
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
                                next_state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                print(f"[SM] PRE_GRASP done (step={state_step}) -> ORIENT",
                                      flush=True)
                                next_state = PipelineState.ORIENT
                                state_step = 0

            elif current_state == PipelineState.ORIENT:
                # 只用 joint6 绕 gripper approach 轴旋转；j1~j5 锁定“进入 ORIENT 时的实际值”。
                # 本版 ORIENT 不再精确复现 AnyGrasp closing 的小竖直倾角，而是：
                #   1) 保持当前实际 approach 方向不变；
                #   2) 将 closing 方向强制设为世界水平（world z 分量 = 0）；
                #   3) 在两个水平解 ±closing 中选择最接近 AnyGrasp 原 closing 的一个。
                # 因而 ORIENT 结束时两根平行手指在理想刚体几何下等高，后续 REACH 只保持该姿态前进。
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None or "R_desired_EE_in_arm" not in grasp_result:
                    next_state = PipelineState.DONE
                else:
                    cur_q = _get_arm_q(jpos)
                    if state_step == 0:
                        from arm_ik_mujoco import (
                            extract_j6_angle,
                            _RX_NEG90 as _RXN90_or,
                            quat_to_rot as _q2r_or_init,
                        )

                        # AnyGrasp 原始目标（gripper_base，arm frame），只用于决定水平 closing 的 ± 方向。
                        _R_gb_anygrasp_or = (grasp_result["R_desired_EE_in_arm"]
                                             @ _RXN90_or)

                        # 以 MuJoCo 当前真实姿态的 approach 作为固定旋转轴。
                        # joint6 只绕该轴转，因此 ORIENT 不改变 approach。
                        _, _R_or_init_w = _gb_pose_world()
                        _approach_w = np.asarray(_R_or_init_w[:, 2], dtype=np.float64)
                        _approach_w /= max(float(np.linalg.norm(_approach_w)), 1e-12)

                        # AnyGrasp 原 closing 转到世界系，用来选择最接近的水平解。
                        _R_robot_or_init = _q2r_or_init(np.asarray(quat_w, dtype=np.float64))
                        _R_anygrasp_w = _R_robot_or_init @ _R_gb_anygrasp_or
                        _closing_any_w = np.asarray(_R_anygrasp_w[:, 1], dtype=np.float64)
                        _closing_any_w /= max(float(np.linalg.norm(_closing_any_w)), 1e-12)

                        # 水平且与 approach 垂直的 closing：c ∝ world_z × approach。
                        # 若 approach 几乎竖直，world_z × approach 退化；此时直接使用
                        # AnyGrasp closing 的水平投影（竖直 approach 下任意水平 closing 都与其正交）。
                        _world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                        _closing_h = np.cross(_world_z, _approach_w)
                        _closing_h_norm = float(np.linalg.norm(_closing_h))
                        if _closing_h_norm < 1e-6:
                            _closing_h = _closing_any_w.copy()
                            _closing_h[2] = 0.0
                            _closing_h_norm = float(np.linalg.norm(_closing_h))
                            if _closing_h_norm < 1e-6:
                                _closing_h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                                _closing_h_norm = 1.0
                        _closing_h /= _closing_h_norm

                        # closing 对平行夹爪有 ± 对称性；取与 AnyGrasp 原方向更接近的一支。
                        if float(np.dot(_closing_h, _closing_any_w)) < 0.0:
                            _closing_h = -_closing_h

                        # 重新正交化，构造合法右手 gripper_base 旋转矩阵：
                        # columns = [binormal(+X), closing(+Y), approach(+Z)].
                        _binormal_h = np.cross(_closing_h, _approach_w)
                        _binormal_h /= max(float(np.linalg.norm(_binormal_h)), 1e-12)
                        _closing_h = np.cross(_approach_w, _binormal_h)
                        _closing_h /= max(float(np.linalg.norm(_closing_h)), 1e-12)
                        _R_gb_level_w = np.column_stack((_binormal_h, _closing_h, _approach_w))
                        _R_gb_desired_or = _R_robot_or_init.T @ _R_gb_level_w

                        # j1~j5 不动，只解 j6；因此实际动作就是绕 approach 轴把双指“调平”。
                        j6_target = extract_j6_angle(cur_q, _R_gb_desired_or)
                        _orient_q_fixed = cur_q.copy()   # 关键：实际姿态，而非 PRE 理论 q
                        _orient_j6_start = float(cur_q[5])

                        _closing_tilt_any = math.degrees(math.asin(float(np.clip(
                            abs(_closing_any_w[2]), 0.0, 1.0))))
                        _closing_tilt_target = math.degrees(math.asin(float(np.clip(
                            abs(_closing_h[2]), 0.0, 1.0))))
                        print(f"[SM] ORIENT level fingers: "
                              f"AnyGrasp closing_tilt={_closing_tilt_any:.2f}deg -> "
                              f"target={_closing_tilt_target:.2f}deg, "
                              f"j6_target={j6_target:.4f} rad "
                              f"({math.degrees(j6_target):.1f} deg), "
                              f"j6_start={_orient_j6_start:.4f} rad", flush=True)

                    _q_cmd_or = _orient_q_fixed.copy()
                    _q_cmd_or[5] = j6_target
                    _arm_step(_q_cmd_or)
                    _gripper_step(close=False)

                    j6_err = abs(float(cur_q[5]) - float(j6_target))
                    if state_step % 25 == 0:
                        print(f"[SM] ORIENT step {state_step}/{BUDGET[PipelineState.ORIENT]}: "
                              f"j6_err={j6_err:.4f} rad", flush=True)
                    state_step += 1

                    _orient_converged = (j6_err < 0.015 and state_step >= 10)
                    _orient_timeout = (state_step >= BUDGET[PipelineState.ORIENT])
                    if _orient_converged or _orient_timeout:
                        if _orient_timeout and not _orient_converged:
                            print(f"[WARN] ORIENT timeout: j6_err={math.degrees(j6_err):.2f}deg "
                                  f"-> ARM_INIT retry", flush=True)
                            grasp_result = None
                            _pg_cmd = None
                            next_state = PipelineState.ARM_INIT
                            state_step = 0
                        else:
                            from arm_ik_mujoco import quat_to_rot as _q2r_or_done
                            _cur_q_orient_done = _get_arm_q(jpos).copy()
                            _, _R_or_actual_w = _gb_pose_world()
                            _R_robot_or = _q2r_or_done(np.asarray(quat_w, dtype=np.float64))
                            _R_gb_desired_w = _R_robot_or @ _R_gb_desired_or

                            # 用 MuJoCo 真值姿态检查，而不是再用 ikpy FK 自己验证自己。
                            # closing axis 对平行夹爪有 ± 方向对称性，因此取绝对点积。
                            _app_dot = float(np.clip(
                                np.dot(_R_or_actual_w[:, 2], _R_gb_desired_w[:, 2]), -1.0, 1.0))
                            _close_dot = float(np.clip(abs(
                                np.dot(_R_or_actual_w[:, 1], _R_gb_desired_w[:, 1])), 0.0, 1.0))
                            _app_err = math.acos(_app_dot)
                            _close_err = math.acos(_close_dot)

                            if _app_err > math.radians(10.0) or _close_err > math.radians(10.0):
                                print(f"[WARN] ORIENT actual pose mismatch: "
                                      f"approach_err={math.degrees(_app_err):.1f}deg "
                                      f"closing_err={math.degrees(_close_err):.1f}deg "
                                      f"-> ARM_INIT retry", flush=True)
                                grasp_result = None
                                _pg_cmd = None
                                next_state = PipelineState.ARM_INIT
                                state_step = 0
                            else:
                                grasp_result["target_angles_orient"] = _cur_q_orient_done.copy()
                                _closing_actual_tilt = math.degrees(math.asin(float(np.clip(
                                    abs(_R_or_actual_w[2, 1]), 0.0, 1.0))))
                                print(f"[SM] ORIENT done: j6={_cur_q_orient_done[5]:.4f} "
                                      f"(target={j6_target:.4f}), "
                                      f"approach_err={math.degrees(_app_err):.2f}deg, "
                                      f"closing_err={math.degrees(_close_err):.2f}deg, "
                                      f"actual_closing_tilt={_closing_actual_tilt:.2f}deg -> REACH",
                                      flush=True)
                                next_state = PipelineState.REACH
                                state_step = 0
                                reach_ctx.reset()

            elif current_state == PipelineState.REACH:
                # 沿 approach 轴前进 0.06m 到接触点（对齐原版逻辑）
                cmd_vx = 0.0; cmd_vy = 0.0; cmd_wz = 0.0
                if grasp_result is None or "pre_t_gb_arm" not in grasp_result:
                    next_state = PipelineState.DONE
                else:
                    if reach_ctx.target_cur is None:
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
                            REACH_ADVANCE   = 0.115
                            TABLE_CLEARANCE = 0.001   # 指尖/指板最低点距桌面的最小高度(m)
                            _re_pre_t_gb = grasp_result["pre_t_gb_arm"]
                            _re_approach = grasp_result["approach_arm"]
                            _re_t_gb_arm = (_re_pre_t_gb
                                            + REACH_ADVANCE * _re_approach)
                            # ── 闭环纠偏状态（REACH 结束时用 gb 实测误差反推命令，见 done 分支）──
                            reach_ctx.target_gb_arm0 = _re_t_gb_arm.copy()  # 原始(clip 后)命令目标，纠偏基准
                            _tc_need = None                      # 指尖最低允许高度（仅本次初始化使用）
                            # ── 桌面 clip: 手不得插进桌面（沿 approach 自动收缩前进量）──
                            # 指尖目标 = 掌心 + (depth + s)·a：depth < 0.1358 时指尖会落到
                            # 掌心之下较浅处，若物体矮（苹果）则该目标可能低于桌面 → 手指压桌、
                            # 机器狗对抗、抓空。这里按桌面高度收缩 advance 兜底。
                            # 桌面高度直接使用 GRASP_PLAN 缓存在 grasp_result 中的世界系点云，
                            # 用指尖 XY 邻域点 z 的**中位数**估计
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
                                _tc_pts_mem = grasp_result.get("table_cloud_world")
                                if _tc_apw[2] < -0.05 and _tc_pts_mem is not None:
                                    _tc_pts = np.asarray(_tc_pts_mem, dtype=np.float64)
                                    _tc_dxy = np.linalg.norm(_tc_pts[:, :2] - _tc_tip_w[:2], axis=1)
                                    for _tc_r in (0.03, 0.05, 0.10):
                                        if int((_tc_dxy < _tc_r).sum()) >= 20:
                                            reach_ctx.table_height = float(np.median(_tc_pts[_tc_dxy < _tc_r, 2]))
                                            break
                                if reach_ctx.table_height is None:
                                    print("[WARN] 桌面 clip: 桌面点云不足或 approach 非向下 -> 不 clip",
                                          flush=True)
                                else:
                                    # 指尖需停在 need_eff 以上；指板外角再低 35.1mm×sin(倾角)
                                    _tc_need = (reach_ctx.table_height + TABLE_CLEARANCE
                                                + 0.0351 * float(np.hypot(_tc_apw[0], _tc_apw[1])))
                                    _tc_dmax = ((_tc_tip_w[2] - _tc_need) / (-_tc_apw[2]))
                                    if 0.0 < _tc_dmax < REACH_ADVANCE:
                                        print(f"[SM] REACH clip: 桌面z={reach_ctx.table_height:.4f} "
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
                            reach_ctx.target_cur = np.asarray(
                                _rq, dtype=np.float64).copy()
                            print(f"[SM] REACH IK={np.round(reach_ctx.target_cur,3)}",
                                  flush=True)
                            # 打印 REACH 目标点（gripper_base 原点）的世界坐标
                            try:
                                from arm_ik_mujoco import fk_gripper as _fk_re_diag, quat_to_rot as _q2r_re
                                _re_R_rob   = _q2r_re(grasp_result["quat_w_scan"])
                                _re_pos_rob = grasp_result["pos_w_scan"]
                                _re_arm_base_w = _re_pos_rob + _re_R_rob @ np.array([0.0, 0.0, 0.0888])
                                _re_T_arm = _fk_re_diag(reach_ctx.target_cur)
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
                            next_state = PipelineState.DONE
                    if reach_ctx.target_cur is not None \
                            and next_state == current_state:
                        from arm_ik_mujoco import _IK_JOINT_LIMITS as _re_lims2
                        cur_q = _get_arm_q(jpos)
                        # 速度限制斜坡：每步最多移动 _PG_MAX_DELTA，避免瞬移穿模
                        _re_delta = np.clip(
                            reach_ctx.target_cur - cur_q,
                            -_PG_MAX_DELTA, +_PG_MAX_DELTA)
                        _q6_re = np.clip(
                            cur_q + _re_delta,
                            [lo for lo, hi in _re_lims2],
                            [hi for lo, hi in _re_lims2])
                        _arm_step(_q6_re)
                        _gripper_step(close=False)
                        if state_step % 50 == 0:
                            _re_err = np.abs(cur_q - reach_ctx.target_cur)
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
                            if reach_ctx.corr_iter < 3:
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
                                    _cc_gb_ana  = _cc_R @ reach_ctx.target_gb_arm0 + _cc_base
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
                                        reach_ctx.corr_iter = 3   # 之后不再纠偏
                                    elif reach_ctx.prev_actual_gb_world is not None and \
                                            float(np.linalg.norm(_gb_act_c - reach_ctx.prev_actual_gb_world)) \
                                            < 0.4 * float(np.linalg.norm(reach_ctx.prev_cmd_move_world)):
                                        # 实际没跟随命令（硬顶物体/到限位）：再纠只会把
                                        # 纠正量越积越大、持续加压 → 回退到纠偏前的目标并停止。
                                        # 判据 = "实测 gb 位移 vs 当轮命令位移"：跟随与不跟随
                                        # 时 |e| 都可能≈原值（第一轮实测 38.8 -> 47.5mm 两者
                                        # 都判不出来），只有位移有区分度。
                                        print(f"[WARN] REACH corr: 实际未跟随命令 "
                                              f"(实测 gb 位移 "
                                              f"{float(np.linalg.norm(_gb_act_c - reach_ctx.prev_actual_gb_world)) * 1000:.1f}mm"
                                              f" < 命令位移 {float(np.linalg.norm(reach_ctx.prev_cmd_move_world)) * 1000:.1f}mm"
                                              f" 的 40%) -> 回退并停止纠偏", flush=True)
                                        reach_ctx.corr_iter = 3
                                        if reach_ctx.prev_target_q is not None:
                                            reach_ctx.target_cur = reach_ctx.prev_target_q.copy()
                                            state_step        = 0
                                            _re_corr_done     = True
                                    elif 0.005 < _cc_en < 0.05:
                                        # 单轮限幅 ≤15mm：一次纠太多 PD 追不上/直接撞物体，
                                        # 但总量不封顶（3 轮内仍可累到完整补偿）。
                                        _cc_step = -_cc_e_w
                                        if _cc_en > 0.015:
                                            _cc_step = _cc_step * (0.015 / _cc_en)
                                            print(f"[SM] REACH corr 限幅: 单轮移动 "
                                                  f"{_cc_en * 1000:.1f} -> 15.0mm", flush=True)
                                        _cc_corr = reach_ctx.corr_world + _cc_step
                                        _cc_gb_cmd_arm = reach_ctx.target_gb_arm0 + _cc_R.T @ _cc_corr
                                        _cc_cur_q = _get_arm_q(jpos).copy()
                                        # 局部增量（纯平移，j6 不动）：不重建朝向，
                                        # 因此不依赖 IK 种子/滚动角，也不会跳到另一分支
                                        _cc_q, _cc_res, _cc_dq = _dls_gb_step(_cc_cur_q,
                                                                              _cc_gb_cmd_arm)
                                        # ── 指板地板（第九轮）：候选臂角的**真实指板最低点**
                                        # 不得低于"桌面 + TABLE_CLEARANCE"（与桌面 clip 同一
                                        # 标准，只是换成精确几何）→ 低于就沿世界 z 抬差额重算。
                                        # 旧公式（按 |e| 猜"预测实际"再反推命令）已删：静差
                                        # 翻号时它把指板放到了桌面下（第一轮 0.6525 < 0.6613）。
                                        if reach_ctx.table_height is not None and _cc_res <= 0.010 and _cc_dq <= 0.35:
                                            try:
                                                _cc_floor_z = float(reach_ctx.table_height) + TABLE_CLEARANCE
                                                for _ in range(2):
                                                    _cc_padz = _pad_min_z_fk(_cc_q)
                                                    _cc_dz   = _cc_floor_z - _cc_padz
                                                    if _cc_dz <= 1e-4:
                                                        break
                                                    print(f"[SM] REACH corr 指板地板: 指板最低点 "
                                                          f"{_cc_padz:.4f} < 桌面+"
                                                          f"{TABLE_CLEARANCE * 1000:.0f}mm "
                                                          f"{_cc_floor_z:.4f} -> 沿世界 z 抬 "
                                                          f"{_cc_dz * 1000:.1f}mm", flush=True)
                                                    _cc_corr = _cc_corr + np.array([0., 0., _cc_dz])
                                                    _cc_gb_cmd_arm = (reach_ctx.target_gb_arm0
                                                                      + _cc_R.T @ _cc_corr)
                                                    _cc_q, _cc_res, _cc_dq = _dls_gb_step(
                                                        _cc_cur_q, _cc_gb_cmd_arm)
                                            except Exception as _cc_fe:
                                                print(f"[WARN] REACH corr 指板地板 FK 失败: "
                                                      f"{_cc_fe}", flush=True)
                                        if _cc_res > 0.010 or _cc_dq > 0.35:
                                            print(f"[WARN] REACH corr: 局部增量不可用 "
                                                  f"(残差 {_cc_res * 1000:.1f}mm, 单关节 "
                                                  f"{np.degrees(_cc_dq):.1f}deg) -> 放弃纠偏",
                                                  flush=True)
                                        else:
                                            reach_ctx.prev_target_q        = reach_ctx.target_cur.copy()  # 回退点
                                            reach_ctx.corr_iter    += 1
                                            reach_ctx.prev_actual_gb_world      = _gb_act_c.copy()       # 下轮判跟随用
                                            reach_ctx.prev_cmd_move_world      = _cc_corr - reach_ctx.corr_world  # 本轮命令位移
                                            reach_ctx.corr_world        = _cc_corr
                                            reach_ctx.target_cur = _cc_q.copy()
                                            state_step        = 0
                                            _re_corr_done     = True
                                            print(f"[SM] REACH corr iter={reach_ctx.corr_iter} "
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
                                _diag_print_reach_done(
                                    env, args.object, jpos, _ARM_IDX, reach_ctx.target_cur)
                                next_state = PipelineState.CLOSE
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
                          f"双指都曾接触后切换恒力夹持 ±1.0N", flush=True)

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
                        "(joint7=-1.00N, joint8=+1.00N; no center lock)",
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
                )
                state_step = 0
                _pg_cmd = None

    except KeyboardInterrupt:
        print("[MJ] KeyboardInterrupt")
    finally:
        bridge.stop()
        env.close()
        print("[MJ] Done")


if __name__ == "__main__":
    main()