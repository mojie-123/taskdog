#!/usr/bin/env python
"""grasp_stability_test.py — 抓取稳定性 A/B 测试 harness（不依赖 GraspNet / ROS2 / policy）

为什么单独写这个：真实管线每次试验都要跑导航 + AnyGrasp 推理（几十秒且结果有随机性），
无法做「同一抓取点、改一处物理参数」的单变量对比。本 harness 只用解析几何给出
自上而下的抓取目标，跳过感知，直接复现 REACH -> CLOSE -> LIFT 的物理接触过程，
用于对比 scene.xml 物理参数与 CLOSE/LIFT 时序改动前后的抓取稳定性。

几何（与管线一致的部分）：
  - 机械臂 arm_base 系与机器人 base 系同向，仅平移 +0.0888m（见 navigate_mujoco.py
    REACH 段的 _re_arm_base_w 计算）
  - approach 方向取世界 -Z（自上而下）；gb +Z 对齐 approach
  - gb 目标 = obj_arm + (0,0,z_grasp) - R_gb @ [0,0,0.1358]
    => j7（指尖原点）落在 obj_arm + (0,0,z_grasp)

复用的真实常量：ARM_SIDE_ANGLES / ARM_PREGRASP_ANGLES / GRIPPER_*_POS /
_PG_MAX_DELTA / _ARM_IDX，以及 arm_ik_mujoco 的 solve_for_gripper_base。

CLOSE 段是本文件自己实现的两段式冻结（首触轻贴 0N → 两指都接触后同刻加深 28mm
→ 5.6N/指，末 15 步单侧兜底）——**与 navigate_mujoco.py 的 CLOSE 分支（L1370-1453）
+ LIFT 沿用冻结目标（L1464-1466）保持同步，改了那边必须改这边**，否则 harness 测的
就不是真实管线行为。

用法：
  python custom_envs/scripts/navigation/grasp_stability_test.py \\
      --object apple --trials 5 --close_steps 100 --lift_steps 200 --lift_ramp 150

  # CLOSE 两段式专用测试（4cm 方块放 (4.9, 6.0)，两夹持面严格平行 → 两指应同刻接触）
  python custom_envs/scripts/navigation/grasp_stability_test.py --object cube --trials 5
"""

import argparse
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "utils")))

import numpy as np

from custom_envs.mujoco.env import MuJoCoEnv
from arm_ik_mujoco import (
    solve_for_gripper_base,
    fk_gripper,
    fk,
    _IK_JOINT_LIMITS,
    _RX_POS90,
    _J7_ORIGIN_IN_GB,
    quat_to_rot,
)

# ── 与 navigate_mujoco.py 保持一致的常量 ──
ARM_SIDE_ANGLES     = np.array([-math.pi / 2, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float64)
ARM_PREGRASP_ANGLES = np.array([-math.pi / 2, 1.5, -1.5, 0.0, 0.0, 0.0], dtype=np.float64)
GRIPPER_OPEN_POS    = np.array([0.035, -0.035], dtype=np.float64)
GRIPPER_CLOSE_POS   = np.array([0.000, 0.000], dtype=np.float64)
_ARM_IDX            = np.array([4, 9, 14, 19, 20, 21], dtype=np.int32)
_PG_MAX_DELTA       = np.array([0.03, 0.05, 0.05, 0.04, 0.04, 0.04], dtype=np.float64)
ARM_BASE_OFFSET_Z   = 0.0888
ARM_INIT_BUDGET     = 300
REACH_BUDGET        = 300
SUCCESS_Z           = 0.75          # 与管线 LIFT 判定一致

# scene.xml 中物体的默认位姿（body pos）
DEFAULT_OBJ_POS = {
    "apple":  (4.9, 4.5, 0.6623),
    "banana": (4.9, 5.0, 0.6800),
    # bowl 已替换为圆柱 r=2.5cm h=6cm：z = 桌面 0.6613 + 半高 0.03 = 0.6913
    "bowl":   (4.9, 5.5, 0.6913),
    "cube":   (4.9, 6.0, 0.6813),
}
# CLOSE 两段式冻结的加深量：必须与 navigate_mujoco.py 的 _CLOSE_FREEZE_S 保持同步
_CLOSE_FREEZE_S = 0.028
_CLOSE_SINGLE_FALLBACK_STEP = 85     # 与管线一致：最后 15 步仍单侧接触 → 兜底加深
FINGER_BODIES = ("link7", "link8")


def _yaw_quat(yaw):
    """yaw -> 四元数 (w,x,y,z)"""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def _build_topdown_R_gb():
    """自上而下抓取的 gb 目标旋转（arm_base 系）。

    列约定（与 compute_desired_ee_rot_in_arm 一致）：
      col2 = gb +Z = approach 方向 = 世界 -Z
      col1 = gb +Y = 夹爪闭合方向（joint7/8 沿 gb ∓Y 滑动）= 取 arm 系 +X
      col0 = col1 x col2
    """
    col2 = np.array([0.0, 0.0, -1.0])
    col1 = np.array([1.0, 0.0, 0.0])
    col0 = np.cross(col1, col2)
    return np.column_stack([col0, col1, col2])


class GraspHarness:
    def __init__(self, args):
        self.args = args
        self.env = MuJoCoEnv(
            robot_init_pos=(args.robot_pos[0], args.robot_pos[1], 0.58),
            render=args.render,
        )
        self.model = self.env.model
        self.data = self.env.data
        import mujoco
        self.mj = mujoco
        # body 名 -> id
        self._body_id = {}
        for nm in list(FINGER_BODIES) + [args.object]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
            self._body_id[nm] = bid
        self._obj_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              "{}_freejoint".format(args.object))
        ]
        self._freeze_pos = np.array([args.robot_pos[0], args.robot_pos[1], 0.58],
                                    dtype=np.float64)
        self._freeze_quat = _yaw_quat(args.robot_yaw)

    # ── 基础控制 ──
    def _arm_q(self):
        return self.env.get_robot_state()["joint_pos"][_ARM_IDX].copy()

    def _step(self, arm_target=None, grip_target=None):
        if arm_target is not None:
            self.env.set_arm_target(np.asarray(arm_target, dtype=np.float64))
        if grip_target is not None:
            self.env.set_gripper_target(np.asarray(grip_target, dtype=np.float64))
        self.env.step(np.zeros(16, dtype=np.float32))
        # FREEZE：与管线一致（env.step 之后强制恢复根位姿）
        self.env.set_root_pose(self._freeze_pos, self._freeze_quat)

    # ── 接触统计 ──
    def contact_stats(self):
        """返回 (手指数, 物体接触对数, 最大接触法向力)"""
        n_touch, max_fn, _ = self._contact_scan()
        return n_touch, max_fn

    def _contact_scan(self):
        """返回 (接触对数, 最大法向力, {link7: bool, link8: bool})。

        按指判定与 navigate_mujoco.py 的 _fingers_contact 同一套逻辑：扫 d.contact
        全部接触对，某条接触对一边是物体、另一边是 link7/link8 即判该指接触（纯几何
        接触，无力度阈值——这是"首触轻贴"能成立的前提）。
        """
        mj, m, d = self.mj, self.model, self.data
        finger_gids = {}
        for nm in FINGER_BODIES:
            bid = self._body_id[nm]
            gids = set()
            if bid >= 0:
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == bid:
                        gids.add(g)
            finger_gids[nm] = gids
        obj_bid = self._body_id[self.args.object]
        obj_gids = {g for g in range(m.ngeom) if m.geom_bodyid[g] == obj_bid}

        touch = {nm: False for nm in FINGER_BODIES}
        n_touch = 0
        max_fn = 0.0
        res = np.zeros(6, dtype=np.float64)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            if (pair & obj_gids):
                for nm in FINGER_BODIES:
                    if pair & finger_gids[nm]:
                        touch[nm] = True
                if pair & (finger_gids["link7"] | finger_gids["link8"]):
                    n_touch += 1
                    mj.mj_contactForce(m, d, i, res)
                    max_fn = max(max_fn, abs(float(res[0])))
        return n_touch, max_fn, touch

    def grip_q(self):
        """当前 joint7/joint8 关节角（m）——直线滑块关节，目标也是米。"""
        return self.env.get_robot_state()["joint_pos"][self.env._grip_all_idx].copy()

    def obj_state(self):
        a = self._obj_qposadr
        pos = self.data.qpos[a:a + 3].copy()
        quat = self.data.qpos[a + 3:a + 7].copy()
        return pos, quat

    # ── 单次试验 ──
    def run_trial(self, trial_idx, z_grasp, close_steps, lift_steps, lift_ramp,
                  grip_delta=None, verbose=True):
        args = self.args
        env = self.env
        if grip_delta is not None:
            env._grip_max_delta = float(grip_delta)

        env.reset()
        # 设置物体初始位姿（允许 --obj_pos 覆盖）
        a = self._obj_qposadr
        self.data.qpos[a:a + 3] = np.array(args.obj_pos, dtype=np.float64)
        env.set_root_pose(self._freeze_pos, self._freeze_quat)
        self.mj.mj_forward(self.model, self.data)

        obj_pos0, obj_quat0 = self.obj_state()
        R0 = quat_to_rot(obj_quat0)
        log = {"trial": trial_idx, "obj_pos0": obj_pos0.copy()}

        # ── 阶段 0：ARM_INIT，插值到 ARM_SIDE_ANGLES ──
        t0 = time.time()
        arm_steps = 0
        for s in range(ARM_INIT_BUDGET):
            cur_q = self._arm_q()
            alpha = min(1.3, s / max(ARM_INIT_BUDGET * 0.6, 1.0))
            self._step(cur_q + alpha * (ARM_SIDE_ANGLES - cur_q), GRIPPER_OPEN_POS)
            arm_steps += 1
            if np.abs(self._arm_q() - ARM_SIDE_ANGLES).max() < 0.02:
                break
        log["arm_init_steps"] = arm_steps

        # ── 阶段 1：IK 求解（自上而下抓取） ──
        R_gb = _build_topdown_R_gb()
        R_rob = quat_to_rot(self._freeze_quat)
        arm_base_w = self._freeze_pos + R_rob @ np.array([0.0, 0.0, ARM_BASE_OFFSET_Z])
        obj_arm = R_rob.T @ (obj_pos0 - arm_base_w)
        target_gb_arm = obj_arm + np.array([0.0, 0.0, z_grasp]) \
            - R_gb @ _J7_ORIGIN_IN_GB
        target_rot_j7 = R_gb @ _RX_POS90

        q_ik = solve_for_gripper_base(
            target_gb_arm, target_rot_j7=target_rot_j7,
            initial_angles=ARM_PREGRASP_ANGLES.copy())
        if q_ik is None or np.any(np.isnan(q_ik)):
            print("[HARNESS] IK failed -> trial aborted", flush=True)
            return None
        q_ik = np.asarray(q_ik, dtype=np.float64).copy()
        # j6 钉回 ARM_SIDE（harness 无 ORIENT 阶段，用 side 值保证可复现）
        q_ik[5] = float(np.clip(ARM_SIDE_ANGLES[5], *_IK_JOINT_LIMITS[5]))
        gb_err = float(np.linalg.norm(fk_gripper(q_ik)[:3, 3] - target_gb_arm))
        log["ik_q"] = q_ik.copy()
        log["ik_gb_err"] = gb_err
        if verbose:
            print(f"[HARNESS] trial {trial_idx}: IK q={np.round(q_ik, 3)} "
                  f"gb_err={gb_err * 1000:.1f}mm target_gb={np.round(target_gb_arm, 4)}",
                  flush=True)

        # ── 阶段 2：REACH（速度限幅斜坡） ──
        reach_steps = 0
        for s in range(REACH_BUDGET):
            cur_q = self._arm_q()
            delta = np.clip(q_ik - cur_q, -_PG_MAX_DELTA, _PG_MAX_DELTA)
            q_cmd = np.clip(cur_q + delta,
                            [lo for lo, _ in _IK_JOINT_LIMITS],
                            [hi for _, hi in _IK_JOINT_LIMITS])
            self._step(q_cmd, GRIPPER_OPEN_POS)
            reach_steps += 1
            if np.abs(self._arm_q() - q_ik).max() < 0.02:
                break
        log["reach_steps"] = reach_steps
        j7_w = R_rob @ fk(q_ik)[:3, 3] + arm_base_w
        log["j7_actual_w"] = j7_w.copy()
        log["reach_pos_err"] = float(np.linalg.norm(j7_w - (obj_pos0 + np.array([0.0, 0.0, z_grasp]))))
        if verbose:
            print(f"[HARNESS]   REACH {reach_steps} steps, "
                  f"j7-obj 误差={log['reach_pos_err'] * 1000:.1f}mm", flush=True)

        # ── 阶段 3：CLOSE（两段式冻结，镜像 navigate_mujoco.py L1370-1453）──
        # 第一段：某指首次接触 → 该指目标冻结在接触时 q 本身（PD 误差≈0 → 挤压力≈0N），
        #   未接触指继续盲闭 0.0；
        # 第二段：两指都接触 → 同一步两侧同时加深 _CLOSE_FREEZE_S（对称挤压，净横向力≈0）；
        # 兜底：最后 15 步仍单侧 → 已接触侧加深 S（速率限制 1mm/步，太晚触发压不出力）。
        # ⚠ 本段必须与 navigate_mujoco.py 的 CLOSE 分支保持同步，改了那边要改这边。
        n_touch_trace, fn_trace = [], []
        cl_frz = [None, None]        # [joint7, joint8] 冻结目标
        cl_qc = [None, None]         # 冻结时记录的 q7/q8
        cl_sq = [False]              # 第二段是否已触发
        frz_steps = [None, None]     # 各指首触的步号（DIAG）
        sq_step = [None]
        for s in range(close_steps):
            _, _, tc = self._contact_scan()
            gq = self.grip_q()
            for k, nm in enumerate(FINGER_BODIES):      # link7 -> 0, link8 -> 1
                if cl_frz[k] is None and tc[nm]:
                    cl_frz[k] = float(gq[k])
                    cl_qc[k] = float(gq[k])
                    frz_steps[k] = s
                    if verbose:
                        print(f"[HARNESS]   CLOSE: {nm} 首触 @step{s} q{gq[k]:+.4f} "
                              f"→ 轻贴冻结", flush=True)
            if (not cl_sq[0]) and cl_frz[0] is not None and cl_frz[1] is not None:
                cl_frz[0] = cl_qc[0] - _CLOSE_FREEZE_S
                cl_frz[1] = cl_qc[1] + _CLOSE_FREEZE_S
                cl_sq[0] = True
                sq_step[0] = s
                if verbose:
                    print(f"[HARNESS]   CLOSE: 两指都已接触 @step{s} → 同时加深 "
                          f"S={_CLOSE_FREEZE_S * 1000:.0f}mm → 目标 "
                          f"[{cl_frz[0]:.4f}, {cl_frz[1]:.4f}] "
                          f"= {200.0 * _CLOSE_FREEZE_S:.1f}N/指", flush=True)
            if (not cl_sq[0]) and s >= _CLOSE_SINGLE_FALLBACK_STEP and \
                    (cl_frz[0] is not None or cl_frz[1] is not None):
                if cl_frz[0] is not None:
                    cl_frz[0] = cl_qc[0] - _CLOSE_FREEZE_S
                if cl_frz[1] is not None:
                    cl_frz[1] = cl_qc[1] + _CLOSE_FREEZE_S
                cl_sq[0] = True
                sq_step[0] = s
                if verbose:
                    print(f"[HARNESS]   [WARN] CLOSE: 只有单侧接触 → 兜底单侧压 "
                          f"S={_CLOSE_FREEZE_S * 1000:.0f}mm", flush=True)
            grip_tgt = np.array([cl_frz[0] if cl_frz[0] is not None else 0.0,
                                 cl_frz[1] if cl_frz[1] is not None else 0.0],
                                dtype=np.float64)
            self._step(q_ik, grip_tgt)
            nt, mf = self.contact_stats()
            n_touch_trace.append(nt)
            fn_trace.append(mf)

        log["close_steps"] = close_steps
        log["close_contacts_end"] = n_touch_trace[-1]
        log["close_max_fn"] = max(fn_trace) if fn_trace else 0.0
        log["close_contact_step"] = list(frz_steps)        # [link7, link8] 首触步
        log["close_squeeze_step"] = sq_step[0]
        log["close_frz_target"] = [cl_frz[0], cl_frz[1]]
        log["close_two_stage"] = bool(cl_frz[0] is not None and cl_frz[1] is not None
                                      and frz_steps[0] is not None
                                      and frz_steps[1] is not None
                                      and cl_qc[0] is not None and cl_qc[1] is not None
                                      # 两指都靠"首触轻贴"进入，而非兜底单侧压
                                      and sq_step[0] is not None
                                      and sq_step[0] < _CLOSE_SINGLE_FALLBACK_STEP)
        log["close_step_gap"] = (abs(frz_steps[0] - frz_steps[1])
                                 if None not in frz_steps else None)
        pos_c, _ = self.obj_state()
        log["pos_after_close"] = pos_c.copy()
        log["grip_q_after_close"] = self.grip_q()

        # ── 阶段 4：LIFT（沿用 CLOSE 冻结目标，镜像管线 L1464-1466）──
        lift_fn, lift_nt = [], []
        lift_grip = np.array([cl_frz[0] if cl_frz[0] is not None else 0.0,
                              cl_frz[1] if cl_frz[1] is not None else 0.0],
                             dtype=np.float64)
        for s in range(lift_steps):
            cur_q = self._arm_q()
            alpha = min(1.3, s / max(float(lift_ramp), 1.0))
            self._step(cur_q + alpha * (ARM_SIDE_ANGLES - cur_q), lift_grip)
            nt, mf = self.contact_stats()
            lift_nt.append(nt)
            lift_fn.append(mf)
        log["lift_steps"] = lift_steps
        log["lift_contacts_end"] = lift_nt[-1] if lift_nt else 0
        log["lift_max_fn"] = max(lift_fn) if lift_fn else 0.0

        pos_f, quat_f = self.obj_state()
        Rf = quat_to_rot(quat_f)
        log["pos_final"] = pos_f.copy()
        log["final_z"] = float(pos_f[2])
        log["delta_xy"] = float(np.linalg.norm(pos_f[:2] - obj_pos0[:2]))
        # 翻滚角：物体自身 +Z 相对初始 +Z 的夹角（apple 滚动报警指标，度）
        cos_t = float(np.clip(R0[:, 2] @ Rf[:, 2], -1.0, 1.0))
        log["tilt_deg"] = float(np.degrees(np.arccos(cos_t)))
        log["success"] = bool(pos_f[2] > SUCCESS_Z)
        log["wall_s"] = time.time() - t0
        log["total_steps"] = arm_steps + reach_steps + close_steps + lift_steps
        log["step_ms"] = log["wall_s"] / max(log["total_steps"], 1) * 1000.0

        if verbose:
            print(f"[HARNESS]   CLOSE端接触={log['close_contacts_end']} "
                  f"maxFn={log['close_max_fn']:.2f}N | "
                  f"LIFT端接触={log['lift_contacts_end']} maxFn={log['lift_max_fn']:.2f}N",
                  flush=True)
            print(f"[HARNESS]   CLOSE: 首触步={log['close_contact_step']} "
                  f"步差={log['close_step_gap']} 加深步={log['close_squeeze_step']} "
                  f"两段式={'OK' if log['close_two_stage'] else '未走通'} | "
                  f"CLOSE后位移={np.linalg.norm(log['pos_after_close'][:2] - obj_pos0[:2]) * 1000:.2f}mm",
                  flush=True)
            print(f"[HARNESS]   final_z={log['final_z']:.4f} tilt={log['tilt_deg']:.1f}deg "
                  f"dxy={log['delta_xy'] * 1000:.1f}mm "
                  f"success={'YES' if log['success'] else 'NO'} "
                  f"({log['step_ms']:.2f} ms/step)", flush=True)
        return log

    def close(self):
        self.env.close()


def main():
    ap = argparse.ArgumentParser(description="抓取稳定性 A/B harness")
    ap.add_argument("--object", default="apple", choices=list(DEFAULT_OBJ_POS.keys()))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--obj_pos", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--robot_pos", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="抓取时机器人 base 世界坐标；默认 (obj_x - standoff, obj_y)，"
                         "即机器人与物体同 y、站在物体 -X 侧 standoff 米处 "
                         "(镜像管线 PAN_VX 对齐 y + SNAP 后的实际抓取站位)")
    ap.add_argument("--standoff", type=float, default=0.4,
                    help="机器人 base 与物体在 x 方向的间距（m）。0.4 时自上而下 IK "
                         "精确可解(gb_err=0)且解落在 j1=-π/2 分支，与 ARM_SIDE/PREGRASP 种子一致")
    ap.add_argument("--robot_yaw", type=float, default=math.pi / 2)
    ap.add_argument("--z_grasp", type=float, default=0.0,
                    help="j7 目标相对物体中心的高度偏移（m）")
    ap.add_argument("--close_steps", type=int, default=100)
    ap.add_argument("--lift_steps", type=int, default=200)
    ap.add_argument("--lift_ramp", type=float, default=150.0)
    ap.add_argument("--grip_delta", type=float, default=None,
                    help="覆盖 env._grip_max_delta（默认用 env 内部的 0.001）")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--repeat_identical", action="store_true",
                    help="每个 trial 之间不扰动（默认同样不扰动，保留以便扩展）")
    args = ap.parse_args()

    if args.obj_pos is None:
        args.obj_pos = list(DEFAULT_OBJ_POS[args.object])
    if args.robot_pos is None:
        args.robot_pos = (args.obj_pos[0] - args.standoff, args.obj_pos[1])

    h = GraspHarness(args)
    print(f"[HARNESS] object={args.object} obj_pos={np.round(args.obj_pos, 4)} "
          f"robot={args.robot_pos} yaw={args.robot_yaw:.4f} "
          f"close={args.close_steps} lift={args.lift_steps} ramp={args.lift_ramp}",
          flush=True)

    logs = []
    try:
        for i in range(args.trials):
            lg = h.run_trial(i, args.z_grasp, args.close_steps, args.lift_steps,
                             args.lift_ramp, grip_delta=args.grip_delta)
            if lg is not None:
                logs.append(lg)
    finally:
        h.close()

    if not logs:
        print("[HARNESS] no valid trial", flush=True)
        return 1

    n = len(logs)
    succ = sum(1 for l in logs if l["success"])
    print("\n=== SUMMARY object={} trials={} ===".format(args.object, n), flush=True)
    print("success      : {}/{} = {:.0f}%".format(succ, n, 100.0 * succ / n), flush=True)
    for key, fmt in (("final_z", "{:.4f}"), ("tilt_deg", "{:.1f}"),
                     ("delta_xy", "{:.4f}"), ("close_contacts_end", "{:.1f}"),
                     ("lift_contacts_end", "{:.1f}"), ("close_max_fn", "{:.2f}"),
                     ("lift_max_fn", "{:.2f}"), ("reach_pos_err", "{:.4f}"),
                     ("step_ms", "{:.2f}"),
                     ("close_step_gap", "{:.1f}"), ("close_squeeze_step", "{:.1f}")):
        vals = np.array([l[key] for l in logs], dtype=np.float64)
        print("  {:<18s} mean={} min={} max={}".format(
            key, fmt.format(vals.mean()), fmt.format(vals.min()), fmt.format(vals.max())),
            flush=True)
    import json
    print("\n[JSON] " + json.dumps([
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in l.items()}
        for l in logs]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
