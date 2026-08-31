#!/usr/bin/env python3
"""grasp_vis.py — 可视化 GraspNet 输出的抓取候选解。

读取:
  /tmp/nav_run.log         — 相机世界坐标、香蕉世界坐标、机器人位姿、扫描时关节角
  /tmp/grasp_result.npz   — GraspNet 候选抓取（相机坐标系）

可视化:
  黄点  — 香蕉世界坐标
  红点  — 相机世界坐标
  蓝箭头 — 夹爪中心（尾）+ 接近方向（头，长5cm）
  空格  — 切换下一组解（置信度从高到低）
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "custom_envs", "utils"))

from arm_ik import (
    cam_to_world, quat_to_rot, fk_gripper, _CAM_OFFSET_ROT,
    solve_for_gripper_base, compute_desired_ee_rot_in_arm,
    world_pos_to_arm_frame, get_chain,
)
ARM_BASE_OFFSET = np.array([0.0, 0.0, 0.0888])  # base_link -> arm_base_link


# ---------------------------------------------------------------------------
# 1. Log 解析
# ---------------------------------------------------------------------------

def parse_log(log_path):
    cam_world = banana_world = pos_w_scan = quat_w_scan = q_scan = None
    v3 = r'\[\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\]'
    v4 = r'\[\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\]'
    np_ = r'[\-\+]?[\d.eE+\-]+'
    v6 = r'\[\s*(' + r')\s+('.join([np_] * 6) + r')\s*\]'

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[DIAG-CAM] Code  cam world' in line:
                m = re.search(v3, line)
                if m: cam_world = np.array([float(m.group(i)) for i in range(1, 4)])
            elif '[DIAG-CAM] Banana world pos' in line:
                m = re.search(v3, line)
                if m: banana_world = np.array([float(m.group(i)) for i in range(1, 4)])
            elif 'ALIGN_YAW freeze anchor set' in line:
                mp = re.search(r'pos=' + v3, line)
                mq = re.search(r'quat=' + v4, line)
                if mp: pos_w_scan  = np.array([float(mp.group(i)) for i in range(1, 4)])
                if mq: quat_w_scan = np.array([float(mq.group(i)) for i in range(1, 5)])
            elif 'cur_q at SCAN' in line:
                m = re.search(v6, line)
                if m: q_scan = np.array([float(m.group(i)) for i in range(1, 7)])

    return cam_world, banana_world, pos_w_scan, quat_w_scan, q_scan


# ---------------------------------------------------------------------------
# 2. 坐标变换
# ---------------------------------------------------------------------------

def transform_grasps(translations, rotations, q_scan, pos_w_scan, quat_w_scan):
    K = len(translations)
    centers_world    = np.zeros((K, 3))
    approaches_world = np.zeros((K, 3))

    T_gb = fk_gripper(q_scan.astype(np.float64))
    R_gb = T_gb[:3, :3]
    R_robot = quat_to_rot(quat_w_scan.astype(np.float64))
    R_cam2world = R_robot @ R_gb @ _CAM_OFFSET_ROT

    for i in range(K):
        centers_world[i] = cam_to_world(
            translations[i].astype(np.float64),
            q_scan.astype(np.float64),
            pos_w_scan.astype(np.float64),
            quat_w_scan.astype(np.float64),
        )
        app = rotations[i, :, 0].astype(np.float64)
        app_w = R_cam2world @ app
        n = np.linalg.norm(app_w)
        if n > 1e-6:
            app_w /= n
        approaches_world[i] = app_w

    return centers_world, approaches_world


# ---------------------------------------------------------------------------
# 2b. IK 求解 + FK 关节位置
# ---------------------------------------------------------------------------

def solve_grasp_ik(t_cam, R_cam, q_scan, pos_w, quat_w):
    """对单个 GraspNet 候选求解 IK，返回 6 个关节角。

    失败时返回 None。
    """
    # 1. 相机坐标系 -> 世界坐标系 -> arm_base_link 坐标系
    t_world = cam_to_world(t_cam.astype(np.float64), q_scan, pos_w, quat_w)
    t_arm   = world_pos_to_arm_frame(t_world, pos_w, quat_w)

    # 2. 计算目标旋转（SCAN 时刻的关节角为参考）
    R_target = compute_desired_ee_rot_in_arm(R_cam.astype(np.float64), q_scan)

    # 3. IK 求解
    try:
        q_sol = solve_for_gripper_base(
            t_arm,
            target_rot_j7=R_target,
            initial_angles=q_scan,
        )
        return q_sol
    except Exception as e:
        print(f'  [IK] 求解失败: {e}')
        return None


def arm_joints_world(q6, pos_w, quat_w):
    """用 ikpy full_kinematics 求机械臂各关节在世界坐标系下的位置。

    参数
    ----
    q6    : (6,) joint1-6 角度
    pos_w : (3,) 机器人 base_link 世界坐标
    quat_w: (4,) 机器人四元数 [w,x,y,z]

    返回
    ----
    pts_world : list of (3,)，长度 = chain 节点数（含 arm_base_link 原点）
    """
    chain = get_chain()
    q9 = np.array([0.0] + list(q6) + [0.0, 0.0], dtype=np.float64)
    transforms = chain.forward_kinematics(q9, full_kinematics=True)

    # transforms[i] 是 (4,4)，坐标系为 arm_base_link
    R_robot = quat_to_rot(quat_w.astype(np.float64))
    arm_base_w = pos_w + R_robot @ ARM_BASE_OFFSET

    pts_world = []
    for T in transforms:
        p_arm = T[:3, 3]
        p_world = R_robot @ p_arm + arm_base_w
        pts_world.append(p_world)
    return pts_world


# ---------------------------------------------------------------------------
# 3. 可视化
# ---------------------------------------------------------------------------

def visualize(cam_world, banana_world, centers_world, approaches_world,
              scores, widths, ik_joints_world_list):
    """交互式 3D 可视化。

    ik_joints_world_list : list of K 元素，每个元素是 arm_joints_world() 的返回值
                           （各关节世界坐标列表），None 表示该解 IK 失败。
    """
    K = len(centers_world)
    current = [0]

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # --- 静态元素 ---
    ax.scatter(*banana_world, color='gold', s=250, depthshade=False, zorder=5,
               label='Banana')
    ax.scatter(*cam_world, color='red', s=180, marker='^', depthshade=False, zorder=5,
               label='Camera')
    ax.plot([cam_world[0], banana_world[0]],
            [cam_world[1], banana_world[1]],
            [cam_world[2], banana_world[2]],
            color='gray', linestyle='--', linewidth=1, alpha=0.4)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # 显示范围：以香蕉为中心，但要包含机械臂根部
    half = 0.9
    cx, cy, cz = banana_world
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(max(0.0, cz - half), cz + half)

    # --- 动态图元句柄 ---
    handles = {'arrow': None, 'dot': None, 'arm_lines': [], 'arm_dots': []}
    ARROW_LEN = 0.05
    # 机械臂颜色渐变：从根部(深橙)到末端(亮橙)
    ARM_COLORS = ['#8B4513', '#CD853F', '#DAA520', '#FFA500', '#FF8C00',
                  '#FF6347', '#FF4500', '#FF0000', '#DC143C']

    def clear_dynamic():
        if handles['arrow'] is not None:
            handles['arrow'].remove()
            handles['arrow'] = None
        if handles['dot'] is not None:
            handles['dot'].remove()
            handles['dot'] = None
        for ln in handles['arm_lines']:
            ln.remove()
        handles['arm_lines'].clear()
        for dt in handles['arm_dots']:
            dt.remove()
        handles['arm_dots'].clear()

    def draw_grasp(idx):
        clear_dynamic()

        # --- 夹爪箭头 ---
        c = centers_world[idx]
        d = approaches_world[idx]
        handles['arrow'] = ax.quiver(
            c[0], c[1], c[2],
            d[0]*ARROW_LEN, d[1]*ARROW_LEN, d[2]*ARROW_LEN,
            color='royalblue', linewidth=2.5, arrow_length_ratio=0.4,
        )
        handles['dot'] = ax.scatter(*c, color='royalblue', s=80,
                                    depthshade=False, zorder=6)

        # --- 机械臂线段 ---
        pts = ik_joints_world_list[idx]
        ik_ok = pts is not None
        if ik_ok:
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                col = ARM_COLORS[i % len(ARM_COLORS)]
                ln, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                              '-', color=col, linewidth=3, alpha=0.85)
                handles['arm_lines'].append(ln)
            for i, p in enumerate(pts):
                col = ARM_COLORS[i % len(ARM_COLORS)]
                dt = ax.scatter(*p, color=col, s=40, depthshade=False, zorder=7)
                handles['arm_dots'].append(dt)

        dist = np.linalg.norm(c - banana_world) * 100
        ik_str = 'IK OK' if ik_ok else 'IK FAIL'
        ax.set_title(
            'Grasp [{}/{}]  score={:.3f}  width={:.1f}cm  '
            'dist_to_banana={:.1f}cm  {}\n'
            'center_world={}\napproach_world={}\n'
            '[Space] 下一组解  共 {} 组'.format(
                idx, K-1, scores[idx], widths[idx]*100, dist, ik_str,
                np.round(c, 3), np.round(d, 3), K),
            fontsize=9,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == ' ':
            current[0] = (current[0] + 1) % K
            draw_grasp(current[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    ax.legend(loc='upper left', fontsize=8)
    draw_grasp(0)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------------------------

def main():
    log_path   = '/tmp/nav_run.log'
    grasp_path = '/tmp/grasp_result.npz'

    print('[grasp_vis] 解析 log ...', flush=True)
    cam_world, banana_world, pos_w_scan, quat_w_scan, q_scan = parse_log(log_path)

    missing = []
    if cam_world    is None: missing.append('Code cam world')
    if banana_world is None: missing.append('Banana world pos')
    if pos_w_scan   is None: missing.append('ALIGN_YAW freeze anchor pos')
    if quat_w_scan  is None: missing.append('ALIGN_YAW freeze anchor quat')
    if q_scan       is None: missing.append('cur_q at SCAN')
    if missing:
        print('[grasp_vis] ERROR: log 中未找到以下字段：')
        for msg in missing:
            print('  -', msg)
        sys.exit(1)

    print('  cam_world    =', np.round(cam_world, 4))
    print('  banana_world =', np.round(banana_world, 4))
    print('  pos_w_scan   =', np.round(pos_w_scan, 4))
    print('  quat_w_scan  =', np.round(quat_w_scan, 4))
    print('  q_scan       =', np.round(q_scan, 4))

    print('[grasp_vis] 读取 grasp_result.npz ...', flush=True)
    gr           = np.load(grasp_path)
    translations = gr['translations']
    rotations    = gr['rotations']
    widths       = gr['widths']
    scores       = gr['scores']
    K = len(scores)
    print('  共 {} 组候选，分数范围 [{:.3f}, {:.3f}]'.format(K, scores.min(), scores.max()))

    print('[grasp_vis] 变换到世界坐标系 ...', flush=True)
    centers_world, approaches_world = transform_grasps(
        translations, rotations, q_scan, pos_w_scan, quat_w_scan,
    )
    for i in range(min(3, K)):
        dist = np.linalg.norm(centers_world[i] - banana_world) * 100
        print('  [{}] score={:.3f}  center={}  approach={}  dist={:.1f}cm'.format(
            i, scores[i], np.round(centers_world[i], 3),
            np.round(approaches_world[i], 3), dist))

    print('[grasp_vis] 对每组候选求解 IK + FK 关节位置 ...', flush=True)
    ik_joints_world_list = []
    for i in range(K):
        q_sol = solve_grasp_ik(
            translations[i], rotations[i], q_scan, pos_w_scan, quat_w_scan
        )
        if q_sol is not None:
            pts = arm_joints_world(q_sol, pos_w_scan, quat_w_scan)
            ik_joints_world_list.append(pts)
            if i < 3:
                print('  [{}] IK OK  q={}'.format(i, np.round(q_sol, 3)))
        else:
            ik_joints_world_list.append(None)
            if i < 3:
                print('  [{}] IK FAIL'.format(i))

    print('[grasp_vis] 打开 3D 窗口（按空格切换解）...', flush=True)
    visualize(cam_world, banana_world, centers_world, approaches_world,
              scores, widths, ik_joints_world_list)


if __name__ == '__main__':
    main()

