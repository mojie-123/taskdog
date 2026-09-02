#!/usr/bin/env python3
"""grasp_vis.py — 可视化 AnyGrasp 输出的抓取候选解。

读取:
  /tmp/nav_run.log         — 香蕉世界坐标、机器人位姿、扫描时关节角、PRE_GRASP失败链
  /tmp/grasp_result.npz   — AnyGrasp 候选抓取（相机坐标系）

可视化:
  金色小点 — 香蕉世界坐标
  灰色方块 — 机械臂 base 世界坐标
  细线     — 机械臂各连杆姿态
  细蓝箭头 — 夹爪中心（尾）+ 接近方向（头，长5cm）
  空格     — 切换下一组解（置信度从高到低）

终端输出:
  所有 AnyGrasp 候选列表
  navigate_to_goal 的 PRE_GRASP 候选执行顺序与失败原因
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
    cam_to_world, quat_to_rot, fk_gripper, _CAM_OFFSET_ROT, _RY_POS90,
    solve_for_gripper_base, compute_desired_ee_rot_in_arm,
    world_pos_to_arm_frame, get_chain, _RX_NEG90,
)
ARM_BASE_OFFSET = np.array([0.0, 0.0, 0.0888])  # base_link -> arm_base_link


# ---------------------------------------------------------------------------
# 1. Log 解析
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """解析 nav_run.log，提取机械臂 base 位置、香蕉坐标、关节角，以及 PRE_GRASP 失败链。

    返回:
        banana_world   : (3,) 世界坐标
        pos_w_scan     : (3,) 扫描时机器人 base_link 世界坐标（即机械臂 base 位置参考）
        quat_w_scan    : (4,) 扫描时四元数 [w,x,y,z]
        q_scan         : (6,) 扫描时关节角
        pregrasp_chain : list of dict，每项格式:
                         {'candidate': int, 'score': float,
                          'result': 'OK'|'FAIL',
                          'reason': str}  # FAIL 时有原因
        sort_line      : str  # 候选排序原始行（含 ft 分数）
        best_line      : str  # Best grasp 原始行
    """
    banana_world = pos_w_scan = quat_w_scan = q_scan = None
    sort_line = best_line = ''
    v3 = r'\[\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\]'
    v4 = r'\[\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\]'
    np_ = r'[\-\+]?[\d.eE+\-]+'
    v6 = r'\[\s*(' + r')\s+('.join([np_] * 6) + r')\s*\]'

    # PRE_GRASP 失败链状态机
    pregrasp_chain = []   # 按执行顺序记录每个候选的结果
    active_cand   = None  # 当前正在执行的候选编号
    active_score  = None
    in_pregrasp   = False # 是否进入过 Best grasp 阶段

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # --- 基础字段 ---
        if '[DIAG-CAM] Banana world pos' in line:
            m = re.search(v3, line)
            if m:
                banana_world = np.array([float(m.group(j)) for j in range(1, 4)])
        elif 'ALIGN_YAW freeze anchor set' in line:
            mp = re.search(r'pos=' + v3, line)
            mq = re.search(r'quat=' + v4, line)
            if mp: pos_w_scan  = np.array([float(mp.group(j)) for j in range(1, 4)])
            if mq: quat_w_scan = np.array([float(mq.group(j)) for j in range(1, 5)])
        elif 'cur_q at SCAN' in line:
            m = re.search(v6, line)
            if m:
                q_scan = np.array([float(m.group(j)) for j in range(1, 7)])

        # --- 候选排序行 ---
        elif '候选排序' in line and 'IK预筛' in line:
            sort_line = line.strip()

        # --- Best grasp 行：初始化失败链第一个候选 ---
        elif '[SM] Best grasp' in line:
            best_line = line.strip()
            in_pregrasp = True
            # 从排序行里提取第一个候选编号和分数
            m_sort = re.search(r'\[(\d+)\].*?s=([\d.]+)', sort_line)
            if m_sort:
                active_cand  = int(m_sort.group(1))
                active_score = float(m_sort.group(2))

        # --- switch to candidate [N] score=X ---
        elif in_pregrasp and 'switch to candidate' in line:
            m_sw = re.search(r'switch to candidate \[(\d+)\] score=([\d.]+)', line)
            if m_sw:
                active_cand  = int(m_sw.group(1))
                active_score = float(m_sw.group(2))

        # --- IK returned None (奇异点) ---
        elif in_pregrasp and 'IK returned None' in line and active_cand is not None:
            reason = 'j2=π 奇异点（IK 返回 None）'
            pregrasp_chain.append({
                'candidate': active_cand, 'score': active_score,
                'result': 'FAIL', 'reason': reason,
            })

        # --- FK check ✗ FAIL ---
        elif in_pregrasp and 'FK check' in line and '✗ FAIL' in line and active_cand is not None:
            m_err = re.search(r'pos_err=([\d.]+)cm', line)
            m_lim = re.search(r'at_limit=(\S+)', line)
            err_str = m_err.group(1) + 'cm' if m_err else '?'
            lim_str = ' (关节触限)' if m_lim and m_lim.group(1) == 'True' else ''
            reason = f'FK 验证失败，pos_err={err_str}{lim_str}'
            pregrasp_chain.append({
                'candidate': active_cand, 'score': active_score,
                'result': 'FAIL', 'reason': reason,
            })

        # --- FK check ✓ OK ---
        elif in_pregrasp and 'FK check' in line and '✓ OK' in line and active_cand is not None:
            m_err = re.search(r'pos_err=([\d.]+)cm', line)
            err_str = m_err.group(1) + 'cm' if m_err else '?'
            pregrasp_chain.append({
                'candidate': active_cand, 'score': active_score,
                'result': 'OK', 'reason': f'FK pos_err={err_str}',
            })
            in_pregrasp = False  # 找到 OK 候选，停止追踪

    return banana_world, pos_w_scan, quat_w_scan, q_scan, pregrasp_chain, sort_line, best_line


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
    R_cam2world = R_robot @ R_gb @ _CAM_OFFSET_ROT  # 相机系 -> arm系（物理安装旋转，蓝色箭头用原始相机安装变换）

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

# AnyGrasp translation = gripper_base wrist origin (not finger-tip midpoint).
# ik_solve_gb / solve_for_gripper_base also expects gripper_base origin as input.
# No backward retreat offset needed; keep at 0.0 to match navigate_to_goal.py.
_GRASP_RETREAT = 0.0  # AnyGrasp translation = wrist origin; no offset needed

def solve_grasp_ik(t_cam, R_cam, q_scan, pos_w, quat_w):
    """对单个 GraspNet 候选求解 IK，返回 6 个关节角。

    与 navigate_to_goal.py PRE_GRASP 逻辑一致：
    - AnyGrasp translation = gripper_base wrist origin，直接作为 IK 目标
    - FK 二次验证：pos_err < 3cm 且不在关节限位
    失败时返回 None。
    """
    from arm_ik import _IK_JOINT_LIMITS
    # 1. 相机坐标系 -> 世界坐标系 -> arm_base_link 坐标系
    t_world = cam_to_world(t_cam.astype(np.float64), q_scan, pos_w, quat_w)
    t_arm   = world_pos_to_arm_frame(t_world, pos_w, quat_w)

    # 2. 计算目标旋转（SCAN 时刻的关节角为参考）
    R_target = compute_desired_ee_rot_in_arm(R_cam.astype(np.float64), q_scan)

    # 3. gripper_base 目标位置 = AnyGrasp wrist origin（无偏移）
    _R_gb = R_target @ _RX_NEG90          # gripper_base 旋转（arm frame）
    _approach_arm = _R_gb[:, 2]           # approach 方向（arm frame，R_gb_desired[:,2] = gb+Z）
    t_arm_gb = t_arm - _GRASP_RETREAT * _approach_arm

    # 4. IK 求解
    try:
        q_sol = solve_for_gripper_base(
            t_arm_gb,
            target_rot_j7=R_target,
            initial_angles=q_scan,
        )
    except Exception as e:
        print(f'  [IK] 求解失败: {e}')
        return None
    if q_sol is None:
        return None

    # 5. FK 二次验证：gripper_base 位置误差 < 3cm，不在关节限位
    T_fk = fk_gripper(q_sol.astype(np.float64))
    _pos_err = float(np.linalg.norm(T_fk[:3, 3] - t_arm_gb))
    _at_lim  = any(
        abs(float(qi) - lo) < 0.01 or abs(float(qi) - hi) < 0.01
        for qi, (lo, hi) in zip(q_sol, _IK_JOINT_LIMITS)
    )
    if _pos_err >= 0.03 or _at_lim:
        return None
    return q_sol


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

def visualize(arm_base_world, banana_world, centers_world, approaches_world,
              scores, widths, ik_joints_world_list, ik_q_list, quat_w_scan):
    """交互式 3D 可视化。

    arm_base_world       : (3,) 机械臂 base 世界坐标（用于显示参考点）
    ik_joints_world_list : list of K 元素，每个元素是 arm_joints_world() 的返回值
                           （各关节世界坐标列表），None 表示该解 IK 失败。
    ik_q_list            : list of K 元素，每个元素是 q_sol (6,) 或 None（IK 失败）。
    quat_w_scan          : (4,) 扫描时机器人四元数 [w,x,y,z]，用于计算三轴世界方向。
    """
    K = len(centers_world)
    current = [0]

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # --- 静态元素 ---
    # 香蕉长方体（YCB 011_banana 标准尺寸：长19cm×宽4cm×高3.5cm，长轴=世界X轴）
    BANANA_HALF = np.array([0.095, 0.020, 0.0175])  # 半长宽高
    bx0, by0, bz0 = banana_world - BANANA_HALF
    bdx, bdy, bdz = BANANA_HALF * 2
    # 画长方体 12 条边
    _bverts = [
        # bottom face
        ([bx0, bx0+bdx], [by0,    by0   ], [bz0,    bz0   ]),
        ([bx0, bx0+bdx], [by0+bdy,by0+bdy],[bz0,   bz0   ]),
        ([bx0,    bx0 ], [by0, by0+bdy  ], [bz0,    bz0   ]),
        ([bx0+bdx,bx0+bdx],[by0,by0+bdy ], [bz0,    bz0   ]),
        # top face
        ([bx0, bx0+bdx], [by0,    by0   ], [bz0+bdz,bz0+bdz]),
        ([bx0, bx0+bdx], [by0+bdy,by0+bdy],[bz0+bdz,bz0+bdz]),
        ([bx0,    bx0 ], [by0, by0+bdy  ], [bz0+bdz,bz0+bdz]),
        ([bx0+bdx,bx0+bdx],[by0,by0+bdy ], [bz0+bdz,bz0+bdz]),
        # vertical edges
        ([bx0,    bx0   ], [by0,    by0   ], [bz0, bz0+bdz]),
        ([bx0+bdx,bx0+bdx],[by0,    by0   ], [bz0, bz0+bdz]),
        ([bx0,    bx0   ], [by0+bdy,by0+bdy],[bz0, bz0+bdz]),
        ([bx0+bdx,bx0+bdx],[by0+bdy,by0+bdy],[bz0, bz0+bdz]),
    ]
    for _e in _bverts:
        ax.plot(_e[0], _e[1], _e[2], color='goldenrod', linewidth=0.8, alpha=0.7)
    # 中心点
    ax.scatter(*banana_world, color='gold', s=20, depthshade=False, zorder=5,
               label='Banana')
    ax.scatter(*arm_base_world, color='dimgray', s=50, marker='s',
               depthshade=False, zorder=5, label='Arm Base')
    ax.plot([arm_base_world[0], banana_world[0]],
            [arm_base_world[1], banana_world[1]],
            [arm_base_world[2], banana_world[2]],
            color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # 显示范围：包含香蕉和机械臂 base
    all_pts = np.vstack([banana_world, arm_base_world])
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
    cz = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2
    half = max(0.9, np.linalg.norm(banana_world - arm_base_world) * 0.7)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(max(0.0, cz - half), cz + half)

    # --- 动态图元句柄 ---
    handles = {'arrow': None, 'dot': None, 'arm_lines': [], 'arm_dots': [],
               'axis_x': None, 'axis_y': None, 'axis_z': None}
    ARROW_LEN = 0.1358  # joint7 origin 到 gripper_base 的距离（13.58cm），使蓝色箭头与机械臂末端段等长
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
        for k in ('axis_x', 'axis_y', 'axis_z'):
            if handles[k] is not None:
                handles[k].remove()
                handles[k] = None

    def draw_grasp(idx):
        clear_dynamic()

        # --- 夹爪箭头（细线） ---
        c = centers_world[idx]
        d = approaches_world[idx]
        handles['arrow'] = ax.quiver(
            c[0], c[1], c[2],
            d[0]*ARROW_LEN, d[1]*ARROW_LEN, d[2]*ARROW_LEN,
            color='royalblue', linewidth=1.0, arrow_length_ratio=0.4,
        )
        handles['dot'] = ax.scatter(*c, color='royalblue', s=25,
                                    depthshade=False, zorder=6)

        # --- 机械臂线段（细线） ---
        pts = ik_joints_world_list[idx]
        ik_ok = pts is not None
        if ik_ok:
            for i in range(len(pts) - 1):
                p0, p1 = pts[i], pts[i + 1]
                col = ARM_COLORS[i % len(ARM_COLORS)]
                ln, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                              '-', color=col, linewidth=1.0, alpha=0.85)
                handles['arm_lines'].append(ln)
            for i, p in enumerate(pts):
                col = ARM_COLORS[i % len(ARM_COLORS)]
                dt = ax.scatter(*p, color=col, s=15, depthshade=False, zorder=7)
                handles['arm_dots'].append(dt)

            # --- gripper_base 三轴（从 gripper_base 原点 = pts[7] 出发） ---
            # 用 IK 解出的关节角做 FK，得到 gripper_base 在 arm_base_link 下的旋转
            # 再旋转到世界坐标系
            from arm_ik import fk_gripper as _fk_gb
            _q_sol = ik_q_list[idx]  # IK 解出的关节角
            _T_gb  = _fk_gb(_q_sol.astype(np.float64))
            _R_gb  = _T_gb[:3, :3]
            _R_rob = quat_to_rot(quat_w_scan.astype(np.float64))
            # gripper_base 原点（世界坐标）= pts[7]
            gb_origin = pts[7]
            AXIS_LEN = 0.06  # 轴箭头长度 6cm
            # X 轴：红色
            ax_x = _R_rob @ _R_gb[:, 0]
            handles['axis_x'] = ax.quiver(
                gb_origin[0], gb_origin[1], gb_origin[2],
                ax_x[0]*AXIS_LEN, ax_x[1]*AXIS_LEN, ax_x[2]*AXIS_LEN,
                color='red', linewidth=1.2, arrow_length_ratio=0.3,
            )
            # Y 轴：绿色
            ax_y = _R_rob @ _R_gb[:, 1]
            handles['axis_y'] = ax.quiver(
                gb_origin[0], gb_origin[1], gb_origin[2],
                ax_y[0]*AXIS_LEN, ax_y[1]*AXIS_LEN, ax_y[2]*AXIS_LEN,
                color='green', linewidth=1.2, arrow_length_ratio=0.3,
            )
            # Z 轴：蓝色
            ax_z = _R_rob @ _R_gb[:, 2]
            handles['axis_z'] = ax.quiver(
                gb_origin[0], gb_origin[1], gb_origin[2],
                ax_z[0]*AXIS_LEN, ax_z[1]*AXIS_LEN, ax_z[2]*AXIS_LEN,
                color='cyan', linewidth=1.2, arrow_length_ratio=0.3,
            )

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
    banana_world, pos_w_scan, quat_w_scan, q_scan, pregrasp_chain, sort_line, best_line = \
        parse_log(log_path)

    missing = []
    if banana_world is None: missing.append('Banana world pos  ([DIAG-CAM] Banana world pos)')
    if pos_w_scan   is None: missing.append('ALIGN_YAW freeze anchor pos')
    if quat_w_scan  is None: missing.append('ALIGN_YAW freeze anchor quat')
    if q_scan       is None: missing.append('cur_q at SCAN')
    if missing:
        print('[grasp_vis] ERROR: log 中未找到以下字段：')
        for msg in missing:
            print('  -', msg)
        sys.exit(1)

    # 机械臂 base 世界坐标 = base_link + ARM_BASE_OFFSET（旋转后）
    R_robot = quat_to_rot(quat_w_scan.astype(np.float64))
    arm_base_world = pos_w_scan + R_robot @ ARM_BASE_OFFSET

    print('  banana_world   =', np.round(banana_world, 4))
    print('  pos_w_scan     =', np.round(pos_w_scan, 4))
    print('  arm_base_world =', np.round(arm_base_world, 4))
    print('  quat_w_scan    =', np.round(quat_w_scan, 4))
    print('  q_scan         =', np.round(q_scan, 4))

    print('[grasp_vis] 读取 grasp_result.npz ...', flush=True)
    gr           = np.load(grasp_path)
    translations = gr['translations']
    rotations    = gr['rotations']
    widths       = gr['widths']
    scores       = gr['scores']
    K = len(scores)

    print('[grasp_vis] 变换到世界坐标系 ...', flush=True)
    centers_world, approaches_world = transform_grasps(
        translations, rotations, q_scan, pos_w_scan, quat_w_scan,
    )

    # ── 打印 AnyGrasp 全部候选列表 ──────────────────────────────────────────
    print()
    print('=' * 70)
    print('[AnyGrasp 候选列表]  共 {} 组  score∈[{:.3f}, {:.3f}]'.format(
        K, scores.min(), scores.max()))
    print('  {:>4}  {:>7}  {:>8}  {:>28}  {:>10}  approach'.format(
        'idx', 'score', 'width/cm', 'center_world(m)', 'dist2banana'))
    print('  ' + '-' * 80)
    for i in range(K):
        dist = np.linalg.norm(centers_world[i] - banana_world) * 100
        print('  [{:>2}]  {:.3f}   {:>6.1f}cm   {}   {:>6.1f}cm   {}'.format(
            i, scores[i], widths[i]*100,
            np.array2string(np.round(centers_world[i], 3), separator=','),
            dist,
            np.array2string(np.round(approaches_world[i], 3), separator=','),
        ))
    print('=' * 70)

    # ── 打印 PRE_GRASP 执行链（来自日志） ───────────────────────────────────
    print()
    print('[PRE_GRASP 执行记录]  (解析自 navigate_to_goal 日志)')
    if sort_line:
        # 只保留排序信息部分
        sort_part = sort_line.split('候选排序')[-1].strip()
        print('  候选排序:', sort_part)
    if best_line:
        best_part = best_line.split('[SM]')[-1].strip()
        print('  {}'.format(best_part))
    if pregrasp_chain:
        print()
        for item in pregrasp_chain:
            c_idx   = item['candidate']
            c_score = item['score']
            result  = item['result']
            reason  = item.get('reason', '')
            if result == 'OK':
                tag = '✓ IK OK'
            else:
                tag = '✗ IK FAIL'
            print('  [{}] score={:.3f}  {}  {}'.format(
                c_idx, c_score, tag, reason))
    else:
        print('  (日志中未找到 PRE_GRASP 候选执行记录)')
    print('=' * 70)
    print()

    print('[grasp_vis] 对每组候选求解 IK + FK 关节位置 ...', flush=True)

    # ── 加载桌面点云（世界坐标系），用于指尖穿桌检测 ──────────────────────
    _table_pts_world = None
    _tc_path = '/tmp/table_cloud.npz'
    if os.path.exists(_tc_path):
        try:
            _tc_data = np.load(_tc_path)
            _table_pts_world = _tc_data['points'].astype(np.float64)
            print(f'[grasp_vis] 桌面点云已加载: {len(_table_pts_world)} pts（世界坐标系）')
        except Exception as _e_tc:
            print('[grasp_vis] WARN: 加载桌面点云失败:', _e_tc)
    else:
        print('[grasp_vis] INFO: /tmp/table_cloud.npz 不存在，跳过穿桌检测')

    print()
    print('  {:>4}  {:>6}  {:>40}  {:>40}  {:>10}  一致?  穿桌?'.format(
        'idx', 'result', 'approach_world(蓝色箭头)', 'fk_approach_world(IK末端X轴)', 'cos_sim'))
    print('  ' + '-' * 120)
    ik_joints_world_list = []
    ik_q_list = []
    tip_penetrate_list = []   # 每个候选是否穿桌
    R_robot = quat_to_rot(quat_w_scan.astype(np.float64))
    _arm_base_w_ik = pos_w_scan + R_robot @ np.array([0., 0., 0.0888])
    for i in range(K):
        q_sol = solve_grasp_ik(
            translations[i], rotations[i], q_scan, pos_w_scan, quat_w_scan
        )
        if q_sol is not None:
            pts = arm_joints_world(q_sol, pos_w_scan, quat_w_scan)
            ik_joints_world_list.append(pts)
            ik_q_list.append(q_sol)
            # 用 IK 解算出的关节角做 FK，得到 gripper_base 的旋转矩阵
            T_gb_sol = fk_gripper(q_sol.astype(np.float64))
            R_gb_sol = T_gb_sol[:3, :3]
            approach_fk_arm   = R_gb_sol[:, 2]   # R_gb_desired[:,2] = gb+Z（夹爪伸出方向，右乘_RY_POS90后）
            approach_fk_world = R_robot @ approach_fk_arm
            app_vis = approaches_world[i]
            cos_sim = float(np.dot(approach_fk_world, app_vis) /
                            (np.linalg.norm(approach_fk_world) * np.linalg.norm(app_vis) + 1e-9))
            ok_str  = '✓ 一致' if cos_sim > 0.95 else ('~ 近似' if cos_sim > 0.8 else '✗ 偏差大')
            # ── 穿桌检测：指尖 XY 邻域内桌面点云最大 Z ──
            penetrate_str = 'N/A'
            penetrate = False
            if _table_pts_world is not None:
                try:
                    from arm_ik import fk as _fk_j7_i
                    _T_j7_i   = _fk_j7_i(q_sol.astype(np.float64))
                    _p_tip_wi = R_robot @ _T_j7_i[:3, 3] + _arm_base_w_ik
                    _xy_d_i   = np.linalg.norm(_table_pts_world[:, :2] - _p_tip_wi[:2], axis=1)
                    _nearby_i = _table_pts_world[_xy_d_i < 0.03]
                    if len(_nearby_i) >= 5:
                        _ltbl_z_i = float(_nearby_i[:, 2].max())
                        penetrate = _p_tip_wi[2] < _ltbl_z_i + 0.01
                        penetrate_str = ('✗穿桌(tbl={:.3f},tip={:.3f})'.format(_ltbl_z_i, _p_tip_wi[2])
                                         if penetrate else
                                         '✓OK(tbl={:.3f},tip={:.3f})'.format(_ltbl_z_i, _p_tip_wi[2]))
                    else:
                        penetrate_str = '?邻域点少({})'.format(len(_nearby_i))
                except Exception as _ep:
                    penetrate_str = 'ERR'
            tip_penetrate_list.append(penetrate)
            print('  [{:>2}]  IK OK   {}  {}  {:>6.3f}  {}  {}'.format(
                i,
                np.array2string(np.round(app_vis, 3), separator=','),
                np.array2string(np.round(approach_fk_world, 3), separator=','),
                cos_sim, ok_str, penetrate_str,
            ))
        else:
            ik_joints_world_list.append(None)
            ik_q_list.append(None)
            tip_penetrate_list.append(False)
            print('  [{:>2}]  IK FAIL'.format(i))
    print()

    print('[grasp_vis] 打开 3D 窗口（按空格切换解）...', flush=True)
    visualize(arm_base_world, banana_world, centers_world, approaches_world,
              scores, widths, ik_joints_world_list, ik_q_list, quat_w_scan)


if __name__ == '__main__':
    main()

