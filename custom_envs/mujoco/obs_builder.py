"""obs_builder.py — 构造 57 维 policy obs，格式和 Isaac Lab 训练时完全一致。

观测向量结构（已从 Isaac Lab [MJ_MIGRATE] 日志和 joint_pos obs 确认）：

  index   字段                维度  scale  说明
  ────────────────────────────────────────────────────────────
  0~2     base_ang_vel         3   0.25   机体角速度（机体系）
  3~5     projected_gravity    3   1.0    重力在机体系的投影（单位向量）
  6       cmd_vx               1   1.0    速度指令 vx
  7       cmd_vy               1   1.0    速度指令 vy
  8       cmd_wz               1   1.0    偏航角速度指令
  9~24    joint_pos_rel       16   1.0    关节角 - default
  25~40   joint_vel_rel       16   0.05   关节角速度
  41~56   last_action         16   1.0    上一步 action

  合计：3+3+3+16+16+16 = 57
"""

import numpy as np

# ─────────────────────────────────────────────
# 关节名称总表（24个，和 Isaac Lab joint_names 完全一致）
# ─────────────────────────────────────────────
ALL_JOINT_NAMES = [
    "fl_hipx_joint",   # [ 0]
    "fr_hipx_joint",   # [ 1]
    "hl_hipx_joint",   # [ 2]
    "hr_hipx_joint",   # [ 3]
    "joint1",          # [ 4]
    "fl_hipy_joint",   # [ 5]
    "fr_hipy_joint",   # [ 6]
    "hl_hipy_joint",   # [ 7]
    "hr_hipy_joint",   # [ 8]
    "joint2",          # [ 9]
    "fl_knee_joint",   # [10]
    "fr_knee_joint",   # [11]
    "hl_knee_joint",   # [12]
    "hr_knee_joint",   # [13]
    "joint3",          # [14]
    "fl_wheel_joint",  # [15]
    "fr_wheel_joint",  # [16]
    "hl_wheel_joint",  # [17]
    "hr_wheel_joint",  # [18]
    "joint4",          # [19]
    "joint5",          # [20]
    "joint6",          # [21]
    "joint7",          # [22]
    "joint8",          # [23]
]

# joint_pos obs 的 16 个关节顺序（实测确认）
JPOS_OBS_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]

_NAME2IDX = {n: i for i, n in enumerate(ALL_JOINT_NAMES)}
JPOS_OBS_IDX = np.array([_NAME2IDX[n] for n in JPOS_OBS_ORDER], dtype=np.int32)

# Default joint pos（顺序和 ALL_JOINT_NAMES 一致，24个）
# ARM_HOME_ANGLES from navigate_to_goal_nav2_whole.py: [0.0, 0.5, -1.0, 0.0, 0.5, 0.0]
DEFAULT_JOINT_POS_ALL = np.array([
     0.000,  # fl_hipx_joint
     0.000,  # fr_hipx_joint
     0.000,  # hl_hipx_joint
     0.000,  # hr_hipx_joint
     0.000,  # joint1
    -0.300,  # fl_hipy_joint
    -0.300,  # fr_hipy_joint
     0.300,  # hl_hipy_joint
     0.300,  # hr_hipy_joint
     0.500,  # joint2  (was 0.300, now matches ARM_HOME_ANGLES)
     0.600,  # fl_knee_joint
     0.600,  # fr_knee_joint
    -0.600,  # hl_knee_joint
    -0.600,  # hr_knee_joint
    -1.000,  # joint3  (was -1.500, now matches ARM_HOME_ANGLES)
     0.000,  # fl_wheel_joint
     0.000,  # fr_wheel_joint
     0.000,  # hl_wheel_joint
     0.000,  # hr_wheel_joint
     0.000,  # joint4
     0.500,  # joint5  (was 0.800, now matches ARM_HOME_ANGLES)
     0.000,  # joint6
     0.035,  # joint7
    -0.035,  # joint8
], dtype=np.float32)

# 只取 JPOS_OBS_IDX 对应的 16 个 default 值
JPOS_OBS_DEFAULT = DEFAULT_JOINT_POS_ALL[JPOS_OBS_IDX]  # shape (16,)


def compute_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """把世界系重力方向 [0,0,-1] 投影到机体系（和 Isaac Lab projected_gravity 一致）。

    参数：quat_wxyz (4,) 顺序 (w, x, y, z)
    返回：(3,) float32
    """
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # R^T（world→body）作用于 g_world
    Rt = np.array([
        [1-2*(y*y+z*z),   2*(x*y+w*z),   2*(x*z-w*y)],
        [  2*(x*y-w*z), 1-2*(x*x+z*z),   2*(y*z+w*x)],
        [  2*(x*z+w*y),   2*(y*z-w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)
    return Rt @ g_world


def build_policy_obs(
    joint_pos: np.ndarray,    # (24,) 当前关节角（顺序 ALL_JOINT_NAMES）
    joint_vel: np.ndarray,    # (24,) 当前关节角速度
    ang_vel_body: np.ndarray, # (3,)  机体角速度（机体系）
    quat_wxyz: np.ndarray,    # (4,)  机体姿态四元数 (w,x,y,z)
    cmd_vx: float,
    cmd_vy: float,
    cmd_wz: float,
    last_action: np.ndarray,  # (16,) 上一步 action
) -> np.ndarray:
    """构造 57 维 policy obs，格式和 Isaac Lab 训练时完全一致。

    返回 (57,) float32 numpy 数组。
    """
    obs_ang_vel = ang_vel_body.astype(np.float32) * 0.25
    obs_gravity = compute_projected_gravity(quat_wxyz)
    obs_cmd     = np.array([cmd_vx, cmd_vy, cmd_wz], dtype=np.float32)
    obs_jpos    = (joint_pos[JPOS_OBS_IDX] - JPOS_OBS_DEFAULT).astype(np.float32)
    obs_jpos[12:16] = 0.0  # 轮子位置置零（与 Isaac Lab joint_pos_rel_without_wheel 一致）
    obs_jvel    = (joint_vel[JPOS_OBS_IDX] * 0.05).astype(np.float32)
    obs_act     = last_action.astype(np.float32)

    obs = np.concatenate([obs_ang_vel, obs_gravity, obs_cmd,
                          obs_jpos, obs_jvel, obs_act])
    assert obs.shape == (57,), f"obs shape mismatch: {obs.shape}"
    return obs
