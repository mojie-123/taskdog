"""env.py — MuJoCo 环境封装，替代 Isaac Lab env/robot 接口。

职责：
  1. 加载 scene.xml（机器人 + 桌子 + 物体）
  2. reset()           — 初始化机器人姿态
  3. step(action)      — PD 控制 + mj_step × substeps
  4. get_robot_state() — 读取位置/四元数/速度/关节角
  5. set_arm_target()  — 机械臂关节目标角
  6. render_camera()   — RGB + depth（供 AnyGrasp/YOLO）
  7. set_root_pose()   — 强制设置根节点位姿（FREEZE 用）

PD 参数（对应 Isaac Lab actuator cfg）：
  腿：kp=80, kd=2 | 轮：kd=0.6 | 臂j1-4：kp=8000,kd=400
  臂j5：kp=15000,kd=600 | 臂j6：kp=8000,kd=400 | 夹爪：kp=30000,kd=20

Action 解码：
  action[0:12]  → 腿 pos target = action*scale + default
  action[12:16] → 轮 vel target = action*5.0
  hipx scale=0.125，hipy/knee scale=0.25
"""

import os
import numpy as np

try:
    import mujoco
    import mujoco.viewer as _mujoco_viewer
except ImportError:
    raise ImportError("请先安装 mujoco：pip install mujoco")

from .obs_builder import ALL_JOINT_NAMES, DEFAULT_JOINT_POS_ALL

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(_HERE, "scene.xml")

# M20_Piper.xml actuator 顺序（关节名，去掉 _motor 后缀）
_ROBOT_CTRL_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7", "joint8",
]

# action[0:12] 腿关节顺序（和 Isaac Lab leg_joint_names 完全一致：按腿分组）
_LEG_ACTION_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
# action[12:16] 轮子顺序
_WHEEL_ACTION_NAMES = [
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]

# 腿 action scale（每条腿：hipx=0.125, hipy=0.250, knee=0.250）
_LEG_SCALE = np.array([
    0.125, 0.250, 0.250,  # fl: hipx, hipy, knee
    0.125, 0.250, 0.250,  # fr: hipx, hipy, knee
    0.125, 0.250, 0.250,  # hl: hipx, hipy, knee
    0.125, 0.250, 0.250,  # hr: hipx, hipy, knee
], dtype=np.float32)
_WHEEL_SCALE = 5.0

# PD 参数
# MuJoCo motor 是直接力矩控制（非 Isaac Lab 隐式关节驱动器），kp 需按 ctrlrange 换算。
# 腿部 ctrlrange=76.4N*m，臂部 ctrlrange=100N*m，夹爪 ctrlrange=10N*m。
_KP_LEG    = 80.0;    _KD_LEG    = 2.0     # 腿关节 kp=80（合理）
_KD_WHEEL  = 0.4                           # 轮子速度控制
_KP_J1234  = 150.0;   _KD_J1234  = 15.0   # 臂 j1-4：kp=150（误差0.3rad -> 45N*m）
_KP_J5     = 150.0;   _KD_J5     = 15.0   # 臂 j5
_KP_J6     = 150.0;   _KD_J6     = 15.0   # 臂 j6
_KP_GRIP   = 200.0;    _KD_GRIP   = 8.0    # 夹爪：kp=80（误差0.1rad -> 8N*m）

_INIT_HEIGHT = 0.58


class MuJoCoEnv:
    """MuJoCo 环境封装，提供和 Isaac Lab 相似的接口。"""

    def __init__(
        self,
        robot_init_pos=(0.0, 0.0, _INIT_HEIGHT),
        robot_init_quat=(1.0, 0.0, 0.0, 0.0),  # (w,x,y,z)
        substeps: int = 4,
        camera_width: int = 640,
        camera_height: int = 480,
        render: bool = False,
    ):
        self._substeps   = substeps
        self._cam_w      = camera_width
        self._cam_h      = camera_height
        self._render_gui = render

        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data  = mujoco.MjData(self._model)

        # 建立关节名 -> qpos/dof 地址表
        self._jnt_qposadr = {}
        self._jnt_dofadr  = {}
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._jnt_qposadr[name] = self._model.jnt_qposadr[i]
                self._jnt_dofadr[name]  = self._model.jnt_dofadr[i]

        # 建立 ctrl 名 -> ctrl 索引表（actuator name 去掉 _motor 后缀）
        self._ctrl_idx = {}
        for i in range(self._model.nu):
            aname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if aname:
                jname = aname.replace("_motor", "")
                self._ctrl_idx[jname] = i

        # ALL_JOINT_NAMES 顺序的 qpos/dof 索引数组
        self._all_qpos_idx = np.array(
            [self._jnt_qposadr[n] for n in ALL_JOINT_NAMES], dtype=np.int32)
        self._all_dof_idx = np.array(
            [self._jnt_dofadr[n] for n in ALL_JOINT_NAMES], dtype=np.int32)

        _n2i = {n: i for i, n in enumerate(ALL_JOINT_NAMES)}
        self._leg_all_idx   = np.array([_n2i[n] for n in _LEG_ACTION_NAMES],   dtype=np.int32)
        self._wheel_all_idx = np.array([_n2i[n] for n in _WHEEL_ACTION_NAMES], dtype=np.int32)
        self._leg_ctrl_idx  = np.array([self._ctrl_idx[n] for n in _LEG_ACTION_NAMES],   dtype=np.int32)
        self._wheel_ctrl_idx= np.array([self._ctrl_idx[n] for n in _WHEEL_ACTION_NAMES], dtype=np.int32)

        _arm_names  = ["joint1","joint2","joint3","joint4","joint5","joint6"]
        _grip_names = ["joint7","joint8"]
        self._arm_names  = _arm_names
        self._grip_names = _grip_names
        self._arm_ctrl_idx  = np.array([self._ctrl_idx[n] for n in _arm_names],  dtype=np.int32)
        self._grip_ctrl_idx = np.array([self._ctrl_idx[n] for n in _grip_names], dtype=np.int32)
        self._arm_all_idx   = np.array([_n2i[n] for n in _arm_names],  dtype=np.int32)
        self._grip_all_idx  = np.array([_n2i[n] for n in _grip_names], dtype=np.int32)
        self._arm_kp  = np.array([_KP_J1234]*4 + [_KP_J5, _KP_J6], dtype=np.float64)
        self._arm_kd  = np.array([_KD_J1234]*4 + [_KD_J5, _KD_J6], dtype=np.float64)

        self._arm_target  = DEFAULT_JOINT_POS_ALL[self._arm_all_idx].copy().astype(np.float64)
        self._grip_target = DEFAULT_JOINT_POS_ALL[self._grip_all_idx].copy().astype(np.float64)
        self._grip_cmd    = DEFAULT_JOINT_POS_ALL[self._grip_all_idx].copy().astype(np.float64)  # 速率限制后的当前命令位置
        _GRIP_MAX_DELTA   = 0.001  # 每 step 最多移动 1mm，避免目标跳变产生冲击
        self._grip_max_delta = _GRIP_MAX_DELTA
        self._grip_min_tau   = 0  # 最小夹爪力矩（N·m），防止接触时误差小导致力不足

        # 1-step action delay buffer（对应训练时 DelayedPDActuator max_delay=1）
        # 腿部 position target 和轮子 velocity target 各延迟 1 个 policy step 执行
        self._leg_target_buf   = None  # (12,) 上一步的 leg_target
        self._wheel_target_buf = None  # (4,)  上一步的 wheel_target_vel

        # freejoint 地址（机器人根节点）
        # 优先通过名字 "robot_freejoint" 查找；若找不到则用第一个无名 freejoint
        self._root_qpos_adr = 0
        self._root_dof_adr  = 0
        _robot_jnt_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "robot_freejoint")
        if _robot_jnt_id >= 0:
            self._root_qpos_adr = self._model.jnt_qposadr[_robot_jnt_id]
            self._root_dof_adr  = self._model.jnt_dofadr[_robot_jnt_id]
        else:
            # fallback: 找第一个无名的 freejoint
            for i in range(self._model.njnt):
                if self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    _name = mujoco.mj_id2name(
                        self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if _name is None:
                        self._root_qpos_adr = self._model.jnt_qposadr[i]
                        self._root_dof_adr  = self._model.jnt_dofadr[i]
                        break

        self._init_pos  = np.array(robot_init_pos,  dtype=np.float64)
        # MuJoCo quat 顺序是 (w,x,y,z)
        self._init_quat = np.array(robot_init_quat, dtype=np.float64)
        self.reset()

        self._viewer = None
        if self._render_gui:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

    # ──────────────────────────────────────────
    # reset
    # ──────────────────────────────────────────
    def reset(self):
        """重置仿真状态：机器人回到初始站立姿态，物体回到初始位置。"""
        mujoco.mj_resetData(self._model, self._data)

        # 设置根节点位置和四元数
        a = self._root_qpos_adr
        self._data.qpos[a:a+3] = self._init_pos
        # MuJoCo freejoint quat 顺序：w x y z
        self._data.qpos[a+3:a+7] = self._init_quat

        # 设置关节初始角（default_joint_pos）
        for i, name in enumerate(ALL_JOINT_NAMES):
            qpa = self._jnt_qposadr.get(name)
            if qpa is not None:
                self._data.qpos[qpa] = float(DEFAULT_JOINT_POS_ALL[i])

        # 重置机械臂目标
        self._arm_target  = DEFAULT_JOINT_POS_ALL[self._arm_all_idx].copy().astype(np.float64)
        self._grip_target = DEFAULT_JOINT_POS_ALL[self._grip_all_idx].copy().astype(np.float64)

        # 重置 action delay buffer
        self._leg_target_buf   = None
        self._wheel_target_buf = None

        mujoco.mj_forward(self._model, self._data)

    # ──────────────────────────────────────────
    # step
    # ──────────────────────────────────────────
    def step(self, action: np.ndarray):
        """执行一个 policy step（substeps 次物理步）。

        参数
        ----
        action : (16,) float，policy 输出的原始 action（未 clip）
        """
        action = np.asarray(action, dtype=np.float64).flatten()[:16]

        # 读取当前关节角和角速度
        jpos = self._data.qpos[self._all_qpos_idx].copy()  # (24,)
        jvel = self._data.qvel[self._all_dof_idx].copy()   # (24,)

        # 腿关节 position target
        leg_default = DEFAULT_JOINT_POS_ALL[self._leg_all_idx].astype(np.float64)
        leg_target  = action[:12].astype(np.float64) * _LEG_SCALE + leg_default

        # 轮子 velocity target
        wheel_target_vel = action[12:16].astype(np.float64) * _WHEEL_SCALE

        for _ in range(self._substeps):
            # 每个物理步重读状态
            jpos = self._data.qpos[self._all_qpos_idx]
            jvel = self._data.qvel[self._all_dof_idx]

            # 腿 PD 力矩
            leg_q   = jpos[self._leg_all_idx]
            leg_qd  = jvel[self._leg_all_idx]
            leg_tau = _KP_LEG * (leg_target - leg_q) + _KD_LEG * (0.0 - leg_qd)
            np.clip(leg_tau, -76.4, 76.4, out=leg_tau)

            # 轮子速度控制
            wheel_qd  = jvel[self._wheel_all_idx]
            wheel_tau = _KD_WHEEL * (wheel_target_vel - wheel_qd)
            np.clip(wheel_tau, -21.6, 21.6, out=wheel_tau)

            # 机械臂 PD 力矩
            arm_q   = jpos[self._arm_all_idx]
            arm_qd  = jvel[self._arm_all_idx]
            arm_tau = self._arm_kp * (self._arm_target - arm_q) \
                    + self._arm_kd * (0.0 - arm_qd)
            np.clip(arm_tau, -220.0, 220.0, out=arm_tau)

            # 夹爪 PD 力矩（带速率限制和最小力矩）
            grip_q   = jpos[self._grip_all_idx]
            grip_qd  = jvel[self._grip_all_idx]
            # 速率限制：_grip_cmd 每步最多朝目标移动 _grip_max_delta
            delta = np.clip(self._grip_target - self._grip_cmd,
                            -self._grip_max_delta, self._grip_max_delta)
            self._grip_cmd += delta
            grip_tau = _KP_GRIP * (self._grip_cmd - grip_q) \
                     + _KD_GRIP * (0.0 - grip_qd)
            # 最小力矩：保持夹持方向上至少 _grip_min_tau 的力矩
            for i in range(len(grip_tau)):
                sign = np.sign(self._grip_cmd[i] - grip_q[i])
                if sign != 0 and abs(grip_tau[i]) < self._grip_min_tau:
                    grip_tau[i] = sign * self._grip_min_tau
            np.clip(grip_tau, -10.0, 10.0, out=grip_tau)

            # 写入 ctrl
            for i, ci in enumerate(self._leg_ctrl_idx):
                self._data.ctrl[ci] = leg_tau[i]
            for i, ci in enumerate(self._wheel_ctrl_idx):
                self._data.ctrl[ci] = wheel_tau[i]
            for i, ci in enumerate(self._arm_ctrl_idx):
                self._data.ctrl[ci] = arm_tau[i]
            for i, ci in enumerate(self._grip_ctrl_idx):
                self._data.ctrl[ci] = grip_tau[i]

            mujoco.mj_step(self._model, self._data)

        # NaN guard：检测 qpos/qvel 中出现 NaN 时自动重置
        if np.any(np.isnan(self._data.qpos)) or np.any(np.isnan(self._data.qvel)):
            import warnings
            warnings.warn(
                "[MuJoCoEnv] NaN detected in qpos/qvel after mj_step — "
                "auto-resetting simulation.",
                RuntimeWarning, stacklevel=2,
            )
            self.reset()

        if self._viewer is not None:
            self._viewer.sync()

    # ──────────────────────────────────────────
    # 状态读取
    # ──────────────────────────────────────────
    def get_robot_state(self) -> dict:
        """返回机器人当前状态字典，供 obs_builder 和状态机使用。

        返回键：
          pos       (3,)  机体世界位置
          quat      (4,)  机体四元数 (w,x,y,z)
          lin_vel   (3,)  机体线速度（世界系）
          ang_vel   (3,)  机体角速度（机体系）
          joint_pos (24,) 关节角（ALL_JOINT_NAMES 顺序）
          joint_vel (24,) 关节角速度
          yaw       float 偏航角（rad）
        """
        a = self._root_qpos_adr
        pos  = self._data.qpos[a:a+3].copy()
        quat = self._data.qpos[a+3:a+7].copy()  # (w,x,y,z)

        d = self._root_dof_adr
        lin_vel_world = self._data.qvel[d:d+3].copy()
        ang_vel_world = self._data.qvel[d+3:d+6].copy()

        # 角速度转到机体系
        w, x, y, z = quat
        Rt = np.array([
            [1-2*(y*y+z*z),   2*(x*y+w*z),   2*(x*z-w*y)],
            [  2*(x*y-w*z), 1-2*(x*x+z*z),   2*(y*z+w*x)],
            [  2*(x*z+w*y),   2*(y*z-w*x), 1-2*(x*x+y*y)],
        ], dtype=np.float64)
        ang_vel_body = Rt @ ang_vel_world

        joint_pos = self._data.qpos[self._all_qpos_idx].copy()
        joint_vel = self._data.qvel[self._all_dof_idx].copy()

        # 偏航角（绕 Z 轴）
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        return {
            "pos":        pos,
            "quat":       quat,
            "lin_vel":    lin_vel_world,
            "ang_vel":    ang_vel_body,
            "joint_pos":  joint_pos,
            "joint_vel":  joint_vel,
            "yaw":        yaw,
        }

    def get_joint_pos_target(self) -> np.ndarray:
        """返回当前机械臂关节目标角（6维，joint1~joint6）。"""
        return self._arm_target.copy()

    def get_gripper_target(self) -> np.ndarray:
        """返回当前夹爪目标角（2维，joint7/joint8）。"""
        return self._grip_target.copy()

    # ──────────────────────────────────────────
    # 机械臂控制
    # ──────────────────────────────────────────
    def set_arm_target(self, q: np.ndarray):
        """设置机械臂关节目标角（6维，joint1~joint6）。"""
        self._arm_target = np.asarray(q, dtype=np.float64).copy()

    def set_gripper_target(self, q: np.ndarray):
        """设置夹爪目标角（2维，joint7/joint8）。"""
        self._grip_target = np.asarray(q, dtype=np.float64).copy()


    # ──────────────────────────────────────────
    # FREEZE 支持
    # ──────────────────────────────────────────
    def set_root_pose(self, pos: np.ndarray, quat: np.ndarray):
        """强制设置机器人根节点位姿并清零速度（对应 Isaac Lab FREEZE 逻辑）。"""
        a = self._root_qpos_adr
        self._data.qpos[a:a+3]   = pos
        self._data.qpos[a+3:a+7] = quat
        d = self._root_dof_adr
        self._data.qvel[d:d+6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def freeze_root(self):
        """清零根节点线速度/角速度（不改变位姿）。"""
        d = self._root_dof_adr
        self._data.qvel[d:d+6] = 0.0

    # ──────────────────────────────────────────
    # 相机渲染
    # ──────────────────────────────────────────
    def render_camera(
        self,
        camera_name: str = "wrist_cam",
    ) -> tuple:
        """渲染指定相机视角，返回 (rgb, depth)。

        rgb   : (H, W, 3) uint8
        depth : (H, W)    float32，单位 m（distance_to_image_plane）

        注意：相机 site 需要在 scene.xml 或 M20_Piper.xml 里定义。
        如果相机名不存在，抛出 ValueError。
        """
        cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id < 0:
            raise ValueError(
                f"Camera '{camera_name}' not found in model. "
                f"Please add a <camera> tag to scene.xml or M20_Piper.xml."
            )

        renderer = mujoco.Renderer(self._model, height=self._cam_h, width=self._cam_w)
        renderer.update_scene(self._data, camera=camera_name)
        rgb = renderer.render().copy()  # (H,W,3) uint8

        renderer.enable_depth_rendering()
        renderer.update_scene(self._data, camera=camera_name)
        depth = renderer.render().copy()  # (H,W) float32
        renderer.disable_depth_rendering()
        renderer.close()

        return rgb, depth

    # ──────────────────────────────────────────
    # 属性
    # ──────────────────────────────────────────
    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def dt(self):
        """每个 policy step 的时间步长（秒）。"""
        return self._model.opt.timestep * self._substeps

    def get_object_pos(self, name: str) -> np.ndarray:
        """返回场景中 freejoint 物体的世界位置 (x, y, z)。

        name 对应 scene.xml 中 <joint name='{name}_freejoint'> 的物体名，
        例如 'banana'、'apple'、'bowl'。
        """
        jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
        if jid < 0:
            raise ValueError(
                f"Joint '{name}_freejoint' not found in model. "
                f"Make sure the object freejoint is named correctly in scene.xml."
            )
        a = self._model.jnt_qposadr[jid]
        return self._data.qpos[a:a + 3].copy()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
