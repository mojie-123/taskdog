"""Independent, finite ``ARM_INIT -> LIFT`` manipulation episode runner.

The implementation reuses the project's validated AnyGrasp conversion, MoveIt
bridge and MuJoCo local servo, but deliberately does not start Nav2 or the
locomotion policy.  Every call begins from a full MuJoCo reset and ends as soon
as Cartesian LIFT is verified.

独立运行一次机械臂抓取；
从 ARM_INIT 开始；
到确认 LIFT 成功后结束；
复用项目已有的 AnyGrasp、MoveIt 和局部伺服代码；
不启动 Nav2；
不运行机器狗移动策略；
每次 episode 都从完整 MuJoCo reset 开始。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import numpy as np

from custom_envs.scripts.N2M.attached_object_specs import (
    attached_box_pose_matrix,   # 计算这个包围盒相对于抓取 TCP 的姿态。
    resolve_attached_box_spec,   # 查询 cube、apple、orange、mug 等物体对应的 MoveIt 包围盒尺寸和中心偏移
)
from custom_envs.scripts.N2M.config import EpisodeConfig
from custom_envs.scripts.N2M.geometry import angular_error, joint1_towards_cube
from custom_envs.scripts.N2M.interactive import InteractiveController, ViewerOverlay
from custom_envs.scripts.N2M.mujoco_state import (
    BaseWeld,   # 锁住底盘
    place_episode_state,   # 重置并放置机器人和cube
    robot_table_contacts,   # 检测机器人是否碰茶几
)


ARM_HOME = np.array([0.0, 0.5, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
ARM_DEPLOY_TEMPLATE = np.array([0.0, 1.5, -1.5, 0.0, 1.2, 0.0], dtype=np.float64)
GRIPPER_OPEN = np.array([0.035, -0.035], dtype=np.float64)


class EpisodeFailure(RuntimeError):   # 一次抓取 episode 的结构化失败异常
    def __init__(self, stage: str, reason: str, **metrics):   # 记录失败阶段、失败原因、相关测量值
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.metrics = metrics


class CandidateAborted(EpisodeFailure):   # EpisodeFailure 的特殊类型，表示当前候选由操作人员主动终止
    def __init__(self, reason="operator_abort"):
        super().__init__("OPERATOR", reason)


@dataclass
class EpisodeResult:   # 一次候选 M 实验的统一输出数据结构
    candidate_id: str
    scene_id: str
    attempt: int   # 第几次尝试
    success: Optional[bool]
    completed_stage: str
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    cube_pose: list
    base_pose: list
    joint1_target: Optional[float]
    joint1_actual: Optional[float]
    joint1_error: Optional[float]
    elapsed_s: float   # 本次实验总耗时
    episode_steps: int   # 执行了多少个 MuJoCo policy step
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:   # 把 dataclass 转换成普通字典
        return asdict(self)


class ManipulationEpisodeRunner:
    def __init__(
        self,
        scene_path: Path,
        config: EpisodeConfig,
        grasp_checkpoint: Optional[str] = None,
        render: bool = False,
        pause_on_stage: bool = False,
        interactive: Optional[InteractiveController] = None,   # 可选的交互控制器
        debug_dir: Optional[Path] = None,
    ):   # 创建并初始化一个可以反复测试候选 M 的 runner
        self.cfg = config
        self.grasp_checkpoint = grasp_checkpoint
        self.render = bool(render)
        self.pause_on_stage = bool(pause_on_stage)
        self.interactive = interactive or InteractiveController()   # 如果调用者没有提供控制器，就自动创建一个默认控制器
        self.debug_dir = Path(debug_dir) if debug_dir else None

        # RGB-D rendering is required even in batch mode.  Use EGL without a
        # viewer and GLX only for the interactive X11 viewer.  An explicit user
        # MUJOCO_GL setting still wins.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "glx" if self.render else "egl"
        if self.render:
            os.environ.setdefault("DISPLAY", ":0")

        from custom_envs.mujoco.env import MuJoCoEnv

        self.env = MuJoCoEnv(
            robot_init_pos=(0.0, 0.0, config.robot_root_z),
            render=self.render,
            key_callback=self.interactive.key_callback if self.render else None,
            scene_path=str(scene_path),
        )
        self.overlay = ViewerOverlay(self.env)
        self.weld = BaseWeld(self.env)
        self.zeros_action = np.zeros(16, dtype=np.float32)
        self.step_count = 0
        self.current_candidate = None
        self.joint1_target = None
        self.moveit = None
        self._moveit_scene_applied = False
        self.attached = False
        self.attached_id = "n2m_carried_cube"
        self._imports_ready = False
        self._prepare_indices()

    def _prepare_indices(self):   # 建立MuJoCo 机器人状态数组 ↔ Piper 机械臂关节之间的索引关系
        import mujoco

        self.arm_idx = np.asarray(self.env._arm_all_idx, dtype=np.int32)
        self.grip_idx = np.asarray(self.env._grip_all_idx, dtype=np.int32)
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.moveit_state_names = self.arm_joint_names + ["joint7", "joint8"]
        joint_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
        ]   # 把六个关节名转换为 MuJoCo joint ID
        if min(joint_ids) < 0:
            raise RuntimeError("Missing one or more Piper arm joints")
        self.arm_lo = np.array([self.env.model.jnt_range[j, 0] for j in joint_ids])   # 读取机械臂各关节的最小角度
        self.arm_hi = np.array([self.env.model.jnt_range[j, 1] for j in joint_ids])   # 读取机械臂各关节的最大角度

    def _prepare_pipeline_imports(self):   # 延迟导入抓取流水线需要的组件
        if self._imports_ready:
            return
        root = Path(__file__).resolve().parents[3]
        navigation = root / "custom_envs" / "scripts" / "navigation"
        utils = root / "custom_envs" / "utils"
        for path in (str(navigation), str(utils)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from custom_envs.utils.moveit_bridge import MoveItROS2Bridge
        from mujoco_moveit_frames import (
            Pose,
            T_gripper_base_grasp_tcp,
            anygrasp_candidate_to_tcp,
            capture_scan_transforms,
            inv_T,
            mujoco_body_pose_world,
            pose_error,
            pose_to_ros_dict,
            tcp_pose_to_gripper_base_pose,
            validate_camera_optical_transform,
        )
        from mujoco_local_servo import MuJoCoGripperBaseServo, finger_object_contact
        from navigate_mujoco_moveit import PipelineState, _run_grasp_plan

        self.MoveItROS2Bridge = MoveItROS2Bridge
        self.Pose = Pose
        self.T_gripper_base_grasp_tcp = T_gripper_base_grasp_tcp
        self.anygrasp_candidate_to_tcp = anygrasp_candidate_to_tcp
        self.capture_scan_transforms = capture_scan_transforms
        self.inv_T = inv_T
        self.mujoco_body_pose_world = mujoco_body_pose_world
        self.pose_error = pose_error
        self.pose_to_ros_dict = pose_to_ros_dict
        self.tcp_pose_to_gripper_base_pose = tcp_pose_to_gripper_base_pose
        self.validate_camera_optical_transform = validate_camera_optical_transform
        self.MuJoCoGripperBaseServo = MuJoCoGripperBaseServo
        self.finger_object_contact = finger_object_contact
        self.PipelineState = PipelineState
        self.run_grasp_plan = _run_grasp_plan
        self._imports_ready = True

    def _ensure_moveit(self):   # 保证真正需要规划时 MoveIt 已经启动
        self._prepare_pipeline_imports()
        if not self.grasp_checkpoint:
            raise EpisodeFailure("SCAN", "grasp_checkpoint_not_provided")
        if self.moveit is None:
            self.moveit = self.MoveItROS2Bridge(request_timeout=10.0)
            self.moveit.start(timeout=35.0)
            self._moveit_scene_applied = False
            diag = self.validate_camera_optical_transform(self.env)
            print(
                "[N2M] wrist camera transform verified: "
                f"{diag['position_error_m'] * 1e6:.2f}um, "
                f"{math.degrees(diag['rotation_error_rad']):.6f}deg",
                flush=True,
            )

    def close(self):   # 安全释放 MoveIt和 MuJoCo资源
        try:
            self._detach_safely()
            if self.moveit is not None:
                self.moveit.stop()
        finally:
            self.env.close()

    def _arm_q(self) -> np.ndarray:   # 读取j1~j6
        return self.env.get_robot_state()["joint_pos"][self.arm_idx].copy()

    def _grip_q(self) -> np.ndarray:   # 读取j7~j8
        return self.env.get_robot_state()["joint_pos"][self.grip_idx].copy()

    def _moveit_state(self):   # 发送给 MoveIt 的当前机器人状态
        state = self.env.get_robot_state()["joint_pos"]
        positions = np.concatenate([state[self.arm_idx], state[self.grip_idx]])
        return self.moveit_state_names, positions.astype(np.float64)

    def _update_moveit_state(self):   # 同步 MoveIt 状态
        if self.moveit is not None:
            names, positions = self._moveit_state()
            self.moveit.update_joint_state(names, positions)   # 发送

    def _step(self, arm_target=None, grip_target=None):   # 推进一个抓取控制步
        if arm_target is not None:   # 更新机械臂 PD 控制器的目标
            self.env.set_arm_target(np.asarray(arm_target, dtype=np.float64))
        if grip_target is not None:   # 夹爪同理
            self.env.set_gripper_target(np.asarray(grip_target, dtype=np.float64))
        if not self.interactive.honor_pause(self.env):
            raise CandidateAborted()
        self.env.step(self.zeros_action)   # 16 维零动作
        self.step_count += 1
        arm_table_hits = robot_table_contacts(self.env, robot_root_body="link1")   # 检查机械臂是否碰桌
        if arm_table_hits:
            raise EpisodeFailure(
                "SAFETY", "arm_table_contact", contacts=arm_table_hits,
                step_count=self.step_count,
            )   # MuJoCo 真实物理几何检测到机械臂碰桌
        self._update_moveit_state()
        self._update_overlay()   # 更新 viewer 中的调试显示
        if self.step_count > int(self.cfg.max_episode_steps):
            raise EpisodeFailure("EPISODE", "step_budget_exceeded", step_count=self.step_count)   # 超时检查，防止某个异常候选无限运行
        if self.render:   # viewer 限速
            time.sleep(0.005)

    def _update_overlay(self, status="candidate"):
        if not self.current_candidate:
            return
        actual = None
        try:
            actual = float(self._arm_q()[0])
        except Exception:
            pass
        self.overlay.update(
            self.current_candidate["cube_pose"][:3],
            self.current_candidate["base_pose"],
            self.joint1_target,
            actual,
            status=status,
        )   # 用于可视化

    def _checkpoint(self, stage: str, details: str = ""):   # 记录阶段检查点
        print(f"[N2M][{stage}] {details}", flush=True)
        self._update_overlay()
        if self.pause_on_stage and self.render:
            if not self.interactive.wait_for_advance(self.env, stage):
                raise CandidateAborted()

    def reset_candidate(self, candidate: dict) -> list[str]:   # 建立一次候选实验的初始状态
        self.current_candidate = candidate   # 保存候选
        self.step_count = 0
        self.joint1_target = None
        self.interactive.clear_candidate_abort()   # 清除上一个候选可能留下的“终止当前候选”状态
        self.weld.active = False   # 重置 Python 侧的底盘锁状态
        self.attached = False
        place_episode_state(
            self.env,
            candidate["cube_pose"],
            candidate["base_pose"],
            self.cfg.robot_root_z,
            self.cfg.robot_root_roll,
            self.cfg.robot_root_pitch,
        )   # 重置并放置场景
        hits = robot_table_contacts(self.env)   # 检查初始碰桌
        self._update_overlay()
        self._checkpoint(
            "BASE_PLACED",
            f"candidate={candidate['candidate_id']} base={np.round(candidate['base_pose'], 4)} "
            f"cube={np.round(candidate['cube_pose'][:3], 4)} contacts={hits}",
        )
        return hits

    def preview_candidate(self, candidate: dict) -> dict:   # 只预览候选，不执行完整抓取
        hits = self.reset_candidate(candidate)
        self.joint1_target = joint1_towards_cube(
            candidate["base_pose"], candidate["cube_pose"]
        )
        self._update_overlay("failure" if hits else "candidate")
        if self.render:
            self.interactive.wait_for_advance(
                self.env,
                f"preview {candidate['candidate_id']} (contacts={hits})",
            )
        return {
            "candidate_id": candidate["candidate_id"],
            "scene_id": candidate["scene_id"],
            "base_pose": candidate["base_pose"],
            "cube_pose": candidate["cube_pose"],
            "joint1_target": self.joint1_target,
            "initial_table_contacts": hits,
        }

    def _run_arm_target(self, target, stage: str, budget: Optional[int] = None):   # 把机械臂平滑移动到目标关节姿态（arm_init阶段使用）
        target = np.clip(np.asarray(target, dtype=np.float64), self.arm_lo, self.arm_hi)   # 把目标限制在关节范围内
        budget = int(budget or self.cfg.arm_stage_budget)
        start = self._arm_q()   # 起始关节角
        ramp_steps = max(1, int(budget * 0.65))   # 前 65% 步数用于平滑插值
        for index in range(budget):
            alpha = min(1.0, float(index + 1) / ramp_steps)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            command = start + alpha * (target - start)
            self._step(command, GRIPPER_OPEN)
            error = float(np.max(np.abs(self._arm_q() - target)))   # 判断是否收敛
            if alpha >= 1.0 and error <= float(self.cfg.arm_joint_tolerance):
                self._checkpoint(stage, f"converged steps={index + 1} max_error={error:.5f}rad")
                return
        error = float(np.max(np.abs(self._arm_q() - target)))
        raise EpisodeFailure(stage, "arm_target_timeout", max_joint_error=error)   # 超时

    def run_arm_init(self, candidate: dict) -> dict:   # ARM_INIT的指标会写入EpisodeResult.metrics
        base_pose = candidate["base_pose"]
        cube_pose = candidate["cube_pose"]
        target = joint1_towards_cube(base_pose, cube_pose)   # j1目标角度
        margin = math.radians(5.0)
        if not (self.arm_lo[0] + margin <= target <= self.arm_hi[0] - margin):
            raise EpisodeFailure(
                "ARM_INIT", "joint1_out_of_range", target=target,
                lower=float(self.arm_lo[0]), upper=float(self.arm_hi[0]),
            )
        self.joint1_target = float(target)
        self.env.disable_gripper_force_hold()

        current = self._arm_q()
        retract = ARM_HOME.copy()
        retract[0] = current[0]
        self._run_arm_target(retract, "ARM_RETRACTED")   # 除了j1都变为ARM_HOME

        rotate = ARM_HOME.copy()
        rotate[0] = target
        self._run_arm_target(rotate, "JOINT1_ROTATED")   # 旋转j1

        deploy = ARM_DEPLOY_TEMPLATE.copy()
        deploy[0] = target
        self._run_arm_target(deploy, "ARM_DEPLOYED")   # 除了j1都变为ARM_DEPLOY_TEMPLATE

        actual = float(self._arm_q()[0])
        error = abs(angular_error(actual, target))   # 检查j1误差
        rgb, depth = self.env.render_camera("wrist_cam")
        object_mask = None
        visible_pixels = None
        if self.cfg.use_mujoco_visibility_gate:   # 真值检测cube是否在视野中，False
            object_mask = self._object_segmentation_mask(
                "wrist_cam", int(rgb.shape[0]), int(rgb.shape[1])
            )
            visible_pixels = int(object_mask.sum())
        if self.debug_dir is not None:
            out = self.debug_dir / candidate["candidate_id"]
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "wrist_rgb.npy", rgb)
            np.save(out / "wrist_depth.npy", depth)
            if object_mask is not None:
                np.save(out / "wrist_object_mask.npy", object_mask)

        visibility_details = (
            f"object_pixels={visible_pixels}"
            if visible_pixels is not None
            else "object_visibility=deferred_to_YOLOE+SAM2_scan"
        )
        details = (
            f"joint1 target={target:+.4f} actual={actual:+.4f} "
            f"error={error:.5f}rad {visibility_details}"
        )
        self._checkpoint("ARM_INIT_DONE", details)
        if error > float(self.cfg.arm_joint_tolerance):
            raise EpisodeFailure("ARM_INIT", "joint1_not_converged", error=error)
        if (
            visible_pixels is not None
            and visible_pixels < int(self.cfg.wrist_min_object_pixels)
        ):
            raise EpisodeFailure(
                "ARM_INIT", "wrist_cube_not_visible",
                object_pixels=visible_pixels,
            )
        return {
            "joint1_target": target,
            "joint1_actual": actual,
            "joint1_error": error,
            "wrist_object_pixels": visible_pixels,
            "deploy_arm_target": deploy.tolist(),
        }

    @staticmethod
    def _ransac_fit_plane(points, n_iter=150, dist_thresh=0.008):   # 从点云数据中拟合桌面
        points = np.asarray(points, dtype=np.float64)
        count = len(points)
        if count < 10:   # 至少10个点
            return None, None, np.zeros(count, dtype=bool)
        rng = np.random.default_rng(42)
        best_mask = np.zeros(count, dtype=bool)
        best_normal = best_d = None
        for _ in range(n_iter):
            a, b, c = points[rng.choice(count, 3, replace=False)]   # 每次取3个点
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-8:   # 排除共线的3个点
                continue
            normal /= norm
            d = float(normal @ a)   # 原点到平面的距离
            mask = np.abs(points @ normal - d) < dist_thresh
            if int(mask.sum()) > int(best_mask.sum()):   # 选择包含点最多的平面
                best_mask, best_normal, best_d = mask, normal, d
        return best_normal, best_d, best_mask

    @classmethod
    def _ransac_remove_plane(cls, points, n_iter=150, dist_thresh=0.008,
                             min_inlier_ratio=0.15):   # 从点云中删除桌面
        normal, _, mask = cls._ransac_fit_plane(points, n_iter, dist_thresh)
        if normal is None or int(mask.sum()) < int(len(points) * min_inlier_ratio):
            return np.ones(len(points), dtype=bool)
        return ~mask

    def _current_tcp_pose_world(self):   # 计算当前TCP世界位姿
        T_w_gb = self.mujoco_body_pose_world(self.env, "gripper_base").matrix()   # gb在world系下的位姿
        T_w_tcp = T_w_gb @ self.T_gripper_base_grasp_tcp() 
        return self.Pose(T_w_tcp[:3, 3].copy(), T_w_tcp[:3, :3].copy())

    def _current_tcp_pose_base(self):   # 计算当前TCP相对底盘的位姿
        T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
        T_b_tcp = self.inv_T(T_w_b) @ self._current_tcp_pose_world().matrix()
        return self.Pose(T_b_tcp[:3, 3].copy(), T_b_tcp[:3, :3].copy())

    def _tcp_error_base(self, target):   # 计算 TCP 位姿误差
        pos, rot = self.pose_error(self._current_tcp_pose_base(), target)
        return float(np.linalg.norm(pos)), float(np.linalg.norm(rot))

    def _object_pose_in_tcp(self):   # 计算物体相对 TCP 的位姿
        T_w_tcp = self._current_tcp_pose_world().matrix()
        T_w_obj = self.mujoco_body_pose_world(self.env, self.cfg.object_name).matrix()
        T_tcp_obj = self.inv_T(T_w_tcp) @ T_w_obj
        return self.Pose(T_tcp_obj[:3, 3].copy(), T_tcp_obj[:3, :3].copy())

    def _attach_current_object(self, stage: str, object_pose_in_tcp=None) -> dict:
        """Attach the configured object's shared box model to ``grasp_tcp``.把目标物体加入 MoveIt 附着碰撞模型"""

        try:
            spec = resolve_attached_box_spec(
                self.cfg.object_name,
                size_override=self.cfg.moveit_attached_object_size,
                center_override=self.cfg.moveit_attached_object_center,
            )
        except ValueError as exc:
            raise EpisodeFailure(
                stage,
                "attached_object_spec_invalid",
                object_name=self.cfg.object_name,
                error=str(exc),
            ) from exc

        object_pose = (
            self._object_pose_in_tcp()
            if object_pose_in_tcp is None
            else object_pose_in_tcp
        )   # 没有传入object_pose_in_tcp，通过 MuJoCo 当前状态计算object_pose = self._object_pose_in_tcp()
        T_tcp_box = attached_box_pose_matrix(
            object_pose.matrix(), spec.center_xyz
        )   # 碰撞盒中心偏移
        box_pose = self.Pose(
            T_tcp_box[:3, 3].copy(), T_tcp_box[:3, :3].copy()
        )
        reply = self.moveit.attach_box(
            self.attached_id,   # 把盒子附着在 grasp_tcp 上，随tcp一起运动，MoveIt 规划时要考虑它和环境、机械臂其他部分的碰撞，允许它与夹爪和末端相关 link 接触
            self.pose_to_ros_dict(box_pose),
            spec.size_xyz,
            link_name="grasp_tcp",
            touch_links=["gripper_base", "grasp_tcp", "link6", "link7", "link8"],
            timeout=6.0,
        )
        if not reply.get("ok", False):
            raise EpisodeFailure(stage, "moveit_attach_failed", reply=reply)
        self.attached = True
        print(
            f"[N2M][{stage}] attached {self.cfg.object_name} box: "
            f"size={np.round(spec.size_xyz, 4)} "
            f"center_object={np.round(spec.center_xyz, 4)}",
            flush=True,
        )
        return {
            "object_pose_in_tcp": object_pose,
            "box_pose_in_tcp": box_pose,
            "spec": spec,
            "reply": reply,
        }

    def _finger_contacts(self):   # j7和j8是否和object接触
        return self.finger_object_contact(self.env, self.cfg.object_name)

    def _object_segmentation_mask(self, camera_name: str, height: int, width: int):   # Mujoco真值分割，当前use_mujoco_visibility_gate=False所以无效
        """Render a lighting-independent mask for the configured target body."""
        import mujoco

        object_bid = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.object_name
        )
        if object_bid < 0:
            raise EpisodeFailure("ARM_INIT", "target_body_not_found")

        def belongs_to_object(body_id):
            current = int(body_id)
            while current > 0:
                if current == object_bid:
                    return True
                current = int(self.env.model.body_parentid[current])
            return current == object_bid

        geom_ids = np.asarray([
            geom_id for geom_id in range(int(self.env.model.ngeom))
            if belongs_to_object(self.env.model.geom_bodyid[geom_id])
        ], dtype=np.int32)
        renderer = mujoco.Renderer(self.env.model, height=height, width=width)
        try:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.env.data, camera=camera_name)
            segmentation = renderer.render().copy()
        finally:
            renderer.close()
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            raise EpisodeFailure(
                "ARM_INIT", "unexpected_segmentation_shape",
                shape=list(segmentation.shape),
            )
        return (
            np.isin(segmentation[..., 0], geom_ids)
            & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        )

    def _finger_normal_forces(self):   # 统计左右手指法向接触力
        import mujoco

        out = {"link7": 0.0, "link8": 0.0}
        model, data = self.env.model, self.env.data
        object_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.object_name
        )
        finger_bids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in out
        }
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            b1 = int(model.geom_bodyid[int(contact.geom1)])
            b2 = int(model.geom_bodyid[int(contact.geom2)])
            if object_bid not in (b1, b2):   # 没碰到物体
                continue
            for name, bid in finger_bids.items():
                if bid in (b1, b2):    # 手指有接触记录
                    force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, data, index, force)
                    out[name] += max(0.0, float(force[0]))   # 多个接触，把力相加
        return out

    def _dense_commands(self, reply):   # 把 MoveIt 轨迹加密（相当于添加目标关节节点，让轨迹更平滑
        path = self.MoveItROS2Bridge.extract_arm_path(
            reply.get("trajectory", {}), self.arm_joint_names
        )
        if len(path) == 0:
            return path
        return self.MoveItROS2Bridge.densify_joint_path(
            path, np.array([0.012, 0.018, 0.018, 0.015, 0.015, 0.018])
        )   # 限制相邻控制命令之间各关节最大变化量

    def _execute_path(self, commands, target_pose, stage, joint_tolerance,
                      position_tolerance=0.015, rotation_tolerance=math.radians(10),
                      grip_target=GRIPPER_OPEN, extra_settle=220):   # 在 MuJoCo 中执行 MoveIt 轨迹
        commands = np.asarray(commands, dtype=np.float64)   # 加密后的六关节指令
        if len(commands) == 0:
            raise EpisodeFailure(stage, "empty_trajectory")
        for command in commands:   # 依次执行关节路点
            self._step(command, grip_target)
        final = commands[-1]   # 保持最后的命令
        for settle in range(int(extra_settle) + 1):   # 还没到达最终目标TCP，还可以再执行至多extra_settle步
            qerr = float(np.max(np.abs(self._arm_q() - final)))   # 最大关节误差
            pos_err, rot_err = self._tcp_error_base(target_pose)   # TCP位姿误差
            if (qerr <= float(joint_tolerance)
                    and pos_err <= float(position_tolerance)
                    and rot_err <= float(rotation_tolerance)):
                self._checkpoint(
                    stage,
                    f"qerr={qerr:.4f} pos={pos_err*1000:.1f}mm "
                    f"rot={math.degrees(rot_err):.2f}deg",
                )
                return {"qerr": qerr, "tcp_pos_error": pos_err, "tcp_rot_error": rot_err}
            if settle < int(extra_settle):
                self._step(final, grip_target)
        raise EpisodeFailure(
            stage, "trajectory_not_settled", qerr=qerr,
            tcp_pos_error=pos_err, tcp_rot_error=rot_err,
        )

    def _scan_and_propose(self):   # 扫描并生成 AnyGrasp 候选
        depth_accum = []
        scan_rgb = None
        deploy = ARM_DEPLOY_TEMPLATE.copy()
        deploy[0] = self.joint1_target   # 保持ARM_INIT阶段的姿态
        for index in range(int(self.cfg.scan_warmup + self.cfg.scan_frames)):
            rgb, depth = self.env.render_camera("wrist_cam")
            if index >= int(self.cfg.scan_warmup):   # 热身阶段不保存点云信息
                if depth.ndim == 3:
                    depth = depth[..., 0]
                depth_accum.append(depth.astype(np.float32))   # 保存多帧深度信息
                if scan_rgb is None:
                    scan_rgb = rgb   # rgb只要一张
            self._step(deploy, GRIPPER_OPEN)   # 保持夹爪张开

        state = self.env.get_robot_state()
        scan_transforms = self.capture_scan_transforms(self.env)
        args = SimpleNamespace(**self.cfg.to_dict())
        args.grasp_checkpoint = self.grasp_checkpoint
        args.object = self.cfg.object_name
        # The reused worker communicates through this legacy temporary file.
        # Removing it first prevents a crashed worker from leaving a stale
        # result that could be mistaken for the current observation.
        Path("/tmp/grasp_result.npz").unlink(missing_ok=True)   # 删除旧的 AnyGrasp 临时结果
        next_state, result = self.run_grasp_plan(
            self.env,
            args,
            depth_accum,   # 深度信息
            scan_rgb,   # rgb
            self._arm_q(),   # 当前关节角
            state["pos"],   # 拍摄扫描图像时的位姿
            state["quat"],
            state["pos"],   # 执行抓取规划时的当前状态
            state["quat"],
            self._ransac_fit_plane,   # 平面拟合函数
            self._ransac_remove_plane,   # 去平面函数
            scan_transforms,   # 坐标变换（world/base/wrist camera系）
            build_support_surface=bool(self.cfg.dynamic_support_surface),   # 是否构建动态支撑面
            use_target_mask=bool(self.cfg.semantic_target_mask),   # 是否使用语义目标 mask
        )
        if result is None or next_state != self.PipelineState.MOVEIT_PRE_GRASP:
            raise EpisodeFailure("GRASP_PLAN", "anygrasp_no_usable_candidate")
        self._checkpoint(
            "SCAN_DONE",
            f"frames={len(depth_accum)} candidates={len(result.get('candidate_idxs', []))}",
        )   # 记录推理结果
        return result

    def _apply_dynamic_support_surface(self, grasp_result):   # 添加动态桌面碰撞体
        """Apply this scan's RANSAC table before checking grasp candidates."""
        if not self.cfg.dynamic_support_surface:
            return

        if not self._moveit_scene_applied:
            reply = self.moveit.apply_default_scene(timeout=6.0)
            self._moveit_scene_applied = True
            if reply.get("ok", False):
                print(
                    "[N2M][MOVEIT] PlanningScene: floor applied; "
                    "legacy fixed tables removed/already absent",
                    flush=True,
                )
            else:
                # N2M deliberately stops its Nav2 bridge before manipulation,
                # so map->base_link may no longer be available when the
                # scene-independent map-frame floor is applied.  The detected
                # table below is expressed in base_link and remains mandatory.
                print(
                    f"[N2M][MOVEIT] map-frame floor/legacy cleanup not "
                    f"acknowledged: {reply}; continuing with the required "
                    "base_link RANSAC support surface",
                    flush=True,
                )

        if grasp_result.get("_support_surface_applied", False):
            return
        surface = grasp_result.get("support_surface")
        if surface is None:
            raise EpisodeFailure(
                "MOVEIT_PRE_GRASP", "ransac_support_surface_missing"
            )

        # ``_run_grasp_plan`` returns the same world/map-frame horizontal box
        # used by the standalone MoveIt entry point.  N2M stops the Nav2 bridge
        # before manipulation, so convert that box to the robot-rooted frame
        # while the base is welded.  This removes any dependency on a live
        # map->base_link TF without changing the RANSAC-derived geometry.
        T_w_b = np.asarray(
            grasp_result["scan_transforms"]["T_world_base"],
            dtype=np.float64,
        )
        T_b_w = self.inv_T(T_w_b)
        T_w_surface = np.eye(4, dtype=np.float64)
        T_w_surface[:3, 3] = np.asarray(
            surface["pose"]["position"], dtype=np.float64
        )
        T_b_surface = T_b_w @ T_w_surface
        surface_pose_base = self.pose_to_ros_dict(
            self.Pose(
                T_b_surface[:3, 3].copy(),
                T_b_surface[:3, :3].copy(),
            )
        )
        reply = self.moveit.apply_support_surface(
            surface_pose_base,
            surface["size_xyz"],
            frame_id="base_link",
            object_id=surface["object_id"],
            timeout=6.0,
        )
        if not reply.get("ok", False):
            raise EpisodeFailure(
                "MOVEIT_PRE_GRASP", "dynamic_support_surface_failed", reply=reply
            )
        grasp_result["_support_surface_applied"] = True
        print(
            f"[N2M][MOVEIT] PlanningScene: RANSAC table applied "
            f"height={surface['table_height']:.4f}m "
            f"size={np.round(surface['size_xyz'], 3)}",
            flush=True,
        )

    def _select_and_execute_pregrasp(self, grasp_result):   # 筛选候选并到达预抓取位姿
        self._apply_dynamic_support_surface(grasp_result)   # 添加桌面
        names, positions = self._moveit_state()   # 当前 MoveIt 状态和扫描变换
        T_b_c = grasp_result["scan_transforms"]["T_base_camera_optical"]
        R_w_b = grasp_result["scan_transforms"]["T_world_base"][:3, :3]
        selected = None
        last_reason = "no_candidate"
        for candidate_index in grasp_result.get("candidate_idxs", []):   # 按分数顺序遍历候选
            try:
                geometry = self.anygrasp_candidate_to_tcp(
                    T_b_c,
                    grasp_result["gr_translations"][candidate_index],
                    grasp_result["gr_rotations"][candidate_index],
                    float(grasp_result["gr_depths"][candidate_index]),
                    insertion=float(self.cfg.moveit_insertion),
                    pregrasp_distance=float(self.cfg.moveit_pregrasp_distance),
                    local_reach_distance=float(self.cfg.moveit_local_reach_distance),
                )
                approach_world = R_w_b @ np.asarray(geometry["approach_base"])
                if float(approach_world[2]) > float(self.cfg.moveit_max_approach_world_z):
                    last_reason = "upward_approach"
                    continue
                target = self.pose_to_ros_dict(geometry["pregrasp_tcp_pose_base"])
                ik = self.moveit.compute_ik(target, names, positions, avoid_collisions=True)
                if not ik.get("ok", False):
                    last_reason = "ik_or_goal_collision"
                    continue
                plan = self.moveit.plan_to_pose(
                    target, names, positions,
                    position_tolerance=float(self.cfg.moveit_position_tolerance),
                    orientation_tolerance=float(self.cfg.moveit_orientation_tolerance),
                    allowed_planning_time=float(self.cfg.moveit_plan_time),
                )
                commands = self._dense_commands(plan) if plan.get("ok", False) else []
                if len(commands) == 0:
                    last_reason = "path_planning_failed"
                    continue
                selected = {"geometry": geometry, "commands": commands,
                            "candidate_index": int(candidate_index)}
                break
            except Exception as exc:
                last_reason = f"candidate_exception:{exc}"
        if selected is None:
            raise EpisodeFailure("MOVEIT_PRE_GRASP", last_reason)

        self._execute_path(
            selected["commands"],
            selected["geometry"]["pregrasp_tcp_pose_base"],
            "MOVEIT_PRE_GRASP",
            self.cfg.moveit_exec_joint_tolerance,
            self.cfg.moveit_exec_tcp_position_tolerance,
            self.cfg.moveit_exec_tcp_orientation_tolerance,
        )   # 执行 pre-grasp 路径
        return selected   # 返回 selected

    def _execute_approach(self, selected):   # 从 pre-grasp 接近到目标前 1.5 cm
        names, positions = self._moveit_state()
        target_pose = selected["geometry"]["near_tcp_pose_base"]
        reply = self.moveit.compute_cartesian_path(
            [self.pose_to_ros_dict(target_pose)],
            names, positions,
            max_step=float(self.cfg.moveit_cart_step),
            jump_threshold=0.0,
            avoid_collisions=True,
        )
        fraction = float(reply.get("fraction", 0.0))
        if not reply.get("ok", False) or fraction < float(self.cfg.moveit_cart_fraction):
            raise EpisodeFailure(
                "MOVEIT_APPROACH", "cartesian_path_failed", fraction=fraction,
            )
        commands = self._dense_commands(reply)
        self._execute_path(
            commands, target_pose, "MOVEIT_APPROACH",
            self.cfg.moveit_approach_joint_tolerance,
            self.cfg.moveit_exec_tcp_position_tolerance,
            self.cfg.moveit_exec_tcp_orientation_tolerance,
            extra_settle=180,
        )   # 直接复用

    def _run_local_reach(self, selected):   # 接触感知的最后局部接近
        servo = self.MuJoCoGripperBaseServo(self.env)   # gripper_base 局部伺服器
        target_tcp_b = selected["geometry"]["contact_tcp_pose_base"]
        target_gb_b = self.tcp_pose_to_gripper_base_pose(target_tcp_b)   # TCP目标转换成gb目标
        first_contact_position = None
        first_contact_side = None
        overforce_steps = 0

        for index in range(int(self.cfg.local_reach_budget)):
            T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
            T_w_gb = T_w_b @ target_gb_b.matrix()   #  gb目标在世界系的位姿
            target_gb_w = self.Pose(T_w_gb[:3, 3], T_w_gb[:3, :3])
            cur_tcp = self._current_tcp_pose_world()
            T_w_tcp_target = T_w_b @ target_tcp_b.matrix()
            target_tcp = self.Pose(T_w_tcp_target[:3, 3], T_w_tcp_target[:3, :3])
            approach = T_w_b[:3, :3] @ target_tcp_b.rotation[:, 2]   # 计算approach方向
            approach /= max(float(np.linalg.norm(approach)), 1e-12)
            delta_to_target = target_tcp.position - cur_tcp.position
            axial_remaining = float(np.dot(delta_to_target, approach))
            total_pos_error = float(np.linalg.norm(delta_to_target))
            lateral_error = float(np.linalg.norm(
                delta_to_target - axial_remaining * approach
            ))
            contacts = self._finger_contacts()  # 判断手指接触
            count = int(contacts["link7"]) + int(contacts["link8"])
            if count == 2:   # 双指接触
                forces = self._finger_normal_forces()
                print(
                    f"[LOCAL_REACH][DIAG] bilateral contact step={index} "
                    f"axial_remaining={axial_remaining*1000:.1f}mm "
                    f"total_pos_err={total_pos_error*1000:.1f}mm "
                    f"lateral_err={lateral_error*1000:.1f}mm "
                    f"Fn7={float(forces['link7']):.2f}N "
                    f"Fn8={float(forces['link8']):.2f}N -> CLOSE",
                    flush=True,
                )
                self._checkpoint("LOCAL_REACH", "bilateral finger contact")
                return

            speed_scale = 1.0
            if count == 1:   # 单侧接触
                side = "link7" if contacts["link7"] else "link8"
                first_on_side = (
                    first_contact_position is None or first_contact_side != side
                )
                if first_on_side:
                    first_contact_position = cur_tcp.position.copy()   # 单侧接触后的继续深入起点
                    first_contact_side = side
                    overforce_steps = 0
                advance = max(
                    0.0,
                    float(np.dot(cur_tcp.position - first_contact_position, approach)),
                )
                force = float(self._finger_normal_forces()[side])   # 单侧力不能太大
                overforce_steps = overforce_steps + 1 if (
                    force >= float(self.cfg.local_single_contact_force_limit)
                ) else 0
                if first_on_side or index % 5 == 0:
                    print(
                        f"[LOCAL_REACH][DIAG] unilateral {side} step={index} "
                        f"axial_remaining={axial_remaining*1000:.1f}mm "
                        f"total_pos_err={total_pos_error*1000:.1f}mm "
                        f"lateral_err={lateral_error*1000:.1f}mm "
                        f"advance={advance*1000:.1f}mm "
                        f"Fn={force:.2f}N "
                        f"overforce={overforce_steps}/"
                        f"{int(self.cfg.local_single_contact_force_confirm_steps)} "
                        f"speed_scale={float(self.cfg.local_single_contact_speed_scale):.2f}",
                        flush=True,
                    )
                if overforce_steps >= int(self.cfg.local_single_contact_force_confirm_steps):
                    print(
                        f"[LOCAL_REACH][DIAG] exit=unilateral_overforce "
                        f"side={side} axial_remaining={axial_remaining*1000:.1f}mm "
                        f"total_pos_err={total_pos_error*1000:.1f}mm "
                        f"lateral_err={lateral_error*1000:.1f}mm "
                        f"advance={advance*1000:.1f}mm Fn={force:.2f}N",
                        flush=True,
                    )
                    raise EpisodeFailure(
                        "LOCAL_REACH", "unilateral_overforce",
                        side=side, force=force, advance=advance,
                    )
                if advance >= float(self.cfg.local_single_contact_max_advance):
                    print(
                        f"[LOCAL_REACH][DIAG] exit=unilateral_advance_limit "
                        f"side={side} axial_remaining={axial_remaining*1000:.1f}mm "
                        f"total_pos_err={total_pos_error*1000:.1f}mm "
                        f"lateral_err={lateral_error*1000:.1f}mm "
                        f"advance={advance*1000:.1f}mm Fn={force:.2f}N -> CLOSE",
                        flush=True,
                    )
                    self._checkpoint(
                        "LOCAL_REACH",
                        f"unilateral bounded advance complete side={side} advance={advance:.4f}m",
                    )   # 单侧接触后又前进了local_single_contact_max_advance，进入CLOSE
                    return
                speed_scale = float(self.cfg.local_single_contact_speed_scale)   # 减速
            else:
                first_contact_position = None
                first_contact_side = None
                overforce_steps = 0
                if index == 0 or index % 10 == 0:
                    print(
                        f"[LOCAL_REACH][DIAG] no contact step={index} "
                        f"axial_remaining={axial_remaining*1000:.1f}mm "
                        f"total_pos_err={total_pos_error*1000:.1f}mm "
                        f"lateral_err={lateral_error*1000:.1f}mm "
                        f"configured_local_reach="
                        f"{float(self.cfg.moveit_local_reach_distance)*1000:.1f}mm "
                        "speed_scale=1.00",
                        flush=True,
                    )

            raw_command, diagnostics = servo.step_world(target_gb_w)   # 局部伺服命令
            current = self._arm_q()
            command = current + speed_scale * (raw_command - current)
            self._step(command, GRIPPER_OPEN)
            if diagnostics.pos_error_m < 0.002 and diagnostics.rot_error_rad < 0.10:    # 位姿误差判定
                print(
                    f"[LOCAL_REACH][DIAG] exit=geometric_target "
                    f"step={index} axial_remaining={axial_remaining*1000:.1f}mm "
                    f"total_pos_err={total_pos_error*1000:.1f}mm "
                    f"lateral_err={lateral_error*1000:.1f}mm "
                    f"servo_pos_err={diagnostics.pos_error_m*1000:.1f}mm "
                    f"servo_rot_err={math.degrees(diagnostics.rot_error_rad):.1f}deg "
                    "-> CLOSE",
                    flush=True,
                )
                self._checkpoint(
                    "LOCAL_REACH",
                    f"geometric target pos={diagnostics.pos_error_m:.4f}m",
                )
                return
        raise EpisodeFailure("LOCAL_REACH", "timeout")

    def _run_close(self):
        arm_hold = self._arm_q()
        finger_targets = [None, None]
        contact_history = []   # 两根手指第一次接触物体后需要保持的位置
        object_history = []   # 保存最近20步两指是否接触
        normal_delta = float(self.env._grip_max_delta)   # 保存最近20步物体位置
        force_hold_started = False   # 是否已经进入2 N恒力夹持
        for _ in range(int(self.cfg.close_budget)):
            contacts = self._finger_contacts()
            grip = self._grip_q()
            contact_history.append((bool(contacts["link7"]), bool(contacts["link8"])))
            contact_history = contact_history[-20:]
            object_history.append(self.env.get_object_pos(self.cfg.object_name).copy())
            object_history = object_history[-20:]

            if finger_targets[0] is None and contacts["link7"]:
                finger_targets[0] = float(grip[0] - 0.003125)   # link7 第一次接触后，其目标设为当前关节位置再闭合 3.125 mm
            if finger_targets[1] is None and contacts["link8"]:
                finger_targets[1] = float(grip[1] + 0.003125)

            touched = int(finger_targets[0] is not None) + int(finger_targets[1] is not None)
            self.env._grip_max_delta = 0.0004 if (touched == 1 and not force_hold_started) else normal_delta
            if touched == 2 and not force_hold_started:
                self.env.set_gripper_force_hold(2.0)
                force_hold_started = True

            target = np.array([
                finger_targets[0] if finger_targets[0] is not None else 0.0,
                finger_targets[1] if finger_targets[1] is not None else 0.0,
            ])
            self._step(arm_hold, target)

        self.env._grip_max_delta = normal_delta
        rate7 = sum(int(x[0]) for x in contact_history) / max(len(contact_history), 1)
        rate8 = sum(int(x[1]) for x in contact_history) / max(len(contact_history), 1)
        motion = (
            float(np.linalg.norm(object_history[-1] - object_history[0]))
            if len(object_history) >= 2 else float("inf")
        )
        self._checkpoint(
            "CLOSE_DONE",
            f"contact_rate={rate7:.2f}/{rate8:.2f} object_motion={motion*1000:.2f}mm",
        )
        if not force_hold_started:
            raise EpisodeFailure(
                "CLOSE", "bilateral_contact_never_established",
                contact_rate=[rate7, rate8], object_motion=motion,
            )
        return {
            "arm_hold": arm_hold,
            "finger_targets": target,
            "contact_rate": [rate7, rate8],
            "object_motion": motion,
        }

    def _detach_safely(self):   # 清除 MoveIt 附着碰撞体
        if not self.attached or self.moveit is None:
            return
        try:
            self.moveit.detach_object(self.attached_id, link_name="grasp_tcp", timeout=6.0)
        except Exception as exc:
            print(f"[N2M] detach warning: {exc}", flush=True)
        self.attached = False

    def _lift_guard(self, initial_object_tcp, miss_steps):   # LIFT阶段
        contacts = self._finger_contacts()
        miss_steps = miss_steps + 1 if not (contacts["link7"] or contacts["link8"]) else 0   # 是否连续失去双指接触
        if miss_steps >= int(self.cfg.moveit_lift_contact_miss_steps):
            raise EpisodeFailure("LIFT", "both_finger_contacts_lost", miss_steps=miss_steps)
        current = self._object_pose_in_tcp().position
        slip = float(np.linalg.norm(current - initial_object_tcp))
        if slip > float(self.cfg.moveit_lift_max_object_slip):   # 物体是否相对 TCP 滑动过多
            raise EpisodeFailure("LIFT", "object_tcp_slip", slip=slip)
        return miss_steps, slip

    def _run_lift(self, close_info):   # 验证夹持并抬升物体
        object_start = self.env.get_object_pos(self.cfg.object_name).copy()
        seen = np.zeros(2, dtype=np.int32)
        hold_start = object_start.copy()
        for _ in range(int(self.cfg.lift_hold_steps)):
            self._step(close_info["arm_hold"], close_info["finger_targets"])
            contacts = self._finger_contacts()
            seen += [int(contacts["link7"]), int(contacts["link8"])]
        rates = seen / max(int(self.cfg.lift_hold_steps), 1)
        hold_move = float(np.linalg.norm(
            self.env.get_object_pos(self.cfg.object_name) - hold_start
        ))
        if rates.min() < 0.40 or hold_move > 0.002:
            raise EpisodeFailure(
                "LIFT", "hold_unstable",
                contact_rate=rates.tolist(), object_motion=hold_move,
            )

        object_tcp = self._object_pose_in_tcp()
        initial_object_tcp = object_tcp.position.copy()
        self._attach_current_object("LIFT", object_pose_in_tcp=object_tcp)

        T_w_b = self.mujoco_body_pose_world(self.env, "base_link").matrix()
        T_w_target = self._current_tcp_pose_world().matrix()
        T_w_target[:3, 3] += np.array([0.0, 0.0, float(self.cfg.moveit_lift_distance)])
        T_b_target = self.inv_T(T_w_b) @ T_w_target
        target = self.Pose(T_b_target[:3, 3].copy(), T_b_target[:3, :3].copy())
        names, positions = self._moveit_state()
        reply = self.moveit.compute_cartesian_path(
            [self.pose_to_ros_dict(target)], names, positions,
            max_step=float(self.cfg.moveit_cart_step),
            jump_threshold=0.0,
            avoid_collisions=True,
            velocity_scaling=0.15,
            acceleration_scaling=0.15,
            timeout=8.0,
        )
        fraction = float(reply.get("fraction", 0.0))
        if not reply.get("ok", False) or fraction < float(self.cfg.moveit_lift_cart_fraction):
            raise EpisodeFailure("LIFT", "cartesian_lift_plan_failed", fraction=fraction)
        commands = self._dense_commands(reply)
        if len(commands) == 0:
            raise EpisodeFailure("LIFT", "empty_lift_trajectory")

        miss_steps = 0
        max_slip = 0.0
        for command in commands:
            self._step(command, close_info["finger_targets"])
            miss_steps, slip = self._lift_guard(initial_object_tcp, miss_steps)
            max_slip = max(max_slip, slip)
        final = commands[-1]
        for _ in range(220):
            qerr = float(np.max(np.abs(self._arm_q() - final)))
            pos_err, rot_err = self._tcp_error_base(target)
            if (qerr <= float(self.cfg.moveit_lift_joint_tolerance)
                    and pos_err <= 0.020 and rot_err <= math.radians(12.0)):
                break
            self._step(final, close_info["finger_targets"])
            miss_steps, slip = self._lift_guard(initial_object_tcp, miss_steps)
            max_slip = max(max_slip, slip)
        else:
            raise EpisodeFailure(
                "LIFT", "lift_execution_not_settled",
                qerr=qerr, tcp_pos_error=pos_err, tcp_rot_error=rot_err,
            )

        object_final = self.env.get_object_pos(self.cfg.object_name).copy()
        rise = float(object_final[2] - object_start[2])
        self._checkpoint(
            "LIFT_DONE",
            f"rise={rise*1000:.1f}mm qerr={qerr:.4f} max_slip={max_slip*1000:.1f}mm",
        )
        if rise < float(self.cfg.lift_success_rise):
            raise EpisodeFailure("LIFT", "insufficient_object_rise", rise=rise)
        self._detach_safely()
        return {
            "object_start": object_start.tolist(),
            "object_final": object_final.tolist(),
            "object_rise": rise,
            "max_object_tcp_slip": max_slip,
            "lift_fraction": fraction,
            "hold_contact_rate": rates.tolist(),
            "hold_object_motion": hold_move,
        }

    def run_episode(self, candidate: dict, attempt: int = 0,
                    stop_after: str = "lift") -> EpisodeResult:   # 串联一次完整抓取实验
        started = time.monotonic()
        arm_info = {}
        completed = "RESET"
        try:
            initial_hits = self.reset_candidate(candidate)
            if initial_hits and self.cfg.reject_initial_table_contact:
                raise EpisodeFailure(
                    "BASE_PLACED", "initial_robot_table_collision", contacts=initial_hits
                )
            if initial_hits:
                print(
                    f"[N2M][BASE_PLACED] warning: retaining candidate with "
                    f"initial table contacts {initial_hits}",
                    flush=True,
                )
            self.weld.enable()
            arm_info = self.run_arm_init(candidate)
            arm_info["initial_table_contacts"] = initial_hits
            completed = "ARM_INIT"
            if stop_after.lower() == "arm_init":
                self._update_overlay("success")
                return self._result(
                    candidate, attempt, None, completed, None, None,
                    started, arm_info, arm_info,
                )

            self._ensure_moveit()
            grasp = self._scan_and_propose()
            completed = "GRASP_PLAN"
            selected = self._select_and_execute_pregrasp(grasp)
            completed = "MOVEIT_PRE_GRASP"
            self._execute_approach(selected)
            completed = "MOVEIT_APPROACH"
            self._run_local_reach(selected)
            completed = "LOCAL_REACH"
            close_info = self._run_close()
            completed = "CLOSE"
            lift_info = self._run_lift(close_info)
            completed = "LIFT"
            metrics = dict(arm_info)
            metrics.update(lift_info)
            metrics["selected_anygrasp_index"] = selected["candidate_index"]
            self._update_overlay("success")
            return self._result(
                candidate, attempt, True, completed, None, None,
                started, arm_info, metrics,
            )
        except EpisodeFailure as failure:
            metrics = dict(arm_info)
            metrics.update(failure.metrics)
            self._update_overlay("failure")
            return self._result(
                candidate, attempt, False, completed,
                failure.stage, failure.reason, started, arm_info, metrics,
            )
        except Exception as exc:
            self._update_overlay("failure")
            return self._result(
                candidate, attempt, False, completed,
                "INTERNAL", repr(exc), started, arm_info, arm_info,
            )
        finally:
            self._detach_safely()
            try:
                self.env.disable_gripper_force_hold()
            except Exception:
                pass

    def _result(self, candidate, attempt, success, completed, failure_stage,
                failure_reason, started, arm_info, metrics):   # 整理实验结果
        actual = arm_info.get("joint1_actual")
        target = arm_info.get("joint1_target", self.joint1_target)
        error = arm_info.get("joint1_error")
        drift_position, drift_rotation = self.weld.drift()
        full_metrics = dict(metrics)
        full_metrics.update({
            "base_weld_drift_m": drift_position,
            "base_weld_drift_rad": drift_rotation,
        })
        return EpisodeResult(
            candidate_id=candidate["candidate_id"],
            scene_id=candidate["scene_id"],
            attempt=int(attempt),
            success=success,
            completed_stage=completed,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            cube_pose=list(candidate["cube_pose"]),
            base_pose=list(candidate["base_pose"]),
            joint1_target=None if target is None else float(target),
            joint1_actual=None if actual is None else float(actual),
            joint1_error=None if error is None else float(error),
            elapsed_s=float(time.monotonic() - started),
            episode_steps=int(self.step_count),
            metrics=full_metrics,
        )
