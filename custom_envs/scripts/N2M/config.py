"""Fixed six-scene pilot configuration and episode defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house.xml"   # 场景路径（仅用于搜索M）
DEFAULT_DATA_DIR = PACKAGE_DIR / "data" / "manipulation_pose_search"   # 数据存储目录（搜索可成功操作的底盘位姿 M 时产生的数据）
DEFAULT_PERCEPTION_CONFIG = (
    PROJECT_ROOT / "custom_envs" / "config" / "sysnav.yaml"
)   # 感知配置文件路径


@dataclass(frozen=True)
class EdgeSpec:   # 桌边定义
    edge_id: str   # 桌边名称
    outward_xy: Tuple[float, float]   # 从桌面内部向外指的二维单位方向（EDGES）


EDGES: Dict[str, EdgeSpec] = {
    "left": EdgeSpec("left", (-1.0, 0.0)),
    "right": EdgeSpec("right", (1.0, 0.0)),
    "bottom": EdgeSpec("bottom", (0.0, -1.0)),
    "top": EdgeSpec("top", (0.0, 1.0)),
}   # 4条边


@dataclass(frozen=True)
class CubeScene:   # 6个固定cube场景
    scene_id: str   # 场景唯一名称
    cube_xyz: Tuple[float, float, float]   # cube位置
    approach_edges: Tuple[str, ...]   # 这个cube位置允许从哪些桌边进行抓取
    category: str   # 场景分类（corner/long_edge/short_edge），主要用于结果分组、数据统计和评估


CUBE_SCENES: Dict[str, CubeScene] = {
    "corner_lb": CubeScene("corner_lb", (5.30, 3.50, 0.475), ("left", "bottom"), "corner"),
    "corner_rt": CubeScene("corner_rt", (9.70, 3.80, 0.475), ("right", "top"), "corner"),
    "long_bottom": CubeScene("long_bottom", (7.50, 3.50, 0.475), ("bottom",), "long_edge"),
    "long_top": CubeScene("long_top", (7.50, 3.80, 0.475), ("top",), "long_edge"),
    "short_left": CubeScene("short_left", (5.30, 3.65, 0.475), ("left",), "short_edge"),
    "short_right": CubeScene("short_right", (9.70, 3.65, 0.475), ("right",), "short_edge"),
}


# Actual base state from a successful legacy ARM_INIT -> LIFT run:
# pos=[4.791462, 3.558308, 0.563874]
# quat(wxyz)=[0.697786, 0.010601, 0.013125, 0.716107]
# The normalized quaternion is RPY ~= (1.9251deg, 0.1796deg, 91.4878deg).
DEFAULT_REFERENCE_BASE = (4.791462, 3.558308, 1.5967632116203363)   # 真实成功位姿
DEFAULT_ROBOT_ROOT_Z = 0.563874   # 机器人底盘根节点的世界高度
DEFAULT_ROBOT_ROOT_ROLL = 0.03359878103266884
DEFAULT_ROBOT_ROOT_PITCH = 0.003133989573390085
REFERENCE_SCENE_ID = "short_left"
REFERENCE_EDGE_ID = "left"


@dataclass
class CandidateGrid:   # 候选M搜索网格
    # Zero-first ordering makes --max-candidates useful during visual inspection:
    # nominal M is always candidate 0, followed by progressively wider offsets.
    # Normal offset is defined positive *into* the table.  The successful
    # reference M is already at the closest acceptable distance, so all newly
    # searched offsets are strictly negative (away from the table).
    normal_offsets: List[float] = field(
        default_factory=lambda: [0.0, -0.05, -0.10, -0.15]
    )   # 法向偏移
    tangent_offsets: List[float] = field(
        default_factory=lambda: [0.0, -0.125, 0.125, -0.25, 0.25]
    )   # 切向偏移
    yaw_offsets_deg: List[float] = field(
        default_factory=lambda: [0.0, -22.5, 22.5, -45.0, 45.0]
    )   # 旋转角度


@dataclass
class EpisodeConfig:   # 抓取episode配置
    object_name: str = "cube"   # 抓取目标
    robot_root_z: float = DEFAULT_ROBOT_ROOT_Z
    robot_root_roll: float = DEFAULT_ROBOT_ROOT_ROLL
    robot_root_pitch: float = DEFAULT_ROBOT_ROOT_PITCH
    arm_joint_tolerance: float = 0.02   # 机械臂关节到达目标时允许的最大关节误差
    arm_stage_budget: int = 260   # ARM_INIT 子阶段最多执行多少个仿真控制步
    scan_warmup: int = 10
    scan_frames: int = 30
    wrist_min_object_pixels: int = 20   # 目标物体至少有20个像素点（use_mujoco_visibility_gate=False时无效）
    # A complete episode includes three deliberately slow ARM_INIT ramps,
    # dense MoveIt paths, local reach, close and lift settling.  Keep this
    # guard comfortably above the normal trajectory length; it is a runaway
    # guard, not a planning budget.
    max_episode_steps: int = 6000   # 一次完整抓取实验允许执行的最大仿真控制步数
    # Collision is a hard precondition for data collection.  A colliding M is
    # logged as BASE_PLACED failure and never reaches ARM_INIT/AnyGrasp.
    reject_initial_table_contact: bool = True   # 把机器人放到候选 \(M\) 后，首先检查机器人是否已经和桌子接触

    # All manipulation entry points use the semantic target mask and the
    # RANSAC-derived support surface by default.  The integrated N2M/SysNav
    # entry points also set these explicitly to document their required mode.
    semantic_target_mask: bool = True   # 是否用YOLOE → SAM2 → 目标 mask → 目标点云流程。在n2m脚本中设置为True
    dynamic_support_surface: bool = True   # 把腕部相机看到的桌面，转换成 MoveIt 能够进行碰撞检测的桌面模型。n2m脚本中设置为True
    use_mujoco_visibility_gate: bool = False   # 使用 MuJoCo 真值分割检查目标是否出现在腕部相机中.n2m脚本中设置为False
    perception_config: str = str(DEFAULT_PERCEPTION_CONFIG)   # YOLOE、SAM2 模型和感知参数配置文件路径
    target_detection_confidence: float = 0.01   # YOLOE检测目标的最低置信度

    # AnyGrasp候选参数
    grasp_topk: int = 1   # 没用，要大于50才有用。候选数量
    moveit_candidate_limit: int = 12   # 语义/几何过滤和分数排序后，最多取12个候选给moveit检查
    moveit_pregrasp_distance: float = 0.115   # contact = palm + (depth + insertion) × approach，从contact退后11.5cm
    moveit_local_reach_distance: float = 0.015   # 前进10cm，在目标前1.5cm停下。最后1.5cm由MuJoCo 局部伺服完成
    moveit_insertion: float = 0.01
    moveit_object_point_dist: float = 0.08   # 计算 AnyGrasp 预测抓取落点到目标点云最近点的距离。小于 8 cm 才认为候选落在目标附近
    moveit_table_margin: float = 0.05   # 把看到的桌面范围再扩大5cm
    moveit_table_thickness: float = 0.05   # MoveIt 桌面碰撞盒向拟合平面下方延伸 5 cm
    moveit_table_min_extent: float = 0.30   # 桌面碰撞盒的 X、Y 边长最少都为 30 cm
    moveit_table_max_tilt_deg: float = 12.0   # 与水平方向夹角大于12度就认为不是桌面（创建的碰撞体是水平盒子，否则与实际碰撞体误差太大）
    moveit_position_tolerance: float = 0.008   # MoveIt 规划到 pre-grasp 时的位置容差
    moveit_orientation_tolerance: float = 0.12   # MoveIt 规划到 pre-grasp 时的姿态容差
    moveit_plan_time: float = 2.5   # MoveIt 对每个候选进行 OMPL 规划时允许使用的时间，默认 2.5 秒
    moveit_cart_step: float = 0.005   # 计算笛卡尔路径时，相邻插值点的最大距离
    moveit_cart_fraction: float = 0.95   # 从 pre-grasp 到 near 位置的笛卡尔路径，至少要完成 95%
    moveit_lift_distance: float = 0.08   # 抓紧后，TCP 沿世界 +Z 抬升 8 cm
    moveit_lift_cart_fraction: float = 0.95   # 抬升笛卡尔路径至少要规划完成 95%
    moveit_lift_joint_tolerance: float = 0.04   # 抬升轨迹执行结束后，实际关节位置相对轨迹终点允许的最大误差
    moveit_lift_max_object_slip: float = 0.015   # 抬升期间，物体相对 TCP 最多允许滑动 1.5 cm
    moveit_lift_contact_miss_steps: int = 8   # 抬升过程中，如果两个手指都没有接触目标，并连续持续 8 个控制步，判定物体已经脱离夹爪
    moveit_exec_joint_tolerance: float = 0.05   # 执行完 pre-grasp 轨迹后，实际六个机械臂关节角与规划轨迹最后一个点的六个关节角允许的最大误差
    moveit_approach_joint_tolerance: float = 0.025   # 执行完笛卡尔接近轨迹、准备交给 LOCAL_REACH 时的关节误差
    moveit_exec_tcp_position_tolerance: float = 0.015   # 执行完 pre-grasp 或 approach 后，实际 TCP 与目标 TCP 的位置误差最大值
    moveit_exec_tcp_orientation_tolerance: float = math.radians(10.0)   # 执行结束后，TCP 姿态误差最多为 10°
    local_single_contact_speed_scale: float = 0.25   # 如果只有一个手指先碰到物体，局部伺服速度降到原来的 25%
    local_single_contact_max_advance: float = 0.008   # 第一次出现单侧接触后，最多允许继续沿 approach 方向深入 8 mm
    local_single_contact_force_limit: float = 1.5   # 单侧接触时，手指法向力安全上限为 1.5 N
    local_single_contact_force_confirm_steps: int = 3   # 单侧力不是一次超过 1.5 N 就立即失败，而是需要连续 3 个控制步超限才中止
    moveit_max_approach_world_z: float = 0.0   # 拒绝从下往上抓的候选
    moveit_attached_object_size: Tuple[float, float, float] | None = None   # MoveIt附着物体碰撞盒尺寸；None时按object_name从公共规格表读取
    moveit_attached_object_center: Tuple[float, float, float] | None = None   # 碰撞盒中心相对目标物体body原点的局部坐标；None时从公共规格表读取

    close_budget: int = 150   # CLOSE阶段BUDGET
    local_reach_budget: int = 300   # REACH阶段BUDGET
    lift_hold_steps: int = 20   # 夹紧后，在真正抬升前先保持 20 个控制步
    lift_success_rise: float = 0.04   # 最终物体高度相对抓取前至少上升这个高度才认为成功

    def to_dict(self):
        return asdict(self)
