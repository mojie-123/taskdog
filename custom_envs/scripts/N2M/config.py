"""Fixed six-scene pilot configuration and episode defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house.xml"
DEFAULT_DATA_DIR = PACKAGE_DIR / "data" / "manipulation_pose_search"


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: str
    outward_xy: Tuple[float, float]


EDGES: Dict[str, EdgeSpec] = {
    "left": EdgeSpec("left", (-1.0, 0.0)),
    "right": EdgeSpec("right", (1.0, 0.0)),
    "bottom": EdgeSpec("bottom", (0.0, -1.0)),
    "top": EdgeSpec("top", (0.0, 1.0)),
}


@dataclass(frozen=True)
class CubeScene:
    scene_id: str
    cube_xyz: Tuple[float, float, float]
    approach_edges: Tuple[str, ...]
    category: str


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
DEFAULT_REFERENCE_BASE = (4.791462, 3.558308, 1.5967632116203363)
DEFAULT_ROBOT_ROOT_Z = 0.563874
DEFAULT_ROBOT_ROOT_ROLL = 0.03359878103266884
DEFAULT_ROBOT_ROOT_PITCH = 0.003133989573390085
REFERENCE_SCENE_ID = "short_left"
REFERENCE_EDGE_ID = "left"


@dataclass
class CandidateGrid:
    # Zero-first ordering makes --max-candidates useful during visual inspection:
    # nominal M is always candidate 0, followed by progressively wider offsets.
    # Normal offset is defined positive *into* the table.  The successful
    # reference M is already at the closest acceptable distance, so all newly
    # searched offsets are strictly negative (away from the table).
    normal_offsets: List[float] = field(
        default_factory=lambda: [0.0, -0.05, -0.10, -0.15]
    )
    tangent_offsets: List[float] = field(
        default_factory=lambda: [0.0, -0.125, 0.125, -0.25, 0.25]
    )
    yaw_offsets_deg: List[float] = field(
        default_factory=lambda: [0.0, -22.5, 22.5, -45.0, 45.0]
    )


@dataclass
class EpisodeConfig:
    object_name: str = "cube"
    robot_root_z: float = DEFAULT_ROBOT_ROOT_Z
    robot_root_roll: float = DEFAULT_ROBOT_ROOT_ROLL
    robot_root_pitch: float = DEFAULT_ROBOT_ROOT_PITCH
    arm_joint_tolerance: float = 0.02
    arm_stage_budget: int = 260
    scan_warmup: int = 10
    scan_frames: int = 30
    wrist_min_object_pixels: int = 20
    # A complete episode includes three deliberately slow ARM_INIT ramps,
    # dense MoveIt paths, local reach, close and lift settling.  Keep this
    # guard comfortably above the normal trajectory length; it is a runaway
    # guard, not a planning budget.
    max_episode_steps: int = 6000
    # Collision is a hard precondition for data collection.  A colliding M is
    # logged as BASE_PLACED failure and never reaches ARM_INIT/AnyGrasp.
    reject_initial_table_contact: bool = True

    grasp_topk: int = 1
    moveit_candidate_limit: int = 12
    moveit_pregrasp_distance: float = 0.115
    moveit_local_reach_distance: float = 0.015
    moveit_insertion: float = 0.005
    moveit_object_point_dist: float = 0.08
    moveit_position_tolerance: float = 0.008
    moveit_orientation_tolerance: float = 0.12
    moveit_plan_time: float = 2.5
    moveit_cart_step: float = 0.005
    moveit_cart_fraction: float = 0.95
    moveit_lift_distance: float = 0.08
    moveit_lift_cart_fraction: float = 0.95
    moveit_lift_joint_tolerance: float = 0.04
    moveit_lift_max_object_slip: float = 0.015
    moveit_lift_contact_miss_steps: int = 8
    moveit_exec_joint_tolerance: float = 0.05
    moveit_approach_joint_tolerance: float = 0.025
    moveit_exec_tcp_position_tolerance: float = 0.015
    moveit_exec_tcp_orientation_tolerance: float = math.radians(10.0)
    local_single_contact_speed_scale: float = 0.25
    local_single_contact_max_advance: float = 0.008
    local_single_contact_force_limit: float = 1.5
    local_single_contact_force_confirm_steps: int = 3
    moveit_max_approach_world_z: float = 0.0
    moveit_attached_object_size: Tuple[float, float, float] | None = None

    close_budget: int = 150
    local_reach_budget: int = 300
    lift_hold_steps: int = 20
    lift_success_rise: float = 0.04

    def to_dict(self):
        return asdict(self)
