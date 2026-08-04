"""AgileX Piper arm — separate articulation (frozen as a decorative payload).

Uses the lightweight ATEC2026 Piper physics file (5.5 KB simplified
collision meshes) so GPU memory stays under 8 GB when mounted on M20 Pro.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_PIPER_DIR = os.path.dirname(__file__)
PIPER_USD_PATH = os.path.join(_PIPER_DIR, "piper", "piper.usd")

# Mount point on M20 Pro's back
PIPER_MOUNT_POS = (0.0, 0.0, 0.72)  # x, y, z — on top of base_link

# Folded / parked joint positions (within limits from URDF)
PIPER_PARKED_POSE = {
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.8,
    "joint6": 0.0,
    "joint7": 0.0175,   # gripper half-open
    "joint8": -0.0175,  # finger half-closed
}

PIPER_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=PIPER_MOUNT_POS,
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=PIPER_PARKED_POSE,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=800.0,
            damping=80.0,
        ),
    },
)
