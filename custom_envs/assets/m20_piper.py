"""M20 Pro + AgileX Piper arm combined articulation configuration.

Joints (26 total, 16 active + 8 arm/gripper frozen):
  12 leg joints (fl/fr/hl/hr × hipx/hipy/knee)
   4 wheel joints (fl/fr/hl/hr_wheel)
   6 arm joints (joint1–joint6, Piper 6-DOF)
   1 gripper (joint7, prismatic)
   1 finger (joint8, prismatic)
   2 fixed (base_to_arm, joint6_to_gripper_base)

The arm + gripper are **frozen** at a folded pose so they do not
interfere with locomotion.  Only the 16 leg + wheel joints appear in
the action space.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# USD path (same location as other Deep Robotics models)
_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "deps", "deep_robotics_model")
)
M20_PIPER_USD = os.path.join(_MODEL_DIR, "M20_Piper", "usd", "M20_Piper.usd")

# ---------------------------------------------------------------------------
# Helper — arm joint names for frozen init
# ---------------------------------------------------------------------------
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]

# Folded pose on the dog's back (within joint limits from URDF).
#   joint1:   base rotation             [-2.618,  2.618]
#   joint2:   shoulder                  [ 0,      3.14 ]
#   joint3:   elbow                     [-2.967,  0     ]
#   joint4:   wrist roll                [-1.745,  1.745]
#   joint5:   wrist pitch               [-1.22,   1.22 ]
#   joint6:   wrist yaw                 [-2.094,  2.094]
#   joint7:   gripper (prismatic)       [ 0,      0.035]
#   joint8:   finger (prismatic)        [-0.035,  0     ]
ARM_FOLDED_POSE = {
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.8,
    "joint6": 0.0,
    "joint7": 0.0175,   # gripper half-open
    "joint8": -0.0175,  # fingers half-closed
}

# ---------------------------------------------------------------------------
# Articulation config
# ---------------------------------------------------------------------------

DEEPROBOTICS_M20_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=M20_PIPER_USD,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            # Legs (same as M20 Pro)
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.3,
            "h[l,r]_hipy_joint": 0.3,
            "f[l,r]_knee_joint": 0.6,
            "h[l,r]_knee_joint": -0.6,
            ".*wheel_joint": 0.0,
            # Arm — folded on the back
            **ARM_FOLDED_POSE,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # --- Legs (same as M20 Pro) ---
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        # --- Wheels (same as M20 Pro) ---
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
        # --- Arm (frozen — not in action space) ---
        "arm": IdealPDActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit=30.0,
            velocity_limit=10.0,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
        ),
        # --- Gripper (frozen — not in action space) ---
        "gripper": IdealPDActuatorCfg(
            joint_names_expr=GRIPPER_JOINT_NAMES,
            effort_limit=10.0,
            velocity_limit=2.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
