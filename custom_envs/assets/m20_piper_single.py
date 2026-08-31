"""Isaac Lab configuration for the converted single-articulation M20 + Piper."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


ASSET_DIR = os.path.join(os.path.dirname(__file__), "m20_piper_single")
M20_PIPER_SINGLE_USD = os.path.join(ASSET_DIR, "M20_Piper.usd")

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
ARM_PARKED_POSE = {
    "joint1": 0.0,
    "joint2": 0.3,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.8,
    "joint6": 0.0,
    "joint7":  0.035,   # fully open
    "joint8": -0.035,   # fully open
}

DEEPROBOTICS_M20_PIPER_SINGLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=M20_PIPER_SINGLE_USD,
        activate_contact_sensors=True,
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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.3,
            "h[l,r]_hipy_joint": 0.3,
            "f[l,r]_knee_joint": 0.6,
            "h[l,r]_knee_joint": -0.6,
            ".*_wheel_joint": 0.0,
            **ARM_PARKED_POSE,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
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
        "wheels": DelayedPDActuatorCfg(
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
        # j1/j2/j3/j4: stiffness=1800 / damping=180 (general arm joints).
        # j5/j6: stiffness=2600 / damping=260 — both show gravity-driven constant
        # velocity drift in extended arm poses:
        #   j5: wrist pitch drifts from scan pose (1.2 rad) under gravity
        #   j6: wrist roll drifts ~-0.003 rad/step = -0.42 rad in 300 steps
        # Higher stiffness ensures PD restoring torque dominates gravity torque.
        # All 6 arm joints: stiffness=2600 / damping=260
        # j4 shows identical gravity-drift pattern to j5/j6 when arm is extended
        # (j2≈π, arm nearly horizontal): constant -0.003 rad/step drift over 300 steps.
        # Raising stiffness from 1800→2600 ensures PD restoring torque dominates gravity
        # torque for all joints, including j1-j4 in extreme IK poses (j2 near π).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4"],
            effort_limit=140.0,
            velocity_limit=5.0,
            stiffness=8000.0,
            damping=400.0,
        ),
        "arm_j5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            effort_limit=220.0,
            velocity_limit=5.0,
            stiffness=15000.0,   # raised from 8000: j5 wrist pitch residual was 15deg under gravity
            damping=600.0,
        ),
        "arm_j6": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit=220.0,
            velocity_limit=5.0,
            stiffness=8000.0,
            damping=400.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_JOINT_NAMES,
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=400.0,
            damping=20.0,
        ),
    },
)