"""Flat-terrain M20 + Piper environment using one converted articulation.

Inherits from DeeproboticsM20ProLidarFlatEnvCfg so that the Mid-360 LiDAR
sensor and its mesh_prim_paths (ground + Shop_Table + banana) are included
identically to the dual-articulation Piper-v0 task, enabling the same
occupancy-grid mapping pipeline.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from custom_envs.assets.m20_piper_single import DEEPROBOTICS_M20_PIPER_SINGLE_CFG
from custom_envs.tasks.deeprobotics_m20_pro.lidar_flat_env_cfg import (
    DeeproboticsM20ProLidarFlatEnvCfg,
)


@configclass
class DeeproboticsM20ProSinglePiperEnvCfg(DeeproboticsM20ProLidarFlatEnvCfg):
    """Single-articulation M20 + Piper with LiDAR, table and banana props.

    Inherits DeeproboticsM20ProLidarFlatEnvCfg so the Mid-360 LiDAR sensor
    and its raycasting targets (ground, Shop_Table, banana) are present out
    of the box — same as the dual-articulation Piper-v0 task.

    The inherited M20 configuration explicitly binds locomotion observations
    and actions to ``joint_names`` (16 M20 joints), so the eight Piper joints
    do not alter the policy dimensions or ordering.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.robot = DEEPROBOTICS_M20_PIPER_SINGLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # table and banana are already provided by TaskdogSceneCfg (inherited
        # from DeeproboticsM20ProLidarFlatEnvCfg) with the correct positions
        # and LiDAR mesh_prim_paths — no need to add them again here.

        # Keep these bindings explicit at the integration boundary. This is
        # intentionally redundant with the parent M20 config so future parent
        # changes cannot silently expose the Piper DOFs to a locomotion policy.
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # The inherited event targets every joint by default. Restrict both
        # joint-level reset events to the 16 locomotion DOFs so Piper joints
        # are never overwritten by the randomisation logic:
        #
        #   randomize_actuator_gains — gain tensors are per-actuator; indexing
        #     all 24 joint IDs can write outside the M20 actuator tensors on CUDA.
        #
        #   randomize_reset_joints   — uses default_joint_pos (typically 0) as
        #     the scale base; resetting Piper joints to 0 collapses joint3 from
        #     -1.5 rad (parked) to 0 (extended), causing the arm to slam into the
        #     robot body and producing the violent shaking observed at runtime.
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names
        )
        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names
        )

        # ---- Wrist RGB-D camera (mounted on gripper_base) ----
        # Attached to the Piper gripper base link. The camera faces forward
        # along the +Z axis of gripper_base (ROS convention), offset 5 cm
        # forward from the link origin so it clears the gripper fingers.
        # Outputs both RGB and depth (distance_to_image_plane) at 640x480,
        # updated every 0.1 s (10 Hz), matching the grasp pipeline cadence.
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_base/wrist_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(0.5, 0.5, -0.5, -0.5),  # (w,x,y,z): rotates ROS +Z => gripper forward
                convention="ros",
            ),
            spawn=PinholeCameraCfg(
                focal_length=1.93,          # ~90 deg HFOV, typical RGBD depth camera
                horizontal_aperture=3.896,  # sensor width = 2*focal_length*tan(HFOV/2)
                clipping_range=(0.1, 5.0),  # 10 cm - 5 m, suitable for table-top grasping
            ),
            data_types=["rgb", "distance_to_image_plane"],
            width=640,
            height=480,
            update_period=0.1,
        )