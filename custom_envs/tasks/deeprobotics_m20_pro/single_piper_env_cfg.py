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
        # Simulates an Intel RealSense D435 mounted on the Piper gripper_base.
        #
        # Coordinate convention (convention="ros": +Z=optical/forward, +X=right, -Y=up):
        #   gripper_base +X (finger-forward) --> camera +Z (optical axis / forward)
        #   gripper_base +Y (finger-left)    --> camera -X (image left)
        #   gripper_base +Z (finger-up)      --> camera -Y (image up)
        # Quaternion derivation: R columns = parent-axis in ROS-camera frame
        #   col0=[0,0,1], col1=[-1,0,0], col2=[0,-1,0]  => (w=0.5, x=0.5, y=-0.5, z=0.5)
        #   (previously wrong: y=-0.5 was listed as -0.5 but z was -0.5 instead of +0.5)
        #
        # Intrinsics match Intel RealSense D435 at 640x480:
        #   fx = fy = 616,  HFOV ≈ 54.9°,  VFOV ≈ 42.6°
        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_base/wrist_camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),        # 5 cm forward of gripper_base origin
                rot=(0.5, 0.5, -0.5, 0.5),  # (w,x,y,z): gripper +X => camera optical +Z
                convention="ros",
            ),
            spawn=PinholeCameraCfg(
                focal_length=1.93,             # D435 focal length (arbitrary unit)
                horizontal_aperture=2.005195,  # => fx = fy = 616 at 640x480
                vertical_aperture=1.503896,    # = horizontal_aperture * 480/640
                clipping_range=(0.1, 5.0),     # 10 cm – 5 m (D435 usable range)
            ),
            data_types=["rgb", "distance_to_image_plane"],
            width=640,
            height=480,
            update_period=0.1,
        )