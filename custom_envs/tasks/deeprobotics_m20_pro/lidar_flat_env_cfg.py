"""M20 Pro flat-terrain + Livox Mid-360 LiDAR + navigation target banana."""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.lidar_rough_env_cfg import (
    DeeproboticsM20ProLidarRoughEnvCfg,
)
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    MySceneCfg,
)

BANANA_USD = os.path.join(
    os.path.dirname(__file__), "..", "..", "objects", "011_banana.usd"
)
TABLE_USD = os.path.join(os.path.dirname(__file__), "..", "..", "objects", "Shop_Table.usd")


def _banana_cfg(pos=(4.9, 5.0, 0.75), rot=(1.0, 0.0, 0.0, 0.0)):
    """RigidObject config for the banana target.

    Placed near the -X edge of the table (table center x=5.0, estimated
    -X edge ~4.52 m) so the Piper arm can reach it without overstretching.
    Spawned at z=0.85 m (approx 15 cm above estimated table surface of ~0.70 m)
    so it falls under gravity and settles on the table.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/banana",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BANANA_USD,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # falls under gravity, lands on table
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _table_cfg(pos=(5.0, 5.0, 0.0), scale=(0.008, 0.008, 0.008)):
    """Kinematic RigidObject for the shop table — static, but has PhysX collision."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shop_Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # static prop, doesn't fall
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class TaskdogSceneCfg(MySceneCfg):
    """MySceneCfg + table + banana for navigation target."""

    table: RigidObjectCfg = _table_cfg()
    banana: RigidObjectCfg = _banana_cfg()


@configclass
class DeeproboticsM20ProLidarFlatEnvCfg(DeeproboticsM20ProLidarRoughEnvCfg):
    """M20 Pro flat terrain + Mid-360 LiDAR + banana for navigation."""

    scene: TaskdogSceneCfg = TaskdogSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # Replace terrain with flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # ---- configure LiDAR targets (MultiMeshRayCaster) ----
        # Ground: static plane (legacy string format, no transform tracking).
        # Table + banana: tracked with local-space raycasting (fixes
        # float32 precision issue for finite meshes at world coordinates).
        if self.scene.mid360_lidar is not None:
            self.scene.mid360_lidar.mesh_prim_paths = [
                "/World/ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Shop_Table",
                    track_mesh_transforms=True,
                    is_shared=True,
                ),
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/banana",
                    track_mesh_transforms=True,
                    is_shared=True,
                ),
            ]

        # Already handled by parent chain (see lidar_rough_env_cfg.py comment)
