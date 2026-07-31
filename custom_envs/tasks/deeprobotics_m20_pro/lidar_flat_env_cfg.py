"""M20 Pro flat-terrain + Livox Mid-360 LiDAR + navigation target sphere."""

from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.lidar_rough_env_cfg import (
    DeeproboticsM20ProLidarRoughEnvCfg,
)
from custom_envs.utils.target_spawner import spawn_target_sphere


@configclass
class DeeproboticsM20ProLidarFlatEnvCfg(DeeproboticsM20ProLidarRoughEnvCfg):
    """M20 Pro flat terrain + Mid-360 LiDAR + target sphere for navigation."""

    def __post_init__(self):
        super().__post_init__()

        # Replace terrain with flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # ---- spawn target sphere ----
        # RayCaster check patched to allow multiple mesh prims.
        ball_prim = spawn_target_sphere(self.scene, pos=(5.0, 2.0, 0.9), radius=0.15)
        if self.scene.mid360_lidar is not None:
            current = list(self.scene.mid360_lidar.mesh_prim_paths)
            if ball_prim not in current:
                self.scene.mid360_lidar.mesh_prim_paths = current + [ball_prim]

        # Already handled by parent chain (see lidar_rough_env_cfg.py comment)
