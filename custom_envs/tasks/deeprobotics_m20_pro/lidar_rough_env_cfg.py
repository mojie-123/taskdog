"""M20 Pro rough-terrain + Livox Mid-360 LiDAR environment."""

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.rough_env_cfg import (
    DeeproboticsM20ProRoughEnvCfg,
)
from custom_envs.utils.lidar_pattern import get_mid360_lidar_pattern_light


@configclass
class DeeproboticsM20ProLidarRoughEnvCfg(DeeproboticsM20ProRoughEnvCfg):
    """M20 Pro rough terrain + Mid-360 LiDAR."""

    def __post_init__(self):
        super().__post_init__()

        # ---- LiDAR sensor (MultiMeshRayCaster for object-space raycasting) ----
        self.scene.mid360_lidar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 0.55)),
            ray_alignment="base",
            pattern_cfg=get_mid360_lidar_pattern_light(),
            max_distance=70.0,
            update_period=0.1,
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # LiDAR sensor is in the scene for mapping/teleop, but NOT in the
        # policy/critic observation (all envs stay at 57-dim input).
        # Note: disable_zero_weight_rewards() is already called by parent chain.
