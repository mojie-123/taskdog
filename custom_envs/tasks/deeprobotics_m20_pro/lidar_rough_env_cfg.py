"""M20 Pro rough-terrain + Livox Mid-360 LiDAR environment."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.rough_env_cfg import (
    DeeproboticsM20ProRoughEnvCfg,
)
from custom_envs.utils.lidar_pattern import get_mid360_lidar_pattern_light
from custom_envs.utils.lidar_observation import lidar_knn_downsample


def _lidar_obs(env, sensor_cfg: SceneEntityCfg, num_points: int = 64, max_range: float = 70.0):
    """Observation function: downsample LiDAR point cloud.

    Called by the IsaacLab observation manager each step.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return lidar_knn_downsample(
        ray_hits_w=sensor.data.ray_hits_w,
        sensor_pos_w=sensor.data.pos_w,
        num_points=num_points,
        max_range=max_range,
    )


@configclass
class DeeproboticsM20ProLidarRoughEnvCfg(DeeproboticsM20ProRoughEnvCfg):
    """M20 Pro rough terrain + Mid-360 LiDAR."""

    def __post_init__(self):
        super().__post_init__()

        # ---- LiDAR sensor ----
        self.scene.mid360_lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.30, 0.0, 0.55)),
            ray_alignment="base",
            pattern_cfg=get_mid360_lidar_pattern_light(),
            max_distance=70.0,
            update_period=0.1,
            debug_vis=False,  # disable for training — saves GPU memory
            mesh_prim_paths=["/World/ground"],
        )

        # ---- LiDAR observation (policy) ----
        self.observations.policy.lidar = ObsTerm(
            func=_lidar_obs,
            params={
                "sensor_cfg": SceneEntityCfg("mid360_lidar"),
                "num_points": 64,
                "max_range": 70.0,
            },
        )

        # ---- LiDAR observation (critic) ----
        self.observations.critic.lidar = ObsTerm(
            func=_lidar_obs,
            params={
                "sensor_cfg": SceneEntityCfg("mid360_lidar"),
                "num_points": 64,
                "max_range": 70.0,
            },
        )

        # Note: disable_zero_weight_rewards() is already called by
        # DeeproboticsM20ProRoughEnvCfg.__post_init__() in the parent chain.
        # Calling it again would crash because it sets rewards to None
        # on the first pass, and the second pass tries to read .weight from None.
