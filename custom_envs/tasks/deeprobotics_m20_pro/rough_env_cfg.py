"""M20 Pro rough-terrain environment configuration.

Inherits from the official rl_training Deeprobotics M20 rough config.
Customize this file to add M20 Pro-specific behaviors (e.g. different
reward weights, domain randomization ranges, observation terms).
"""

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)


@configclass
class DeeproboticsM20ProRoughEnvCfg(DeeproboticsM20RoughEnvCfg):
    """M20 Pro rough-terrain locomotion environment config.

    Currently identical to the M20 config. Override __post_init__ here
    to customize observation space, action scale, reward weights,
    domain randomization, terrain difficulty, etc.
    """

    def __post_init__(self):
        # Call parent first (this sets up all M20 defaults)
        super().__post_init__()

        # --- Customize below for M20 Pro ---
        # Examples of what you can change:
        #
        #   self.observations.policy.base_lin_vel.scale = 2.0
        #   self.rewards.track_lin_vel_xy_exp.weight = 8.0
        #   self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        #   self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.25)
        #
        # For now, we keep the same configuration as the official M20.
        # This proves the inheritance chain works end-to-end.

        # Disable zero-weight rewards (cleanup inherited from parent pattern)
        self.disable_zero_weight_rewards()
