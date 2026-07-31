"""M20 Pro flat-terrain environment configuration."""

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import (
    DeeproboticsM20FlatEnvCfg,
)


@configclass
class DeeproboticsM20ProFlatEnvCfg(DeeproboticsM20FlatEnvCfg):
    """M20 Pro flat-terrain locomotion environment config."""

    def __post_init__(self):
        super().__post_init__()

        # --- Customize below for M20 Pro ---
        # Same customization points as rough_env_cfg.

        self.disable_zero_weight_rewards()
