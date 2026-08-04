"""Register M20 Pro Gym environments.

This file registers two environment IDs:
    Flat-Deeprobotics-M20Pro-v0   — Flat terrain locomotion
    Rough-Deeprobotics-M20Pro-v0  — Rough terrain locomotion

Both inherit from the official M20 configs and can be trained
using the same scripts (just change --task).
"""

import gymnasium as gym

from . import agents  # noqa: F401 (make agent configs available)

##
# Register Gym environments
##

gym.register(
    id="Flat-Deeprobotics-M20Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20ProFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20ProRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProRoughPPORunnerCfg",
    },
)

##
# LiDAR environments (Mid-360 on M20 Pro)
##

gym.register(
    id="Flat-Deeprobotics-M20Pro-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_flat_env_cfg:DeeproboticsM20ProLidarFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lidar_cfg:DeeproboticsM20ProFlatLidarPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20Pro-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_rough_env_cfg:DeeproboticsM20ProLidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lidar_cfg:DeeproboticsM20ProRoughLidarPPORunnerCfg",
    },
)

##
# Piper arm environment (M20 Pro + AgileX Piper + RealSense)
##

gym.register(
    id="Flat-Deeprobotics-M20Pro-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper_env_cfg:DeeproboticsM20ProPiperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg",
    },
)
