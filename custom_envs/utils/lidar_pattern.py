"""Livox Mid-360 LiDAR pattern for RayCaster.

Two versions:
- full:   40 ch × 1.0° = 14,400 rays  (high fidelity, needs >8GB VRAM at 4096 envs)
- light:  16 ch × 2.0° =  2,880 rays  (training-optimized, fits 8GB VRAM at 4096 envs)
"""

from isaaclab.sensors.ray_caster import patterns


def get_mid360_lidar_pattern() -> patterns.LidarPatternCfg:
    """Full-resolution pattern (14,400 rays). For viz / single-env playback."""
    return patterns.LidarPatternCfg(
        channels=40,
        vertical_fov_range=(-7.0, 52.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=1.0,
    )


def get_mid360_lidar_pattern_light() -> patterns.LidarPatternCfg:
    """Lightweight pattern for RL training (2,880 rays).

    16 channels × 180 steps (2° resolution) = 2,880 rays.
    Fits in 8 GB VRAM with 4096 parallel environments.
    """
    return patterns.LidarPatternCfg(
        channels=16,
        vertical_fov_range=(-7.0, 52.0),
        horizontal_fov_range=(-180.0, 180.0),
        horizontal_res=2.0,
    )
