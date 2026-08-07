"""M20 Pro + Piper arm (dual articulation with sub-step sync) + LiDAR + table.

Piper is loaded as a separate articulation (PIPER_ARM_CFG) with gravity
disabled and strong damping.  A physics sub-step callback syncs Piper's
root state to M20's base_link position + mounting offset at EVERY physics
sub-step (decimation times per env.step), eliminating the one-frame lag
and collision overlap that caused shaking/drifting in the naive sync.
"""

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from dataclasses import replace

from custom_envs.tasks.deeprobotics_m20_pro.lidar_flat_env_cfg import (
    DeeproboticsM20ProLidarFlatEnvCfg,
    TaskdogSceneCfg,
)
from custom_envs.assets.piper_arm import PIPER_ARM_CFG

# Mounting offset in M20 base_link frame.
# base_link collision box is 0.75×0.09×0.14, so body top = +0.07.
# Offset of 0.12m places Piper base ~5 cm above M20 body.
MOUNT_OFFSET = (0.0, 0.0, 0.12)


@configclass
class DeeproboticsM20ProPiperEnvCfg(DeeproboticsM20ProLidarFlatEnvCfg):
    """M20 Pro + Piper arm on flat terrain — dual articulation with sub-step sync."""

    scene: TaskdogSceneCfg = TaskdogSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # Add Piper as a separate articulation.
        # disable_gravity + strong damping in PIPER_ARM_CFG keep it
        # stable between sub-step syncs.
        piper_cfg = replace(PIPER_ARM_CFG, prim_path="{ENV_REGEX_NS}/piper_arm")
        self.scene.piper = piper_cfg


# ---------------------------------------------------------------------------
# Sub-step sync callback — registered after env creation
# ---------------------------------------------------------------------------

def setup_piper_sync(env) -> callable:
    """Register a physics sub-step callback that syncs Piper to M20.

    The callback runs *before every physics sub-step* (dt = sim.dt),
    typically 4× per env.step() when decimation=4.  This eliminates the
    one-frame lag of a per-env-step sync and keeps Piper locked to M20's
    base_link with zero relative drift.

    Args:
        env: The gym environment (after gym.make + env.reset).

    Returns:
        The callback function (can be used with remove_physics_callback).
    """
    robot = env.unwrapped.scene["robot"]
    piper = env.unwrapped.scene["piper"]
    device = robot.device

    # Cache base_link body index (fixed per articulation).
    base_idx = robot.body_names.index("base_link")

    # Mount offset as a tensor on the correct device.
    mount_offset = torch.tensor(MOUNT_OFFSET, device=device, dtype=torch.float32)

    def _sync(dt: float) -> None:
        """Sync Piper root state to M20 base_link + mount offset."""
        # M20 base_link world pose & velocity.
        base_pos = robot.data.body_pos_w[0, base_idx]       # (3,)
        base_quat = robot.data.body_quat_w[0, base_idx]     # (4,) wxyz
        base_vel = robot.data.body_vel_w[0, base_idx]       # (6,) lin+ang

        # Compute Piper root position in world frame.
        mount_world = base_pos + quat_apply(base_quat, mount_offset)

        # Build new Piper root state and write to sim.
        piper_state = piper.data.root_state_w[0].clone()
        piper_state[0:3] = mount_world
        piper_state[3:7] = base_quat
        piper_state[7:13] = base_vel
        piper.write_root_state_to_sim(piper_state.unsqueeze(0))

    # Register on the SimulationContext — fires before each physics sub-step.
    # env may be wrapped (e.g. OrderEnforcing); always go through .unwrapped.
    env.unwrapped.sim.add_physics_callback("piper_sync", _sync)

    return _sync
