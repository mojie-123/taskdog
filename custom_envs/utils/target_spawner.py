"""Target-object helpers — spawn a banana as a physics RigidObject."""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

BANANA_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "objects", "011_banana.usd"
)


def spawn_banana(scene, pos=(10.0, 10.0, 0.03), rot=(1.0, 0.0, 0.0, 0.0)):
    """Add a physics-enabled banana RigidObject to the scene config.

    The banana spawns at *pos* and drops to the ground under gravity.
    Call inside __post_init__.

    Args:
        scene: self.scene (InteractiveSceneCfg instance).
        pos:   (x, y, z) world spawn position.
        rot:   (w, x, y, z) quaternion orientation.

    Returns:
        prim_path string for LiDAR mesh_prim_paths.
        With num_envs=1, {ENV_REGEX_NS} resolves to /World/envs/env_0.
    """
    prim_path = "/World/envs/env_0/banana"
    scene.banana = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=BANANA_USD_PATH,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            mesh_collision_props=sim_utils.MeshCollisionPropertiesCfg(
                mesh_approximation="convexDecomposition",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )
    return prim_path


def get_target_position():
    """Read the banana's world position from the USD stage.

    Only callable while Isaac Sim is running.
    Returns (x, y, z) in meters, or None on failure.
    """
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path("/World/envs/env_0/banana")
        attr = prim.GetAttribute("xformOp:translate")
        if attr.IsValid():
            v = attr.Get()
            return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        pass
    return None
