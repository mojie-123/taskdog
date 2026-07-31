"""Target-object helpers — spawn a visible sphere and query its position."""

from isaaclab.sim.spawners import SphereCfg
from isaaclab.assets import AssetBaseCfg


def spawn_target_sphere(scene, pos=(5.0, 2.0, 0.3), radius=0.15):
    """Add a static red sphere to the scene config (call inside __post_init__).

    The sphere is visual-only — no MDL material (avoids unavailable asset paths).
    Use a simple diffuse color instead.  RayCaster in IsaacLab 2.3.2 only supports
    a single mesh prim, so the sphere is NOT added to LiDAR mesh_prim_paths.
    Navigation goals are specified manually via --goal.

    Args:
        scene:  self.scene (InteractiveSceneCfg instance).
        pos:    (x, y, z) world position.
        radius: sphere radius (m).

    Returns:
        prim_path string.
    """
    prim_path = "/World/target_ball"
    scene.target_sphere = AssetBaseCfg(
        prim_path=prim_path,
        spawn=SphereCfg(radius=radius),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )
    return prim_path


def get_target_position():
    """Read the sphere's world position from the USD stage.

    Only callable while Isaac Sim is running.
    Returns (x, y, z) in meters, or None on failure.
    """
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
        prim = get_prim_at_path("/World/target_ball")
        attr = prim.GetAttribute("xformOp:translate")
        if attr.IsValid():
            v = attr.Get()
            return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        pass
    return None
