"""M20 Pro + Piper arm (separate articulation) + LiDAR + table + banana."""

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from custom_envs.tasks.deeprobotics_m20_pro.lidar_flat_env_cfg import (
    DeeproboticsM20ProLidarFlatEnvCfg,
    TaskdogSceneCfg,
)

PIPER_USD = "/home/mojie/taskdog/custom_envs/assets/piper/piper.usd"

@configclass
class DeeproboticsM20ProPiperEnvCfg(DeeproboticsM20ProLidarFlatEnvCfg):
    """M20 Pro + Piper arm on flat terrain."""

    scene: TaskdogSceneCfg = TaskdogSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()


_piper_mounted = False

def _strip_physics(prim):
    """Recursively remove physics APIs from a prim and all its children."""
    for api in ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI",
                "PhysxRigidBodyAPI", "PhysicsArticulationRootAPI",
                "PhysxJointAPI", "PhysicsJointStateAPI"]:
        if prim.HasAPI(api):
            try:
                prim.RemoveAPI(api)
            except Exception:
                pass
    for child in prim.GetChildren():
        _strip_physics(child)

def piper_mount_and_follow(env):
    """Create Piper visual (first call) and track robot position."""
    global _piper_mounted
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        stage = omni.usd.get_context().get_stage()
        piper_path = "/World/envs/env_0/PiperArm"

        if not _piper_mounted or not stage.GetPrimAtPath(piper_path):
            prim = UsdGeom.Xform.Define(stage, piper_path)
            prim.GetPrim().GetReferences().AddReference(PIPER_USD)
            _piper_mounted = True
            # Strip physics APIs from all Piper children so PhysX
            # ignores them (pure visual, no collision interference).
            _strip_physics(prim.GetPrim())
            kids = list(prim.GetPrim().GetChildren())
            print(f"[PIPER] Created {piper_path}, children={len(kids)} (physics stripped)", flush=True)

        robot = env.unwrapped.scene["robot"]
        pos_w = robot.data.root_pos_w[0].cpu().numpy()
        quat_w = robot.data.root_quat_w[0].cpu().numpy()
        piper = stage.GetPrimAtPath(piper_path)
        if piper:
            # Position
            piper.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(pos_w[0]), float(pos_w[1]), float(pos_w[2])))
            # Rotation — track robot heading
            piper.GetAttribute("xformOp:orient").Set(
                Gf.Quatd(float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])))
    except Exception as e:
        print(f"[PIPER] Error: {e}", flush=True)
