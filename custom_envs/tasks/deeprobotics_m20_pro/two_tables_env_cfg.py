"""Two-table variant of the Single-Piper environment.

Changes vs. the baseline ``Flat-Deeprobotics-M20Pro-Piper-Single-v0``:
  * Original table (Shop_Table, center 5,5) is stretched along the world-Y
    axis by setting y_scale = 0.012 instead of the default 0.008.
  * A second table (Shop_Table_2, center 9,8) is added at original size
    (scale 0.008 uniformly).
  * Both tables are registered as LiDAR raycasting targets so the
    occupancy-grid mapping pipeline can see them.

No existing files are modified; this file only inherits and extends.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.lidar_flat_env_cfg import (
    TABLE_USD,
    TaskdogSceneCfg,
    _table_cfg,
)
from custom_envs.tasks.deeprobotics_m20_pro.single_piper_env_cfg import (
    DeeproboticsM20ProSinglePiperEnvCfg,
)

# ── YCB object USD paths ──────────────────────────────────────────────────────
_OBJ_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "objects")
APPLE_USD = os.path.join(_OBJ_DIR, "013_apple", "013_apple.usd")
BOWL_USD  = os.path.join(_OBJ_DIR, "024_bowl",  "024_bowl.usd")


def _apple_cfg(pos=(4.9, 4.5, 0.75), rot=(1.0, 0.0, 0.0, 0.0)):
    """RigidObject config for the YCB apple (013_apple) on Table 1."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=APPLE_USD,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _bowl_cfg(pos=(4.9, 5.5, 0.75), rot=(1.0, 0.0, 0.0, 0.0)):
    """RigidObject config for the YCB bowl (024_bowl) on Table 1."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOWL_USD,
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
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _table2_cfg(pos=(8.0, 6.0, 0.0), scale=(0.008, 0.008, 0.008)):
    """Kinematic RigidObject for the second shop table at (8, 6) — original size.

    Uses a distinct prim_path (Shop_Table_2) so it coexists with the first
    table (Shop_Table) inside the same environment namespace.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shop_Table_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # static prop, doesn't fall
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class TwoTablesSceneCfg(TaskdogSceneCfg):
    """TaskdogSceneCfg with the original table stretched in Y and a second table.

    Field overrides / additions:
      table   — same prim_path (Shop_Table), y_scale increased from 0.008 to
                0.012 (1.5× longer along world-Y).  The prim_path stays
                identical so LiDAR mesh_prim_paths in the parent chain remain
                valid without any changes.
      table_2 — new field; spawned at (9, 8, 0) with original uniform scale.
    """

    # Override: original table stretched 1.5× along Y (y_scale 0.008 → 0.012).
    table: RigidObjectCfg = _table_cfg(
        pos=(5.0, 5.0, 0.0),
        scale=(0.008, 0.024, 0.008),
    )

    # New: second table at (8, 6) with original uniform scale.
    table_2: RigidObjectCfg = _table2_cfg()

    # YCB objects on Table 1 (same z=0.75 as banana, tested)
    apple: RigidObjectCfg = _apple_cfg()
    bowl:  RigidObjectCfg = _bowl_cfg()


@configclass
class DeeproboticsM20ProSinglePiperTwoTablesEnvCfg(DeeproboticsM20ProSinglePiperEnvCfg):
    """Single-articulation M20 + Piper with two tables.

    Table 1 (Shop_Table)  : center (5, 5), scale (0.008, 0.024, 0.008) — stretched Y.
    Table 2 (Shop_Table_2): center (9, 8), scale (0.008, 0.008, 0.008) — original.
    Banana stays at its inherited position on Table 1.

    LiDAR mesh_prim_paths from the parent chain already include
    /World/ground, Shop_Table, and banana.  We append Shop_Table_2 so
    the occupancy-grid mapper can perceive the second table as well.
    """

    scene: TwoTablesSceneCfg = TwoTablesSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # The parent chain (DeeproboticsM20ProLidarFlatEnvCfg.__post_init__)
        # already sets mesh_prim_paths = [ground, Shop_Table, banana].
        # Append Shop_Table_2 so the LiDAR also raycasts against it.
        if self.scene.mid360_lidar is not None:
            self.scene.mid360_lidar.mesh_prim_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Shop_Table_2",
                    track_mesh_transforms=True,
                    is_shared=True,
                )
            )
            self.scene.mid360_lidar.mesh_prim_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/apple",
                    track_mesh_transforms=True,
                    is_shared=True,
                )
            )
            self.scene.mid360_lidar.mesh_prim_paths.append(
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/bowl",
                    track_mesh_transforms=True,
                    is_shared=True,
                )
            )
