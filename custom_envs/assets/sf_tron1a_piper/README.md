# SF_TRON1A + Piper model

This directory contains a new combined model. Existing M20, Piper, and house
scene files are read-only inputs and are never modified.

## Mount assumption

- TRON `+X` is forward.
- Piper uses the same mounting orientation as the current M20 + Piper model.
- `joint1=0` faces TRON `+X`.
- The missing adapter is represented by an ideal fixed connection.
- Temporary mount translation: `(0, 0, 0.0165)` metres. This places the bottom
  of the Piper base at the highest top surface of the TRON body mesh.

Edit only `mount_config.json` and rerun `build_model.py` when the real adapter
dimensions are available.

## Rebuild

```bash
python custom_envs/assets/sf_tron1a_piper/build_model.py
```

Generated files:

- `SF_TRON1A_Piper.urdf`: combined single-tree URDF.
- `Piper_current_project.urdf`: standalone copy of the current Piper subtree.
- `SF_TRON1A_Piper.xml`: combined MuJoCo robot model.

## View in the house scene

```bash
python custom_envs/scripts/assets/view_sf_tron1a_piper.py
```

The viewer loads `scene_house_N2M.xml` without changing it, replaces the M20
body in a temporary scene, and places SF_TRON1A + Piper at `(1.6, 1.6, 0.85)`.
It intentionally does not step physics because this visualization utility does
not include a TRON balance controller.

The script disables GLFW framebuffer scaling before launching MuJoCo's official
viewer. This keeps the viewport and mouse coordinates consistent on HiDPI or
fractionally-scaled desktops, and uses the official right-button and wheel
controls directly. The passive viewer intentionally does not call `mj_step`,
so the biped stays in its initial pose even though no balance controller is
loaded.

Compilation-only check:

```bash
python custom_envs/scripts/assets/view_sf_tron1a_piper.py --validate-only
```

The script automatically uses the existing `env_isaaclab` Python when the
current Python environment does not provide the `mujoco` package.

## Sources

- SF_TRON1A: LimX Dynamics `tron1-robot-description`, Apache-2.0. The retained
  license is at `vendor/tron1/LICENSE`.
- Piper: the exact arm/gripper geometry and kinematics already used by this
  project, copied into this new directory without modifying the source model.
