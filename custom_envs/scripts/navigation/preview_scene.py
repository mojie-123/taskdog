#!/usr/bin/env python
"""Isaac Sim scene previewer — opens any registered task without loading a policy.

This script is designed for quick visual inspection of scene layouts
(object positions, table sizes, etc.) without requiring a trained checkpoint.

Usage
-----
# Inspect the TwoTables scene (physics running, banana falls onto table)
python custom_envs/scripts/navigation/preview_scene.py \\
    --task Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0  --enable_cameras

# Inspect with physics frozen — only check static prop placement
python custom_envs/scripts/navigation/preview_scene.py \\
    --task Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0 --pause  --enable_cameras

# Inspect a different env
python custom_envs/scripts/navigation/preview_scene.py \\
    --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 --pause

Controls
--------
    Ctrl+C / close Isaac Sim window   quit

Notes
-----
* No checkpoint is required — a zero action tensor is sent every step so the
  robot stays in its default joint configuration (it may fall over; that is
  expected without a locomotion policy).
* Use --pause if you only care about prop positions and do not want gravity.
* --task accepts any ID registered in custom_envs/tasks/__init__.py.
"""

# ---------------------------------------------------------------------------
# Stage 1 — argument parsing (including AppLauncher flags).
#
# Importing the AppLauncher *class* here is safe — it only loads a Python
# module and does NOT start Omniverse.  The actual Isaac Sim launch happens
# at AppLauncher(args) below.
#
# AppLauncher.add_app_launcher_args() must be called BEFORE parse_known_args()
# so that flags like --enable_cameras / --headless / --device are recognised
# and land in the args Namespace that is passed to AppLauncher(args).
# Without this call, --enable_cameras is silently ignored and camera-enabled
# scenes raise: "A camera was spawned without the --enable_cameras flag."
# ---------------------------------------------------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher  # class import only — no Omniverse yet

parser = argparse.ArgumentParser(
    description="Isaac Sim scene previewer (no policy required)."
)
parser.add_argument(
    "--task",
    type=str,
    default="Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0",
    help="Registered gym task ID to preview (default: TwoTables).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments to spawn (default: 1).",
)
parser.add_argument(
    "--pause",
    action="store_true",
    default=False,
    help=(
        "Freeze physics after reset and only render the GUI. "
        "Useful for checking static prop placement without gravity."
    ),
)
# Register all AppLauncher flags (--enable_cameras, --headless, --device, …)
# into our parser so they are parsed into args and forwarded to AppLauncher.
AppLauncher.add_app_launcher_args(parser)
args, _unknown = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Stage 2 — launch Isaac Sim / Omniverse.
# Everything below this point may import isaaclab / omni freely.
# ---------------------------------------------------------------------------

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Stage 3 — all isaaclab / gym imports (after SimulationApp is live).
# ---------------------------------------------------------------------------
import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

import custom_envs.tasks  # noqa: F401, E402  (registers all custom gym IDs)

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n[preview] Loading task: '{args.task}'")

    # ------------------------------------------------------------------
    # Build env config from registry
    # ------------------------------------------------------------------
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs

    # Disable training-time noise / randomisation — not useful for preview.
    env_cfg.observations.policy.enable_corruption = False

    # Time-out would reset the env automatically; suppress it.
    env_cfg.terminations.time_out = None

    # Joint randomisation events can cause the robot to jump on reset;
    # disable them so the scene starts from a clean default pose.
    for _event_name in (
        "randomize_apply_external_force_torque",
        "randomize_reset_joints",
        "randomize_actuator_gains",
    ):
        if hasattr(env_cfg.events, _event_name):
            setattr(env_cfg.events, _event_name, None)

    # ------------------------------------------------------------------
    # Create the environment and run one reset so all prims are spawned.
    # ------------------------------------------------------------------
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    print(f"[preview] Scene spawned successfully.")
    print(f"[preview] Mode : {'PAUSED — physics frozen' if args.pause else 'RUNNING — zero-action steps'}")
    print( "[preview] Quit : close the Isaac Sim window or press Ctrl+C\n")

    # ------------------------------------------------------------------
    # Main render / step loop
    # ------------------------------------------------------------------
    try:
        if args.pause:
            # Physics is frozen; only keep the GUI alive.
            while simulation_app.is_running():
                simulation_app.update()
        else:
            # Send zero actions every step so joints don't receive garbage
            # commands.  The robot will likely fall — that is fine for a
            # preview.  The scene geometry (tables, banana) is fully visible.
            zero_action = torch.zeros(
                env.action_space.shape,
                device=env.unwrapped.device,
            )
            while simulation_app.is_running():
                env.step(zero_action)

    except KeyboardInterrupt:
        print("\n[preview] Ctrl+C received — shutting down.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
