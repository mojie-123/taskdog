#!/usr/bin/env python3
"""Convert the lightweight M20 + Piper URDF into an isolated USD asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from custom_envs.scripts.assets.build_m20_piper_urdf import OUTPUT_DIR, build


def main() -> Path:
    urdf_path = build()
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(OUTPUT_DIR),
        usd_file_name="M20_Piper.usd",
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        root_link_name="base_link",
        merge_fixed_joints=False,
        collision_from_visuals=False,
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        joint_drive=None,
        link_density=0.0,
    )
    output = Path(UrdfConverter(cfg).usd_path)
    if not output.is_file():
        raise FileNotFoundError(output)
    print(f"Converted {urdf_path} -> {output} ({output.stat().st_size} bytes)")
    return output


try:
    main()
finally:
    simulation_app.close()