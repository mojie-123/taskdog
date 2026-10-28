#!/usr/bin/env python3
"""Show the SF_TRON1A + Piper model inside scene_house_N2M.xml.

The source scene is never changed.  This script creates a temporary merged
scene, replaces only the M20 robot body/actuators, and opens a static MuJoCo
viewer.  Physics is intentionally not stepped because no TRON balance
controller is part of this model-viewing utility.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "custom_envs/assets/sf_tron1a_piper"
ROBOT_XML = MODEL_DIR / "SF_TRON1A_Piper.xml"
CONFIG_PATH = MODEL_DIR / "mount_config.json"
DEFAULT_SCENE = REPO_ROOT / "custom_envs/mujoco/scene_house_N2M.xml"
FALLBACK_PYTHON = Path("/home/mojie/anaconda3/envs/env_isaaclab/bin/python")


def _import_mujoco(interactive: bool):
    try:
        viewer = None
        if interactive:
            # Configure GLFW before importing/launching MuJoCo's viewer.  This
            # keeps the native window, framebuffer, and mouse coordinates in
            # the same scale on HiDPI/fractionally-scaled Linux desktops.
            import glfw  # type: ignore

            if not glfw.init():
                raise RuntimeError("GLFW initialization failed")
            glfw.window_hint(glfw.SCALE_FRAMEBUFFER, glfw.FALSE)

        import mujoco  # type: ignore
        if interactive:
            from mujoco import viewer as mujoco_viewer  # type: ignore

            viewer = mujoco_viewer

        return mujoco, viewer
    except ModuleNotFoundError as exc:
        current = Path(sys.executable).resolve()
        if FALLBACK_PYTHON.is_file() and current != FALLBACK_PYTHON.resolve():
            os.execv(
                str(FALLBACK_PYTHON),
                [str(FALLBACK_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        raise SystemExit(
            "Python environment does not provide mujoco. Activate the project "
            "MuJoCo environment or set the script's FALLBACK_PYTHON."
        ) from exc


def _absolutize_scene_assets(scene_root: ET.Element, scene_path: Path) -> None:
    compiler = scene_root.find("compiler")
    mesh_dir = scene_path.parent
    texture_dir = scene_path.parent
    if compiler is not None:
        if compiler.attrib.get("meshdir"):
            candidate = Path(compiler.attrib["meshdir"])
            mesh_dir = candidate if candidate.is_absolute() else scene_path.parent / candidate
        if compiler.attrib.get("texturedir"):
            candidate = Path(compiler.attrib["texturedir"])
            texture_dir = (
                candidate if candidate.is_absolute() else scene_path.parent / candidate
            )

    asset = scene_root.find("asset")
    if asset is None:
        return
    for element in asset:
        filename = element.attrib.get("file")
        if not filename or Path(filename).is_absolute():
            continue
        base = texture_dir if element.tag == "texture" else mesh_dir
        element.attrib["file"] = str((base / filename).resolve())


def _replace_section(
    scene_root: ET.Element, robot_root: ET.Element, section_name: str
) -> None:
    old_section = scene_root.find(section_name)
    new_section = robot_root.find(section_name)
    if old_section is not None:
        scene_root.remove(old_section)
    if new_section is not None:
        scene_root.append(copy.deepcopy(new_section))


def build_temporary_scene(
    scene_path: Path,
    x: float,
    y: float,
    z: float,
    mount_z: float,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    scene_root = ET.parse(scene_path).getroot()
    robot_root = ET.parse(ROBOT_XML).getroot()
    _absolutize_scene_assets(scene_root, scene_path)

    scene_worldbody = scene_root.find("worldbody")
    robot_body = robot_root.find("./worldbody/body[@name='base_link']")
    if scene_worldbody is None or robot_body is None:
        raise RuntimeError("Scene or robot model is missing its worldbody/root body")

    old_robot = scene_worldbody.find("./body[@name='base_link']")
    if old_robot is None:
        raise RuntimeError("Source scene does not contain the M20 base_link body")
    insertion_index = list(scene_worldbody).index(old_robot)
    scene_worldbody.remove(old_robot)

    robot_body = copy.deepcopy(robot_body)
    robot_body.attrib["pos"] = f"{x:.9g} {y:.9g} {z:.9g}"
    arm_body = robot_body.find("./body[@name='arm_base_link']")
    if arm_body is None:
        raise RuntimeError("Combined robot model is missing arm_base_link")
    arm_body.attrib["pos"] = f"0 0 {mount_z:.9g}"
    scene_worldbody.insert(insertion_index, robot_body)

    scene_asset = scene_root.find("asset")
    robot_asset = robot_root.find("asset")
    if scene_asset is None or robot_asset is None:
        raise RuntimeError("Scene or robot model is missing its asset section")
    for asset in robot_asset:
        copied = copy.deepcopy(asset)
        filename = copied.attrib.get("file")
        if filename and not Path(filename).is_absolute():
            copied.attrib["file"] = str((MODEL_DIR / filename).resolve())
        scene_asset.append(copied)

    _replace_section(scene_root, robot_root, "actuator")
    _replace_section(scene_root, robot_root, "sensor")

    temp_dir = tempfile.TemporaryDirectory(prefix="sf_tron1a_piper_")
    generated_path = Path(temp_dir.name) / "scene_house_sf_tron1a_piper.xml"
    ET.indent(scene_root, space="  ")
    ET.ElementTree(scene_root).write(
        generated_path, encoding="utf-8", xml_declaration=True
    )
    return temp_dir, generated_path


def _set_joint_positions(mujoco, model, data) -> None:
    display_pose = {
        "joint1": 0.0,
        "joint2": 0.5,
        "joint3": -1.0,
        "joint4": 0.0,
        "joint5": 0.5,
        "joint6": 0.0,
        "joint7": 0.035,
        "joint8": -0.035,
    }
    for name, value in display_pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Generated model is missing joint {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def _parse_args() -> argparse.Namespace:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spawn = config["robot_spawn"]
    mount_z = config["piper_mount"]["xyz"][2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--x", type=float, default=spawn["x"])
    parser.add_argument("--y", type=float, default=spawn["y"])
    parser.add_argument("--z", type=float, default=spawn["z"])
    parser.add_argument(
        "--mount-z",
        type=float,
        default=mount_z,
        help="Temporary Piper base height in the TRON base frame.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Compile and inspect the merged scene without opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mujoco, viewer = _import_mujoco(interactive=not args.validate_only)
    scene_path = args.scene.expanduser().resolve()
    temp_dir, merged_scene = build_temporary_scene(
        scene_path, args.x, args.y, args.z, args.mount_z
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(merged_scene))
        data = mujoco.MjData(model)
        _set_joint_positions(mujoco, model, data)

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        arm_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "arm_base_link"
        )
        if base_id < 0 or arm_id < 0:
            raise RuntimeError("Compiled model is missing base_link or arm_base_link")
        print(
            "Loaded SF_TRON1A + Piper: "
            f"nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}"
        )
        print(f"base_link world position: {data.xpos[base_id].tolist()}")
        print(f"arm_base_link world position: {data.xpos[arm_id].tolist()}")
        print("Piper mount rotation: rpy=(0, 0, 0), joint1=0 faces TRON +X")

        if args.validate_only:
            return

        assert viewer is not None
        print(
            "Opening the official passive MuJoCo viewer with "
            "glfw.SCALE_FRAMEBUFFER disabled. Physics is not stepped, so the "
            "uncontrolled biped remains in its initial standing pose."
        )
        with viewer.launch_passive(model, data) as passive_viewer:
            passive_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            passive_viewer.cam.trackbodyid = base_id
            passive_viewer.cam.lookat[:] = data.xpos[base_id]
            passive_viewer.cam.distance = 2.6
            passive_viewer.cam.azimuth = 145.0
            passive_viewer.cam.elevation = -18.0
            while passive_viewer.is_running():
                passive_viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
