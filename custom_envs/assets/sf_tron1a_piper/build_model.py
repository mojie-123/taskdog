#!/usr/bin/env python3
"""Build new SF_TRON1A + current-project Piper URDF and MJCF models.

This generator is deliberately read-only with respect to every existing model.
It reads the vendored SF_TRON1A description and the current Piper subtree, then
writes only files next to this script.
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[2]
TRON_URDF = MODEL_DIR / "vendor/tron1/SF_TRON1A.urdf"
TRON_MJCF = MODEL_DIR / "vendor/tron1/SF_TRON1A.xml"
PIPER_SOURCE_URDF = (
    REPO_ROOT / "custom_envs/assets/m20_piper_single/SOURCE_M20_Piper.urdf"
)
PIPER_SOURCE_MJCF = REPO_ROOT / "custom_envs/mujoco/scene_house_N2M.xml"
CONFIG_PATH = MODEL_DIR / "mount_config.json"

PIPER_ROOT_LINK = "arm_base_link"
PIPER_LINK_NAMES = (
    "arm_base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
    "link7",
    "link8",
)


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _fmt(values: list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _extract_piper_urdf() -> tuple[list[ET.Element], list[ET.Element]]:
    source = ET.parse(PIPER_SOURCE_URDF).getroot()
    links = {link.attrib["name"]: link for link in source.findall("link")}
    joints = source.findall("joint")

    outgoing: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent")
        if parent is None:
            continue
        outgoing.setdefault(parent.attrib["link"], []).append(joint)

    selected_links: set[str] = set()
    selected_joints: list[ET.Element] = []
    queue: deque[str] = deque([PIPER_ROOT_LINK])
    while queue:
        link_name = queue.popleft()
        if link_name in selected_links:
            continue
        if link_name not in links:
            raise RuntimeError(f"Piper source is missing link {link_name!r}")
        selected_links.add(link_name)
        for joint in outgoing.get(link_name, []):
            child = joint.find("child")
            if child is None:
                continue
            selected_joints.append(copy.deepcopy(joint))
            queue.append(child.attrib["link"])

    if selected_links != set(PIPER_LINK_NAMES):
        raise RuntimeError(
            "Unexpected Piper subtree: "
            f"missing={set(PIPER_LINK_NAMES) - selected_links}, "
            f"extra={selected_links - set(PIPER_LINK_NAMES)}"
        )

    copied_links = [
        copy.deepcopy(link)
        for link in source.findall("link")
        if link.attrib["name"] in selected_links
    ]
    for link in copied_links:
        for mesh in link.findall(".//mesh"):
            mesh.attrib["filename"] = (
                f"meshes/piper/{Path(mesh.attrib['filename']).name}"
            )

    return copied_links, selected_joints


def _append_tool_frames(robot: ET.Element) -> None:
    wrist_link = ET.SubElement(robot, "link", {"name": "wrist_camera_optical_frame"})
    wrist_link.tail = "\n"
    wrist_joint = ET.SubElement(
        robot,
        "joint",
        {"name": "gripper_base_to_wrist_camera_optical", "type": "fixed"},
    )
    ET.SubElement(wrist_joint, "parent", {"link": "gripper_base"})
    ET.SubElement(wrist_joint, "child", {"link": "wrist_camera_optical_frame"})
    ET.SubElement(
        wrist_joint,
        "origin",
        {"xyz": "-0.05 0 0.06", "rpy": "0 0 -1.5707963267948966"},
    )

    tcp_link = ET.SubElement(robot, "link", {"name": "grasp_tcp"})
    tcp_link.tail = "\n"
    tcp_joint = ET.SubElement(
        robot,
        "joint",
        {"name": "gripper_base_to_grasp_tcp", "type": "fixed"},
    )
    ET.SubElement(tcp_joint, "parent", {"link": "gripper_base"})
    ET.SubElement(tcp_joint, "child", {"link": "grasp_tcp"})
    ET.SubElement(tcp_joint, "origin", {"xyz": "0 0 0.1358", "rpy": "0 0 0"})


def _validate_urdf_tree(robot: ET.Element) -> None:
    link_names = [link.attrib["name"] for link in robot.findall("link")]
    joint_names = [joint.attrib["name"] for joint in robot.findall("joint")]
    children = [
        joint.find("child").attrib["link"]
        for joint in robot.findall("joint")
        if joint.find("child") is not None
    ]
    if len(link_names) != len(set(link_names)):
        raise RuntimeError("Combined URDF contains duplicate link names")
    if len(joint_names) != len(set(joint_names)):
        raise RuntimeError("Combined URDF contains duplicate joint names")
    roots = set(link_names) - set(children)
    if roots != {"base_link"}:
        raise RuntimeError(f"Combined URDF is not a single tree: roots={roots}")


def build_urdf(config: dict) -> tuple[Path, Path]:
    tron = ET.parse(TRON_URDF).getroot()
    tron.attrib["name"] = "SF_TRON1A_Piper"

    # Produce a simulator-neutral generated model.  The untouched vendor file
    # still contains its Gazebo plugins and is retained under vendor/tron1/.
    for child in list(tron):
        if child.tag in {"gazebo", "transmission", "ros2_control"}:
            tron.remove(child)

    for link in tron.findall("link"):
        if link.attrib.get("name") == "base_Link":
            link.attrib["name"] = "base_link"
        for mesh in link.findall(".//mesh"):
            mesh.attrib["filename"] = (
                f"meshes/tron/{Path(mesh.attrib['filename']).name}"
            )
    for reference in tron.findall(".//parent") + tron.findall(".//child"):
        if reference.attrib.get("link") == "base_Link":
            reference.attrib["link"] = "base_link"

    piper_links, piper_joints = _extract_piper_urdf()
    for element in piper_links + piper_joints:
        tron.append(element)
    _append_tool_frames(tron)

    mount = config["piper_mount"]
    fixed_joint = ET.SubElement(
        tron, "joint", {"name": "tron_to_piper_joint", "type": "fixed"}
    )
    ET.SubElement(fixed_joint, "parent", {"link": "base_link"})
    ET.SubElement(fixed_joint, "child", {"link": PIPER_ROOT_LINK})
    ET.SubElement(
        fixed_joint,
        "origin",
        {"xyz": _fmt(mount["xyz"]), "rpy": _fmt(mount["rpy"])},
    )
    _validate_urdf_tree(tron)

    combined_path = MODEL_DIR / "SF_TRON1A_Piper.urdf"
    _write_xml(tron, combined_path)

    standalone_piper = ET.Element("robot", {"name": "Piper_current_project"})
    for element in piper_links + piper_joints:
        standalone_piper.append(copy.deepcopy(element))
    _append_tool_frames(standalone_piper)
    piper_path = MODEL_DIR / "Piper_current_project.urdf"
    _write_xml(standalone_piper, piper_path)
    return combined_path, piper_path


def _prefix_mesh_references(body: ET.Element, prefix: str) -> None:
    for geom in body.findall(".//geom"):
        mesh = geom.attrib.get("mesh")
        if mesh:
            geom.attrib["mesh"] = f"{prefix}{mesh}"


def _make_robot_assets() -> ET.Element:
    asset = ET.Element("asset")
    for path in sorted((MODEL_DIR / "meshes/tron").glob("*.STL")):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"tron_{path.stem}", "file": f"meshes/tron/{path.name}"},
        )
    for path in sorted((MODEL_DIR / "meshes/piper").glob("*.STL")):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"piper_{path.stem}", "file": f"meshes/piper/{path.name}"},
        )

    source_scene = ET.parse(PIPER_SOURCE_MJCF).getroot()
    source_asset = source_scene.find("asset")
    if source_asset is None:
        raise RuntimeError("Source scene has no asset section")
    required_materials = {f"{name}_mat" for name in PIPER_LINK_NAMES}
    for material in source_asset.findall("material"):
        old_name = material.attrib.get("name")
        if old_name not in required_materials:
            continue
        copied = copy.deepcopy(material)
        copied.attrib["name"] = f"piper_{old_name}"
        asset.append(copied)
    return asset


def _extract_piper_mjcf(mount_xyz: list[float]) -> ET.Element:
    scene = ET.parse(PIPER_SOURCE_MJCF).getroot()
    source_body = scene.find("./worldbody/body[@name='base_link']")
    if source_body is None:
        raise RuntimeError("Current scene is missing base_link")
    arm = source_body.find("./body[@name='arm_base_link']")
    if arm is None:
        raise RuntimeError("Current scene is missing arm_base_link")
    arm = copy.deepcopy(arm)
    arm.attrib["pos"] = _fmt(mount_xyz)
    _prefix_mesh_references(arm, "piper_")
    for geom in arm.findall(".//geom"):
        material = geom.attrib.get("material")
        if material:
            geom.attrib["material"] = f"piper_{material}"

    gripper = arm.find(".//body[@name='gripper_base']")
    if gripper is None:
        raise RuntimeError("Current Piper MJCF is missing gripper_base")
    ET.SubElement(
        gripper,
        "site",
        {
            "name": "grasp_tcp",
            "pos": "0 0 0.1358",
            "quat": "1 0 0 0",
            "size": "0.003",
            "rgba": "0 1 0 0.8",
        },
    )
    return arm


def build_mjcf(config: dict) -> Path:
    official = ET.parse(TRON_MJCF).getroot()
    source_body = official.find("./worldbody/body[@name='base_Link']")
    if source_body is None:
        raise RuntimeError("Official SF_TRON1A MJCF is missing base_Link")
    robot_body = copy.deepcopy(source_body)
    robot_body.attrib["name"] = "base_link"
    robot_body.attrib["pos"] = "0 0 0.85"
    free_joint = robot_body.find("joint[@type='free']")
    if free_joint is None:
        raise RuntimeError("Official SF_TRON1A MJCF is missing its free joint")
    free_joint.attrib["name"] = "robot_freejoint"
    _prefix_mesh_references(robot_body, "tron_")

    for geom in robot_body.findall(".//geom"):
        if geom.attrib.pop("class", None) == "visual":
            geom.attrib["type"] = "mesh"
            geom.attrib.setdefault("contype", "0")
            geom.attrib.setdefault("conaffinity", "0")
            geom.attrib.setdefault("group", "1")

    ET.SubElement(
        robot_body,
        "site",
        {
            "name": "base_lock_site",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "size": "0.002",
            "rgba": "0 0 0 0",
        },
    )
    robot_body.append(_extract_piper_mjcf(config["piper_mount"]["xyz"]))

    root = ET.Element("mujoco", {"model": "SF_TRON1A_Piper"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.001", "gravity": "0 0 -9.81", "integrator": "implicitfast"},
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", {"armature": "0.01", "damping": "0.01"})
    ET.SubElement(
        defaults,
        "geom",
        {
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": "0.6 0.005 0.0001",
            "margin": "0.001",
        },
    )
    root.append(_make_robot_assets())
    worldbody = ET.SubElement(root, "worldbody")
    worldbody.append(robot_body)

    actuators = ET.SubElement(root, "actuator")
    official_actuators = official.find("actuator")
    if official_actuators is None:
        raise RuntimeError("Official SF_TRON1A MJCF is missing actuators")
    for actuator in official_actuators:
        actuators.append(copy.deepcopy(actuator))

    source_scene = ET.parse(PIPER_SOURCE_MJCF).getroot()
    source_actuators = source_scene.find("actuator")
    if source_actuators is None:
        raise RuntimeError("Current Piper scene is missing actuators")
    for actuator in source_actuators:
        if actuator.attrib.get("joint") in {f"joint{i}" for i in range(1, 9)}:
            actuators.append(copy.deepcopy(actuator))

    official_sensors = official.find("sensor")
    if official_sensors is not None:
        sensors = ET.SubElement(root, "sensor")
        for sensor in official_sensors:
            sensors.append(copy.deepcopy(sensor))

    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "link7", "body2": "link8"})
    ET.SubElement(
        contact, "exclude", {"body1": "link7", "body2": "gripper_base"}
    )
    ET.SubElement(
        contact, "exclude", {"body1": "link8", "body2": "gripper_base"}
    )

    output = MODEL_DIR / "SF_TRON1A_Piper.xml"
    _write_xml(root, output)
    return output


def main() -> None:
    config = _load_config()
    combined_urdf, piper_urdf = build_urdf(config)
    combined_mjcf = build_mjcf(config)
    print(f"Built {combined_urdf}")
    print(f"Built {piper_urdf}")
    print(f"Built {combined_mjcf}")


if __name__ == "__main__":
    main()
