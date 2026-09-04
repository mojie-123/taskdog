#!/usr/bin/env python3
"""Build the lightweight, single-tree M20 + Piper URDF used for USD conversion."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_URDF = REPO_ROOT / "deps/deep_robotics_model/M20_Piper/urdf/M20_Piper.urdf"
OUTPUT_DIR = REPO_ROOT / "custom_envs/assets/m20_piper_single"
OUTPUT_URDF = OUTPUT_DIR / "M20_Piper_lightweight.urdf"

# Primitive collision approximations in each Piper link frame. Visual STL
# meshes are retained; only the expensive mesh colliders are replaced.
# Values: shape, origin xyz, origin rpy, shape attributes.
PIPER_COLLIDERS = {
    "arm_base_link": ("cylinder", "0 0 0.045", "0 0 0", {"radius": "0.075", "length": "0.09"}),
    "link1": ("cylinder", "0 0 0", "0 0 0", {"radius": "0.065", "length": "0.10"}),
    "link2": ("box", "0.14 0 0", "0 0 0", {"size": "0.28 0.07 0.07"}),
    "link3": ("box", "0 -0.125 0", "0 0 0", {"size": "0.07 0.25 0.07"}),
    "link4": ("cylinder", "0 0 0", "0 0 0", {"radius": "0.060", "length": "0.10"}),
    "link5": ("box", "0 -0.050 0", "0 0 0", {"size": "0.07 0.11 0.07"}),
    "link6": ("cylinder", "0 0 0", "0 0 0", {"radius": "0.045", "length": "0.05"}),
    "gripper_base": ("box", "0 0 0.032", "0 0 0", {"size": "0.075 0.075 0.064"}),
    # link7 and link8 (gripper fingers) intentionally omitted:
    # their collision geometry is kept as the original STL mesh collider
    # for accurate finger-object contact detection during grasping.
}


def _replace_collision(link: ET.Element, spec: tuple[str, str, str, dict[str, str]]) -> None:
    """Replace every collider on a link with one primitive collider."""
    for collision in list(link.findall("collision")):
        link.remove(collision)
    shape, xyz, rpy, dimensions = spec
    collision = ET.SubElement(link, "collision", {"name": "simplified_collision"})
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, shape, dimensions)


def build() -> Path:
    """Generate and statically validate the conversion input URDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE_URDF.read_text(encoding="utf-8")
    # Full-width decimal typo was already fixed in source URDF; skip replacement.
    root = ET.fromstring(source_text)

    for mesh in root.findall(".//mesh"):
        mesh_path = Path(mesh.attrib["filename"])
        if not mesh_path.is_absolute():
            resolved = (SOURCE_URDF.parent / mesh_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            mesh.attrib["filename"] = str(resolved)

    replaced = set()
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name in PIPER_COLLIDERS:
            _replace_collision(link, PIPER_COLLIDERS[name])
            replaced.add(name)
    if replaced != set(PIPER_COLLIDERS):
        raise RuntimeError(f"Missing Piper links: {set(PIPER_COLLIDERS) - replaced}")

    links = {link.attrib["name"] for link in root.findall("link")}
    joints = root.findall("joint")
    children = [joint.find("child").attrib["link"] for joint in joints]
    roots = links - set(children)
    if roots != {"base_link"} or len(children) != len(set(children)):
        raise RuntimeError(f"URDF is not a single tree: roots={roots}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    shutil.copy2(SOURCE_URDF, OUTPUT_DIR / "SOURCE_M20_Piper.urdf")
    return OUTPUT_URDF


if __name__ == "__main__":
    output = build()
    print(f"Built {output}")