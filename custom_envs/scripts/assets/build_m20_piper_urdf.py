#!/usr/bin/env python3
"""Build the lightweight, single-tree M20 + Piper URDF used for USD conversion."""

from __future__ import annotations

import shutil   # 把原始文件复制过来（到SOURCE_M20_Piper.urdf）
import xml.etree.ElementTree as ET   # xml解析模块
from pathlib import Path  # 路径操作


REPO_ROOT = Path(__file__).resolve().parents[3]   # __file__ 是当前脚本的路径，.resolve() 转为绝对路径，.parents[3] 向上跳 3 级目录，得到仓库根目录
SOURCE_URDF = REPO_ROOT / "deps/deep_robotics_model/M20_Piper/urdf/M20_Piper.urdf"   # 原文件位置
OUTPUT_DIR = REPO_ROOT / "custom_envs/assets/m20_piper_single"   # 输出urdf的目标位置
OUTPUT_URDF = OUTPUT_DIR / "M20_Piper_lightweight.urdf"   # 输出urdf的目标urdf名称

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
}   # Piper 机械臂各连杆的简化碰撞几何体（基础几何体，xyz，rpy，几何体参数设置）。注意link7和link8没换


def _replace_collision(link: ET.Element, spec: tuple[str, str, str, dict[str, str]]) -> None:
    """Replace every collider on a link with one primitive collider."""
    for collision in list(link.findall("collision")):   # 找到该连杆下所有 <collision> 元素并删除
        link.remove(collision)
    shape, xyz, rpy, dimensions = spec
    collision = ET.SubElement(link, "collision", {"name": "simplified_collision"})   # 这里说明collision是简化版本
    ET.SubElement(collision, "origin", {"xyz": xyz, "rpy": rpy})   # collision下添加origin
    geometry = ET.SubElement(collision, "geometry")   # collision下添加geometry
    ET.SubElement(geometry, shape, dimensions)   # geometry下添加几何体及其属性


def build() -> Path:
    """Generate and statically validate the conversion input URDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_text = SOURCE_URDF.read_text(encoding="utf-8")   # 读取源 URDF 文件的文本内容
    # Full-width decimal typo was already fixed in source URDF; skip replacement.
    root = ET.fromstring(source_text)   # 将 XML 文本解析为 ElementTree 的根元素

    for mesh in root.findall(".//mesh"):   # 遍历urdf中所有<mesh>
        mesh_path = Path(mesh.attrib["filename"])
        if not mesh_path.is_absolute():   # 把所有mesh变为绝对路径
            resolved = (SOURCE_URDF.parent / mesh_path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            mesh.attrib["filename"] = str(resolved)

    replaced = set()
    for link in root.findall("link"):   # 执行替换
        name = link.attrib["name"]
        if name in PIPER_COLLIDERS:
            _replace_collision(link, PIPER_COLLIDERS[name])
            replaced.add(name)
    if replaced != set(PIPER_COLLIDERS):
        raise RuntimeError(f"Missing Piper links: {set(PIPER_COLLIDERS) - replaced}")

    # 验证urdf是单树结构
    links = {link.attrib["name"] for link in root.findall("link")}   # 所有link名称
    joints = root.findall("joint")   # 所有Joint名称
    children = [joint.find("child").attrib["link"] for joint in joints]   # 所有joint的child
    roots = links - set(children)   # 根连杆就是所有连杆-所有子连杆！
    if roots != {"base_link"} or len(children) != len(set(children)):
        raise RuntimeError(f"URDF is not a single tree: roots={roots}")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    shutil.copy2(SOURCE_URDF, OUTPUT_DIR / "SOURCE_M20_Piper.urdf")
    return OUTPUT_URDF


if __name__ == "__main__":
    output = build()
    print(f"Built {output}")