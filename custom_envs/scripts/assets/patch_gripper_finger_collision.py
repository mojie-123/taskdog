#!/usr/bin/env python3
"""
patch_gripper_finger_collision.py — 为夹爪手指 link7/link8 补写 mesh 碰撞体

=== 背景 ===
Isaac Sim 的 UrdfConverter 无法处理 URDF 中 <collision><geometry><mesh> 格式的碰撞体。
对于 link7/link8（夹爪手指），URDF 里指定了 STL mesh 作为碰撞体，但转换后生成的 USD
里 link7/link8 的碰撞体 prim 是一个空的 Xform，没有几何数据，也没有 PhysicsCollisionAPI，
导致仿真中两根手指完全没有碰撞体，无法物理接触香蕉。

=== 本脚本做什么 ===
本脚本直接修改 URDF->USD 转换产生的两个文件：
  1. M20_Piper_base.usd   — 把 /colliders/link7/link7（空 Xform）改为 Mesh 类型，
                           从 /visuals/link7/link7/node_STL_BINARY_/mesh 复制几何数据
  2. M20_Piper_physics.usd — 在 /M20_Piper/link7/collisions/link7 上添加
                             PhysicsCollisionAPI 和 physics:approximation
  link8 完全对称，做同样的操作。

=== USD 文件结构说明 ===
URDF->USD 转换后，M20_Piper.usd 由三个子文件组成（sublayer）：
  - M20_Piper_base.usd   ：存储所有几何体形状（mesh 顶点、primitive 尺寸等）
  - M20_Piper_physics.usd：存储物理属性（刚体、关节、碰撞声明等）
  - M20_Piper_sensor.usd ：存储传感器
每个 link 的碰撞体在 base.usd 里的路径：/colliders/<link_name>/<child_prim>
对应的物理声明在 physics.usd 里的路径：/M20_Piper/<link_name>/collisions/<child_prim>

=== 运行方式 ===
  本脚本不需要启动 Isaac Sim，只需要 Isaac Sim 内置的 pxr 库。

  conda activate env_isaaclab

  PXR_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
  LD_LIBRARY_PATH=$PXR_PATH/bin:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \\
  PYTHONPATH=$PXR_PATH \\
  python3 custom_envs/scripts/assets/patch_gripper_finger_collision.py

  # 只预览不写文件：
  ... python3 patch_gripper_finger_collision.py --dry-run

  # 指定不同的 USD 目录：
  ... python3 patch_gripper_finger_collision.py --usd-dir /path/to/configuration

=== 注意事项 ===
  1. 每次重新运行 convert_m20_piper_urdf.py 之后，必须重新运行本脚本。
  2. 脚本运行前会自动备份原文件（.usd.bak 后缀）。
  3. approximation 使用 convexDecomposition，首次加载时 PhysX 需额外预计算。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]  # .../custom_envs/scripts/assets -> repo root
_DEFAULT_USD_DIR = _REPO_ROOT / "custom_envs/assets/m20_piper_single/configuration"

# 要补写碰撞体的 link 列表
_FINGER_LINKS = ["link7", "link8"]

# visuals mesh 的 prim 路径模板（base.usd stage root 下）
_VISUAL_MESH_PATH = "/visuals/{link}/{link}/node_STL_BINARY_/mesh"

# colliders 几何体 prim 路径模板（base.usd stage root 下）
# URDF Importer 为 mesh collision 生成的子 prim 名称与 link 同名
_COLLIDER_PRIM_PATH = "/colliders/{link}/{link}"

# physics.usd 里碰撞声明的实际 SdfLayer spec 路径模板
# （/M20_Piper/<link>/collisions 是 instanceable，其子 prim 是 instance proxy，
#  无法直接写。必须操作 prototype 对应的 SdfLayer spec，路径在 /colliders/<link>/<link>）
_PHYSICS_COLL_PATH = "/colliders/{link}/{link}"

# 从 visual mesh 复制到 collider mesh 的属性
_MESH_ATTRS = [
    "points",
    "faceVertexCounts",
    "faceVertexIndices",
    "normals",
    "extent",
    "doubleSided",
    "orientation",
    "subdivisionScheme",
]


def _patch_base_usd(base_usd: Path, dry_run: bool) -> None:
    """
    修改 M20_Piper_base.usd：
    把 /colliders/<link>/<link> 从空 Xform 改为 Mesh，
    并从 /visuals/<link>/<link>/node_STL_BINARY_/mesh 复制几何属性。
    """
    from pxr import Sdf, Usd  # noqa: PLC0415

    print(f"[base.usd] 打开 {base_usd}")
    stage = Usd.Stage.Open(str(base_usd))
    layer = stage.GetRootLayer()

    for link in _FINGER_LINKS:
        src_path = _VISUAL_MESH_PATH.format(link=link)
        dst_path = _COLLIDER_PRIM_PATH.format(link=link)

        src_prim = stage.GetPrimAtPath(src_path)
        if not src_prim.IsValid():
            print(f"  [ERROR] 源 prim 不存在: {src_path}")
            sys.exit(1)

        dst_prim = stage.GetPrimAtPath(dst_path)
        if not dst_prim.IsValid():
            print(f"  [ERROR] 目标 prim 不存在: {dst_path}")
            sys.exit(1)

        print(f"  [{link}] src={src_path}")
        print(f"  [{link}] dst={dst_path}  当前 typeName={dst_prim.GetTypeName()}")

        if dry_run:
            print(f"  [{link}] [DRY-RUN] 将把 typeName 改为 Mesh 并复制几何属性")
            continue

        # 1. 通过 SdfLayer 修改 prim spec 的 typeName
        sdf_dst = Sdf.Path(dst_path)
        prim_spec = layer.GetPrimAtPath(sdf_dst)
        if prim_spec is None:
            # prim 可能只存在于 sublayer，在 root layer 创建 over spec
            parent_spec = layer.GetPrimAtPath(Sdf.Path("/colliders/" + link))
            prim_spec = Sdf.PrimSpec(parent_spec, link, Sdf.SpecifierOver)
        prim_spec.typeName = "Mesh"
        print(f"  [{link}] typeName -> Mesh")

        # 2. 复制几何属性（重新获取 prim，typeName 已改）
        dst_prim = stage.GetPrimAtPath(dst_path)
        for attr_name in _MESH_ATTRS:
            src_attr = src_prim.GetAttribute(attr_name)
            if not src_attr.IsValid():
                continue
            value = src_attr.Get()
            if value is None:
                continue
            type_name = src_attr.GetTypeName()
            dst_attr = dst_prim.GetAttribute(attr_name)
            if not dst_attr.IsValid():
                dst_attr = dst_prim.CreateAttribute(attr_name, type_name)
            dst_attr.Set(value)
            print(f"  [{link}] 复制属性 {attr_name}")

        # purpose 改为 guide（与其他 collider prim 一致）
        purpose_attr = dst_prim.GetAttribute("purpose")
        if not purpose_attr.IsValid():
            purpose_attr = dst_prim.CreateAttribute(
                "purpose", Sdf.ValueTypeNames.Token
            )
        purpose_attr.Set("guide")
        print(f"  [{link}] purpose -> guide")

    if not dry_run:
        layer.Save()
        print(f"[base.usd] 已保存")


def _patch_physics_usd(physics_usd: Path, dry_run: bool) -> None:
    """
    修改 M20_Piper_physics.usd：
    在 /colliders/<link>/<link> spec 上添加 PhysicsCollisionAPI
    及 physics:approximation = convexDecomposition。

    注意：/M20_Piper/<link>/collisions 是 instanceable prim，其子 prim
    是 instance proxy，不能直接通过 Usd.Stage 写入。必须直接操作 SdfLayer
    的 /colliders/<link>/<link> spec（即 prototype 在 physics layer 上的
    实际 spec 路径）。
    """
    from pxr import Sdf  # noqa: PLC0415

    print(f"[physics.usd] 打开 {physics_usd}")
    layer = Sdf.Layer.FindOrOpen(str(physics_usd))
    if layer is None:
        print(f"  [ERROR] 无法打开 SdfLayer: {physics_usd}")
        sys.exit(1)

    for link in _FINGER_LINKS:
        spec_path_str = _PHYSICS_COLL_PATH.format(link=link)
        sdf_path = Sdf.Path(spec_path_str)

        prim_spec = layer.GetPrimAtPath(sdf_path)
        if prim_spec is None:
            # spec 不存在，在 /colliders/<link> 下创建 over spec
            parent_path = Sdf.Path("/colliders/" + link)
            parent_spec = layer.GetPrimAtPath(parent_path)
            if parent_spec is None:
                # 递归建父 spec
                colliders_spec = layer.GetPrimAtPath(Sdf.Path("/colliders"))
                if colliders_spec is None:
                    colliders_spec = Sdf.PrimSpec(layer.GetPseudoRoot(), "colliders",
                                                  Sdf.SpecifierOver)
                parent_spec = Sdf.PrimSpec(colliders_spec, link, Sdf.SpecifierOver)
            prim_spec = Sdf.PrimSpec(parent_spec, link, Sdf.SpecifierOver)

        # 读取当前 apiSchemas
        schemas_field = prim_spec.GetInfo("apiSchemas")
        current_items = []
        if schemas_field is not None:
            current_items = list(schemas_field.explicitItems)
        print(f"  [{link}] spec_path={spec_path_str}")
        print(f"  [{link}] 当前 apiSchemas: {current_items}")

        if dry_run:
            print(f"  [{link}] [DRY-RUN] 将添加 PhysicsCollisionAPI + physics:approximation")
            continue

        # 追加 PhysicsCollisionAPI
        if "PhysicsCollisionAPI" not in current_items:
            current_items.append("PhysicsCollisionAPI")
        prim_spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(current_items))
        print(f"  [{link}] apiSchemas -> {current_items}")

        # physics:collisionEnabled = True
        ce_key = "physics:collisionEnabled"
        ce_attr_spec = prim_spec.attributes.get(ce_key)
        if ce_attr_spec is None:
            ce_attr_spec = Sdf.AttributeSpec(
                prim_spec, ce_key, Sdf.ValueTypeNames.Bool
            )
        ce_attr_spec.default = True
        print(f"  [{link}] physics:collisionEnabled -> True")

        # physics:approximation = "convexDecomposition"
        approx_key = "physics:approximation"
        approx_attr_spec = prim_spec.attributes.get(approx_key)
        if approx_attr_spec is None:
            approx_attr_spec = Sdf.AttributeSpec(
                prim_spec, approx_key, Sdf.ValueTypeNames.Token
            )
        approx_attr_spec.default = "convexDecomposition"
        print(f"  [{link}] physics:approximation -> convexDecomposition")

    if not dry_run:
        layer.Save()
        print(f"[physics.usd] 已保存")


def _verify(base_usd: Path, physics_usd: Path) -> None:
    """用 strings 快速确认写入结果（strings 输出有前缀压缩，做子串匹配）。"""
    print("\n=== 验证 ===")
    checks = {
        base_usd: ["Mesh"],
        physics_usd: ["PhysicsCollisionAPI", "convexDecomposition"],
    }
    all_ok = True
    for usd_file, keywords in checks.items():
        result = subprocess.run(
            ["strings", str(usd_file)], capture_output=True, text=True
        )
        raw = result.stdout
        for kw in keywords:
            found = kw in raw
            status = "OK" if found else "NOT FOUND"
            print(f"  {usd_file.name}: {kw:30s} [{status}]")
            if not found:
                all_ok = False
    if not all_ok:
        print("  [WARNING] 部分 token 未找到，可能是 USDC 前缀压缩导致 strings 匹配失败。")
        print("  建议用 Isaac Sim Script Editor 打开 USD 并检查 prim 树。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 link7/link8 夹爪手指补写 mesh 碰撞体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--usd-dir",
        type=Path,
        default=_DEFAULT_USD_DIR,
        help="包含 M20_Piper_base.usd 和 M20_Piper_physics.usd 的目录 (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要做的操作，不写入任何文件",
    )
    args = parser.parse_args()

    usd_dir = args.usd_dir.resolve()
    base_usd = usd_dir / "M20_Piper_base.usd"
    physics_usd = usd_dir / "M20_Piper_physics.usd"

    for f in [base_usd, physics_usd]:
        if not f.is_file():
            print(f"[ERROR] 文件不存在: {f}")
            print("  请先运行 convert_m20_piper_urdf.py 生成 USD 文件")
            sys.exit(1)

    if not args.dry_run:
        for f in [base_usd, physics_usd]:
            bak = f.with_suffix(".usd.bak")
            if bak.exists():
                print(f"[备份] {bak.name} 已存在，跳过覆盖")
            else:
                shutil.copy2(f, bak)
                print(f"[备份] {f.name} -> {bak.name}")

    print()
    _patch_base_usd(base_usd, args.dry_run)
    print()
    _patch_physics_usd(physics_usd, args.dry_run)

    if not args.dry_run:
        print()
        _verify(base_usd, physics_usd)
        print("\n完成。请重启 Isaac Sim 使修改生效。")
    else:
        print("\n[DRY-RUN 完成] 未写入任何文件。")


if __name__ == "__main__":
    main()
