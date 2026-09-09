#!/usr/bin/env python3
"""Convert the lightweight M20 + Piper URDF into an isolated USD asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)   # 创建参数解析器，使用文档字符串作为描述
AppLauncher.add_app_launcher_args(parser)   # 向解析器添加 Isaac Lab 应用所需的参数（如 --headless、--enable_cameras 等）
args = parser.parse_args()   # 解析命令行参数
app_launcher = AppLauncher(args)   # 创建 AppLauncher 实例
simulation_app = app_launcher.app   # 获取底层的仿真应用对象（simulation_app）

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg   # UrdfConverter：执行 URDF → USD 转换的核心类。UrdfConverterCfg：转换配置类


from custom_envs.scripts.assets.build_m20_piper_urdf import OUTPUT_DIR, build


def main() -> Path:
    urdf_path = build()   # 生成urdf
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(OUTPUT_DIR),
        usd_file_name="M20_Piper.usd",
        force_usd_conversion=True,   # 强制重新转换，即使目标文件已存在
        make_instanceable=True,   # 生成可实例化的 USD 资产（节省内存，支持多机器人场景
        fix_base=False,   # 不固定基座，允许机器人整体移动（移动基座机器人
        root_link_name="base_link",   # 指定根连杆名称
        merge_fixed_joints=False,   # False	不合并固定关节，保留原始关节结构
        collision_from_visuals=False,   # 不从视觉网格生成碰撞体（因为 URDF 中已有专门的碰撞体）
        collider_type="convex_hull",   # 碰撞体类型为凸包（比原始网格更高效）
        self_collision=False,   # 禁用自碰撞（减少计算量，机械臂通常不需要自碰撞检测）
        replace_cylinders_with_capsules=False,   # 不将圆柱体替换为胶囊体
        joint_drive=None,   # 使用默认关节驱动配置
        link_density=0.0,   # 使用 URDF 中定义的密度（0 表示不覆盖）
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