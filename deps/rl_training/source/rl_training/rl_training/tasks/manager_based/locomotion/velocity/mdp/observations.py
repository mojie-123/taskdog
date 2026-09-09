# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # 所有joint的配置
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),   # 轮子joint的配置
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    # ``wheel_asset_cfg.joint_ids`` are global articulation indices, while
    # ``joint_pos_rel`` may already be a subset selected by ``asset_cfg``.
    # Convert the wheel IDs to indices local to that subset. This matters for
    # combined articulations whose selected locomotion joints are non-contiguous.
    if isinstance(asset_cfg.joint_ids, slice):   # 此时 joint_pos_rel 的列索引与全局关节索引完全一致，所以可以直接使用 wheel_asset_cfg.joint_ids
        wheel_joint_ids = wheel_asset_cfg.joint_ids
    elif isinstance(wheel_asset_cfg.joint_ids, slice):   # 所有关节都是轮子
        wheel_joint_ids = slice(None)
    else:
        wheel_global_ids = set(wheel_asset_cfg.joint_ids)
        wheel_joint_ids = [
            local_id for local_id, global_id in enumerate(asset_cfg.joint_ids) if global_id in wheel_global_ids
        ]
    joint_pos_rel[:, wheel_joint_ids] = 0   # 轮子置零
    return joint_pos_rel   # 这个列表里只有asset_cfg.joint_ids对应的值（可能不是所有关节），其中轮子部分置零


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    # 计算当前 episode 的相位（phase），并将其编码为正弦和余弦值，常用于步态控制或周期性运动的策略观测
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)   # 每个环境当前 episode 已经执行了多少步
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time   # env.step_dt：仿真步长（如 0.02s）；cycle_time：周期时间（如 0.5s 表示一个完整步态周期）
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)   # 正余弦编码
    return phase_tensor
