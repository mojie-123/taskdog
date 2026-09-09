# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""
"""
__init__.py        ← 聚合器：把上游 IsaacLab 的 mdp + 本目录所有自定义函数合并为统一命名空间
commands.py        ← 指令生成器：决定每 episode 给机器人下发什么速度目标
observations.py    ← 观测函数：把仿真状态裁剪/变换成 policy 网络的输入特征
rewards.py         ← 奖励函数库：核心训练信号，1428行，轮腿专属设计
events.py          ← 事件/扰动：初始化与在线域随机化（DR）
curriculums.py     ← 课程调度器：根据训练进度自动调整地形难度和指令范围
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

# 注意：三层继承叠加，后者覆盖前者同名符号
