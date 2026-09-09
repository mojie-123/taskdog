# learn.md — M20 Pro 接入 IsaacLab 完整实现过程

> **目的**: 记录从零开始在 `custom_envs/` 下搭建 M20 Pro 自定义环境的每一步修改，理解 IsaacLab 环境注册机制、继承链、以及如何将自定义机器人接入训练流程。

---

## 目录

1. [修改总览](#1-修改总览)
2. [环境注册机制概述](#2-环境注册机制概述)
3. [修改 A：修复 `handle_deprecated_rsl_rl_cfg` 兼容性问题](#3-修改-a修复-handle_deprecated_rsl_rl_cfg-兼容性问题)
4. [修改 B：创建 custom_envs Python 包基础结构](#4-修改-b创建-custom_envs-python-包基础结构)
5. [修改 C：编写 M20 Pro 环境配置文件](#5-修改-c编写-m20-pro-环境配置文件)
6. [修改 D：编写 PPO 算法配置文件](#6-修改-d编写-ppo-算法配置文件)
7. [修改 E：编写 Gym 环境注册文件](#7-修改-e编写-gym-环境注册文件)
8. [修改 F：安装 custom_envs 包](#8-修改-f安装-custom_envs-包)
9. [修改 G：将 custom_envs 接入训练脚本](#9-修改-g将-custom_envs-接入训练脚本)
10. [完整数据流](#10-完整数据流)
11. [后续开发指南](#11-后续开发指南)
12. [分析 & 实现完成](#12-分析--实现完成)

---

## 1. 修改总览

本次共涉及 **12 个文件**（创建 7 个、修改 3 个、删除 1 个、新增系统文件 1 个）：

| 文件 | 操作 | 类别 |
|------|------|------|
| `deps/rl_training/.../train.py` | 修改 (3处) | 兼容性修复 + 注册新环境 |
| `deps/rl_training/.../play.py` | 修改 (3处) | 兼容性修复 + 注册新环境 + checkpoint 路径修复 |
| `deps/rl_training/.../list_envs.py` | 修改 (1处) | 注册新环境 |
| `custom_envs/__init__.py` | 重写 | 包初始化 |
| `custom_envs/tasks/__init__.py` | 重写 | 自动发现子包 |
| `custom_envs/tasks/deeprobotics_m20_pro/__init__.py` | 重写 | Gym 环境注册 |
| `custom_envs/tasks/deeprobotics_m20_pro/rough_env_cfg.py` | 创建 | 崎岖地形配置 |
| `custom_envs/tasks/deeprobotics_m20_pro/flat_env_cfg.py` | 创建 | 平坦地形配置 |
| `custom_envs/tasks/deeprobotics_m20_pro/agents/rsl_rl_ppo_cfg.py` | 创建 | PPO 算法配置 |
| `custom_envs/setup.py` | 创建后删除 | pip 安装尝试（失败，改用 .pth） |
| `.../site-packages/taskdog.pth` | 创建 | Python 路径注入（替代 pip install） |

---

## 2. 环境注册机制概述

### 2.1 IsaacLab 的环境是如何被发现的？

IsaacLab 使用 OpenAI Gym 的注册表（registry）机制来管理环境。整个流程如下：

```
train.py 启动
    │
    ├─① AppLauncher 启动 Isaac Sim / Omniverse
    │
    ├─② import rl_training.tasks    ← Python 导入包
    │     │
    │     └─ rl_training/tasks/__init__.py
    │          │  import_packages(__name__, _BLACKLIST_PKGS)
    │          │  自动遍历 tasks/ 下所有子目录，逐个 import
    │          │
    │          ├─ import ...config/wheeled/deeprobotics_m20
    │          │     └─ __init__.py 执行:
    │          │         gym.register(
    │          │             id="Rough-Deeprobotics-M20-v0",     ← 环境名称
    │          │             entry_point="...ManagerBasedRLEnv",  ← 环境类
    │          │             kwargs={
    │          │                 "env_cfg_entry_point": "....rough_env_cfg:DeeproboticsM20RoughEnvCfg",
    │          │                 "rsl_rl_cfg_entry_point": "....rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    │          │             }
    │          │         )
    │          │
    │          ├─ import ...config/wheeled/deeprobotics_m20/agents
    │          └─ ... (其他子包)
    │
    ├─③ import custom_envs.tasks     ← 我们的新环境也是同样的流程!
    │
    ├─④ gym.make("Rough-Deeprobotics-M20Pro-v0", cfg=env_cfg)
    │     │  Gym 根据注册的 entry_point 找到 ManagerBasedRLEnv
    │     │  根据 env_cfg_entry_point 加载 DeeproboticsM20ProRoughEnvCfg
    │     │  根据 rsl_rl_cfg_entry_point 加载 DeeproboticsM20ProRoughPPORunnerCfg
    │     ▼
    └─⑤ 环境实例化 → PPO 训练循环
```

### 2.2 关键概念

| 概念 | 说明 | 类比 |
|------|------|------|
| `gym.register()` | 向 Gym 全局注册表添加一个环境 ID | 注册一个"服务名" |
| `entry_point` | 环境类的 Python 路径，`gym.make()` 时实例化 | 服务实现类 |
| `env_cfg_entry_point` | 环境配置类路径 `module:ClassName` | 服务的配置 |
| `rsl_rl_cfg_entry_point` | PPO 算法配置类路径 | 训练算法的配置 |
| `import_packages()` | IsaacLab 工具函数，自动递归 import 子包 | 批量 import |
| `@configclass` | IsaacLab 装饰器，让 dataclass 支持嵌套和继承 | 配置类的语法糖 |

### 2.3 继承链

我们的 M20 Pro 环境通过继承复用上游代码：

```
isaaclab.envs.ManagerBasedRLEnvCfg          ← IsaacLab 基础环境类
    └─ LocomotionVelocityRoughEnvCfg         ← rl_training 速度跟踪基类
        └─ DeeproboticsM20RoughEnvCfg        ← 官方 M20 配置
            └─ DeeproboticsM20ProRoughEnvCfg  ← ★ 我们的 M20 Pro 配置
                 (在 __post_init__ 中定制)
```

这样我们只需要写 ~30 行代码，所有底层逻辑（地形生成、奖励函数、Domain Randomization、终止条件）全部继承自上游。

---

## 3. 修改 A：修复 `handle_deprecated_rsl_rl_cfg` 兼容性问题

### 3.1 问题

运行 `train.py` 时报错：

```
ImportError: cannot import name 'handle_deprecated_rsl_rl_cfg' from 'isaaclab_rl.rsl_rl'
```

### 3.2 原因

`rl_training` 仓库开发时使用的是 IsaacLab 2.4+（或 `main` 分支），其中 `isaaclab_rl.rsl_rl` 模块包含了 `handle_deprecated_rsl_rl_cfg` 函数。但我们 `env_isaaclab` 环境安装的是 **IsaacLab 2.3.2**，该版本还没有这个函数。

这个函数的作用是：当用户使用旧版 RSL-RL 配置格式（flat policy config）时，自动转换为新格式（分离的 actor/critic config）。我们的环境使用的就是新格式，所以不需要这个函数。

### 3.3 修改文件

**文件**: `deps/rl_training/scripts/reinforcement_learning/rsl_rl/train.py`  
**位置**: 第 102 行

**修改前**:
```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
```

**修改后**:
```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    # Fallback for IsaacLab <2.4 where this function doesn't exist
    def handle_deprecated_rsl_rl_cfg(cfg, installed_version):
        return cfg
```

**解释**:
- 先导入肯定存在的 `RslRlOnPolicyRunnerCfg` 和 `RslRlVecEnvWrapper`
- 单独尝试导入 `handle_deprecated_rsl_rl_cfg`，如果失败（`ImportError`），定义一个什么都不做的替代函数，直接返回原配置
- 后续代码 `agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)` 在两种情况下都能正常工作

---

**文件**: `deps/rl_training/scripts/reinforcement_learning/rsl_rl/play.py`  
**位置**: 第 107-112 行

**修改前**:
```python
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
```

**修改后**:
```python
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    # Fallback for IsaacLab <2.4 where this function doesn't exist
    def handle_deprecated_rsl_rl_cfg(cfg, installed_version):
        return cfg
```

**解释**: 与 train.py 完全相同的修改逻辑，因为 play.py 在第 184 行也调用了 `handle_deprecated_rsl_rl_cfg`。

---

## 4. 修改 B：创建 custom_envs Python 包基础结构

### 4.1 为什么需要 `setup.py`？

要让 `import custom_envs` 工作，custom_envs 必须是一个 Python 包。通过 `pip install -e`（开发模式安装），我们把它注册到 Python 的 `site-packages` 中，这样任何脚本都可以直接 `import custom_envs`。

### 4.2 修改文件

**文件**: `custom_envs/setup.py`（创建）

```python
"""Installation script for custom_envs python package."""

from setuptools import setup, find_packages

setup(
    name="custom_envs",
    packages=find_packages(),
    version="0.1.0",
    description="Custom IsaacLab environments for M20 Pro robot dog",
    python_requires=">=3.10",
    zip_safe=False,
)
```

**解释**:
- `find_packages()` — 自动发现当前目录下所有含 `__init__.py` 的子目录作为包
- `version="0.1.0"` — 初始版本号，后续每次大改可以递增
- `zip_safe=False` — IsaacLab 需要读取文件系统中的配置文件，不能从 zip 包加载
- 注意：这里故意**没有**在 `install_requires` 中声明 `isaaclab`/`isaaclab_tasks` 等依赖，因为这些包不在 PyPI 上，声明了反而会导致 `pip install` 失败

---

**文件**: `custom_envs/__init__.py`（重写）

**修改前**（空文件）:
```python

```

**修改后**:
```python
"""Custom IsaacLab environments for DeepRobotics M20 Pro.

This package provides custom RL training environments for the M20 Pro robot dog.
It inherits from the official rl_training Deeprobotics M20 configurations,
allowing customization without modifying upstream code.
"""

from . import tasks  # noqa: F401
```

**解释**:
- `from . import tasks` — 当有人 `import custom_envs` 时，自动导入 `tasks` 子包
- `# noqa: F401` — 告诉 linter "这个 import 看起来没被使用，但它是故意的"（我们确实只为了触发其 `__init__.py` 执行）
- 这个 import 会触发 `tasks/__init__.py` 的执行，进而触发 `import_packages()` 自动发现

---

**文件**: `custom_envs/tasks/__init__.py`（重写）

**修改前**（空文件）:
```python

```

**修改后**:
```python
"""Task implementations for custom environments.

Auto-discovers and imports all task configurations so that
gym.register() calls in each task's __init__.py are executed.
"""

from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]

# Import all configs in sub-packages (this triggers gym.register() calls)
import_packages(__name__, _BLACKLIST_PKGS)
```

**解释**:
- `import_packages(__name__, _BLACKLIST_PKGS)` — IsaacLab 提供的工具函数，自动扫描当前包（`custom_envs.tasks`）下所有子目录，对每个含 `__init__.py` 的子目录执行 `import`
- `_BLACKLIST_PKGS = ["utils"]` — 排除 `utils/` 目录（它包含工具函数而非任务配置，不需要注册环境）
- 这个自动发现机制意味着：将来如果你添加 `custom_envs/tasks/navigation/` 子目录，不需要修改这个文件，新任务会被自动发现

---

## 5. 修改 C：编写 M20 Pro 环境配置文件

### 5.1 环境配置的作用

环境配置文件定义了 RL 训练的一切：机器人用什么 USD 模型、观测空间包含哪些量、动作如何定义、奖励函数怎么算、地形有多难、Domain Randomization 范围多大等等。

### 5.2 rough_env_cfg.py

**文件**: `custom_envs/tasks/deeprobotics_m20_pro/rough_env_cfg.py`（创建）

```python
"""M20 Pro rough-terrain environment configuration.

Inherits from the official rl_training Deeprobotics M20 rough config.
Customize this file to add M20 Pro-specific behaviors (e.g. different
reward weights, domain randomization ranges, observation terms).
"""

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20RoughEnvCfg,
)


@configclass
class DeeproboticsM20ProRoughEnvCfg(DeeproboticsM20RoughEnvCfg):
    """M20 Pro rough-terrain locomotion environment config.

    Currently identical to the M20 config. Override __post_init__ here
    to customize observation space, action scale, reward weights,
    domain randomization, terrain difficulty, etc.
    """

    def __post_init__(self):
        # Call parent first (this sets up all M20 defaults)
        super().__post_init__()

        # --- Customize below for M20 Pro ---
        # Examples of what you can change:
        #
        #   self.observations.policy.base_lin_vel.scale = 2.0
        #   self.rewards.track_lin_vel_xy_exp.weight = 8.0
        #   self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        #   self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.25)
        #
        # For now, we keep the same configuration as the official M20.
        # This proves the inheritance chain works end-to-end.

        # Disable zero-weight rewards (cleanup inherited from parent pattern)
        self.disable_zero_weight_rewards()
```

**逐行解释**:

| 代码 | 解释 |
|------|------|
| `from isaaclab.utils import configclass` | `@configclass` 装饰器，让类支持 IsaacLab 的嵌套 dataclass 语法 |
| `from rl_training....rough_env_cfg import DeeproboticsM20RoughEnvCfg` | 导入官方 M20 配置作为父类 |
| `class DeeproboticsM20ProRoughEnvCfg(DeeproboticsM20RoughEnvCfg)` | **继承**：获得 M20 的所有配置（机器人模型、场景、观测、动作、奖励、终止条件、domain randomization） |
| `def __post_init__(self)` | IsaacLab 配置类的初始化钩子。`super().__post_init__()` 先执行父类的所有设置，然后我们追加自定义。这个顺序很重要！ |
| `self.disable_zero_weight_rewards()` | 父类定义的方法，遍历所有奖励项，把 weight=0 的设为 None（避免计算浪费） |
| 注释区 | 列举了常见的定制点，需要改什么就取消注释对应的行 |

### 5.3 flat_env_cfg.py

**文件**: `custom_envs/tasks/deeprobotics_m20_pro/flat_env_cfg.py`（创建）

```python
"""M20 Pro flat-terrain environment configuration."""

from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.flat_env_cfg import (
    DeeproboticsM20FlatEnvCfg,
)


@configclass
class DeeproboticsM20ProFlatEnvCfg(DeeproboticsM20FlatEnvCfg):
    """M20 Pro flat-terrain locomotion environment config."""

    def __post_init__(self):
        super().__post_init__()
        self.disable_zero_weight_rewards()
```

**解释**:
- 逻辑与 rough 版本完全一致，但继承自 `DeeproboticsM20FlatEnvCfg`（平坦地形版本）
- 父类的 `__post_init__` 会自动：把地形改成平面、移除高度扫描传感器、移除地形课程学习
- Flat 配置比 Rough 简单很多，因为不需要处理地形复杂性

---

## 6. 修改 D：编写 PPO 算法配置文件

### 6.1 算法配置的作用

PPO 配置定义了**训练算法**的超参数，与环境配置（定义 MDP）是独立的两件事：

- 环境配置 = "机器人要学什么"（观测是什么、奖励是什么）
- 算法配置 = "怎么学"（网络多大、学习率多少、训练多少步）

### 6.2 修改文件

**文件**: `custom_envs/tasks/deeprobotics_m20_pro/agents/rsl_rl_ppo_cfg.py`（创建）

```python
"""PPO algorithm configuration for M20 Pro environments."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20ProRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for M20 Pro rough terrain."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20pro_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20ProFlatPPORunnerCfg(DeeproboticsM20ProRoughPPORunnerCfg):
    """PPO runner config for M20 Pro flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20pro_flat"
```

**逐行解释**:

| 参数 | 值 | 解释 |
|------|-----|------|
| `num_steps_per_env` | 24 | 每个环境每轮收集 24 步数据，总共 24×4096 ≈ 10 万 transitions 做一次 PPO 更新 |
| `max_iterations` | 20000 / 5000 | 总训练迭代数。Rough 需要更多步数（地形难），Flat 收敛快 |
| `save_interval` | 100 | 每 100 轮保存一次 checkpoint（`model_100.pt`, `model_200.pt`...） |
| `experiment_name` | `"deeprobotics_m20pro_rough"` | 日志目录名（用于 TensorBoard 和 checkpoint 路径） |
| `empirical_normalization` | False | 是否对观测做 running mean/std 归一化 |
| `clip_actions` | 100 | PPO 的动作裁剪上限（这里是很大的值，实际不裁剪） |
| `actor_hidden_dims` | [512, 256, 128] | Actor 网络结构（3 层 MLP） |
| `critic_hidden_dims` | [512, 256, 128] | Critic 网络结构（与 Actor 相同） |
| `activation` | `"elu"` | 激活函数 |
| `learning_rate` | 1e-3 | 初始学习率（adaptive scheduler 会自动调整） |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ 参数 |
| `clip_param` | 0.2 | PPO clip 范围 |
| `entropy_coef` | 0.003 | 熵正则系数（鼓励探索） |
| `desired_kl` | 0.01 | 自适应 LR 的目标 KL 散度 |

---

## 7. 修改 E：编写 Gym 环境注册文件

### 7.1 这是最关键的文件

这个文件把环境配置类和算法配置类"绑定"到一个环境 ID 上，`train.py` 通过 `--task` 参数找到对应的配置。

**文件**: `custom_envs/tasks/deeprobotics_m20_pro/__init__.py`（重写）

**修改前**（空文件）:
```python

```

**修改后**:
```python
"""Register M20 Pro Gym environments.

This file registers two environment IDs:
    Flat-Deeprobotics-M20Pro-v0   — Flat terrain locomotion
    Rough-Deeprobotics-M20Pro-v0  — Rough terrain locomotion

Both inherit from the official M20 configs and can be trained
using the same scripts (just change --task).
"""

import gymnasium as gym

from . import agents  # noqa: F401 (make agent configs available)

##
# Register Gym environments
##

gym.register(
    id="Flat-Deeprobotics-M20Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20ProFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20ProRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProRoughPPORunnerCfg",
    },
)
```

**逐行解释**:

| 代码 | 解释 |
|------|------|
| `import gymnasium as gym` | OpenAI Gymnasium 库，提供 `gym.register()` 和 `gym.make()` |
| `from . import agents` | 导入 agents 子包，确保 PPO 配置类在 Python 命名空间中可见 |
| `gym.register(...)` | 向全局注册表添加一个环境。`gym.make("Flat-Deeprobotics-M20Pro-v0")` 会根据注册信息创建环境 |
| `id="Flat-Deeprobotics-M20Pro-v0"` | 环境唯一标识符。`--task` 参数传的就是这个字符串 |
| `entry_point="isaaclab.envs:ManagerBasedRLEnv"` | 环境实现类。冒号前是模块路径，冒号后是类名。`ManagerBasedRLEnv` 是 IsaacLab 的标准 RL 环境类 |
| `disable_env_checker=True` | 跳过 Gym 的环境兼容性检查（IsaacLab 环境不完全遵循 Gym API 规范） |
| `kwargs={...}` | 传递给环境类的额外参数。IsaacLab 用 kwargs 传递配置类路径 |
| `env_cfg_entry_point` | 环境配置类的路径，格式 `module.submodule:ClassName`。`train.py` 会解析这个字符串，用 hydra 实例化对应的配置类 |
| `rsl_rl_cfg_entry_point` | PPO 算法配置类的路径，格式同上 |
| `f"{__name__}.flat_env_cfg:..."` | `__name__` 是当前模块名（如 `custom_envs.tasks.deeprobotics_m20_pro`），动态构建完整路径 |

### 7.2 `gym.register()` 完整调用链

```
train.py --task=Flat-Deeprobotics-M20Pro-v0
    │
    ├─ gym.make("Flat-Deeprobotics-M20Pro-v0", cfg=env_cfg)
    │     │  Gym 查注册表 → 找到 id="Flat-Deeprobotics-M20Pro-v0"
    │     │  读取 entry_point → "isaaclab.envs:ManagerBasedRLEnv"
    │     │  读取 kwargs → {
    │     │      "env_cfg_entry_point": "custom_envs.tasks.deeprobotics_m20_pro.flat_env_cfg:DeeproboticsM20ProFlatEnvCfg",
    │     │      "rsl_rl_cfg_entry_point": "custom_envs.tasks.deeprobotics_m20_pro.agents.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg",
    │     │  }
    │     │
    │     ▼
    │  实例化 ManagerBasedRLEnv(cfg=DeeproboticsM20ProFlatEnvCfg)
    │     │
    │     ├─ 加载场景: M20 USD 模型 + 平坦地形
    │     ├─ 配置观测: 57 维 (base_ang_vel + gravity + commands + joint_pos + joint_vel + actions)
    │     ├─ 配置动作: 16 维 (12 腿位置 + 4 轮速度)
    │     ├─ 配置奖励: 速度跟踪 + 姿态惩罚 + 关节平滑 ...
    │     ├─ 配置 DR: 质量 ±15%、摩擦 0.35~1.5 ...
    │     └─ 返回 env 对象
    │
    └─ RslRlVecEnvWrapper(env) → OnPolicyRunner(env, agent_cfg) → runner.learn()
```

---

## 8. 修改 F：安装 custom_envs 包（已修正为 .pth 方式）

### 8.1 第一版尝试：`pip install -e`（失败）

最初尝试用 `pip install -e /home/mojie/taskdog/custom_envs` 安装。安装后 `pip show custom_envs` 显示已安装，但实际 `import custom_envs` 报 `ModuleNotFoundError`。

### 8.2 根因

`setup.py` 放在 `custom_envs/` 目录**内部**（与 `__init__.py` 同级）。`find_packages()` 从 `setup.py` 所在目录开始搜索，找到了 `tasks`、`utils` 等子包，但**没有将 `custom_envs/` 本身注册为根包**。

生成的 `__editable___custom_envs_0_1_0_finder.py` 中的 `MAPPING` 只有子包路径映射，缺少根包的路径入口：

```python
MAPPING: dict[str, str] = {
    'tasks': '/home/mojie/taskdog/custom_envs/tasks',
    'utils': '/home/mojie/taskdog/custom_envs/utils'
}
```

没有 `'custom_envs': '/home/mojie/taskdog/custom_envs'`，所以 `import custom_envs` 永远找不到。

### 8.3 第二版方案：`.pth` 路径注入

放弃 `pip install -e`，改用 Python 标准的 `.pth` 文件机制：

```bash
# 1. 卸载 broken 的 pip 安装
pip uninstall custom_envs -y

# 2. 删除 setup.py 和 egg-info（不再需要）
rm custom_envs/setup.py
rm -rf custom_envs/custom_envs.egg-info

# 3. 创建 .pth 文件，将项目根目录注入 sys.path
echo "/home/mojie/taskdog" > /home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/taskdog.pth
```

### 8.4 `.pth` 原理

Python 启动时自动扫描 `site-packages/` 下所有 `.pth` 文件，将其中每一行路径追加到 `sys.path`。这样：

```
Python 启动
  → 扫描 site-packages/
  → 找到 taskdog.pth
  → sys.path.append("/home/mojie/taskdog")
  → import custom_envs 时
     在 sys.path 中找到 /home/mojie/taskdog/custom_envs/
     → import 成功
```

### 8.5 为什么这样做比 `pip install -e` 更好

| 方式 | 优点 | 缺点 |
|------|------|------|
| `pip install -e` | 标准做法 | 要求 `setup.py` 在包外一层，即需要 `custom_envs/setup.py` + `custom_envs/custom_envs/__init__.py` 的双层结构 |
| `.pth` 文件 | 简单直接，不改目录结构 | 手动管理，不经过 pip |

由于我们已经按单层结构组织了 `custom_envs/`（`__init__.py` 直接在 `custom_envs/` 下），`.pth` 方式比 `pip install -e` 更适合当前布局。

### 8.6 修改后的 custom_envs 目录

```
custom_envs/                        ← 通过 .pth 加入 sys.path
├── __init__.py                     ← from . import tasks
├── tasks/
│   ├── __init__.py                 ← import_packages() 自动发现
│   └── deeprobotics_m20_pro/
│       ├── __init__.py             ← gym.register()
│       ├── flat_env_cfg.py
│       ├── rough_env_cfg.py
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py
└── utils/
    └── __init__.py
```

> `setup.py` 和 `custom_envs.egg-info/` 已删除。

---

## 9. 修改 G：将 custom_envs 接入训练脚本

### 9.1 为什么需要这一步？

`pip install -e` 让 `custom_envs` 可以被 Python import，但它不会**自动**被执行。我们需要在训练/回放/环境列表脚本中**主动 import** custom_envs，触发 `gym.register()` 调用。

这就好比：你安装了一个 WordPress 插件，但还需要在主程序中"激活"它。

### 9.2 修改文件

**文件**: `deps/rl_training/scripts/reinforcement_learning/rsl_rl/train.py`  
**位置**: 第 112 行附近

**修改前**:
```python
import rl_training.tasks  # noqa: F401
```

**修改后**:
```python
import rl_training.tasks  # noqa: F401
import custom_envs.tasks  # noqa: F401 (register M20Pro environments)
```

**解释**: 增加一行 `import custom_envs.tasks`，触发 `custom_envs/tasks/__init__.py` → `import_packages()` → `gym.register()`

---

**文件**: `deps/rl_training/scripts/reinforcement_learning/rsl_rl/play.py`  
**位置**: 第 121 行附近

**修改前**:
```python
import rl_training.tasks  # noqa: F401
```

**修改后**:
```python
import rl_training.tasks  # noqa: F401
import custom_envs.tasks  # noqa: F401 (register M20Pro environments)
```

**解释**: 与 train.py 相同的修改。这样 `play.py` 回放时也能使用 M20Pro 环境。

---

**文件**: `deps/rl_training/scripts/tools/list_envs.py`  
**位置**: 第 37 行附近

**修改前**:
```python
import rl_training.tasks  # noqa: F401
```

**修改后**:
```python
import rl_training.tasks  # noqa: F401
import custom_envs.tasks  # noqa: F401 (register M20Pro environments)
```

**解释**: 这样 `list_envs.py` 列出的环境中会包含 `Flat-Deeprobotics-M20Pro-v0` 和 `Rough-Deeprobotics-M20Pro-v0`。

---

## 10. 完整数据流

### 10.1 启动到训练

```
$ conda activate env_isaaclab
$ python scripts/reinforcement_learning/rsl_rl/train.py \
      --task=Flat-Deeprobotics-M20Pro-v0 \
      --headless \
      --num_envs=2048

    ┌────────────────────────────────────────────┐
    │ ① AppLauncher 启动 Isaac Sim (Omniverse)     │
    │    - 加载 Carbonite 运行时                   │
    │    - 初始化 PhysX 物理引擎                    │
    │    - 检测 GPU (RTX 4060 Laptop 8GB)          │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │ ② 环境注册                                  │
    │    import rl_training.tasks                 │
    │      └→ gym.register("Rough-Deeprobotics-M20-v0", ...)  ← 官方 M20
    │      └→ gym.register("Flat-Deeprobotics-M20-v0", ...)    ← 官方 M20
    │    import custom_envs.tasks                 │
    │      └→ gym.register("Flat-Deeprobotics-M20Pro-v0", ...)  ← ★ 我们的
    │      └→ gym.register("Rough-Deeprobotics-M20Pro-v0", ...) ← ★ 我们的
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │ ③ hydra 解析配置                            │
    │    --task=Flat-Deeprobotics-M20Pro-v0       │
    │      └→ 查 gym 注册表                       │
    │      └→ env_cfg = DeeproboticsM20ProFlatEnvCfg()  │
    │           └→ super().__post_init__()  (M20 flat 配置)
    │           └→ disable_zero_weight_rewards()
    │      └→ agent_cfg = DeeproboticsM20ProFlatPPORunnerCfg()
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │ ④ 环境实例化                                │
    │    env = gym.make(task, cfg=env_cfg)        │
    │      └→ ManagerBasedRLEnv.__init__()        │
    │           ├→ 加载 USD 场景 (M20 + 平面)      │
    │           ├→ 创建 2048 个并行环境             │
    │           ├→ 配置 57 维观测                   │
    │           ├→ 配置 16 维动作                   │
    │           ├→ 配置 15+ 个奖励项                │
    │           ├→ 配置 Domain Randomization        │
    │           └→ 配置终止条件                     │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │ ⑤ RSL-RL 包装与训练                         │
    │    env = RslRlVecEnvWrapper(env)            │
    │    runner = OnPolicyRunner(env, agent_cfg)   │
    │    runner.learn(max_iterations=5000)         │
    │      └→ PPO 循环 × 5000 次                   │
    │           ├→ 收集 rollout (24 steps × 2048)  │
    │           ├→ GAE 优势估计                     │
    │           ├→ 5 epochs minibatch SGD          │
    │           └→ 每 100 次保存 checkpoint         │
    └───────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────────────────┐
    │ ⑥ 输出                                      │
    │    logs/rsl_rl/deeprobotics_m20pro_flat/    │
    │    └→ <timestamp>/                          │
    │         ├→ model_100.pt                     │
    │         ├→ model_200.pt                     │
    │         ├→ ...                               │
    │         ├→ params/env.yaml                  │
    │         ├→ params/agent.yaml                │
    │         └→ events.out.tfevents.*            │
    └────────────────────────────────────────────┘
```

### 10.2 你将来如何定制

所有定制都在 `rough_env_cfg.py` 的 `__post_init__` 中完成，在 `super().__post_init__()` 之后追加代码：

```python
def __post_init__(self):
    super().__post_init__()  # ← 保留这一行! M20 的所有配置都在这里加载

    # ===== 你的定制从这里开始 =====

    # ① 改观测: 加回 base_lin_vel
    self.observations.policy.base_lin_vel = ObsTerm(...)

    # ② 改奖励权重
    self.rewards.track_lin_vel_xy_exp.weight = 8.0  # 更强调速度跟踪

    # ③ 改速度命令范围
    self.commands.base_velocity.ranges.lin_vel_x = (-3.0, 3.0)

    # ④ 改 Domain Randomization
    self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.7, 1.3)

    # ⑤ 改地形难度
    self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.3)

    # ===== 你的定制到这里结束 =====

    self.disable_zero_weight_rewards()  # ← 保留这一行! 清理无效奖励
```

不需要修改其他任何文件。

---

## 11. 后续开发指南

### 11.1 如果要加新任务

在 `custom_envs/tasks/` 下新建目录，例如：

```
custom_envs/tasks/
├── deeprobotics_m20_pro/        ← 已有: locomotion
└── m20pro_grasping/             ← 新任务: 抓取
    ├── __init__.py              ← gym.register("M20Pro-Grasping-v0")
    ├── grasp_env_cfg.py
    └── agents/
        └── rsl_rl_ppo_cfg.py
```

不需要修改 `tasks/__init__.py`，`import_packages()` 会自动发现新子包。

### 11.2 如果要改 PPO 超参数

直接修改 `custom_envs/tasks/deeprobotics_m20_pro/agents/rsl_rl_ppo_cfg.py`。由于 `pip install -e` 是开发模式，修改后下次启动训练立刻生效。

### 11.3 如果要加自定义奖励函数

在 `custom_envs/utils/` 下新建 `rewards.py`，然后在 `rough_env_cfg.py` 中 `import` 并使用：

```python
from custom_envs.utils.rewards import my_custom_reward

# 在 __post_init__ 中:
self.rewards.my_term = RewTerm(func=my_custom_reward, weight=1.0, params={...})
```

---

## 12. 修改 H：play.py checkpoint 路径解析修复

### 12.1 问题

运行 play 命令时报错，即使传了 `--checkpoint=model_1900.pt`，报错仍指向一个不存在的 `model_4999.pt`：

```
FileNotFoundError: Unable to find the file: model_4999.pt
```

### 12.2 根因

原来的 play.py 使用了 `retrieve_file_path()` 来解析 `--checkpoint` 参数：

```python
# 原来的代码 (有问题)
if args_cli.checkpoint:
    resume_path = retrieve_file_path(args_cli.checkpoint)
else:
    resume_path = get_checkpoint_path(...)
```

`retrieve_file_path()` 是 IsaacLab 的通用文件查找函数，用于解析绝对路径或 Nucleus Server URL（如 `omniverse://`）。当传入裸文件名 `model_1900.pt` 时：
- 它只在**当前工作目录**查找该文件（`os.path.abspath("model_1900.pt")` → `cwd/model_1900.pt`）
- 它不知道要去 `logs/rsl_rl/deeprobotics_m20_flat/<run_dir>/` 下拼接路径
- 同时 Hydra 配置系统与 argparse 存在参数传递的交互问题，可能导致 `args_cli.checkpoint` 被 Hydra 的内部 config store 覆盖为其他值

### 12.3 修改内容

**文件**: `deps/rl_training/scripts/reinforcement_learning/rsl_rl/play.py`
**位置**: 第 230-243 行

**修改后**:

```python
if args_cli.checkpoint:
    # If checkpoint is a bare filename, resolve it relative to the run directory
    if os.path.isabs(args_cli.checkpoint) or args_cli.checkpoint.startswith(
        ("http://", "https://", "omniverse://")
    ):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # Treat as a filename inside the run directory
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, args_cli.checkpoint
        )
elif args_cli.load_run:
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
else:
    resume_path = get_checkpoint_path(
        log_root_path, run_dir=".*", checkpoint=agent_cfg.load_checkpoint
    )
print(f"[INFO] Resolved checkpoint path: {resume_path}")
```

### 12.4 逻辑说明

`--checkpoint` 参数现在有三种解析路径：

```
用户传 --checkpoint=XXX
    │
    ├─ 绝对路径? (/home/xxx/model.pt)
    │    → retrieve_file_path() — 直接查找本地文件或 Nucleus Server
    │
    ├─ URL? (http://, https://, omniverse://)
    │    → retrieve_file_path() — 从 Nucleus Server 下载后返回本地路径
    │
    └─ 裸文件名? (model_1900.pt)
         → get_checkpoint_path(log_root_path, load_run, checkpoint)
             拼接: logs/rsl_rl/<experiment_name>/<load_run>/<filename>
             例: logs/rsl_rl/deeprobotics_m20_flat/
                  2026-07-18_10-57-32/model_1900.pt
```

其中三个关键参数来源：
- `log_root_path` = `logs/rsl_rl/<agent_cfg.experiment_name>`（在 PPO 配置中定义，如 `deeprobotics_m20_flat`）
- `agent_cfg.load_run` = 来自 `--load_run` CLI 参数（通过 `update_rsl_rl_cfg` 从 `args_cli.load_run` 赋值）
- `agent_cfg.load_checkpoint` = 来自 `--checkpoint` CLI 参数（同上）

### 12.5 调试输出

新增了 `print(f"[INFO] Resolved checkpoint path: {resume_path}")` 行。无论走哪个分支，最终解析出的完整路径都会被打印，方便排查路径问题。

---

## 13. 分析 & 实现完成

### 一、custom_envs/ 各文件解读

```
custom_envs/                              ← 通过 .pth 文件加入 sys.path
├── __init__.py                           ← 包入口，自动导入 tasks 模块
│
├── tasks/
│   ├── __init__.py                       ← 自动扫描子包，触发 gym.register()
│   │
│   └── deeprobotics_m20_pro/             ← ★ 你的第一个 M20 Pro 任务
│       ├── __init__.py                   ← 注册两个 Gym 环境 ID
│       ├── rough_env_cfg.py              ← 崎岖地形环境 (继承 M20 官方配置)
│       ├── flat_env_cfg.py               ← 平坦地形环境 (继承 M20 官方配置)
│       └── agents/
│           ├── __init__.py               ← 模块标记 (空)
│           └── rsl_rl_ppo_cfg.py         ← PPO 超参数配置
│
└── utils/
    └── __init__.py                       ← 未来工具函数 (空)
```

**数据流**:
```
pip install -e custom_envs
    │
    ▼
import custom_envs.tasks
    │  tasks/__init__.py → import_packages() → 遍历子包
    ▼
import ...deeprobotics_m20_pro
    │  执行 __init__.py → gym.register() →
    │    Flat-Deeprobotics-M20Pro-v0    (平地步态)
    │    Rough-Deeprobotics-M20Pro-v0   (崎岖地形步态)
    ▼
train.py --task=Flat-Deeprobotics-M20Pro-v0
    │  读取 env_cfg_entry_point → DeeproboticsM20ProFlatEnvCfg
    │  读取 rsl_rl_cfg_entry_point → DeeproboticsM20ProFlatPPORunnerCfg
    ▼
环境初始化 → PPO 训练
```

### 二、关键设计决策

| 层级 | 做了什么 | 为什么这样设计 |
|------|----------|----------------|
| `rough_env_cfg.py` | 继承 `DeeproboticsM20RoughEnvCfg`，`__post_init__` 调用 `super().__post_init__()` | 不重复造轮子，所有自定义都在 `super()` 之后追加 |
| `__init__.py` | 通过 `gym.register()` 注册新的环境 ID | 遵循 IsaacLab 的注册模式，`train.py` 通过 `--task` 自动发现 |
| `__init__.py` (tasks) | 通过 `gym.register()` 注册新的环境 ID | 遵循 IsaacLab 的注册模式，`train.py` 通过 `--task` 自动发现 |
| `taskdog.pth` | 将 `/home/mojie/taskdog` 加入 `sys.path` | `setup.py` 在包内部导致 `pip install -e` 失败，`.pth` 方式更简单可靠 |
| 环境命名 | `Flat-Deeprobotics-M20Pro-v0` | 遵循 `{Terrain}-{Brand}-{Robot}-v{Version}` 约定 |

### 三、修改的上游文件

| 文件 | 修改内容 |
|------|----------|
| `train.py` | +兼容性修复 (`handle_deprecated_rsl_rl_cfg`)；+`import custom_envs.tasks` |
| `play.py` | +兼容性修复 (`handle_deprecated_rsl_rl_cfg`)；+`import custom_envs.tasks`；+checkpoint 路径修复 |
| `list_envs.py` | +`import custom_envs.tasks` |

### 四、验证命令

```bash
conda activate env_isaaclab
cd /home/mojie/taskdog/deps/rl_training

# 1. 列出环境 (应看到 M20Pro)
python scripts/tools/list_envs.py

# 2. M20 Pro 平坦地形训练测试
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Flat-Deeprobotics-M20Pro-v0 \
    --headless \
    --num_envs=2048 \
    --max_iterations=200

# 3. M20 Pro 崎岖地形训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20Pro-v0 \
    --headless \
    --num_envs=2048 \
    --max_iterations=200
```


