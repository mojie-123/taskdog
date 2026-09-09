# 与Flat-Deeprobotics-M20-v0这个环境有关的文件有哪些（如各种cfg的位置以及mdp等）

## `Flat-Deeprobotics-M20-v0` 相关文件全览

### 一、继承树总览

```
ManagerBasedRLEnvCfg                          ← IsaacLab 最顶层基类
  └─ LocomotionVelocityRoughEnvCfg            ← rl_training 速度跟踪任务基类
       └─ DeeproboticsM20RoughEnvCfg          ← M20 崎岖地形配置
            └─ DeeproboticsM20FlatEnvCfg      ← M20 平坦地形配置 ✓（注册为 Flat-Deeprobotics-M20-v0）

RslRlOnPolicyRunnerCfg                        ← RSL-RL PPO 基类
  └─ DeeproboticsM20RoughPPORunnerCfg         ← M20 PPO 训练配置
       └─ DeeproboticsM20FlatPPORunnerCfg     ← M20 Flat PPO 训练配置 ✓
```

---

### 二、文件详细说明

#### 📁 Layer 1：环境注册（Gym 入口）

| 文件路径 | 作用 |
|---------|------|
| `deps/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py` | **Gym 注册文件**。调用 `gym.register(id="Flat-Deeprobotics-M20-v0", ...)` 和 `gym.register(id="Rough-Deeprobotics-M20-v0", ...)`，把环境 ID 写入全局注册表，并绑定 `env_cfg_entry_point` 和 `rsl_rl_cfg_entry_point`。 |

---

#### 📁 Layer 2：M20 环境配置（机器人特有）

| 文件路径 | 作用 |
|---------|------|
| `deps/rl_training/.../config/wheeled/deeprobotics_m20/rough_env_cfg.py` | **`DeeproboticsM20RoughEnvCfg`**。M20 崎岖地形环境的核心配置，定义了：① `DeeproboticsM20ActionsCfg`（12腿关节位置控制 + 4轮速度控制）；② `DeeproboticsM20RewardsCfg`（新增轮子专属奖励项）；③ 关节名称列表（`leg_joint_names`/`wheel_joint_names`/`hipx`/`hipy`/`knee`）；④ `__post_init__`：挂载机器人模型、配置观测/奖励/DR/命令范围/终止条件。 |
| `deps/rl_training/.../config/wheeled/deeprobotics_m20/flat_env_cfg.py` | **`DeeproboticsM20FlatEnvCfg`**（`✓ 最终被注册的类`）。继承 Rough，在 `__post_init__` 中做三件事：① 地形改为平面（`terrain_type="plane"`，`terrain_generator=None`）；② 删除高度扫描传感器；③ 关闭地形课程学习。 |

---

#### 📁 Layer 3：PPO 算法配置

| 文件路径 | 作用 |
|---------|------|
| `deps/rl_training/.../config/wheeled/deeprobotics_m20/agents/__init__.py` | 仅版权声明，无实质内容（让 Python 认识 `agents` 为子包）。 |
| `deps/rl_training/.../config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py` | **`DeeproboticsM20RoughPPORunnerCfg`** 和 **`DeeproboticsM20FlatPPORunnerCfg`**。定义网络 `[512,256,128]`、`clip_param=0.2`、`entropy_coef=0.003`、`num_steps_per_env=24`、Flat 版 `max_iterations=5000`、`experiment_name="deeprobotics_m20_flat"`。 |

---

#### 📁 Layer 4：速度跟踪任务基类（rl_training 公共层）

| 文件路径 | 作用 |
|---------|------|
| `deps/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` | **`LocomotionVelocityRoughEnvCfg`**（M20 的爷爷类）。定义了 40+ 奖励项模板、观测项模板（policy/critic 两组）、`MySceneCfg`（场景/地形/高度扫描/接触传感器）、`CommandsCfg`、`EventCfg`（Domain Randomization）、`TerminationsCfg`、`CurriculumCfg`。`__post_init__` 设置 `decimation=4`、`sim.dt=0.005s`。所有 M20/Lite3 配置都通过继承它再在 `__post_init__` 中覆盖来工作。 |
| `deps/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/` | **MDP 函数模块目录**。包含自定义的 `observations.py`、`rewards.py`、`commands.py`、`__init__.py`。`rough_env_cfg.py` 通过 `import ... mdp as mdp` 引用其中的函数（如 `mdp.joint_pos_rel_without_wheel`、`mdp.feet_slide_ang_z_cmd` 等）。 |

---

#### 📁 Layer 5：机器人资产（URDF/USD + 执行器参数）

| 文件路径 | 作用 |
|---------|------|
| `deps/rl_training/source/rl_training/rl_training/assets/deeprobotics.py` | **`DEEPROBOTICS_M20_CFG`**（`ArticulationCfg`）。定义机器人实体的一切物理参数：① USD 模型路径（`M20/M20_usd/M20.usd`）；② 初始姿态（高度 0.58 m，各关节初始角度）；③ 执行器模型：腿关节用 `DelayedPDActuatorCfg`（stiffness=80, damping=2），轮子用 `DelayedPDActuatorCfg`（stiffness=0, damping=0.6，纯速度控制）。 |
| `deps/rl_training/source/rl_training/rl_training/assets/__init__.py` | 导出 `ISAACLAB_ASSETS_DATA_DIR`（USD 文件的绝对路径根目录）。 |

---

### 三、文件调用关系一图流

```
 train.py
    │  gym.make("Flat-Deeprobotics-M20-v0")
    ▼
 deeprobotics_m20/__init__.py          ← Gym 注册表入口
    ├─→ flat_env_cfg.py                 ← DeeproboticsM20FlatEnvCfg
    │       └─ rough_env_cfg.py         ← DeeproboticsM20RoughEnvCfg
    │               ├─ velocity_env_cfg.py  ← LocomotionVelocityRoughEnvCfg (40+奖励/观测/DR模板)
    │               ├─ mdp/             ← 自定义 MDP 函数
    │               └─ assets/deeprobotics.py ← DEEPROBOTICS_M20_CFG (USD + 执行器)
    └─→ agents/rsl_rl_ppo_cfg.py        ← DeeproboticsM20FlatPPORunnerCfg
```

---

### 四、关键数值快速参考

| 参数 | 值 | 定义位置 |
|------|----|---------|
| 仿真步长 | 0.005 s（200 Hz） | `velocity_env_cfg.py` |
| 策略频率 | 50 Hz（decimation=4） | `velocity_env_cfg.py` |
| 并行环境数 | 4096 | `velocity_env_cfg.py` |
| 腿关节数 | 12（位置控制，scale=0.25） | `rough_env_cfg.py` |
| 轮关节数 | 4（速度控制，scale=20.0） | `rough_env_cfg.py` |
| 初始高度 | 0.58 m | `assets/deeprobotics.py` |
| 执行器类型 | DelayedPD（腿）/ DelayedPD纯速度（轮） | `assets/deeprobotics.py` |
| PPO 最大迭代 | 5000（Flat）/20000（Rough） | `agents/rsl_rl_ppo_cfg.py` |
| 网络结构 | `[512, 256, 128]` Actor & Critic | `agents/rsl_rl_ppo_cfg.py` |


# 在train.py中的import rl_training.tasks 这行代码是怎么找到在/home/mojie/taskdog/deps/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py处的注册信息的？

## 核心问题：一行 `import rl_training.tasks` 是如何找到深层 `__init__.py` 的？

---

### 第一步：Python 是怎么找到 `rl_training` 这个包的？

这是前置条件，`import rl_training.tasks` 首先要求 `rl_training` 已经安装到 Python 环境中。

`deps/rl_training/source/rl_training/` 下有 `setup.py`（已被 `pip install -e .` 安装），**效果**：Python 的 `sys.path` 中加入了 `deps/rl_training/source/rl_training/`，使得 `import rl_training` 能找到 `deps/rl_training/source/rl_training/rl_training/` 目录。

```
sys.path  →  .../deps/rl_training/source/rl_training/
                └── rl_training/          ← Python 找到这个包
                    ├── __init__.py
                    ├── tasks/
                    │   └── __init__.py   ← ★ 下一步进这里
                    └── assets/
```

---

### 第二步：`import rl_training.tasks` 做了什么？

Python 的 import 机制规则：`import a.b.c` 会**依次执行**：
1. `a/__init__.py`
2. `a/b/__init__.py`
3. `a/b/c/__init__.py`

所以 `import rl_training.tasks` 依次执行：

```
rl_training/__init__.py          ← 执行（内容：from . import tasks 等）
     ↓
rl_training/tasks/__init__.py    ← 执行 ★ 关键！
```

**`tasks/__init__.py` 的内容**：

```python
from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
# __name__ 此时 = "rl_training.tasks"
```

---

### 第三步：`import_packages()` 是如何递归发现 `deeprobotics_m20/__init__.py` 的？

`import_packages()` 是 IsaacLab 提供的工具函数，其内部逻辑本质上是：

```python
def import_packages(package_name, blacklist_pkgs=None):
    package = importlib.import_module(package_name)  # 导入 "rl_training.tasks"
    for sub_module in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs):
        importlib.import_module(sub_module.name)     # 递归 import 每个子包
```

它调用 `pkgutil.walk_packages()`（或类似实现）**递归遍历** `rl_training/tasks/` 目录下所有含 `__init__.py` 的子目录，并对每个子包执行 `importlib.import_module()`。

**完整遍历路径**：

```
import_packages("rl_training.tasks", ["utils"])
    │
    ├─ 发现 tasks/manager_based/           → import rl_training.tasks.manager_based
    │       └─ __init__.py 执行: import gymnasium as gym
    │
    ├─ 发现 tasks/manager_based/locomotion/ → import rl_training.tasks.manager_based.locomotion
    │       └─ __init__.py 执行: from .velocity import *   ← ★
    │
    ├─ 发现 .../locomotion/velocity/        → import ...velocity
    │       └─ __init__.py 执行: 仅 docstring，无代码
    │
    ├─ 发现 .../velocity/config/            → import ...velocity.config
    │       └─ __init__.py 执行: 仅注释，无代码（但必须存在！）
    │
    ├─ 发现 .../config/wheeled/             → import ...config.wheeled
    │       └─ __init__.py 执行: 仅注释（但必须存在！）
    │
    └─ 发现 .../wheeled/deeprobotics_m20/  → import ...wheeled.deeprobotics_m20
            └─ __init__.py 执行: ★★★
                import gymnasium as gym
                from . import agents
                gym.register(id="Flat-Deeprobotics-M20-v0", ...)
                gym.register(id="Rough-Deeprobotics-M20-v0", ...)
```

`__init__.py` 在路径上**每一层都必须存在**，哪怕内容只有注释。缺少任何一层，`import_packages()` 就无法进入更深的目录，环境注册就会失败。

---

### 第四步：`import custom_envs.tasks` 的链路是完全平行的

`custom_envs` 也已经 `pip install -e .`，同样的机制：

```
import custom_envs.tasks
    └─ custom_envs/tasks/__init__.py 执行:
           from isaaclab_tasks.utils import import_packages
           import_packages("custom_envs.tasks", ["utils"])
               └─ 遍历 custom_envs/tasks/
                   └─ 发现 deeprobotics_m20_pro/
                       └─ __init__.py 执行:
                           gym.register("Flat-Deeprobotics-M20Pro-v0", ...)
                           gym.register("Rough-Deeprobotics-M20Pro-v0", ...)
```

---

### 总结：完整的调用链（一图流）

```
train.py
│
├── import rl_training.tasks
│       │
│       └── tasks/__init__.py
│               └── import_packages("rl_training.tasks")   ← IsaacLab 提供的自动扫描函数
│                       │  递归遍历 tasks/ 下所有含 __init__.py 的目录
│                       │
│                       └──→ .../deeprobotics_m20/__init__.py 被执行
│                                   gym.register("Flat-Deeprobotics-M20-v0", ...)
│                                   gym.register("Rough-Deeprobotics-M20-v0", ...)
│                                          ↑
│                               写入 gym.envs.registry 全局注册表
│
└── import custom_envs.tasks
        │
        └── tasks/__init__.py
                └── import_packages("custom_envs.tasks")
                        └──→ .../deeprobotics_m20_pro/__init__.py 被执行
                                    gym.register("Flat-Deeprobotics-M20Pro-v0", ...)
                                    gym.register("Rough-Deeprobotics-M20Pro-v0", ...)
```

---

### 关键前提：为什么每层都需要 `__init__.py`？

| 目录 | `__init__.py` 内容 | 缺少会怎样 |
|------|-------------------|----------|
| `tasks/` | `import_packages()` | 直接失效，不会扫描子目录 |
| `manager_based/` | `import gymnasium` | `import_packages` 无法进入 |
| `locomotion/` | `from .velocity import *` | `velocity/` 不被发现 |
| `velocity/` | 仅 docstring | 仅作为占位（让 Python 认识这是包） |
| `config/` | 仅注释 | 仅占位 |
| `wheeled/` | 仅注释 | 仅占位 |
| `deeprobotics_m20/` | `gym.register(...)` | **注册入口**，这才是目的 |

**一句话总结**：`import rl_training.tasks` → 触发 `tasks/__init__.py` → `import_packages()` → 用 `pkgutil.walk_packages` 递归遍历整个 `tasks/` 目录树 → 对每个子包调用 `importlib.import_module()` → 最终执行到 `deeprobotics_m20/__init__.py` 里的 `gym.register()` → 环境 ID 写入全局注册表 → 之后 `gym.make("Flat-Deeprobotics-M20-v0")` 就能找到它。