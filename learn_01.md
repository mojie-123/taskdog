# learn_01.md — rl_training 代码架构深度学习

> **目的**: 逐一解析 rl_training 仓库中每个目录、每个 `__init__.py`、每个脚本的作用与实现原理,理解整个训练/回放/导出的代码运行机制。

---

## 目录

- [1. assets 目录详解](#1-assets-目录详解)
- [2. tasks 目录下所有 `__init__.py` 详解](#2-tasks-目录下所有-__init__py-详解)
- [3. scripts 目录每个脚本的运行机制](#3-scripts-目录每个脚本的运行机制)
- [4. Flat-Deeprobotics-M20Pro-v0 环境搭建与注册全流程](#4-flat-deeprobotics-m20pro-v0-环境搭建与注册全流程)
- [5. checkpoint 保存与加载机制](#5-checkpoint-保存与加载机制)

---

## 1. assets 目录详解

### 1.1 路径

```
deps/rl_training/source/rl_training/rl_training/assets/
├── __init__.py          ← 资产模块入口
└── deeprobotics.py      ← Deep Robotics 机器人 ArticulationCfg 定义
```

### 1.2 `__init__.py` — 资产模块入口

```python
import os
import toml

# 计算关键路径
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)
# __file__ = .../rl_training/rl_training/assets/__init__.py
# os.path.dirname = .../rl_training/rl_training/assets/
# "../../" = 往上两级 → .../rl_training/  (即 source/rl_training/)
# 所以 ISAACLAB_ASSETS_EXT_DIR = .../source/rl_training/

ISAACLAB_ASSETS_DATA_DIR = os.path.join(
    ISAACLAB_ASSETS_EXT_DIR, "../../deep_robotics_model"
)
# .../source/rl_training/ + ../../deep_robotics_model
# = .../rl_training/deep_robotics_model/  (即 rl_training 仓库根目录下的子模块)
```

**这两个路径变量是全局常量**,在整个 rl_training 包中通过 `from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR` 被引用。它们的作用:

| 变量 | 指向 | 用途 |
|------|------|------|
| `ISAACLAB_ASSETS_EXT_DIR` | `source/rl_training/` | 扩展根目录,用于加载 `config/extension.toml` 元数据 |
| `ISAACLAB_ASSETS_DATA_DIR` | `rl_training/deep_robotics_model/` (子模块) | 3D 模型文件根目录,deeprobotics.py 用此路径拼接 USD 文件路径 |

### 1.3 `deeprobotics.py` — 机器人物理资产定义

这个文件定义了两个 `ArticulationCfg` 对象:`DEEPROBOTICS_LITE3_CFG` 和 `DEEPROBOTICS_M20_CFG`。

每个 `ArticulationCfg` 包含以下关键字段:

```
DEEPROBOTICS_M20_CFG
│
├── spawn: UsdFileCfg
│   ├── usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_usd/M20.usd"
│   │   实际路径 = .../deep_robotics_model/M20/M20_usd/M20.usd
│   ├── activate_contact_sensors = True     ← 启用接触力传感器
│   ├── rigid_props                         ← 刚体属性 (阻尼/速度上限等)
│   └── articulation_props                  ← 关节求解器设置
│
├── init_state: InitialStateCfg
│   ├── pos = (0.0, 0.0, 0.58)              ← 初始世界坐标
│   ├── joint_pos = {regex: angle}           ← 初始关节角度 (站姿)
│   └── joint_vel = {".*": 0.0}             ← 初始关节速度
│
├── soft_joint_pos_limit_factor = 0.9       ← 关节限位软约束比例
│
└── actuators: dict[str, DelayedPDActuatorCfg]
    ├── "joint"  → hipx/hipy/knee 的控制参数
    │   ├── effort_limit = 76.4 N·m
    │   ├── velocity_limit = 22.4 rad/s
    │   ├── stiffness = 80.0, damping = 2.0  (PD 控制器)
    │   └── min_delay=0, max_delay=1          (通信延迟随机化)
    │
    └── "wheel"  → wheel 的控制参数
        ├── effort_limit = 21.6 N·m
        ├── velocity_limit = 79.3 rad/s
        ├── stiffness = 0.0, damping = 0.6   (轮子只有速度阻尼)
        ├── armature = 0.00243216             (电机转子惯量)
        └── min_delay=0, max_delay=1
```

**这个文件的作用**: 将 USD 模型文件与物理参数绑定,IsaacLab 加载场景时根据这个配置创建机器人的 PhysX 刚体动力学模型。

### 1.4 为什么上层 `__init__.py` 没有 `import .assets`?

```
rl_training/__init__.py
    from .tasks import *     ← 只导入了 tasks, 没有导入 assets
    from .ui_extension_example import *
```

**原因**: `assets` 不是通过 import 自动触发的模块。它提供的是**被其他模块引用**的常量 (`ISAACLAB_ASSETS_DATA_DIR`) 和配置类 (`DEEPROBOTICS_M20_CFG`)。具体调用链是:

```
train.py
    → import rl_training.tasks          ← 触发 tasks/__init__.py → import_packages()
    → 导入 config/wheeled/deeprobotics_m20/__init__.py
        → gym.register() 不直接用到 assets
    → 导入 config/wheeled/deeprobotics_m20/rough_env_cfg.py
        → from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG
            ↑ 这里才真正 import assets! (按需导入)
```

`assets` 是**被动等待引用**的模块,不需要在包初始化时主动导入。`import_packages()` 会遍历 `tasks/` 下的子包并触发 `gym.register()`,但 `assets/` 不在 `tasks/` 下,所以不会被遍历到。

---

## 2. tasks 目录下所有 `__init__.py` 详解

### 2.1 目录树与 `__init__.py` 位置

```
rl_training/tasks/                                          ← ①
├── __init__.py               ← import_packages() 自动发现
├── manager_based/                                         ← ②
│   ├── __init__.py           ← 仅 import gymnasium
│   └── locomotion/                                        ← ③
│       ├── __init__.py       ← from .velocity import *
│       └── velocity/                                      ← ④
│           ├── __init__.py   ← 仅 docstring (空操作)
│           ├── velocity_env_cfg.py
│           ├── mdp/
│           │   └── __init__.py ← 重新导出 isaaclab 的 MDP 函数 + 自己的 MDP 函数
│           └── config/                                    ← ⑤
│               ├── __init__.py     ← 空 (不暴露配置)
│               ├── quadruped/                             ← ⑥
│               │   ├── __init__.py ← 空 (标记为 Python 包)
│               │   └── deeprobotics_lite3/                ← ⑦ ★
│               │       ├── __init__.py ← gym.register() × 2
│               │       ├── flat_env_cfg.py
│               │       ├── rough_env_cfg.py
│               │       └── agents/                        ← ⑧
│               │           ├── __init__.py ← 空
│               │           └── rsl_rl_ppo_cfg.py
│               └── wheeled/                               ← ⑥
│                   ├── __init__.py ← 空
│                   └── deeprobotics_m20/                  ← ⑦ ★
│                       ├── __init__.py ← gym.register() × 2
│                       ├── flat_env_cfg.py
│                       ├── rough_env_cfg.py
│                       └── agents/                        ← ⑧
│                           ├── __init__.py ← 空
│                           └── rsl_rl_ppo_cfg.py
```

### 2.2 逐层解析

#### ① `rl_training/tasks/__init__.py` — 自动发现引擎

```python
from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
```

**这是整个注册机制的心脏。** `import_packages()` 的实现 (见 `isaaclab_tasks/utils/importer.py`):

```python
def import_packages(package_name, blacklist_pkgs=None):
    package = importlib.import_module(package_name)  # 先导入 tasks 本身
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs):
        pass  # 遍历并导入所有子包
```

`_walk_packages` 使用 `pkgutil.iter_modules` 遍历 `tasks/` 目录下的所有子目录,**递归** import:
- 发现 `manager_based/` → import
- 进入 `manager_based/`,发现 `locomotion/` → import
- 进入 `locomotion/`,发现 `velocity/` → import
- 进入 `velocity/`,发现 `config/` → import
- 进入 `config/`,发现 `quadruped/` 和 `wheeled/` → import
- 进入 `quadruped/`,发现 `deeprobotics_lite3/` → import → **触发 `gym.register()`**
- 进入 `wheeled/`,发现 `deeprobotics_m20/` → import → **触发 `gym.register()`**

`_BLACKLIST_PKGS = ["utils"]` 的作用:跳过名称中包含 `"utils"` 的子包,因为它们包含的是工具函数而非任务注册代码。

#### ② `manager_based/__init__.py` — 仅声明 gym

```python
import gymnasium as gym
```

只在模块作用域导入 gymnasium,**没有** `import_packages()`。原因:这个 `import gymnasium as gym` 确保在 `gym.register()` 被调用之前 gymnasium 已经在命名空间中可用。实际递归搜索由 ① 完成。

#### ③ `locomotion/__init__.py` — 转发 velocity

```python
from .velocity import *
```

将 `velocity/` 子包中的符号导出到 `locomotion/` 命名空间。这层转发使得可以通过 `rl_training.tasks.manager_based.locomotion.SomeClass` 访问 `velocity` 中的类。

#### ④ `velocity/__init__.py` — 仅 docstring

```python
"""Locomotion environments with velocity-tracking commands."""
```

**空操作**。它只是声明这个目录是 Python 包,实际的工作由 `config/` 子目录下的注册代码完成。因为 ① 中 `import_packages()` 只遍历到 `config/` 这一层就够了——`config/` 下的 `quadruped/` 和 `wheeled/` 会被继续遍历。

#### ⑤ `config/__init__.py` — 空包声明

```python
# We leave this file empty since we don't want to expose any configs directly.
# We still need this file to import the "config" module in the parent package.
```

**关键作用**: 如果没有这个文件,`config/` 目录就不会被 Python 识别为包,`import_packages()` 就无法进入 `config/` 下的 `quadruped/` 和 `wheeled/` 子目录。

#### ⑥ `quadruped/__init__.py` 和 `wheeled/__init__.py` — 分类标记

同样为空,仅作为 Python 包标记。作用是把四足机器人和轮式机器人分组。将来如果添加新的四足机器人 (如 Go2),只需在 `quadruped/` 下新建目录即可。

#### ⑦ `deeprobotics_lite3/__init__.py` 和 `deeprobotics_m20/__init__.py` — 环境注册

**这是真正干活的文件。** 当 `import_packages()` 遍历到这个子包时,Python 执行这个文件,其中 `gym.register()` 把环境 ID 和配置类绑定:

```python
import gymnasium as gym
from . import agents   # 确保 agents 子包被导入 (PPO 配置类在此)

gym.register(
    id="Rough-Deeprobotics-M20-v0",              # --task 参数传的这个
    entry_point="isaaclab.envs:ManagerBasedRLEnv", # 环境实现类
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        # "rl_training.tasks....deeprobotics_m20.rough_env_cfg:DeeproboticsM20RoughEnvCfg"
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        # "rl_training.tasks....deeprobotics_m20.agents.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg"
    },
)
```

执行后,Gym 全局注册表中就多了一条记录,key 是 `"Rough-Deeprobotics-M20-v0"`,value 包含如何构造环境和如何加载配置的完整信息。

#### ⑧ `agents/__init__.py` — 子模块标记

空文件,仅声明 `agents/` 是 Python 子包。当 ⑦ 执行 `from . import agents` 时,Python 能找到这个子包并导入其中的 `rsl_rl_ppo_cfg.py`。

### 2.3 总结:__init__.py 的三层结构

```
第一层 (tasks/__init__.py):
    import_packages() — 自动递归发现所有子包,相当于 "自动 import 引擎"

第二层 (config/__init__.py, quadruped/__init__.py, wheeled/__init__.py):
    空文件 — 纯 Python 包标记,"让 import_packages() 能找到我"

第三层 (deeprobotics_xxx/__init__.py):
    gym.register() — 执行实际的环境注册,将配置类绑定到环境 ID
```

---

## 3. scripts 目录每个脚本的运行机制

### 3.1 脚本清单

```
deps/rl_training/scripts/
├── reinforcement_learning/
│   ├── rl_utils.py            ← play.py 的辅助函数 (camera_follow)
│   └── rsl_rl/
│       ├── cli_args.py        ← 共享的 CLI 参数定义
│       ├── train.py           ← 训练入口
│       └── play.py            ← 回放入口
└── tools/
    ├── list_envs.py           ← 列出已注册的环境
    ├── export_onnx_fast.py    ← ONNX 导出 (无需 Isaac Sim)
    └── compare_runs.py        ← 对比训练 run
```

### 3.2 `cli_args.py` — 命令行参数定义

**作用**: 定义 RSL-RL 相关的命令行参数,以及参数到配置对象的映射函数。

**关键函数**:

#### `add_rsl_rl_args(parser)` (第 22 行)

```python
def add_rsl_rl_args(parser: argparse.ArgumentParser):
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument("--experiment_name", type=str, default=None, ...)
    arg_group.add_argument("--run_name", type=str, default=None, ...)
    arg_group.add_argument("--resume", action="store_true", default=False, ...)
    arg_group.add_argument("--load_run", type=str, default=None, ...)
    arg_group.add_argument("--checkpoint", type=str, default=None, ...)
    arg_group.add_argument("--logger", type=str, default=None, ...)
    arg_group.add_argument("--log_project_name", type=str, default=None, ...)
```

**这些参数如何被识别**:

```
$ python train.py --task=Rough-Deeprobotics-M20-v0 --headless --num_envs=4096 --load_run=xxx
                  │                                        │         │            │
                  │                                        │         │            └→ args_cli.load_run
                  │                                        │         └→ args_cli.num_envs
                  │                                        └→ args_cli.headless (AppLauncher)
                  └→ args_cli.task (train.py 自己的 --task)
```

`argparse` 解析流程:
1. `parser = argparse.ArgumentParser()` — 创建解析器
2. `parser.add_argument("--task", ...)` — 注册 train.py 自己的参数
3. `cli_args.add_rsl_rl_args(parser)` — 注册 RSL-RL 参数 (`--load_run`, `--checkpoint` 等)
4. `AppLauncher.add_app_launcher_args(parser)` — 注册 Isaac Sim 参数 (`--headless`, `--device` 等)
5. `args_cli, hydra_args = parser.parse_known_args()` — 解析!已知参数进 `args_cli`,未知的进 `hydra_args`

#### `update_rsl_rl_cfg(agent_cfg, args_cli)` (第 66 行)

```python
def update_rsl_rl_cfg(agent_cfg, args_cli):
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run       # ← 关键!
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint  # ← 关键!
    ...
```

**为什么需要这个函数**: Hydra 会从注册表中重新构建 `agent_cfg` (带默认值),这个函数把 CLI 参数覆盖到 Hydra 生成的配置对象上,实现 "CLI 参数 > Hydra 默认值" 的优先级。

### 3.3 `train.py` — 训练入口

**完整执行流程**:

```
① 解析命令行参数 (argparse + parse_known_args)
    sys.argv = [..., '--task=Rough-Deeprobotics-M20-v0', '--headless', '--num_envs=2048']
    └→ args_cli.task = "Rough-Deeprobotics-M20-v0"
    └→ args_cli.headless = True
    └→ args_cli.num_envs = 2048
    └→ hydra_args = [] (所有参数都已知)

② sys.argv = [sys.argv[0]] + hydra_args  (重置,为 Hydra 准备)

③ AppLauncher(args_cli) → 启动 Isaac Sim/Omniverse
    simulation_app = app_launcher.app

④ RSL-RL 版本检查
    要求 rsl-rl-lib >= 3.0.1

⑤ import rl_training.tasks     ← 触发 gym.register() (所有官方环境)
    import custom_envs.tasks    ← 触发 gym.register() (M20Pro 自定义环境)

⑥ @hydra_task_config 装饰器
    ├── register_task_to_hydra(task_name, "rsl_rl_cfg_entry_point")
    │   ├── load_cfg_from_registry("Rough-Deeprobotics-M20-v0", "env_cfg_entry_point")
    │   │   └→ DeeproboticsM20RoughEnvCfg 实例
    │   └── load_cfg_from_registry("Rough-Deeprobotics-M20-v0", "rsl_rl_cfg_entry_point")
    │       └→ DeeproboticsM20RoughPPORunnerCfg 实例
    │
    ├── 转为 dict → 存入 Hydra ConfigStore → hydra.main() 启动
    │
    └── hydra_main(hydra_env_cfg)
        ├── env_cfg.from_dict(hydra_env_cfg["env"])   ← 恢复环境配置
        ├── agent_cfg.from_dict(hydra_env_cfg["agent"]) ← 恢复算法配置
        └── main(env_cfg, agent_cfg) ← 调用下面的 main()

⑦ main(env_cfg, agent_cfg):
    ├── agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)  ← CLI 覆盖 Hydra 默认值
    ├── env_cfg.scene.num_envs = args_cli.num_envs (if set)
    ├── env_cfg.seed = agent_cfg.seed
    ├── log_root_path = "logs/rsl_rl/<experiment_name>"
    ├── log_dir = log_root_path / <timestamp>_<run_name>
    │
    ├── env = gym.make(args_cli.task, cfg=env_cfg)        ← 创建环境
    ├── env = RslRlVecEnvWrapper(env)                      ← RSL-RL 包装
    ├── runner = OnPolicyRunner(env, train_cfg, log_dir)   ← PPO Runner
    ├── runner.learn(max_iterations)                       ← 开始训练
    └── env.close()
```

### 3.4 `play.py` — 回放入口

**与 train.py 的区别**:

| 环节 | train.py | play.py |
|------|----------|---------|
| 环境数量 | 4096 (默认) | 50 (默认) |
| Domain Randomization | 开启 (质量/摩擦/PD 参数随机化) | 关闭 (`enable_corruption=False`, 移除推机器人事件) |
| 地形课程 | 开启 | 关闭 (`curriculum=False`) |
| 速度命令 | 自动随机生成 (含 curriculum) | 随机生成 或 键盘手动控制 (`--keyboard`) |
| 模型加载 | 不加载 (从头训练) | 加载 checkpoint (`--load_run` + `--checkpoint`) |
| 模型导出 | 不导出 | 自动导出 ONNX + JIT 到 `exported/` 目录 |

**Checkpoint 路径解析** (第 230-243 行,修正后):

```python
if args_cli.checkpoint:
    if os.path.isabs(args_cli.checkpoint) or args_cli.checkpoint.startswith(
        ("http://", "https://", "omniverse://")
    ):
        resume_path = retrieve_file_path(args_cli.checkpoint)  # 绝对路径或 URL
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, args_cli.checkpoint
        )  # 裸文件名 → 拼接到 log_root_path/load_run/filename
elif args_cli.load_run:
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
else:
    resume_path = get_checkpoint_path(log_root_path, run_dir=".*", checkpoint=agent_cfg.load_checkpoint)
```

**键盘控制模式** (`--keyboard`,第 212-225 行):

```python
if args_cli.keyboard:
    env_cfg.scene.num_envs = 1         # 强制单环境
    env_cfg.terminations.time_out = None  # 取消超时终止
    config = Se2KeyboardCfg(
        v_x_sensitivity=...,            # 按键灵敏度 = 速度范围的一半
        v_y_sensitivity=...,
        omega_z_sensitivity=...,
    )
    controller = Se2Keyboard(config)
    # 将观测中的速度命令替换为键盘输入
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.tensor(controller.advance(), ...)
    )
```

### 3.5 `list_envs.py` — 列出环境

```python
AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import rl_training.tasks     ← 触发 gym.register()
import custom_envs.tasks     ← 触发 M20Pro gym.register()

for task_spec in gym.registry.values():
    if "Deeprobotics" in task_spec.id:  ← 只显示 Deeprobotics 环境
        table.add_row([task_spec.id, ...])
print(table)
```

### 3.6 `export_onnx_fast.py` — ONNX 导出

**独特之处**: **不需要 Isaac Sim 运行**,直接从 `.pt` checkpoint 重建网络并导出 ONNX。

```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
sd = ckpt["model_state_dict"]  # 或 ckpt["actor_state_dict"]
obs_dim = sd["actor.0.weight"].shape[1]   # 从权重推断输入维度
action_dim = sd["log_std"].shape[0]       # 从 log_std 推断输出维度

actor = _build_actor(sd)  # 用 Linear + ELU 重建 MLP
dummy_obs = torch.zeros(1, obs_dim)
torch.onnx.export(actor, dummy_obs, output_path, ...)
```

**附加的元数据** (嵌入 ONNX 模型):

```python
metadata = {
    "joint_names":       "fl_hipx_joint,fl_hipy_joint,...",
    "joint_stiffness":   "80.0,80.0,...",
    "joint_damping":     "2.0,2.0,...",
    "default_joint_pos": "0.0,-0.6,1.0,...",
    "action_scale":      "0.125,0.25,...",
}
```

这些元数据在真机部署时被 `sdk_deploy` 读取,用于将 ONNX 输出的动作映射到物理关节。

---

## 4. Flat-Deeprobotics-M20Pro-v0 环境搭建与注册全流程

### 4.1 答案:可以跑通

是的,`Flat-Deeprobotics-M20Pro-v0` 已经可以跑通。下面是完整实现路径。

### 4.2 我们创建了哪些文件

```
custom_envs/
├── __init__.py                              ← ① 包入口
├── tasks/
│   ├── __init__.py                          ← ② 自动发现引擎
│   └── deeprobotics_m20_pro/
│       ├── __init__.py                      ← ③ gym.register() 注册
│       ├── flat_env_cfg.py                  ← ④ 平坦地形环境配置
│       ├── rough_env_cfg.py                 ← ⑤ 崎岖地形环境配置
│       └── agents/
│           ├── __init__.py                  ← ⑥ 包标记
│           └── rsl_rl_ppo_cfg.py            ← ⑦ PPO 算法配置
└── utils/
    └── __init__.py                          ← ⑧ 工具模块标记
```

### 4.3 每一步的详细实现

#### ① `custom_envs/__init__.py`

```python
from . import tasks  # noqa: F401
```

当 `import custom_envs.tasks` 执行时,Python 先执行 `custom_envs/__init__.py`,这行代码触发 `tasks/` 的导入。

#### ② `custom_envs/tasks/__init__.py`

```python
from isaaclab_tasks.utils import import_packages
_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
```

完全复制了 rl_training 的自动发现模式。`import_packages("custom_envs.tasks")` 会遍历 `custom_envs/tasks/` 下除 `utils/` 外的所有子包。它发现 `deeprobotics_m20_pro/` 并将其导入。

#### ③ `custom_envs/tasks/deeprobotics_m20_pro/__init__.py`

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Flat-Deeprobotics-M20Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20ProFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg",
    },
)
```

`__name__` 在运行时被 Python 解析为 `"custom_envs.tasks.deeprobotics_m20_pro"`,`agents.__name__` 解析为 `"custom_envs.tasks.deeprobotics_m20_pro.agents"`。

所以实际注册的是:
- `env_cfg_entry_point` = `"custom_envs.tasks.deeprobotics_m20_pro.flat_env_cfg:DeeproboticsM20ProFlatEnvCfg"`
- `rsl_rl_cfg_entry_point` = `"custom_envs.tasks.deeprobotics_m20_pro.agents.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg"`

#### ④ `flat_env_cfg.py`

```python
from rl_training....flat_env_cfg import DeeproboticsM20FlatEnvCfg

@configclass
class DeeproboticsM20ProFlatEnvCfg(DeeproboticsM20FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()  # ← 这行加载了 M20 的全部配置!
        self.disable_zero_weight_rewards()
```

`super().__post_init__()` 做了:加载 M20 USD 模型、配置 57 维观测、16 维动作、15+ 奖励项、Domain Randomization、地形改为平面……**全部复用 M20 的配置**。

#### ⑤ `rough_env_cfg.py`

同逻辑,继承自 `DeeproboticsM20RoughEnvCfg`。

#### ⑥ `agents/__init__.py`

空文件,仅作为 Python 子包标记。

#### ⑦ `rsl_rl_ppo_cfg.py`

```python
@configclass
class DeeproboticsM20ProFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    experiment_name = "deeprobotics_m20pro_flat"  # ← 决定了日志目录名
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        ...
    )
```

### 4.4 如何使 custom_envs 可被导入

```bash
# 在 conda env 的 site-packages 中创建 .pth 文件
echo "/home/mojie/taskdog" > ~/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/taskdog.pth
```

Python 启动时扫描所有 `.pth` 文件,将其中路径追加到 `sys.path`。这样 `import custom_envs` 就能在 `/home/mojie/taskdog/custom_envs/` 找到这个包。

### 4.5 完整的调用链 (从命令到训练)

```
$ python scripts/.../train.py --task=Flat-Deeprobotics-M20Pro-v0 --headless --num_envs=2048

 ① argparse 解析:
    args_cli.task = "Flat-Deeprobotics-M20Pro-v0"
    args_cli.headless = True
    args_cli.num_envs = 2048

 ② AppLauncher → Isaac Sim 启动

 ③ import rl_training.tasks
    └→ import_packages → gym.register(Flat-Deeprobotics-M20-v0, ...)
    └→ gym.register(Rough-Deeprobotics-M20-v0, ...)

 ④ import custom_envs.tasks
    └→ import_packages → 发现 deeprobotics_m20_pro/
    └→ import ...deeprobotics_m20_pro/
        └→ __init__.py 执行:
            gym.register("Flat-Deeprobotics-M20Pro-v0", ...)
            gym.register("Rough-Deeprobotics-M20Pro-v0", ...)

 ⑤ @hydra_task_config("Flat-Deeprobotics-M20Pro-v0", "rsl_rl_cfg_entry_point")
    └→ load_cfg_from_registry("Flat-Deeprobotics-M20Pro-v0", "env_cfg_entry_point")
        │  查 gym 注册表 → kwargs["env_cfg_entry_point"]
        │  = "custom_envs.tasks...flat_env_cfg:DeeproboticsM20ProFlatEnvCfg"
        │  → import + 实例化 → DeeproboticsM20ProFlatEnvCfg()
        │      └→ super().__post_init__() → M20 全部配置加载
        │      └→ disable_zero_weight_rewards()
        │
    └→ load_cfg_from_registry("Flat-Deeprobotics-M20Pro-v0", "rsl_rl_cfg_entry_point")
        │  = "custom_envs.tasks...agents.rsl_rl_ppo_cfg:DeeproboticsM20ProFlatPPORunnerCfg"
        │  → import + 实例化

 ⑥ gym.make("Flat-Deeprobotics-M20Pro-v0", cfg=env_cfg)
    └→ 创建 2048 个并行环境,加载 M20 USD 模型

 ⑦ RslRlVecEnvWrapper + OnPolicyRunner → PPO 训练循环
```

---

## 5. checkpoint 保存与加载机制

### 5.1 checkpoint 保存规则

**保存位置**: `logs/rsl_rl/<experiment_name>/<timestamp>/model_<N>.pt`

**保存频率**: 由 PPO 配置中的 `save_interval` 决定。在 M20 Flat 的配置中:

```python
class DeeproboticsM20FlatPPORunnerCfg:
    save_interval = 100   # ← 每 100 个 iteration 保存一次
```

**保存逻辑** (RSL-RL 的 `OnPolicyRunner.learn()`,位于 `rsl_rl/runners/on_policy_runner.py` 第 159 行):

```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    for it in range(...):        # it = 0, 1, 2, ..., 5000
        # ... PPO 训练一步 ...

        if it % self.save_interval == 0:   # ← 每 save_interval 步
            self.save(os.path.join(
                self.log_dir, f"model_{it}.pt"
            ))
            # 例: logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_100.pt
            #     logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_200.pt
            #     ...
```

**保存的内容** (`OnPolicyRunner.save()` 第 289 行):

```python
def save(self, path: str):
    saved_dict = {
        "model_state_dict": self.alg.policy.state_dict(),  # Actor + Critic 权重
        "optimizer_state_dict": self.alg.optimizer.state_dict(),  # 优化器状态
        # ...
    }
    torch.save(saved_dict, path)
```

**文件名规则**:
- `model_0.pt` — 初始权重 (训练前)
- `model_100.pt` — 第 100 个 iteration
- `model_200.pt` — 第 200 个 iteration
- ...
- `model_5000.pt` — 最终模型

**目录名规则**: `log_root_path / <timestamp>[_<run_name>]`
- `log_root_path` = `logs/rsl_rl/<experiment_name>`
- `experiment_name` 在 PPO 配置中定义 (如 `"deeprobotics_m20_flat"`)
- `<timestamp>` 自动生成 (`datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`)
- 可选 `<run_name>` 通过 `--run_name` CLI 参数追加

### 5.2 play.py 中 checkpoint 的识别与加载

**CLI 参数传递**:

```
--load_run=2026-07-18_10-57-32 --checkpoint=model_1900.pt
    │                                     │
    └→ args_cli.load_run                  └→ args_cli.checkpoint
```

在 `cli_args.update_rsl_rl_cfg()` 中:
```python
agent_cfg.load_run = "2026-07-18_10-57-32"
agent_cfg.load_checkpoint = "model_1900.pt"
```

**路径拼接** (`get_checkpoint_path()` 函数):

```python
get_checkpoint_path(
    log_path="logs/rsl_rl/deeprobotics_m20_flat",  # 来自 agent_cfg.experiment_name
    run_dir="2026-07-18_10-57-32",                  # 来自 --load_run
    checkpoint="model_1900.pt"                      # 来自 --checkpoint
)
# 内部逻辑:
# ① 在 log_path 下找匹配 run_dir 的目录 → logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32
# ② 在目录下找匹配 checkpoint 的文件 → model_1900.pt
# ③ 返回完整路径: logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_1900.pt
```

**模型加载** (play.py 第 263-264 行):

```python
ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
ppo_runner.load(resume_path)  # ← 加载 .pt 文件
```

`OnPolicyRunner.load()` 内部 (rsl_rl 第 310 行):

```python
def load(self, path: str):
    loaded_dict = torch.load(path)
    self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
```

### 5.3 完整时间线

```
训练                                  回放
────                                  ────
$ python train.py                     $ python play.py
    --task=Flat-...M20-v0                --task=Flat-...M20-v0
    --num_envs=4096                      --num_envs=10
    (无 --load_run)                      --load_run=2026-07-18_10-57-32
                                         --checkpoint=model_1900.pt

  agent_cfg.experiment_name               agent_cfg.experiment_name
    = "deeprobotics_m20_flat"               = "deeprobotics_m20_flat"  (从注册的 PPO 配置读取)

  log_dir = logs/rsl_rl/                  agent_cfg.load_run
    deeprobotics_m20_flat/                  = "2026-07-18_10-57-32"  (从 --load_run)
    2026-07-18_10-57-32/  ← 新建
                                         agent_cfg.load_checkpoint
  save_interval = 100                      = "model_1900.pt"  (从 --checkpoint)

  iteration 0:                            解析路径:
    save → model_0.pt                      logs/rsl_rl/deeprobotics_m20_flat/
  iteration 100:                             2026-07-18_10-57-32/
    save → model_100.pt                       model_1900.pt  ← 加载此文件
  ...
  iteration 1900:
    save → model_1900.pt  ← 你回放时用的就是这个
  iteration 2000:
    save → model_2000.pt
```

### 5.4 `get_checkpoint_path()` 的灵活匹配

`get_checkpoint_path()` 支持**正则表达式**匹配。默认值 `load_checkpoint = "model_.*.pt"` 会匹配任意 `model_XXX.pt` 文件,按**字母序取最后一个**(即数字最大的):

```python
# 如果目录下有 model_100.pt, model_200.pt, ..., model_1900.pt
# 不传 --checkpoint 时,agent_cfg.load_checkpoint = "model_.*.pt" (默认)
# get_checkpoint_path 匹配所有 model_*.pt,选 model_1900.pt (最大数字)
```

如果不传 `--load_run`,默认 `load_run = ".*"` 会匹配所有 run 目录,选字母序最后一个(即最新的时间戳):

```python
# logs/rsl_rl/deeprobotics_m20_flat/
#   2026-07-18_09-00-00/  ← 较旧
#   2026-07-18_10-57-32/  ← 较新 → 被选中
```
