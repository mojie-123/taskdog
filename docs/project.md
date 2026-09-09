# M20 Pro IsaacLab 项目 — 技术详解

> **创建日期**: 2026-07-18  
> **配套文档**: work.md（调研报告与工作方案）  
> **本文档定位**: 项目结构详解、代码使用方式、模型架构分析、URDF 导入机制  

---

## 目录

1. [环境检查与依赖安装](#1-环境检查与依赖安装)
2. [完整项目结构详解](#2-完整项目结构详解)
3. [URDF 导入机制详解](#3-urdf-导入机制详解)
4. [rl_training 模型架构分析](#4-rl_training-模型架构分析)
5. [脚本调用方式与关键参数](#5-脚本调用方式与关键参数)
6. [配置文件关键参数速查](#6-配置文件关键参数速查)

---

## 1. 环境检查与依赖安装

### 1.1 现有环境 `env_isaaclab` 检查结果

```
✅ 结论: 环境完全兼容 M20 Pro 开发，可直接使用
```

| 关键依赖 | 要求版本 | 实际版本 | 状态 |
|----------|----------|----------|------|
| Python | 3.11 | 3.11.15 | ✅ |
| Isaac Lab | 2.3.2 | 2.3.2.post1 | ✅ |
| Isaac Sim | 5.1.0 | 5.1.0.0 | ✅ |
| PyTorch | ≥2.0 | 2.7.1+cu128 | ✅ |
| RSL-RL | ≥3.0.1 | 3.0.1 | ✅ (train.py 第71行确认) |
| ONNX | ≥1.14 | 1.20.1 | ✅ |
| NumPy | ≥1.22 | 1.26.0 | ✅ |

**注意**: README 中写 RSL-RL 5.0.1，但实际 `train.py` 第 71 行硬编码的最低版本要求是 `3.0.1`，环境中的 3.0.1 完全满足。

### 1.2 安装 M20 Pro 训练环境

只需两步：

```bash
# 1. 激活环境
source /home/mojie/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# 2. 安装 rl_training (开发模式)
cd /home/mojie/taskdog/deps/rl_training
pip install -e source/rl_training

# 3. 验证安装
python scripts/tools/list_envs.py
# 期望输出中包含:
#   Flat-Deeprobotics-M20-v0
#   Rough-Deeprobotics-M20-v0
#   Flat-Deeprobotics-Lite3-v0
#   Rough-Deeprobotics-Lite3-v0
```

### 1.3 关键路径配置

`rl_training` 通过 `source/rl_training/rl_training/assets/__init__.py` 第 25 行自动解析模型路径：

```python
ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "../../deep_robotics_model")
```

这意味着模型文件相对于 `source/rl_training/` 位于 `../../deep_robotics_model/`，即仓库根目录下的 `deep_robotics_model/` 子模块。**由于我们 git clone 时使用了 `--recurse-submodules`，这个路径已经正确**。

### 1.4 快速跑通验证

```bash
conda activate env_isaaclab
cd /home/mojie/taskdog/deps/rl_training

# 列出可用环境
python scripts/tools/list_envs.py

# 短期训练测试 (100 步，验证不报错)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless \
    --num_envs=64 \
    --max_iterations=100
```

---

## 2. 完整项目结构详解

### 2.1 顶层目录

```
/home/mojie/taskdog/
├── work.md                    # 调研报告与工作方案 (阅读入口)
├── project.md                 # 本文档 (技术详解)
├── README.md                  # 项目概述
├── .gitignore                 # Git 忽略规则
│
├── deps/                      # 上游依赖仓库 (独立 git clone，非 submodule)
├── custom_envs/               # 自定义环境扩展代码
├── scripts/                   # 实用脚本
├── configs/                   # 实验超参数配置
├── logs/                      # 训练日志与模型检查点
├── exported/                  # 导出的 ONNX 模型
├── notebooks/                 # Jupyter 分析笔记本
├── docker/                    # Docker 部署配置
└── docs/                      # 补充文档
```

### 2.2 `deps/` — 依赖仓库详解

#### 2.2.1 `deps/deep_robotics_model/` — 3D 模型资产库

**作用**: 提供 M20 机器人的物理仿真模型（几何/质量/惯量/关节/碰撞体定义）。

**许可证**: BSD-3-Clause

**M20 目录下的关键文件**:

| 文件 | 作用 |
|------|------|
| `M20/urdf/M20.urdf` | URDF 格式模型定义。所有 link/joint 的原始数据源。**碰撞体用简化的几何体（盒子/圆柱）**。 |
| `M20/urdf/meshes/*.STL` | 17 个 STL 网格文件（base_link + 4腿×4连杆）。URDF 中的 `<visual>` 标签引用这些文件用于渲染。 |
| `M20/mjcf/M20.xml` | MuJoCo 格式模型。与 URDF 内容等价，但用 MuJoCo XML 语法。用于 MuJoCo 仿真器或 MJX。 |
| `M20/mjcf/meshes/*.STL` | MJCF 格式使用的 STL 网格（与 URDF 共用同一套）。 |
| `M20/usd/M20.usd` | **IsaacLab 实际加载的文件**。USD (Universal Scene Description) 格式，是 Omniverse/Isaac Sim 的原生格式。 |
| `M20/usd/configuration/M20_base.usd` | 基础视觉几何定义 |
| `M20/usd/configuration/M20_physics.usd` | 物理属性定义（质量、惯量、碰撞体） |
| `M20/usd/configuration/M20_robot.usd` | 机器人关节层级定义 |
| `M20/usd/configuration/M20_sensor.usd` | 传感器配置（接触传感器等） |

**数据流**: `M20.urdf`（原始定义）→ Isaac Sim URDF Importer 转换 → `M20.usd`（IsaacLab 加载）

**其他机器人**: 该仓库还包含 `Lite3/`, `M20S/`, `M20_Piper/`, `X30/`, `DR02/`（pro + standard），共 7 款机器人。

#### 2.2.2 `deps/rl_training/` — RL 训练框架

**作用**: 基于 IsaacLab 的强化学习训练代码库，提供 M20 和 Lite3 的 locomotion 任务。

**许可证**: BSD-3-Clause（主代码）+ Apache-2.0（robot_lab 衍生代码）

**核心代码文件详解**:

##### 训练入口与脚本

| 文件 | 作用 |
|------|------|
| `scripts/reinforcement_learning/rsl_rl/train.py` | **训练主入口**。解析命令行参数 → 启动 Isaac Sim → 创建 Gym 环境 → 包装 RSL-RL → 启动 PPO 训练循环。 |
| `scripts/reinforcement_learning/rsl_rl/play.py` | **策略回放入口**。加载训练好的 checkpoint → 在仿真器中运行策略 → 支持键盘控制和视频录制。 |
| `scripts/reinforcement_learning/rsl_rl/cli_args.py` | CLI 参数定义（`--task`, `--num_envs`, `--headless` 等）。 |
| `scripts/tools/list_envs.py` | 列出所有已注册的 Gym 环境 ID。用于验证安装是否成功。 |
| `scripts/tools/export_onnx_fast.py` | **ONNX 导出**。从 .pt checkpoint 重建 actor 网络 → 导出 ONNX → 附加机器人元数据。**不需要 Isaac Sim 运行**。 |
| `scripts/tools/compare_runs.py` | 对比多个训练 run 的 TensorBoard 指标。 |

##### 资产定义

| 文件 | 作用 |
|------|------|
| `source/rl_training/rl_training/assets/__init__.py` | 定义 `ISAACLAB_ASSETS_DATA_DIR` 路径（指向 `deep_robotics_model/` 子模块）。加载 `extension.toml` 元数据。 |
| `source/rl_training/rl_training/assets/deeprobotics.py` | **机器人 ArticulationCfg 定义**。定义 `DEEPROBOTICS_M20_CFG` 和 `DEEPROBOTICS_LITE3_CFG`，包括：USD 加载路径、初始关节位置、关节限位软约束、**两套执行器配置**（腿关节 vs 轮关节）。 |
| `source/rl_training/rl_training/assets/utils/usd_converter.py` | USD 模型转换工具。 |

##### 环境配置（M20 专属）

| 文件 | 作用 |
|------|------|
| `.../config/wheeled/deeprobotics_m20/__init__.py` | **Gym 环境注册**。注册 `Flat-Deeprobotics-M20-v0` 和 `Rough-Deeprobotics-M20-v0` 两个环境 ID，绑定对应的环境配置和算法配置。 |
| `.../config/wheeled/deeprobotics_m20/rough_env_cfg.py` | **M20 崎岖地形环境配置**。继承自 `LocomotionVelocityRoughEnvCfg`，覆盖：观测定义（去掉 base_lin_vel 和 height_scan）、动作定义（位置控制 12 腿关节 + 速度控制 4 轮关节）、奖励权重（速度跟踪、关节惩罚、足端滑移惩罚、旋转步态奖励等）、domain randomization 范围、终止条件。 |
| `.../config/wheeled/deeprobotics_m20/flat_env_cfg.py` | **M20 平坦地形环境配置**。继承 rough 配置，覆盖：地形改为平面、移除高度扫描传感器、移除地形课程学习。 |
| `.../config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py` | **PPO 算法配置**。定义网络架构 `[512, 256, 128]`、训练超参数（learning rate=1e-3, clip=0.2, entropy=0.003）、训练步数等。 |
| `.../config/wheeled/deeprobotics_m20/agents/__init__.py` | 代理配置模块标记。 |

##### 基础环境框架（M20 配置的父类）

| 文件 | 作用 |
|------|------|
| `.../velocity/velocity_env_cfg.py` | **基类环境配置**（`LocomotionVelocityRoughEnvCfg`）。定义了所有的观测项模板（policy 和 critic 两组）、奖励项模板（40+ 种）、终止条件、事件（domain randomization）、命令生成、课程学习、场景设置（地形/高度扫描/接触传感器）。每个 M20/Lite3 的具体环境通过 `__post_init__` 覆盖具体参数。 |
| `.../velocity/mdp/__init__.py` | MDP 模块入口。重新导出 `isaaclab.envs.mdp.*` 和 `isaaclab_tasks.manager_based.locomotion.velocity.mdp.*`，并覆盖自定义模块。 |
| `.../velocity/mdp/commands.py` | 速度命令生成逻辑（`UniformThresholdVelocityCommandCfg`）。 |
| `.../velocity/mdp/curriculums.py` | 课程学习函数（地形难度渐进、步态级别、命令范围渐进）。 |
| `.../velocity/mdp/events.py` | Domain randomization 事件（质量随机化、摩擦随机化、推机器人等）。 |
| `.../velocity/mdp/observations.py` | 自定义观测函数。关键：`joint_pos_rel_without_wheel` — 计算关节相对位置时排除轮关节的偏移。 |
| `.../velocity/mdp/rewards.py` | 自定义奖励函数。包括 M20 专用的 `feet_air_time_ang_z_cmd_M20`（旋转步态滞空奖励）、`joint_pos_penalty_except_turn_side_cmd`（转弯时豁免关节惩罚）等。 |

##### 包安装入口

| 文件 | 作用 |
|------|------|
| `source/rl_training/setup.py` | `pip install -e` 的入口。声明依赖 `isaaclab`, `cusrl[all]`, `pinocchio`, `xacrodoc` 等。 |
| `source/rl_training/config/extension.toml` | Isaac Lab 扩展元数据（版本号 1.0.0、作者 Bo Peng、依赖声明）。 |

#### 2.2.3 `deps/sdk_deploy/` — 真机部署 SDK

**作用**: 将训练好的 ONNX 策略部署到真实 M20 Pro 机器人上。

**许可证**: BSD-3-Clause

**语言**: C++ 97.9% + Python 1.2%

**M20 部署目录** (`src/M20_sdk_deploy/`):

| 目录/文件 | 作用 |
|-----------|------|
| `include/` | C++ 头文件（ONNX Runtime 推理、关节控制接口） |
| `interface/` | 通信接口定义（与机器人主控的 ROS2 通信） |
| `M20_description/` | 机器人 URDF 描述（用于部署端的运动学计算） |
| `policy/` | ONNX 策略加载与推理引擎 |
| `run_policy/` | 策略运行主程序 |
| `scripts/` | 部署启动脚本 |
| `state_machine/` | 有限状态机（待机/行走/停止切换） |
| `third_party/` | 第三方依赖 |

**⚠️ 免责声明**: README 中明确声明 "Damage caused by using SDK is not covered under warranty!" — 使用 SDK 造成的机器人损坏不在保修范围内。

### 2.3 `custom_envs/` — 自定义环境扩展

**作用**: 在不修改上游 `rl_training` 代码的前提下，创建 M20 Pro 的定制化环境配置。

```
custom_envs/
├── __init__.py                              # 包标记
├── tasks/
│   ├── __init__.py                          # 包标记
│   └── deeprobotics_m20_pro/
│       ├── __init__.py                      # Gym 环境注册 (待编写)
│       ├── flat_env_cfg.py                  # 平坦地形配置 (待编写)
│       ├── rough_env_cfg.py                 # 崎岖地形配置 (待编写)
│       ├── stair_env_cfg.py                 # 楼梯环境 (待编写)
│       └── agents/
│           ├── __init__.py                  # 算法配置标记 (待编写)
│           └── rsl_rl_ppo_cfg.py            # PPO 算法配置 (待编写)
└── utils/
    ├── __init__.py                          # 工具模块标记
    ├── terrain.py                           # 自定义地形生成 (待编写)
    └── sensors.py                           # 自定义传感器配置 (待编写)
```

**设计模式**: 继承 rl_training 的 `LocomotionVelocityRoughEnvCfg`，在 `__post_init__` 中覆盖需要定制的参数。通过 `pip install -e .` 注册到 IsaacLab 环境中。

### 2.4 其他目录（待开发）

| 目录 | 用途 |
|------|------|
| `scripts/` | 训练/回放/评估 shell 封装脚本、超参数搜索脚本 |
| `configs/` | YAML 格式的实验配置（不同地形难度、不同 reward 权重组合） |
| `logs/` | 训练日志（TensorBoard event files、模型 checkpoint .pt 文件） |
| `exported/` | 导出的 ONNX 策略文件（用于真机部署） |
| `notebooks/` | Jupyter Notebook（模型结构可视化、训练曲线分析、步态分析） |
| `docker/` | Docker 镜像（Isaac Sim 容器化部署） |
| `docs/` | 补充文档（环境安装指南、训练调优指南、真机部署指南） |

---

## 3. URDF 导入机制详解

### 3.1 URDF ↔ USD 转换链

URDF 文件不能直接被 IsaacLab 加载。IsaacLab/Isaac Sim 的原生格式是 **USD (Universal Scene Description)**。转换链路如下：

```
SolidWorks CAD 模型
    │
    ▼
M20.urdf  (手工/自动导出)
    │  ├── <link>    → 连杆定义 (质量、惯量、碰撞体、视觉几何)
    │  ├── <joint>   → 关节定义 (类型、限位、驱动参数)
    │  └── <visual>  → 引用 meshes/*.STL 做可视化渲染
    │
    │  Isaac Sim URDF Importer 转换
    ▼
M20.usd  (IsaacLab 实际加载)
    ├── M20.usd              → 主文件 (组装子模块)
    ├── M20_base.usd         → 视觉几何 (高精度 mesh)
    ├── M20_physics.usd      → 物理属性 (碰撞体/质量/惯量)
    ├── M20_robot.usd        → 关节层级 (articulation hierarchy)
    └── M20_sensor.usd       → 传感器配置 (接触力等)
```

### 3.2 URDF 在训练流程中的角色

**URDF 是「设计蓝图」，不是运行时加载的文件**。

```
训练时:
  deeprobotics.py (ArticulationCfg)
    │  usd_path = ".../M20/M20_usd/M20.usd"  ← 加载的是 USD，不是 URDF
    ▼
  Isaac Sim 解析 USD
    │  ├── 读取 articulation 层级 (joint chain)
    │  ├── 读取 rigid body 属性 (mass/inertia)
    │  ├── 读取碰撞体 (collision shapes)
    │  └── 读取执行器参数 (effort/velocity limits)
    ▼
  PhysX 物理引擎构建刚体动力学模型
    │  ├── 关节驱动力矩 = stiffness*(q_des - q) + damping*(qd_des - qd)
    │  └── 接触力计算 (轮-地、腿-地碰撞)
    ▼
  RL 策略输出关节目标 → PD 控制器 → 关节力矩 → 仿真步进

ONNX 导出时:
  export_onnx_fast.py
    │  └── 将 URDF 中的关节名称、刚度/阻尼、默认位置等硬编码为 ROBOT_CONFIGS
    │     (M20_CFG 第 95-103 行)
    ▼
  ONNX 模型 metadata: joint_names, stiffness, damping, default_joint_pos, action_scale
    │
    ▼
  真机部署时 SDK 读取 metadata 做关节映射
```

### 3.3 URDF 关键内容解读

打开 `deps/deep_robotics_model/M20/urdf/M20.urdf`：

```xml
<!-- 每条腿的关节链: base_link → hipx → hipy → knee → wheel -->
<!-- fl = front-left, fr = front-right, hl = hind-left, hr = hind-right -->

<!-- 髋关节横摆 (绕X轴, 外展/内收) -->
<joint name="fl_hipx_joint" type="revolute">
    <parent link="base_link"/>
    <child link="fl_hipx"/>
    <origin xyz="0.3141 0.0685 0.0965" rpy="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit effort="76.4" velocity="22.4" lower="-0.436" upper="0.611"/>
</joint>

<!-- 髋关节俯仰 (绕Y轴, 前摆/后摆) -->
<joint name="fl_hipy_joint" type="revolute">
    <parent link="fl_hipx"/>
    <child link="fl_hipy"/>
    <axis xyz="0 -1 0"/>
    <limit effort="76.4" velocity="22.4" lower="-2.583" upper="2.286"/>
</joint>

<!-- 膝关节俯仰 (绕Y轴) -->
<joint name="fl_knee_joint" type="revolute">
    <parent link="fl_hipy"/>
    <child link="fl_knee"/>
    <axis xyz="0 -1 0"/>
    <limit effort="76.4" velocity="22.4" lower="-2.792" upper="2.809"/>
</joint>

<!-- 轮关节 (绕Y轴, 连续旋转, 无角度限位) -->
<joint name="fl_wheel_joint" type="continuous">
    <parent link="fl_knee"/>
    <child link="fl_wheel"/>
    <axis xyz="0 -1 0"/>
    <limit effort="21.6" velocity="79.3"/>
</joint>
```

### 3.4 训练代码中与 URDF 对应的部分

`deeprobotics.py` 中的 `DEEPROBOTICS_M20_CFG` 与 URDF 的对应关系：

| URDF 定义 | deeprobotics.py 配置 | 作用 |
|-----------|---------------------|------|
| `<limit effort="76.4">` | `actuators["joint"].effort_limit=76.4` | 腿关节力矩上限 |
| `<limit velocity="22.4">` | `actuators["joint"].velocity_limit=22.4` | 腿关节速度上限 |
| `<limit effort="21.6">` | `actuators["wheel"].effort_limit=21.6` | 轮关节力矩上限 |
| `<limit velocity="79.3">` | `actuators["wheel"].velocity_limit=79.3` | 轮关节速度上限 |
| 默认关节位置（初始站姿） | `init_state.joint_pos` | 仿真初始化姿态 |
| `<limit lower/upper>` | `soft_joint_pos_limit_factor=0.9` | 到达限位的 90% 时开始施加惩罚 |
| 关节刚度/阻尼（PD 控制器） | `stiffness=80.0, damping=2.0` | 将动作指令转换为关节力矩 |
| 轮关节的 armature | `armature=0.00243216` | 电机转子惯量等效到关节空间 |

---

## 4. rl_training 模型架构分析

### 4.1 训练了什么？

**一句话**: 训练一个**速度跟踪 locomotion 策略**。给定目标线速度和角速度命令，策略输出 16 个关节的动作指令，使机器人以命令速度稳定移动。

### 4.2 模型输入 (Observation Space)

**Policy 网络输入**: 57 维

| 观测分量 | 维度 | 说明 |
|----------|------|------|
| base_ang_vel | 3 | 基座角速度 (roll, pitch, yaw rates) |
| projected_gravity | 3 | 重力方向在基座坐标系中的投影 (反映姿态) |
| velocity_commands | 3 | 目标速度命令 (vx, vy, ωz) |
| joint_pos | 16 | 关节相对位置 (当前值 - 默认值)，轮关节偏移置零 |
| joint_vel | 16 | 关节速度 |
| actions | 16 | 上一步的动作输出 (提供时序信息) |

**Critic 网络输入**: 相同结构（57 维），但不添加噪声。Critic 只在训练时使用，用于估计 value function。

**注意**: M20 rough 环境移除了 `base_lin_vel`（基座线速度）和 `height_scan`（高度扫描）观测。前者是因为线速度估计不准反而有害，后者是盲走策略（blind locomotion）不需要地形感知。

### 4.3 模型输出 (Action Space)

**16 维**，分为两组：

| 动作分量 | 维度 | 动作类型 | 目标范围 | 说明 |
|----------|------|----------|----------|------|
| 腿关节位置 (hipx/hipy/knee ×4腿) | 12 | **位置增量** | clip=±100 (但 scale 限制了实际范围) | hipx 动作 scale=0.125，其余 scale=0.25 |
| 轮关节速度 (wheel ×4腿) | 4 | **速度目标** | clip=±100 (scale=5.0) | 直接输出目标转速 |

**动作如何转换为关节力矩**:

```
网络输出 (16维)
    │
    ├── 前12维 (腿关节) → 乘以 scale(0.125 or 0.25) → 加到默认关节位置 → target_pos
    │       └── PD控制器: τ = stiffness*(target_pos - current_pos) + damping*(0 - current_vel)
    │
    └── 后4维 (轮关节) → 乘以 scale(5.0) → target_vel
            └── PD控制器: τ = stiffness*(0) + damping*(target_vel - current_vel)
                (轮关节 stiffness=0，所以只有阻尼项)
```

### 4.4 网络架构 (PPO Actor-Critic)

```
Actor 网络 (策略网络):
  Input(57) → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → ELU → Linear(16)
                                                                              │
                                                                     action_mean (16)
                                                                     action_std  (16) ← 可学习参数

Critic 网络 (价值网络):
  Input(57) → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → ELU → Linear(1)
                                                                              │
                                                                         value (1)

PPO 超参数:
  learning_rate:   1e-3 (adaptive scheduler)
  clip_param:      0.2
  entropy_coef:    0.003
  gamma:           0.99
  lam (GAE):       0.95
  num_steps/env:   24
  num_epochs:      5
  num_minibatches: 4
  max_iterations:  20000 (rough) / 5000 (flat)
```

### 4.5 奖励函数设计

M20 的奖励函数包含以下核心项（非零权重项）：

**速度跟踪** (核心):
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| track_lin_vel_xy_exp | 5.0 | 跟踪 xy 线速度命令，指数奖励 |
| track_ang_vel_z_exp | 3.0 | 跟踪 z 轴角速度命令，指数奖励 |

**姿态稳定**:
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| flat_orientation_l2 | -50.0 | 惩罚基座倾斜 |
| lin_vel_z_l2 | -2.0 | 惩罚垂直线速度 |

**关节限制**:
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| joint_torques_l2 | -2.5e-5 | 惩罚腿关节力矩 |
| joint_acc_l2 | -2e-7 | 惩罚腿关节加速度（平滑性） |
| wheel acc l2 | -1e-7 | 惩罚轮关节加速度 |

**步态与接触**:
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| feet_slide_ang_z_cmd | -2.0 | 惩罚转弯时足端滑移 |
| undesired_contacts | -1.0 | 惩罚轮以外部位接触地面 |
| contact_forces | -1.5e-4 | 惩罚过大接触力 |
| feet_air_time_ang_z_M20 | 50.0 | 奖励转弯时对角轮滞空（rotational gait） |
| rotation_gait_status | 1.5 | 奖励旋转步态状态 |
| rotation_gait_symmetry | 10.0 | 奖励旋转步态对称性 |

**动作平滑性**:
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| action_rate_l2 | -0.01 | 惩罚动作变化率 |
| action_smooth_l2 | -0.01 | 惩罚动作二阶平滑性 |

**对称性**:
| 奖励项 | 权重 | 说明 |
|--------|------|------|
| joint_mirror | -0.03 | 惩罚左右关节不对称 |
| stand_still | -2.0 | 无速度命令时保持站姿 |
| bad_orientation_penalty | -1000 | 严重倾斜时强力惩罚 |

### 4.6 支持哪些机器人？

**rl_training 只支持 2 款机器人**:

| 机器人 | 环境 ID | 类型 | 关节数 |
|--------|---------|------|--------|
| Deeprobotics Lite3 | `Flat/Rough-Deeprobotics-Lite3-v0` | 四足（足式） | 12 DOF (3/腿) |
| Deeprobotics M20 | `Flat/Rough-Deeprobotics-M20-v0` | 四足（轮足混合） | 16 DOF (4/腿) |

**如果需要支持 M20 Pro 的更多特性**（如 SLAM 感知），可以：
1. 直接使用 M20 的环境（机械结构完全相同）
2. 通过 `custom_envs/` 扩展添加摄像头/LiDAR 观测
3. 参考 `robot_lab` 仓库获取更多任务类型（模仿学习、AMP 舞蹈等）

### 4.7 训练数据流全貌

```
┌─────────────────────────────────────────────────────────────────────┐
│                         训练循环 (train.py)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① 环境重置 (4096 并行环境)                                          │
│     └── Domain Randomization (质量±15%, 摩擦0.3-1.0, 质心±3cm ...)   │
│                                                                      │
│  ② 观测采集 (57维 × 4096 环境)                                       │
│     └── 关节位置/速度 + IMU + 速度命令 + 上一步动作                     │
│                                                                      │
│  ③ 策略推理 (Actor 网络)                                              │
│     └── 输入 57维 → 输出 16维 (12 腿位置 + 4 轮速度)                   │
│                                                                      │
│  ④ PD 控制器转换为力矩                                                │
│     └── τ = stiffness*(q_des - q) + damping*(qd_des - qd)            │
│                                                                      │
│  ⑤ PhysX 物理仿真 (4 步 @ 0.005s = 0.02s)                            │
│                                                                      │
│  ⑥ 奖励计算                                                           │
│     └── 速度跟踪 + 姿态惩罚 + 关节惩罚 + 步态奖励 + ...                │
│                                                                      │
│  ⑦ PPO 更新 (每 24 steps × 4096 envs = 98304 transitions)            │
│     └── GAE advantage 估计 → 5 epochs minibatch SGD                  │
│                                                                      │
│  ⑧ 循环 ①~⑦ 共 20000 iterations                                      │
│     └── 每 100 iterations 保存 checkpoint                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 脚本调用方式与关键参数

### 5.1 `train.py` — 训练入口

```bash
# 基础用法
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless

# 完整参数
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \   # 环境 ID
    --headless \                          # 无 GUI 模式
    --num_envs=4096 \                     # 并行环境数 (默认: 环境配置中的值)
    --max_iterations=20000 \              # 最大训练迭代数 (默认: PPO 配置中的值)
    --seed=42 \                           # 随机种子
    --video \                             # 录制训练视频
    --video_length=200 \                  # 视频帧数
    --video_interval=2000 \               # 视频录制间隔
    --distributed \                       # 启用分布式训练
    --resume \                            # 从 checkpoint 恢复
    --load_run=<RUN_NAME> \               # 恢复的 run 名称
    --checkpoint=model_5000.pt            # 恢复的 checkpoint

# 多 GPU 训练
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless --distributed \
    --num_envs=4096

# 多节点训练
python -m torch.distributed.run \
    --nnodes=2 --node_rank=0 \
    --master_addr=<IP> --master_port=<PORT> \
    --nproc_per_node=4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless --distributed --num_envs=8192
```

**输出**:
- `logs/rsl_rl/deeprobotics_m20_rough/<timestamp>/` — 训练日志
  - `model_<N>.pt` — 模型检查点
  - `params/env.yaml` — 环境配置 dump
  - `params/agent.yaml` — 算法配置 dump
  - `events.out.tfevents.*` — TensorBoard events

### 5.2 `play.py` — 策略回放入口

```bash
# 基础回放
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=10 \
    --load_run=<RUN_NAME> \
    --checkpoint=model_10000.pt

# 键盘控制模式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=1 \
    --keyboard \
    --load_run=<RUN_NAME>

# 录制视频
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=10 \
    --video --video_length=200 \
    --load_run=<RUN_NAME>

# 键盘控制说明 (--keyboard 模式):
#   ↑ / ↓        — 前进/后退 (x 线速度)
#   ← / →        — 左移/右移 (y 线速度)
#   A / D        — 左转/右转 (z 角速度)
#   W / S        — 抬升/降低身体高度
#   Q / E        — 减少/增加步频
#   R            — 重置环境
```

### 5.3 `export_onnx_fast.py` — ONNX 导出

```bash
# M20 策略导出
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/<RUN>/model_10000.pt \
    --robot m20 \
    --output_path /home/mojie/taskdog/exported/m20_policy.onnx

# 不附加元数据 (仅导出裸 MLP)
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path <path> \
    --robot m20 \
    --output_path <path> \
    --no_metadata

# ONNX 模型信息:
#   input:  "obs"     shape=(1, 57)  dtype=float32
#   output: "actions" shape=(1, 16)  dtype=float32
#   metadata: joint_names, stiffness, damping, default_joint_pos, action_scale
```

### 5.4 `list_envs.py` — 环境列表

```bash
python scripts/tools/list_envs.py
# 输出所有已注册的 Gym 环境 ID，包括:
#   Flat-Deeprobotics-M20-v0
#   Rough-Deeprobotics-M20-v0
#   Flat-Deeprobotics-Lite3-v0
#   Rough-Deeprobotics-Lite3-v0
```

---

## 6. 配置文件关键参数速查

### 6.1 仿真参数

| 参数 | 值 | 定义位置 | 说明 |
|------|-----|----------|------|
| `sim.dt` | 0.005 s | velocity_env_cfg.py:920 | 物理仿真步长 (200 Hz) |
| `decimation` | 4 | velocity_env_cfg.py:917 | 策略推理频率 = 200/4 = 50 Hz |
| `episode_length_s` | 20.0 s | velocity_env_cfg.py:918 | 每个 episode 的时长 |
| `env_spacing` | 2.5 m | velocity_env_cfg.py:903 | 并行环境之间的间距 |
| `num_envs` | 4096 | velocity_env_cfg.py:903 | 并行环境数量（可在 CLI 覆盖） |

### 6.2 机器人控制参数

| 参数 | 值 | 定义位置 | 说明 |
|------|-----|----------|------|
| 腿关节 stiffness | 80.0 N·m/rad | deeprobotics.py:101 | PD 控制器 P 增益 |
| 腿关节 damping | 2.0 N·m·s/rad | deeprobotics.py:102 | PD 控制器 D 增益 |
| 轮关节 stiffness | 0.0 | deeprobotics.py:112 | 轮关节无位置控制 |
| 轮关节 damping | 0.6 N·m·s/rad | deeprobotics.py:113 | 轮关节速度阻尼 |
| 腿关节 torque 上限 | 76.4 N·m | deeprobotics.py:99 | |
| 腿关节 velocity 上限 | 22.4 rad/s | deeprobotics.py:100 | |
| 轮关节 torque 上限 | 21.6 N·m | deeprobotics.py:110 | |
| 轮关节 velocity 上限 | 79.3 rad/s | deeprobotics.py:111 | |
| 初始站立高度 | 0.58 m | deeprobotics.py:84 | 基座离地高度 |
| soft_joint_pos_limit_factor | 0.9 | deeprobotics.py:95 | 到达限位 90% 开始软约束 |

### 6.3 速度命令范围 (M20)

| 参数 | 默认范围 | 说明 |
|------|----------|------|
| lin_vel_x | (-2.0, 2.0) m/s | 前进/后退 |
| lin_vel_y | (-1.0, 1.0) m/s | 左移/右移 |
| ang_vel_z | (-2.0, 2.0) rad/s | 左转/右转 |
| heading | (-π, π) | 目标朝向角 |

### 6.4 Domain Randomization 范围

| 参数 | 范围 | 说明 |
|------|------|------|
| 质量缩放 (base) | +(-1.0, 3.0) kg | 基座附加质量 |
| 质量缩放 (其他) | (0.85, 1.15)× | 连杆质量缩放 |
| 惯量缩放 | (0.85, 1.15)× | 连杆惯量缩放 |
| 质心偏移 | ±3cm (xy), ±2cm (z) | 质心随机偏移 |
| 静摩擦 | (0.35, 1.5) | 轮-地静摩擦 |
| 动摩擦 | (0.35, 1.5) | 轮-地动摩擦 |
| 弹性恢复系数 | (0.0, 0.7) | 碰撞弹性 |
| PD 刚度缩放 | (0.85, 1.15)× | 执行器刚度随机化 |
| PD 阻尼缩放 | (0.85, 1.15)× | 执行器阻尼随机化 |
| 外部推力 | ±10 N, ±10 N·m | 随机外部扰动 |
| 初始姿态扰动 | ±1m (xy), ±0.3rad (roll/pitch), ±π (yaw) | 环境重置时 |

### 6.5 地形参数 (Rough)

| 参数 | 值 | 说明 |
|------|-----|------|
| 盒子高度 | (0.025, 0.2) m | 随机台阶/障碍物 |
| 随机粗糙度 noise | (0.01, 0.16) m | 地形的随机起伏 |
| 地形网格分辨率 | 0.1 m | 高度扫描的分辨率 |

---

## 附录: 快速参考卡片

```
┌────────────────────────────────────────────────────────────┐
│              M20 Pro IsaacLab 速查卡片                       │
├────────────────────────────────────────────────────────────┤
│ Conda 环境:     conda activate env_isaaclab                 │
│ 项目根目录:      cd /home/mojie/taskdog                      │
│ 训练仓库:        cd deps/rl_training                         │
│ 模型仓库:        cd deps/deep_robotics_model                 │
│                                                            │
│ 列出环境:        python scripts/tools/list_envs.py          │
│ 开始训练:        python scripts/.../train.py                │
│                  --task=Rough-Deeprobotics-M20-v0           │
│                  --headless --num_envs=4096                  │
│ 回放策略:        python scripts/.../play.py                 │
│ 导出 ONNX:       python scripts/tools/export_onnx_fast.py   │
│ 查看日志:        tensorboard --logdir logs/                 │
│                                                            │
│ 观测维度:        57 (策略输入)                               │
│ 动作维度:        16 (12 腿位置 + 4 轮速度)                    │
│ 网络架构:        [512, 256, 128] ELU                        │
│ 控制频率:        50 Hz (decimation=4, dt=0.005s)            │
│ 关节数:          16 (4腿 × (hipx+hipy+knee+wheel))          │
│ 总质量:          ~33.6 kg                                   │
└────────────────────────────────────────────────────────────┘
```

---

## 7. 自定义环境安装方式 (2026-07-18 修正)

### 7.1 问题背景

最初尝试使用 `pip install -e /home/mojie/taskdog/custom_envs` 安装 `custom_envs` 包。安装后 `pip show custom_envs` 显示已安装，但 `import custom_envs` 报 `ModuleNotFoundError`。

### 7.2 根因分析

`setup.py` 放在 `custom_envs/` 目录**内部**，与 `__init__.py` 同级。当 `pip install -e` 运行时：

1. `find_packages()` 从 `setup.py` 所在目录（即 `custom_envs/`）开始搜索
2. 它找到了 `tasks`、`tasks.deeprobotics_m20_pro`、`tasks.deeprobotics_m20_pro.agents`、`utils` 这些子包
3. 但它**没有**将 `custom_envs/` 本身注册为根包，因为 `setup.py` 就在根包目录内

生成的 `__editable___custom_envs_0_1_0_finder.py` 中的 `MAPPING` 只有子包映射，缺少根包的路径入口，导致 `import custom_envs` 失败。

### 7.3 解决方案：使用 `.pth` 文件

放弃 `pip install -e` 方式，改用 Python 的 `.pth` 路径注入机制：

```bash
# 在 conda 环境的 site-packages 下创建 .pth 文件
echo "/home/mojie/taskdog" > /home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/taskdog.pth
```

**原理**: Python 启动时会自动扫描 `site-packages/` 下所有 `.pth` 文件，将其中每一行路径追加到 `sys.path`。这样 `import custom_envs` 就能在 `/home/mojie/taskdog/custom_envs/` 找到这个包。

**对比**:

| 方式 | 优点 | 缺点 |
|------|------|------|
| `pip install -e` | 标准方式 | `setup.py` 必须在包外一层 |
| `.pth` 文件 | 简单直接,无需移动文件 | 手动管理,不经过 pip |
| `sys.path.insert()` | 显式 | 每个脚本都要加,hardcode 路径 |
| 修改 `PYTHONPATH` 环境变量 | 全局生效 | 每次 shell 都要 export |

### 7.4 当前 custom_envs 最终结构

```
custom_envs/                        ← 通过 .pth 加入 sys.path
├── __init__.py                     ← from . import tasks
├── tasks/
│   ├── __init__.py                 ← import_packages() 自动发现
│   └── deeprobotics_m20_pro/
│       ├── __init__.py             ← gym.register(Flat/Rough-Deeprobotics-M20Pro-v0)
│       ├── flat_env_cfg.py         ← 继承 DeeproboticsM20FlatEnvCfg
│       ├── rough_env_cfg.py        ← 继承 DeeproboticsM20RoughEnvCfg
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py   ← PPO [512,256,128] ELU
└── utils/
    └── __init__.py
```

> 注意: `setup.py` 和 `custom_envs.egg-info/` 已删除，不再使用 pip 方式安装。

### 7.5 如何添加新的自定义路径

如果将来需要在其他位置添加 Python 模块，同样的方法：

```bash
echo "/your/other/path" >> /home/mojie/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/taskdog.pth
```

或者创建新的 `.pth` 文件（文件名任意，只要以 `.pth` 结尾）。

