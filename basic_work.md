# 云深处 M20 Pro 机器狗接入 IsaacLab 仿真平台 — 完整工作方案

> **创建日期**: 2026-07-17
> **任务目标**: 将云深处（Deep Robotics）Lynx M20 Pro 机器狗放入 NVIDIA IsaacLab 仿真平台，搭建完整的强化学习训练 + 部署工作流

---

## 目录

1. [调研结论摘要](#1-调研结论摘要)
2. [M20 Pro 技术规格](#2-m20-pro-技术规格)
3. [开源生态全景](#3-开源生态全景)
4. [URDF/USD 模型详解](#4-urdfusd-模型详解)
5. [与宇树（Unitree）生态对比](#5-与宇树unitree生态对比)
6. [可行路径分析](#6-可行路径分析)
7. [推荐实施方案](#7-推荐实施方案)
8. [详细项目结构](#8-详细项目结构)
9. [环境搭建步骤](#9-环境搭建步骤)
10. [训练与部署流程](#10-训练与部署流程)
11. [常见问题与注意事项](#11-常见问题与注意事项)
12. [参考资料](#12-参考资料)

---

## 1. 调研结论摘要

### 核心结论：M20 Pro 完全可以接入 IsaacLab，且有成熟的开源生态直接可用

**好消息：**
- Deep Robotics（云深处科技）官方提供了完整的开源 RL 训练生态，**不需要从零搭建**
- M20 已有 URDF/MJCF/USD 三种格式的官方仿真模型（BSD-3-Clause 许可证）
- 官方 `rl_training` 仓库基于 IsaacLab 2.3.2，开箱即用，支持训练→ONNX 导出→真机部署
- 第三方 `robot_lab` 仓库也支持 M20，且扩展了更多任务类型
- M20 Pro 与 M20 使用**完全相同的机械平台和关节结构**，仿真模型完全通用

**需要关注的点：**
- M20 是**轮足混合**（wheel-legged）设计，与 Unitree Go2 等纯足式机器人不同，RL 策略需要考虑轮关节控制
- 仿真模型的高清版需要从 Google Drive 下载，GitHub 仅含低清版
- robot_lab 的 M20 支持标记为"config optimization in progress"，尚在完善中

---

## 2. M20 Pro 技术规格

### 2.1 基本参数

| 参数 | M20 (标准版) | M20 Pro |
|------|-------------|---------|
| 外形尺寸（站立） | 820 × 430 × 570 mm | 820 × 430 × 570 mm |
| 重量（含电池） | ~33 kg | ~33 kg |
| 自由度 (DOF) | **16 DOF**（4腿 × 4关节） | **16 DOF**（4腿 × 4关节） |
| 额定载荷 | 15 kg | 15 kg |
| 最大静态载荷 | 50 kg | 50 kg |
| 最大速度 | 5 m/s（实验室）/ 2 m/s（运行） | 5 m/s（实验室）/ 2 m/s（运行） |
| 续航（空载） | 3 小时 / 15 km | 3 小时 / 15 km |
| 越障高度 | 25 cm（连续楼梯）/ 80 cm（单级） | 25 cm（连续楼梯）/ 80 cm（单级） |
| 最大爬坡 | 45° | 45° |
| 防护等级 | IP66 | IP66 |
| 工作温度 | -20°C 至 55°C | -20°C 至 55°C |
| 计算平台 | **双**八核 64 位处理器 | **三**八核 64 位处理器 |
| 存储/内存 | 128 GB / 16 GB | 128 GB / 16 GB |
| SLAM 自主导航 | ❌ | ✅ |
| 自主充电 | ❌ | ✅（选配） |
| 扩展接口 | 72V 电源、千兆以太网 | 72V/24V 电源、千兆以太网、USB 3.0 |
| 传感器 | 2×96线激光雷达、2×广角相机 | 2×96线激光雷达、2×广角相机 |

### 2.2 M20 vs M20 Pro 关键区别

```
M20 Pro = M20 机械平台 + 算力升级（双核→三核）+ SLAM 导航 + 自主充电 + 扩展接口
```

**仿真层面: M20 Pro 与 M20 完全一致**。两者的机械结构、关节配置、质量分布完全相同，差异仅在计算硬件和上层功能。因此 M20 的 URDF/USD 模型可以直接用于 M20 Pro 的仿真。

### 2.3 关节结构

每腿 4 关节，总共 16 个关节：

```
base_link
├── fl_hipx_joint (revolute, X轴, ±0.436~0.611 rad)  → 髋关节横摆
│   └── fl_hipy_joint (revolute, Y轴, ±2.286~2.583 rad) → 髋关节俯仰
│       └── fl_knee_joint (revolute, Y轴, ±2.792~2.809 rad) → 膝关节
│           └── fl_wheel_joint (continuous, Y轴, 无限)      → 轮关节
├── fr_hipx → fr_hipy → fr_knee → fr_wheel  (对称)
├── hl_hipx → hl_hipy → hl_knee → hl_wheel  (对称)
└── hr_hipx → hr_hipy → hr_knee → hr_wheel  (对称)
```

| 关节类型 | 力矩上限 | 速度上限 | 备注 |
|----------|----------|----------|------|
| 腿关节 (hipx/hipy/knee) | 76.4 N·m | 22.4 rad/s | 位置/力矩控制 |
| 轮关节 (wheel) | 21.6 N·m | 79.3 rad/s | 速度控制 |

---

## 3. 开源生态全景

Deep Robotics 官方维护着与宇树同样完整的开源生态：

```
┌──────────────────────────────────────────────────────┐
│                  Deep Robotics 开源生态                  │
├──────────────────────────────────────────────────────┤
│                                                        │
│  ┌─────────────────────┐                               │
│  │ deep_robotics_model │  3D 模型仓库                   │
│  │ (URDF/MJCF/USD)     │  BSD-3-Clause                 │
│  └────────┬────────────┘                               │
│           │ 模型文件                                     │
│           ▼                                            │
│  ┌─────────────────────┐                               │
│  │ rl_training         │  RL 训练仓库 (基于 IsaacLab)    │
│  │ (IsaacLab 2.3.2)    │  BSD-3-Clause + Apache-2.0    │
│  └────────┬────────────┘                               │
│           │ 策略导出 (ONNX)                              │
│           ▼                                            │
│  ┌─────────────────────┐                               │
│  │ sdk_deploy          │  真机部署 SDK (ROS2/C++)       │
│  │ (M20 + Lite3)       │  BSD-3-Clause                 │
│  └─────────────────────┘                               │
│                                                        │
│  ┌─────────────────────┐                               │
│  │ robot_lab (第三方)    │  RL 扩展库 (支持 20+ 机器人)    │
│  │ fan-ziqi/robot_lab  │  Apache-2.0                   │
│  └─────────────────────┘                               │
│                                                        │
└──────────────────────────────────────────────────────┘
```

### 3.1 核心仓库一览

| 仓库 | 地址 | 用途 | 许可证 | Stars |
|------|------|------|--------|-------|
| **deep_robotics_model** | [GitHub](https://github.com/DeepRoboticsLab/deep_robotics_model) | 7款机器人 3D 模型 (URDF/MJCF/USD) | BSD-3 | ~50 |
| **rl_training** | [GitHub](https://github.com/DeepRoboticsLab/rl_training) | IsaacLab RL 训练 (M20 + Lite3) | BSD-3 + Apache-2.0 | ~232 |
| **sdk_deploy** | [GitHub](https://github.com/DeepRoboticsLab/sdk_deploy) | Sim-to-Real 部署 SDK (C++/ROS2) | BSD-3 | ~90 commits |
| **robot_lab** | [GitHub](https://github.com/fan-ziqi/robot_lab) | 多机器人 RL 扩展库 | Apache-2.0 | 活跃 |

### 3.2 官方社区

- **Discord**: https://discord.gg/gdM9mQutC8
- **B站官号**: [@云深处实验室](https://space.bilibili.com/3546975261690117)
- **B站教程**: [山猫M20 具身智能开发第一期](https://www.bilibili.com/video/BV17S2VBJEN2/)

---

## 4. URDF/USD 模型详解

### 4.1 模型获取

```bash
# 方式一: 克隆官方模型仓库 (standalone)
git clone https://github.com/DeepRoboticsLab/deep_robotics_model.git

# 模型目录结构 (standalone 仓库):
# M20/
# ├── urdf/
# │   ├── M20.urdf          # URDF 模型文件
# │   └── meshes/           # STL 网格文件 (17个)
# │       ├── base_link.STL
# │       ├── fl_hipx.STL, fl_hipy.STL, fl_knee.STL, fl_wheel.STL
# │       ├── fr_hipx.STL, fr_hipy.STL, fr_knee.STL, fr_wheel.STL
# │       ├── hl_hipx.STL, hl_hipy.STL, hl_knee.STL, hl_wheel.STL
# │       └── hr_hipx.STL, hr_hipy.STL, hr_knee.STL, hr_wheel.STL
# ├── mjcf/
# │   ├── M20.xml            # MuJoCo 模型
# │   └── meshes/            # STL 网格 (17个, 与 urdf 共用)
# └── usd/
#     ├── M20.usd             # USD 主文件 (Isaac Sim/Lab 推荐)
#     └── configuration/      # USD 子模块引用
#         ├── M20_base.usd
#         ├── M20_physics.usd
#         ├── M20_robot.usd
#         └── M20_sensor.usd

# ⚠️ 注意: rl_training 子模块内部有另一套目录结构:
# M20/M20_urdf/urdf/M20.urdf, M20/M20_mjcf/mjcf/M20.xml, M20/M20_usd/M20.usd
# (命名嵌套更深, 但内容相同)

# 方式二: 高清模型 (推荐用于 IsaacLab)
# 从 Google Drive 下载链接 (见仓库 README)
# 高清模型碰撞体、惯量参数经过精调,适合 sim-to-real 迁移
```

### 4.2 质量与惯量参数

| 连杆 | 质量 (kg) | Ixx | Iyy | Izz |
|------|-----------|-----|-----|-----|
| base_link | 15.882 | 0.0881 | 0.5338 | 0.5639 |
| hipx (×4) | 0.2614 | 0.000186 | 0.000371 | 0.000288 |
| hipy (×4) | 2.493 | ~0.039 | ~0.038 | ~0.004 |
| knee (×4) | 1.26 | ~0.013 | ~0.013 | ~0.0015 |
| wheel (×4) | 0.638 | 0.001535 | 0.002853 | 0.001535 |
| **总计** | **~33.6 kg** | | | |

### 4.3 关节限位

| 关节 | 下限 (rad) | 上限 (rad) | 等效角度 |
|------|------------|------------|----------|
| hipx (前腿) | -0.436 ~ -0.611 | 0.611 ~ 0.436 | ±25°~35° |
| hipy | -2.583 ~ -2.286 | 2.286 ~ 2.583 | ±131°~148° |
| knee | -2.792 ~ -2.809 | 2.809 ~ 2.792 | ±160° |
| wheel | continuous | continuous | 无限旋转 |

---

## 5. 与宇树（Unitree）生态对比

| 维度 | Unitree (宇树) | Deep Robotics (云深处) | 评价 |
|------|----------------|------------------------|------|
| **官方模型仓库** | unitree_rl_lab 内含 | deep_robotics_model (独立) | 相当 |
| **URDF 模型** | ✅ Go2/B2/H1/G1 等 | ✅ M20/Lite3/X30 等 | 相当 |
| **IsaacLab 训练** | unitree_rl_lab (IsaacLab 2.x) | rl_training (IsaacLab 2.3.2) | 相当 |
| **Sim-to-Real SDK** | 内置在 unitree_rl_lab | sdk_deploy (独立仓库) | 相当 |
| **ONNX 导出** | ✅ | ✅ | 相当 |
| **第三方生态** | robot_lab 支持 Go2/B2/G1/H1 等 | robot_lab 支持 M20/Lite3 | 相当 |
| **社区规模** | 更大 (GitHub Stars 更多) | 正在成长 | 宇树稍领先 |
| **机器人类型** | 纯四足 + 人形 | 轮足混合 + 四足 | M20 独特性更高 |
| **文档完善度** | 较完善 | 中英文 README + B站视频 | 相当 |

**结论：Deep Robotics 的生态几乎与宇树对齐，完全可以直接用于二次开发。** 宇树的所有核心组件（模型、训练框架、部署 SDK）Deep Robotics 都有对应实现。

---

## 6. 可行路径分析

### 路径 A：使用官方 rl_training 仓库（强烈推荐 ⭐⭐⭐⭐⭐）

```
优点:
  ✅ 官方维护，与 M20 硬件完全匹配
  ✅ 已集成 IsaacLab 2.3.2 + Isaac Sim 5.1.0
  ✅ 预置 Rough-Deeprobotics-M20-v0 任务
  ✅ 包含 ONNX 导出脚本 (无需 Isaac Sim)
  ✅ 包含 sdk_deploy 真机部署流程
  ✅ 中英文文档 + B站视频教程
  ✅ BSD-3-Clause 许可证

缺点:
  ⚠️ 仅支持 Lite3 和 M20 两款机器人
  ⚠️ 依赖特定 IsaacLab 版本 (2.3.2)

适用场景: 目标明确在 M20 Pro 上做 RL 训练和真机部署
```

### 路径 B：使用第三方 robot_lab 仓库（推荐 ⭐⭐⭐⭐）

```
优点:
  ✅ 支持 20+ 机器人，生态系统更丰富
  ✅ RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 环境
  ✅ 支持更多任务类型 (BeyondMimic, AMP Dance 等)
  ✅ 活跃维护, 多人贡献
  ✅ Apache-2.0 许可证

缺点:
  ⚠️ M20 配置标记为 "optimization in progress"
  ⚠️ Sim2Real 尚未验证
  ⚠️ 非官方，可能有参数不匹配

适用场景: 研究多种机器人对比、算法创新、多任务扩展
```

### 路径 C：从零搭建自定义 IsaacLab 环境（推荐 ⭐⭐⭐）

```
优点:
  ✅ 完全控制, 可深度定制
  ✅ 适合特殊需求 (如添加传感器、修改动力学)
  ✅ 学习价值最高

缺点:
  ❌ 工作量大 (估计 2-4 周)
  ❌ 需要深入理解 IsaacLab API
  ❌ 容易出错 (URDF→USD 转换、碰撞体配置等)
  ❌ 参数调优费时

适用场景: 教学目的、特殊科研需求、Deep Robotics 未覆盖的机器人
```

### 路径 D：混合方案 — 官方仓库 + 自定义扩展（推荐 ⭐⭐⭐⭐⭐）

```
策略: 以官方 rl_training 为基础, 按需融入 robot_lab 的任务和自定义模块

步骤:
  1. 先用 rl_training 跑通训练→导出→部署完整流程
  2. 理解代码结构后再定制环境 (修改 reward/observation/domain randomization)
  3. 可选: 移植 robot_lab 中的高级任务到 rl_training 框架

优点:
  ✅ 快速起步 + 深度定制能力
  ✅ 降低调试难度 (有官方基准可对比)
  ✅ 官方更新可合并

这是最务实的路径
```

---

## 7. 推荐实施方案

### 采用路径 D（混合方案），分三阶段实施

```
阶段一 (1-2 天): 环境搭建 + 跑通 Baseline
  ├── 安装 IsaacLab 2.3.2 + Isaac Sim 5.1.0
  ├── 克隆 rl_training + deep_robotics_model
  ├── 验证环境: 列出可用任务列表
  └── 跑通训练: Rough-Deeprobotics-M20-v0

阶段二 (3-7 天): 策略训练 + 评估
  ├── 参数调优 (reward 权重, domain randomization)
  ├── 多地形训练 (flat + rough)
  ├── 导出 ONNX 模型
  ├── 可视化评估 (play + video recording)
  └── Sim-to-Sim 验证 (PhysX vs Newton 物理引擎)

阶段三 (7-14 天): 定制开发 + 进阶
  ├── 自定义环境 (修改观测/动作/奖励)
  ├── 添加传感器仿真 (LiDAR/相机)
  ├── 感知策略训练 (blind → perceptive)
  ├── 多模态运动 (行走+爬楼梯+越障)
  └── (如有真机) ONNX 部署到 M20 Pro
```

---

## 8. 详细项目结构

> **当前状态**: 已拉取所有依赖仓库，目录骨架已搭建（未编写自定义代码）

```
/home/mojie/taskdog/                          # 项目根目录
│
├── work.md                                   # 本文档 (工作方案)
├── README.md                                 # 项目说明
├── .gitignore                                # Git 忽略规则
│
├── deps/                                     # 依赖仓库 (独立克隆)
│   │
│   ├── deep_robotics_model/                  # 官方 3D 模型仓库
│   │   ├── M20/                              # ★ 目标机器人
│   │   │   ├── urdf/
│   │   │   │   ├── M20.urdf                  # URDF 模型定义
│   │   │   │   └── meshes/                   # STL 网格 (17个)
│   │   │   ├── mjcf/
│   │   │   │   ├── M20.xml                   # MuJoCo 模型
│   │   │   │   └── meshes/                   # STL 网格 (17个)
│   │   │   └── usd/
│   │   │       ├── M20.usd                   # USD 主文件 (IsaacLab 推荐)
│   │   │       └── configuration/            # USD 子模块
│   │   │           ├── M20_base.usd
│   │   │           ├── M20_physics.usd
│   │   │           ├── M20_robot.usd
│   │   │           └── M20_sensor.usd
│   │   ├── M20S/                             # M20S 模型
│   │   ├── M20_Piper/                        # M20 机械臂版模型
│   │   ├── Lite3/                            # Lite3 模型
│   │   ├── X30/                              # X30 模型
│   │   ├── DR02/                             # DR02 Pro/Standard
│   │   ├── images/                           # 预览图
│   │   ├── LICENSE.txt                       # BSD-3-Clause
│   │   └── README.md
│   │
│   ├── rl_training/                          # 官方 RL 训练仓库
│   │   ├── deep_robotics_model/              # 子模块 (模型文件, 内嵌版本)
│   │   │   └── M20/
│   │   │       ├── M20_urdf/urdf/M20.urdf    # URDF (内嵌版本)
│   │   │       ├── M20_mjcf/mjcf/M20.xml     # MJCF (内嵌版本)
│   │   │       └── M20_usd/M20.usd           # USD (内嵌版本)
│   │   ├── source/rl_training/
│   │   │   ├── setup.py                      # 包安装入口
│   │   │   ├── config/                       # 全局配置
│   │   │   └── rl_training/                  # Python 包 (嵌套!)
│   │   │       ├── __init__.py
│   │   │       ├── assets/
│   │   │       │   └── deeprobotics.py       # 机器人 ArticulationCfg 定义
│   │   │       └── tasks/manager_based/locomotion/velocity/
│   │   │           ├── velocity_env_cfg.py   # 基础环境配置类
│   │   │           ├── config/
│   │   │           │   ├── quadruped/
│   │   │           │   │   └── deeprobotics_lite3/  # Lite3 环境
│   │   │           │   │       ├── flat_env_cfg.py
│   │   │           │   │       ├── rough_env_cfg.py
│   │   │           │   │       └── agents/rsl_rl_ppo_cfg.py
│   │   │           │   └── wheeled/
│   │   │           │       └── deeprobotics_m20/     # ★ M20 环境
│   │   │           │           ├── __init__.py
│   │   │           │           ├── flat_env_cfg.py   # 平坦地形
│   │   │           │           ├── rough_env_cfg.py  # 崎岖地形
│   │   │           │           └── agents/
│   │   │           │               ├── __init__.py
│   │   │           │               └── rsl_rl_ppo_cfg.py
│   │   │           └── mdp/                 # MDP 组件
│   │   │               ├── commands.py
│   │   │               ├── curriculums.py
│   │   │               ├── events.py
│   │   │               ├── observations.py
│   │   │               └── rewards.py
│   │   ├── scripts/
│   │   │   ├── reinforcement_learning/rsl_rl/
│   │   │   │   ├── train.py                  # 训练入口
│   │   │   │   ├── play.py                   # 回放入口
│   │   │   │   └── cli_args.py               # 命令行参数
│   │   │   └── tools/
│   │   │       ├── list_envs.py              # 列出可用环境
│   │   │       ├── export_onnx_fast.py       # ONNX 导出
│   │   │       └── compare_runs.py           # 运行对比
│   │   ├── docs/imgs/                        # 文档截图
│   │   ├── LICENSE                           # BSD-3-Clause
│   │   ├── LICENSE-robot_lab                 # Apache-2.0
│   │   └── README.md
│   │
│   └── sdk_deploy/                           # 真机部署 SDK
│       ├── src/
│       │   ├── M20_sdk_deploy/               # ★ M20 部署代码
│       │   │   ├── include/                  # C++ 头文件
│       │   │   ├── interface/                # 通信接口
│       │   │   ├── M20_description/          # 机器人描述
│       │   │   ├── policy/                   # 策略加载
│       │   │   ├── run_policy/               # 策略运行
│       │   │   ├── scripts/                  # 部署脚本
│       │   │   ├── state_machine/            # 状态机
│       │   │   └── third_party/              # 第三方库
│       │   ├── Lite3_sdk_deploy/             # Lite3 部署代码
│       │   ├── lite3_sdk_service/            # Lite3 模式切换
│       │   ├── lite3_transfer/               # Lite3 UDP-ROS2 转换
│       │   └── drdds/                        # DRDDS 通信格式
│       ├── img/                              # 文档图片
│       ├── LICENSE                           # BSD-3-Clause
│       └── README.md
│
├── custom_envs/                              # 自定义环境扩展 (待开发)
│   ├── __init__.py                           # ✅ 已创建
│   ├── tasks/
│   │   ├── __init__.py                       # ✅ 已创建
│   │   └── deeprobotics_m20_pro/             # M20 Pro 专属配置 (待开发)
│   │       ├── __init__.py                   # ✅ 已创建
│   │       ├── flat_env_cfg.py               # 平坦地形 (待编写)
│   │       ├── rough_env_cfg.py              # 崎岖地形 (待编写)
│   │       ├── stair_env_cfg.py              # 楼梯环境 (待编写)
│   │       └── agents/
│   │           ├── __init__.py               # ✅ 已创建
│   │           └── rsl_rl_ppo_cfg.py         # PPO 配置 (待编写)
│   └── utils/
│       ├── __init__.py                       # ✅ 已创建
│       ├── terrain.py                        # 自定义地形 (待编写)
│       └── sensors.py                        # 传感器配置 (待编写)
│
├── scripts/                                  # 实用脚本 (待开发)
│   └── .gitkeep
│
├── configs/                                  # 实验配置 (待开发)
│   └── .gitkeep
│
├── logs/                                     # 训练日志 (gitignore)
│   └── .gitkeep
│
├── exported/                                 # 导出模型 (gitignore)
│   └── .gitkeep
│
├── notebooks/                                # Jupyter 分析笔记本 (待开发)
│   └── .gitkeep
│
├── docker/                                   # Docker 部署 (待开发)
│   └── .gitkeep
│
└── docs/                                     # 文档 (待开发)
    └── .gitkeep
```

### 8.1 关键路径速查

| 资源 | 实际路径 |
|------|----------|
| M20 URDF 模型 | `deps/deep_robotics_model/M20/urdf/M20.urdf` |
| M20 USD 模型 | `deps/deep_robotics_model/M20/usd/M20.usd` |
| M20 MJCF 模型 | `deps/deep_robotics_model/M20/mjcf/M20.xml` |
| M20 资产定义 (ArticulationCfg) | `deps/rl_training/source/rl_training/rl_training/assets/deeprobotics.py` |
| M20 Rough 环境配置 | `deps/rl_training/source/rl_training/rl_training/tasks/.../config/wheeled/deeprobotics_m20/rough_env_cfg.py` |
| M20 Flat 环境配置 | `deps/rl_training/source/rl_training/rl_training/tasks/.../config/wheeled/deeprobotics_m20/flat_env_cfg.py` |
| M20 PPO 算法配置 | `deps/rl_training/source/rl_training/rl_training/tasks/.../config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py` |
| 训练入口脚本 | `deps/rl_training/scripts/reinforcement_learning/rsl_rl/train.py` |
| 回放入口脚本 | `deps/rl_training/scripts/reinforcement_learning/rsl_rl/play.py` |
| ONNX 导出脚本 | `deps/rl_training/scripts/tools/export_onnx_fast.py` |
| M20 部署 SDK | `deps/sdk_deploy/src/M20_sdk_deploy/` |

---

## 9. 环境搭建步骤

### 9.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |
| GPU | NVIDIA RTX 3070 (8GB VRAM) | RTX 4090 (24GB VRAM) |
| RAM | 32 GB | 64 GB |
| 磁盘 | 100 GB | 500 GB SSD |
| NVIDIA 驱动 | 535+ | 550+ |
| CUDA | 12.x | 12.4+ |

### 9.2 安装 Isaac Lab

```bash
# 1. 安装 Isaac Lab (官方推荐 conda 方式)
# 参考: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# 创建 conda 环境
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

# 安装 Isaac Sim 5.1.0 (通过 Omniverse Launcher 或 pip)
# 安装 Isaac Lab 2.3.2
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2

# 使用 Isaac Lab 提供的安装脚本
./isaaclab.sh --install  # 或 python tools/install_isaaclab.py
```

### 9.3 安装 M20 训练环境

```bash
# 2. 回到项目目录
cd /home/mojie/taskdog

# 3. 克隆官方仓库 (作为 git submodules)
git init
git submodule add https://github.com/DeepRoboticsLab/deep_robotics_model.git deps/deep_robotics_model
git submodule add https://github.com/DeepRoboticsLab/rl_training.git deps/rl_training
git submodule add https://github.com/DeepRoboticsLab/sdk_deploy.git deps/sdk_deploy

# 4. 下载高清模型 (可选但推荐)
# 从 deep_robotics_model README 中的 Google Drive 链接下载
# 将高清模型放到 IsaacLab 的 assets 目录
# 或配置环境变量: export ISAACLAB_ASSETS_DATA_DIR=/path/to/assets

# 5. 安装 rl_training
conda activate isaaclab
cd deps/rl_training
python -m pip install -e source/rl_training

# 6. 验证安装
python scripts/tools/list_envs.py
# 应看到: Rough-Deeprobotics-M20-v0, Rough-Deeprobotics-Lite3-v0
```

### 9.4 安装 robot_lab (可选)

```bash
# 7. 克隆 robot_lab
cd /home/mojie/taskdog/deps
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab

# 8. 安装
conda activate isaaclab
python -m pip install -e source/robot_lab

# 9. 验证
python scripts/tools/list_envs.py
# 应看到: RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0
```

### 9.5 一键安装脚本

```bash
#!/bin/bash
# install.sh — 一键安装脚本

set -e

PROJECT_DIR="/home/mojie/taskdog"
CONDA_ENV="isaaclab"

echo "=== 安装 DeepRobotics M20 IsaacLab 环境 ==="

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# 克隆依赖
cd ${PROJECT_DIR}
git submodule update --init --recursive

# 安装 rl_training
cd deps/rl_training
python -m pip install -e source/rl_training

# 验证
python scripts/tools/list_envs.py

echo "=== 安装完成 ==="
echo "运行训练: cd deps/rl_training && python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless"
```

---

## 10. 训练与部署流程

### 10.1 训练

```bash
# 激活环境
conda activate isaaclab
cd /home/mojie/taskdog/deps/rl_training

# === 基础训练 ===

# 无头模式训练 (推荐服务器使用)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless \
    --num_envs=4096

# 带 GUI 训练 (调试用)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=1024

# === 分布式训练 ===

# 多 GPU (单机)
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless --distributed \
    --num_envs=4096

# 多 GPU 多节点
python -m torch.distributed.run \
    --nnodes=2 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=<PORT> \
    --nproc_per_node=4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless --distributed \
    --num_envs=8192

# === 从检查点恢复 ===
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless \
    --resume \
    --load_run <RUN_NAME> \
    --checkpoint model_5000.pt

# === 监控训练 ===
tensorboard --logdir logs/rsl_rl/deeprobotics_m20_rough/
```

### 10.2 回放与评估

```bash
# 策略回放 (带 GUI)
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=10 \
    --load_run <RUN_NAME> \
    --checkpoint model_10000.pt

# 键盘控制模式
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=1 \
    --keyboard \
    --load_run <RUN_NAME>

# 录制视频
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --num_envs=10 \
    --video \
    --video_length 200 \
    --load_run <RUN_NAME>

# 平坦地形回放
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Flat-Deeprobotics-M20-v0 \
    --num_envs=10
```

### 10.3 ONNX 模型导出

```bash
# 导出 ONNX (不需要 Isaac Sim!)
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/<RUN>/model_10000.pt \
    --robot m20 \
    --output_path /home/mojie/taskdog/exported/m20_policy.onnx

# ONNX 模型自带元数据:
#   - 关节名称、刚度、阻尼
#   - 默认关节位置
#   - 动作缩放因子
#   - 观测缩放因子
```

### 10.4 真机部署 (M20 Pro)

```bash
# 部署流程详见 sdk_deploy 仓库

# 1. 构建部署 SDK
cd /home/mojie/taskdog/deps/sdk_deploy

# 2. 将 ONNX 模型部署到机器人
# 需要:
#   - 机器人开机并与开发机在同一网络
#   - ROS2 环境配置
#   - 参考 src/M20_sdk_deploy/README.md 中的详细步骤

# 3. 运行部署
# 详细步骤见 deps/sdk_deploy/src/M20_sdk_deploy/README.md
```

### 10.5 训练配置关键参数速查

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 并行环境数 | 4096 | 越大训练越快 |
| 仿真时间步 | 0.005 s (200 Hz) | |
| 控制解耦 | 4 (50 Hz 策略) | 每 4 步仿真执行 1 次策略 |
| Episode 长度 | 20 s | |
| 环境间距 | 2.5 m | |
| 观测空间 | 48 维 | 关节位置/速度 + 基座状态 + 命令 |
| 动作空间 | 16 维 | 12 个腿关节位置 + 4 个轮关节速度 |
| 腿关节刚度 | 80.0 N·m/rad | P 增益 |
| 腿关节阻尼 | 2.0 N·m·s/rad | D 增益 |
| 轮关节阻尼 | 0.6 N·m·s/rad | |
| PPO clip 参数 | 0.2 | |
| 学习率 | 1e-3 | |
| γ (折扣因子) | 0.99 | |
| λ (GAE) | 0.95 | |

---

## 11. 常见问题与注意事项

### 11.1 M20 是轮足混合机器人

这是与 Unitree Go2 等纯足式机器人最大的区别：

- **leg_joint_names** = 12 个关节 (hipx + hipy + knee × 4条腿)
- **wheel_joint_names** = 4 个关节 (wheel × 4条腿)
- 腿关节使用**位置控制** (JointPositionAction)
- 轮关节使用**速度控制** (JointVelocityAction)
- 平地移动时，轮子提供主要推进力；崎岖地形时，腿部提供越障能力

### 11.2 高清 vs 低清模型

- GitHub 仓库中的低清模型碰撞体为简化几何体 (盒子、圆柱)
- 高清模型有更精确的碰撞体和惯量参数
- **Sim-to-Real 迁移强烈建议使用高清模型**

### 11.3 物理引擎选择

- Isaac Sim 5.1.0 默认使用 PhysX 5
- 可尝试 Newton 物理引擎 (beta)，宣称在 RTX 4090 上有高达 152× 加速
- Sim-to-Real 迁移前建议在两种引擎上分别验证

### 11.4 版本兼容性

```
关键约束:
  Isaac Lab 2.3.2 ↔ Isaac Sim 5.1.0 ↔ RSL-RL 5.0.1 ↔ Python 3.11

不要混用版本! 例如:
  ✗ Isaac Lab 1.x 的 API 与 2.x 不兼容
  ✗ robot_lab main 分支可能使用了 IsaacLab main 的 API
```

### 11.5 与 Unitree 工作流的主要差异

| 方面 | Unitree Go2 | DeepRobotics M20 |
|------|-------------|-------------------|
| 运动模式 | 纯足式行走 | 轮足混合 |
| 控制模式 | 位置控制 (12 关节) | 位置 (12) + 速度控制 (4 轮) |
| 平地步态 | 对角小跑 (trot) | 轮式驱动 + 腿姿态调整 |
| 崎岖地形 | 步行越障 | 混合步态 (轮+腿) |
| 能耗 | 纯关节驱动 | 平地轮式省电 |

---

## 12. 参考资料

### 官方仓库
- [DeepRoboticsLab/deep_robotics_model](https://github.com/DeepRoboticsLab/deep_robotics_model) — 3D 模型
- [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) — RL 训练
- [DeepRoboticsLab/sdk_deploy](https://github.com/DeepRoboticsLab/sdk_deploy) — 真机部署 SDK
- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) — RL 扩展库

### 官方文档
- [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Lab 四足机器人教程](https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/)
- [Deep Robotics 产品页面](https://www.deeprobotics.us/products/)
- [M20 Pro 用户手册](https://www.deeprobotics.us/wp-content/uploads/2025/08/Lynx-M20-Pro-User-Manual-V1.0.2-0.pdf)
- [M20 产品规格书](https://www.deeprobotics.us/wp-content/uploads/2025/08/DEEPRobotics-LYNX-M20-ENV-05.20.2025.pdf)

### 教程视频
- [B站: 山猫M20 具身智能开发第一期 (ROS2 + RL)](https://www.bilibili.com/video/BV17S2VBJEN2/)
- [B站: 四足运控从入门到精通 (RL训练)](https://www.bilibili.com/video/BV1xKabz9E2d/)
- [B站官号: @云深处实验室](https://space.bilibili.com/3546975261690117)

### 模型查看器
- [在线 URDF/MJCF 查看器](https://viewer.robotsfan.com) — 拖拽模型文件夹即可查看

### 学术引用
```bibtex
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  year = {2024}
}
```

---

## 附录 A: 快速启动检查清单

- [ ] 硬件满足最低要求 (Ubuntu 22.04, RTX 3070+, 32GB+)
- [ ] NVIDIA 驱动 + CUDA 12.x 安装完毕
- [ ] Isaac Lab 2.3.2 + Isaac Sim 5.1.0 安装成功
- [ ] 运行 `./isaaclab.sh -p source/standalone/demos/quadrupeds.py` 验证基本环境
- [ ] 克隆 rl_training 并安装
- [ ] `python scripts/tools/list_envs.py` 能看到 M20 环境
- [ ] 执行一次短训练 (100 iterations) 确认不报错
- [ ] 回放训练结果,确认机器人运动正常
- [ ] 导出 ONNX 模型,验证文件生成
- [ ] (可选) 阅读 sdk_deploy 文档,了解真机部署流程

---

> **最后更新**: 2026-07-17
> **维护者**: 项目组成员
> **状态**: 调研完成，进入实施阶段
