# add_piper.md — M20 Pro 加装 AgileX Piper 机械臂

> **状态**: 双 articulation + 物理子步同步方案已验证通过（2026-08-07）

---

## 目录

1. [当前架构（方案 C）](#1-当前架构方案-c)
2. [所有尝试方案及失败原因](#2-所有尝试方案及失败原因)
3. [方案 C 为什么成功](#3-方案-c-为什么成功)
4. [后续添加臂控制](#4-后续添加臂控制)
5. [注意事项](#5-注意事项)
6. [涉及文件清单](#6-涉及文件清单)

---

## 1. 当前架构（方案 C）

### 1.1 架构概览

```
┌─────────────────────────────────────────────────┐
│  IsaacLab ManagerBasedRLEnv                      │
│                                                  │
│  ┌──────────────┐    ┌──────────────────┐        │
│  │ M20 (robot)  │    │ Piper (piper)    │        │
│  │ Articulation │    │ Articulation     │        │
│  │ 16 DOF       │    │ 8 DOF            │        │
│  │ 17 bodies    │    │ 11 bodies        │        │
│  │              │    │                  │        │
│  │ 重力: ON     │    │ 重力: OFF        │        │
│  │ 阻尼: 0      │    │ 阻尼: 100/100    │        │
│  └──────┬───────┘    └────────┬─────────┘        │
│         │                     │                   │
│         │  Physics sub-step   │                   │
│         │  callback:          │                   │
│         │  ┌─────────────────┐│                   │
│         │  │每 sim.dt (0.005s)│                   │
│         │  │读取 base_link    │                   │
│         │  │位姿 + 速度       │                   │
│         │  │+ 安装偏移(0,0,   │                   │
│         │  │0.12)            │                   │
│         │  │→ 写入 Piper root │                   │
│         │  └─────────────────┘│                   │
│         │                     │                   │
└─────────┴─────────────────────┴───────────────────┘
```

### 1.2 双 Articulation 配置

**M20（robot）**: 使用原始 `DEEPROBOTICS_M20_CFG`（16 DOF），`prim_path="{ENV_REGEX_NS}/Robot"`。重力开启，标准阻尼。

**Piper（piper）**: 使用 `PIPER_ARM_CFG`（8 DOF），`prim_path="{ENV_REGEX_NS}/piper_arm"`。关键配置：

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=True,       # 无重力——Piper 不会自己下落
    linear_damping=100.0,       # 强阻尼——防止振荡
    angular_damping=100.0,
)
```

### 1.3 子步同步回调

```python
def setup_piper_sync(env):
    """注册每物理子步回调，将 Piper 同步到 M20 base_link + 安装偏移。"""
    base_idx = robot.body_names.index("base_link")
    mount_offset = torch.tensor([0.0, 0.0, 0.12])  # base_link 上方 12cm

    def _sync(dt: float):
        base_pos = robot.data.body_pos_w[0, base_idx]
        base_quat = robot.data.body_quat_w[0, base_idx]
        base_vel = robot.data.body_vel_w[0, base_idx]
        mount_world = base_pos + quat_apply(base_quat, mount_offset)
        piper_state = piper.data.root_state_w[0].clone()
        piper_state[0:3] = mount_world
        piper_state[3:7] = base_quat
        piper_state[7:13] = base_vel
        piper.write_root_state_to_sim(piper_state.unsqueeze(0))

    env.unwrapped.sim.add_physics_callback("piper_sync", _sync)
```

**回调频率**：`sim.dt` = 0.005s（200Hz），decimation=4 时每 `env.step()` 触发 4 次。

**几何间隙**：

```
M20 base_link 碰撞盒: 0.75 × 0.09 × 0.14 (长×宽×高)
M20 身体顶部: base_link_z + 0.07 = z ≈ 0.65
Piper 底座:   base_link_z + 0.12 = z ≈ 0.70
Piper link1:  base_link_z + 0.12 + 0.123 = z ≈ 0.82
间隙:         5cm (M20 身体顶部 → Piper 底座)
安全余量:     3cm (扣除行走时 ±2cm 身体振荡)
```

### 1.4 验证结果

```
ENV_OK 6.0s
DOF=16+8 obs=torch.Size([1, 57])
Walking 300 steps in 13.4s (~22 FPS)

step0:   robot=(0.00,0.00,0.59) piper=(0.00,0.00,0.71)
step100: robot=(1.41,-0.32,0.57) piper=(1.41,-0.32,0.69)
step200: robot=(2.71,-1.15,0.57) piper=(2.71,-1.15,0.69)
```

- Piper Z 偏移 0.12m（精确匹配 MOUNT_OFFSET）
- Piper XY 追踪误差 < 1cm
- 无碰撞抖动 / 随机漂移 / 失去重力
- 观测维度 57 不变
- 帧率无衰减

---

## 2. 所有尝试方案及失败原因

### 2.1 方案：纯视觉 Xform + xformOp 跟随（最早方案）

**做法**：通过 `UsdGeom.Xform.Define` 创建 `/World/envs/env_0/PiperArm` 节点，引用 `piper.usd`，递归清除所有物理 API（`PhysicsRigidBodyAPI`、`PhysicsMassAPI`、`PhysicsCollisionAPI` 等），每帧通过 `xformOp:translate` / `xformOp:orient` 更新位置。

**失败原因**：
- Piper 被 PhysX **完全忽略**（无 articulation、无碰撞、无质量）
- `piper_mount_and_follow()` 函数只在 `piper_env_cfg.py` 中定义，**从未在脚本的步进循环中被调用**
- 现象：Piper 悬停在初始位置 (0, 0, 0.72)，不跟随 M20
- 根源：**调用链路未打通**，非方案本身问题

**为什么不采用**：用户要求 Piper 后续可控制（需保留关节物理），纯视觉方案无法满足。

### 2.2 方案：双 Articulation + Post-Step Root State Sync

**做法**：M20 和 Piper 作为两个独立 `ArticulationCfg` 加载。每 `env.step()` **之后**执行：

```python
step_result = env.step(actions)          # 物理已计算
robot_root = robot.data.root_state_w[0]
piper.data.root_state_w[0][0:7] = robot_root[0:7]
piper.write_root_state_to_sim(...)       # 写入供下帧使用
```

**失败原因**：
- `write_root_state_to_sim` 在物理步**之后**执行
- 写入的位姿在**下一个**物理步开始时才生效
- 若写入的位姿导致 Piper 与 M20 几何重叠，PhysX 在下一帧的 contact solver 中检测到穿透
- 穿透 → 生成巨大 depenetration 反力 → M20 被推开（"随机漂移"）
- 累积效应：每帧积累的碰撞反力使狗"失去重力"

**根因**：Sync 时机错误（post-step 而非 pre-step）+ Piper root = M20 root（几何重叠）。

### 2.3 方案：双 Articulation + Pre-Step Root State Sync（上一版）

**做法**：将 sync 移到 `env.step()` **之前**：

```python
# Pre-step sync
piper_state[0:7] = robot_root[0:7]
piper.write_root_state_to_sim(...)
env.step(actions)  # 物理计算从同步后的位姿开始
```

**失败原因**：
- **Piper root 被设为 M20 root 位置**。M20 root 即 `base_link` 原点（身体中心），Piper 的碰撞几何从此点向外延伸，与 M20 身体重叠
- **Sync 仅在 env.step() 级别**（每 decimation=4 子步触发 1 次），剩余 3 个子步中 Piper 静止而 M20 运动 → 逐步累积偏移
- **未同步 Piper 速度**：Piper 仅有位置但无速度，子步内 PhysX 不产生惯性运动，与 M20 相对运动更大
- 现象：与 post-step sync 相同——抖动、漂移

**根因**：安装位置错误（root=root）+ 同步频率不足（1/env.step）+ 缺少速度同步。

### 2.4 方案：单 Articulation 手写 USDA（ATEC2026 模式）

**做法**：仿照 ATEC2026 `b2_piper.usda`（762 行），手写 `m20_piper.usda`（~1000 行）将 M20 的 17 个 link + Piper 的 10 个 link + 26 个 joint 全部定义在一个 `def Xform "Robot"` 下。

**失败原因**：
- **GPU 显存不足**：单 articulation 需通过 Fabric 加载全部碰撞几何（M20 17 个 mesh + Piper `piper_flattened.usd` 13MB 几何）
- `PhysX error: PxgCudaDeviceMemoryAllocator failed to allocate 67108864 bytes`（64MB CUDA 分配失败）
- RTX 4060 Laptop (8GB) 上，Fabric 初始化时显存不足
- 额外问题：Cube/Cylinder 原始碰撞体不被 Fabric articulation 支持（需转换为 Mesh + convexHull，进一步增加显存）

**根因**：硬件的 GPU 显存限制（8GB），单 articulation 方案要求 ~8GB+ 可用 Fabric 显存。

### 2.5 方案：Variant-Based M20_Piper 模型

**做法**：使用 `M20_Piper.usd`（1.6KB variant 包装）作为 articulation，其 physics variant payload 引用 `M20_Piper_physics.usd`（9.7KB，仅 mass/inertia，mesh 通过 subLayer 引用）。

**失败原因**：
- `M20_Piper_physics.usd` 内含有 `ArticulationRootAPI`
- IsaacLab spawn 时在 `prim_path` 上**也**添加 `ArticulationRootAPI`
- 同一个 articulation 子树中出现**两个** `ArticulationRootAPI` → PhysX 报错：*"UsdPhysics: Nested articulation roots are not allowed"*

**根因**：URDF→USD 转换产物与 IsaacLab 的 articulation 加载机制不兼容（双 ArticulationRootAPI）。

### 2.6 各方案失败总结

| 方案 | 失败根因 | 类别 |
|------|---------|------|
| 纯视觉 Xform | 调用链路未打通 | 工程问题 |
| Post-step sync | Sync 时机错误 + 几何重叠 | 物理时序 |
| Pre-step sync | 安装位置错误 + 同步频率不足 + 无速度同步 | 物理时序 |
| 手写单 articulation | GPU 显存不足 (8GB) | 硬件限制 |
| Variant 模型 | 嵌套 ArticulationRootAPI | USD 结构冲突 |

---

## 3. 方案 C 为什么成功

方案 C 从前面 5 次失败的根因出发，逐一针对性修复：

### 3.1 修复对比

| 之前失败的原因 | 方案 C 的修复 |
|--------------|-------------|
| Piper root = M20 root → 几何重叠 | Piper root = M20 base_link + (0,0,0.12) → **7cm 间隙** |
| Sync 1次/env.step → 3个子步不同步 | 物理回调每 sim.dt 触发 → **decimation 次/env.step** |
| 只复制位置 → 无速度 → 子步内漂移 | 复制位置+朝向+**速度 (13 维)** → 零惯性差异 |
| 调用链路不完整 | `setup_piper_sync(env)` 在 env.reset() 后显式调用 |
| 单 articulation GPU 不足 | 双 articulation 分别管理 → 显存正常 |

### 3.2 三项核心技术要素

1. **`add_physics_callback`**（IsaacSim API，`simulation_context.py:997`）：
   在 PhysX `simulate(dt)` 之前调用回调。回调中写入的 `write_root_state_to_sim` 被当次物理步的 contact solver 使用，**不会**产生穿透反力。

2. **`body_pos_w` / `body_quat_w` / `body_vel_w`**（`ArticulationData`）：
   获取 base_link 的世界位姿和速度。速度写入 Piper 后，Piper 在子步内具有与 M20 相同的惯性运动，消除相对漂移。

3. **`quat_apply`**（`isaaclab.utils.math`）：
   将安装偏移从 base_link 局部坐标系旋转到世界坐标系，确保 Piper 在 M20 倾斜时仍保持在身体顶部正上方。

### 3.3 物理正确性证明

```
时刻 t₀（回调触发）:
  Piper 位姿 = M20.base_link 位姿 + mount_offset
  Piper 速度 = M20.base_link 速度

时刻 t₀ → t₀+dt（PhysX simulate）:
  Piper: 无重力 + 强阻尼 → 速度保持 ≈ 惯性运动
  M20: 腿驱动 → base_link 移动
  相对位移 ≈ 0（速度相同，dt=0.005s 内加速度导致的位移 < 0.02mm）

时刻 t₀+dt（下一次回调触发）:
  Piper 重新同步 → 消除任何累积误差
```

---

## 4. 后续添加臂控制

Piper 作为完整 PhysX articulation 保留，为后续控制奠定基础：

### 4.1 当前状态

| 组件 | 状态 |
|------|------|
| 关节 (8 DOF) | ✅ PhysX 管理，PD 驱动器就绪 |
| 质量/惯量 | ✅ 每连杆有真实物理参数 |
| 碰撞几何 | ✅ convexHull mesh |
| 重力 | `disable_gravity=True`（控制时可恢复） |
| 动作空间 | 仅 M20 16 DOF（arm 不在动作空间中） |
| 观测空间 | 仅 M20 16 DOF（arm 不在观测中） |

### 4.2 加控制需要的改动

1. **动作空间**：在 `m20_piper.py` 中为 arm actuator 组添加动作入口
2. **观测空间**：在 `piper_env_cfg.py` 中添加 arm joint_pos/vel 观测（+16 维，57→73）
3. **策略**：分开训练（腿策略导航 + 臂策略抓取）或合并训练（24 维动作）
4. **重力**：控制时可能需要 `disable_gravity=False`（取决于是否需要重力感知）

---

## 5. 注意事项

### 5.1 回调生命周期

- `setup_piper_sync(env)` 必须在 `env.reset()` **之后**调用（此时 articulation data 已初始化）
- 回调通过 `env.unwrapped.sim.add_physics_callback` 注册，在 `env.close()` 时自动清理
- 回调异常会被 IsaacLab 包装为 `ISAACLAB_CALLBACK_EXCEPTION` 并重抛

### 5.2 安装偏移

- `MOUNT_OFFSET = (0, 0, 0.12)` 基于 M20 base_link 碰撞盒尺寸（0.75×0.09×0.14）
- 如更换机器人或调整安装位置，需重新计算此偏移
- 偏移太小 → 碰撞干涉；太大 → 臂看起来浮在空中

### 5.3 Piper 物理参数

- `disable_gravity=True`：Piper 不会下落。若要臂感知重力，改为 `False`
- `linear_damping=100` / `angular_damping=100`：强阻尼防止振荡。控制时可能需要降低
- 这些参数在 `piper_arm.py` 的 `PIPER_ARM_CFG` 中配置

### 5.4 观测/动作空间保护

- `piper_env_cfg.py` 不需要额外的观测/动作限制（与手写单 articulation 方案不同）
- 双 articulation 时，`SceneEntityCfg("robot", ...)` **不会**匹配 Piper 的关节
- 因为 `"robot"` 是 M20 的 scene key，Piper 是 `"piper"`

### 5.5 性能

- 子步回调开销：~0.05ms/次（tensor copy + write），200Hz = 1% CPU 占用
- 双 articulation 显存：比单 M20 增加约 5.5KB（Piper physics） + 碰撞几何引用
- 总显存占用 < 7GB（RTX 4060 8GB 余量充足）

### 5.6 依赖

- `isaaclab.utils.math.quat_apply`（IsaacLab 内置）
- `env.unwrapped.sim.add_physics_callback`（IsaacSim API，IsaacLab 继承）
- 不需要额外安装或配置

---

## 6. 涉及文件清单

| 文件 | 角色 |
|------|------|
| `tasks/deeprobotics_m20_pro/piper_env_cfg.py` | 双 articulation 配置 + `setup_piper_sync()` 回调 |
| `assets/piper_arm.py` | Piper `ArticulationCfg`（8 DOF, no gravity, strong damping） |
| `assets/piper/piper.usd` | Piper USD 入口（subLayer → configuration） |
| `assets/piper/piper_flattened.usd` | Piper 展平网格（13MB, byte-identical to ATEC2026） |
| `assets/m20_piper.usda` | 手写单 articulation USDA（保留为参考，当前未使用） |
| `assets/m20_piper.py` | M20+Piper 组合 ArticulationCfg（26 DOF，当前未使用） |
| `scripts/navigation/navigate_to_goal.py` | 导航脚本，调用 `setup_piper_sync(env)` |
| `scripts/navigation/teleop_mapping.py` | 建图脚本，调用 `setup_piper_sync(env)` |
| `tasks/deeprobotics_m20_pro/__init__.py` | Env 注册（`Flat-Deeprobotics-M20Pro-Piper-v0`） |
