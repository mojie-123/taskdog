# add_piper.md — M20 Pro 加装 AgileX Piper 机械臂

> **状态**: 视觉安装完成，臂控制待后续实现

---

## 1. 当前方案（已实现并验证）

### 实现方式

Piper 臂通过 `piper_mount_and_follow()` 函数在 env 创建后手动插入到 USD stage 中，作为纯视觉 Xform 图元：

1. 用 `UsdGeom.Xform.Define` 在 `/World/envs/env_0/PiperArm` 创建 Xform prim
2. 用 `AddReference` 引用 `assets/piper/piper.usd`（ATEC2026 的轻量 Piper 模型，物理文件仅 5.5KB）
3. 递归清除所有物理 API（`PhysicsRigidBodyAPI`、`PhysicsMassAPI` 等），使 PhysX 完全忽略
4. 每帧通过 `piper_mount_and_follow()` 读取 M20 的 `root_pos_w` + `root_quat_w`，更新 Piper 的世界坐标和朝向，使其跟随狗身移动

### 效果

- ✅ 机械臂在 Isaac Sim 界面中可见，位于狗背上
- ✅ 随狗身移动和旋转
- ✅ 无物理碰撞干扰，不影响狗的行走
- ✅ 不占用 GPU 显存（纯视觉，PhysX 不管理）
- ❌ 关节不可控（无 articulation）

### 涉及文件

| 文件 | 作用 |
|------|------|
| `tasks/.../piper_env_cfg.py` | Piper env 配置 + `piper_mount_and_follow()` |
| `scripts/navigation/teleop_mapping.py` | 循环中调用 `piper_mount_and_follow()` |
| `scripts/navigation/navigate_to_goal.py` | 同上 |
| `assets/piper/` | ATEC2026 Piper 模型（usd + 贴图 + 物理配置） |

---

## 2. 尝试过的方案及失败原因

### 方案 A：加载为独立 Articulation（失败）

在 `piper_env_cfg.py` 的 `__post_init__` 中创建第二个 `ArticulationCfg`，`prim_path="{ENV_REGEX_NS}/PiperArm"`。

**失败原因**：`piper.usd` 包含 `root_joint`（固定关节连接 world → base_link）。当 Piper 是场景中唯一的 articulation 时（ATEC2026 Task E），PhysX 可以处理这个 root_joint。但当 M20 也作为 articulation 存在时，Piper 的 root_joint 与 PhysX 的多 articulation 管理冲突，报错 `Failed to create articulation at .../root_joint`。

### 方案 B：加载为 RigidObjectCfg（失败）

将 Piper 作为 `RigidObjectCfg` 放在独立 prim path。

**失败原因**：`piper.usd` 包含多个子 link（base_link, link1~link8, gripper_base 等），不满足 RigidObject 要求"单一刚体"的条件。报错 `Found multiple rigid bodies`。

### 方案 C：纯视觉 Xform + 无物理（当前方案，成功）

将 Piper 作为纯视觉 `UsdGeom.Xform` 加载，清除所有物理 API。这是唯一可行的方案。代价是 Piper 关节不可控、无物理交互。

### 方案 D：创建组合 USD 文件（部分完成）

参考 ATEC2026 的 `b2_piper.usda`（761 行，将 B2 的 12 个腿关节 + Piper 的 10 个臂关节全部定义在单一 USD 文件中，subLayer 引用 Piper 扁平化视觉网格）。

为 M20 生成了 30KB 的 `m20_piper.usda`（27 links + 26 joints），但无法被 Isaac Lab 加载——因为手工编写的 USD 缺少 PhysX 所需的高精碰撞几何引用（每个连杆都需要 `</Flattened_Prototype_X>` 形式的碰撞网格引用，这些引用来自 `piper_flattened.usd` 但 M20 的身体碰撞体不在该文件中）。

### 方案 E：GUI 导出组合文件（部分完成）

在 Isaac Sim GUI 中将 Piper 拖入 M20 的 base_link 下，用 File → Save As 导出为 `m20_piper_gui.usda`（22KB）。

**问题**：导出的文件是"叠加层"（over），需要 M20.usd 作为 base layer 存在。Isaac Lab 的 articulation 扫描器期望`def`（独立定义）而非`over`。文件缺少 `ArticulationRootAPI`，补上后仍因 USD 组合机制的差异无法被解析为单一 articulation。

### 方案 F：关 Fabric 用 CPU PhysX（失败）

将 `use_fabric=False` 关闭 GPU 物理。

**失败原因**：Isaac Lab 2.3.2 深度依赖 Fabric，关闭后物理引擎崩溃，机器人不动。

---

## 3. 后续：添加臂控制需要做的事

### 3.1 USD 层面

真正需要的是类似 `b2_piper.usda` 的**自包含组合 articulation 文件**。这需要：

- 在 Isaac Sim GUI 中将 M20 和 Piper 组装好
- 用 `File → Export → Flatten` 展平所有引用到一个文件中
- 确认导出的文件具有 `IsaacRobotAPI` + `ArticulationRootAPI` 且所有关节/连杆是 `def`（非 `over`）
- Isaac Lab 能将其解析为单一 articulation（`r.num_joints` 应返回 26）

### 3.2 动作空间扩展

当前动作空间是 16 DOF（12 腿 + 4 轮）。需要增加 8 DOF（6 臂关节 + 2 夹爪）：

```python
self.actions.joint_pos_leg.joint_names = leg_joint_names   # 12
self.actions.joint_vel_wheel.joint_names = wheel_joint_names # 4
self.actions.joint_pos_arm.joint_names = arm_joint_names    # 8 (新增)
```

### 3.3 观测空间扩展

当前 57 维。需增加臂关节状态（+16 维 = 73 维）和末端相机（可选）：

```python
self.observations.policy.arm_joint_pos = ObsTerm(...)  # 8
self.observations.policy.arm_joint_vel = ObsTerm(...)  # 8
```

### 3.4 训练

两种策略（ATEC2026 选择分开训练）：

| 方式 | 说明 |
|------|------|
| **分开训练** | 腿策略（现有 `model_4999.pt`）管导航，臂策略管抓取。导航到目标后切换 |
| **合并训练** | 一个 24 维动作网络同时控制腿和臂 |

### 3.5 脚本修改

`navigate_to_goal.py` 加入状态机：
```
导航阶段 → 到达目标 0.5m 内 → 切换 arm_policy → 抓取香蕉
```

### 3.6 已有基础

- ✅ `assets/piper/` 中有完整轻量 Piper 模型（ATEC2026）
- ✅ `assets/piper_arm.py` 中有正确的 articulation 配置（关节名/限位/初始位姿）
- ✅ `assets/m20_piper_gui.usda` 中有 GUI 导出的组合文件可供调试
- ✅ 所有脚本已支持 Piper env ID
