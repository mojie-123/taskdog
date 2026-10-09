# MuJoCo 版项目工作流：从建图到导航、抓取、再导航

这份文档只讲本项目的 **MuJoCo 实现**。IsaacLab 版本仍然保留，但这里只在需要说明“代码从哪里迁移而来”时提到它。

本文按数据实际流动的顺序解释以下问题：

1. MuJoCo 建图是怎样完成的；
2. ROS 2 在本项目中负责什么；
3. Nav2 如何收到地图和机器人位姿，又如何把速度送回 MuJoCo；
4. MoveIt 如何收到关节状态、计算机械臂路径，再把路径交还给 MuJoCo；
5. 为什么运行完整任务需要终端 1 和终端 2；
6. 每个相关文件在整条链路中的位置。

---

## 1. 先记住一句话

本项目不是“ROS 2 直接控制 MuJoCo”，而是：

> MuJoCo 是机器人和物理世界的唯一真值来源；Nav2 负责算底盘应该怎样走，MoveIt 负责算机械臂应该怎样动；Python 主程序把这些高层结果转换成 MuJoCo 能执行的控制目标。

因此三个系统的职责边界是：

| 系统 | 负责 | 不负责 |
| --- | --- | --- |
| MuJoCo | 物理仿真、碰撞、接触、相机、机器人真实位姿和关节状态、执行力矩 | 不做全局路径规划 |
| Nav2 | 读取静态地图、计算底盘路径、跟踪路径、输出 `vx/wz` | 不直接控制轮子，不执行机械臂动作 |
| MoveIt | 机械臂 IK、碰撞检查、OMPL 路径规划、笛卡尔路径规划 | 不直接给 MuJoCo 施加力矩，本项目也没有启动 `ros2_control` |
| `navigate_mujoco_moveit.py` | 编排“导航→抓取→导航→放置”的状态机，连接所有模块 | 它不是 ROS 2 server，也不是物理引擎 |

完整的数据闭环可以简化为：

```text
MuJoCo 真实位姿
    │
    ├── JSON ──> ROS 2 bridge ──> /tf、/odom ──> Nav2
    │                                            │
    │                                            └── /cmd_vel
    │                                                  │
    └<─ JSON <── ROS 2 bridge <────────────────────────┘
          │
          └── vx/wz ──> 57维策略观测 ──> 行走策略 ──> 16维动作 ──> MuJoCo

MuJoCo 机械臂关节状态
    │
    ├── JSON ──> MoveIt bridge ──> /joint_states、MoveIt services
    │                                      │
    └<─ JSON <── 机械臂 JointTrajectory <──┘
          │
          └── 轨迹加密 ──> env.set_arm_target() ──> MuJoCo PD ──> 真实机械臂运动
```

---

## 2. 整个项目分成两个阶段

### 2.1 离线建图阶段

运行 `teleop_mapping_mujoco.py`，手动驾驶机器狗走遍场景。MuJoCo 雷达射线得到世界坐标下的击中点，程序把这些点融合成二维占用栅格，并输出 Nav2 可以读取的 PGM/YAML。

这一阶段 **不使用 ROS 2、Nav2 或 MoveIt**，所以单独建图时不需要启动终端 1。

```text
键盘按住 W/S/A/D/Q/E
        │
        v
速度命令 vx/vy/wz
        │
        v
57维策略观测 ──> rsl_rl 行走策略 ──> 16维腿/轮动作 ──> MuJoCo
                                                        │
MuJoCo mj_multiRay <── 雷达位姿 <───────────────────────┘
        │
        v
世界坐标击中点 ──> OccupancyGrid ──> NPZ ──> PGM + YAML
                         └──────────> 3D 点云 NPY
```

### 2.2 在线执行阶段

先在终端 1 启动 Nav2，若使用 MoveIt 则同时启动 `move_group`。再在终端 2 启动 MuJoCo 主状态机。

```text
终端 1：长期运行的 ROS 2 服务端
    map_server + Nav2 各 server + lifecycle manager
    + move_group + robot_state_publisher（MoveIt 版本）

终端 2：任务本体
    MuJoCo + 行走策略 + 状态机
    + ROS 2 bridge 子进程
    + MoveIt bridge 子进程
    + AnyGrasp 子进程
```

---

## 3. 建图链路逐步解释

### 3.1 入口：`teleop_mapping_mujoco.py`

文件：`custom_envs/scripts/navigation/teleop_mapping_mujoco.py`

它完成五件事：

1. 创建 `MuJoCoEnv` 并加载指定 MJCF 场景；
2. 加载训练好的 rsl_rl 行走策略；
3. 用按键生成期望速度 `vx/vy/wz`；
4. 周期性调用 MuJoCo 雷达并融合占用栅格；
5. 保存 NPZ、点云 NPY，并自动转成 Nav2 PGM/YAML。

键盘命令并不直接修改机器狗坐标。它们只是进入策略观测，行走策略再决定 12 个腿部位置目标和 4 个轮子速度目标。

```text
W/S -> cmd_vx
A/D -> cmd_vy
Q/E -> cmd_wz
          │
          v
build_policy_obs(...) -> policy(...) -> env.step(action)
```

`MuJoCoTeleopViewer` 自己接管 GLFW 按下和松开事件，因此：

- 按住键时才持续运动；
- 松开键会立即从速度命令中移除；
- W/S/A/D/Q/E 不再进入 MuJoCo 默认 viewer 的渲染快捷键处理。

### 3.2 MuJoCo 环境：`env.py`

文件：`custom_envs/mujoco/env.py`

`MuJoCoEnv` 是所有 MuJoCo 入口共同依赖的底层封装：

- `MjModel.from_xml_path()` 加载 `scene.xml` 或 `--scene` 指定的场景；
- `reset()` 设置机器人根位姿和所有关节初值；
- `get_robot_state()` 读取位置、四元数、速度、关节角和偏航角；
- `step(action)` 把策略动作解码成腿/轮目标，叠加机械臂和夹爪 PD，再推进物理；
- `set_arm_target()`、`set_gripper_target()` 接收状态机或 MoveIt 返回的目标；
- `render_camera()` 输出腕部 RGB 和深度，供 AnyGrasp 使用；
- 物理 weld 接口用于抓取阶段固定底盘。

这里是“真正执行动作”的位置。Nav2 和 MoveIt 都不会绕过它直接改 MuJoCo 状态。

### 3.3 策略观测：`obs_builder.py`

文件：`custom_envs/mujoco/obs_builder.py`

它把 MuJoCo 状态整理为训练时一致的 57 维观测：

```text
3  机体角速度
3  机体系重力方向
3  速度指令 vx/vy/wz
16 关节位置相对值
16 关节速度
16 上一步动作
--
57 维
```

这层非常重要：如果关节顺序、缩放或四元数约定与 IsaacLab 训练时不一致，即使 Nav2 输出正确速度，机器狗也可能不按预期运动。

### 3.4 策略加载：`policy_loader.py`

文件：`custom_envs/mujoco/policy_loader.py`

它读取：

- `model_4999.pt` 等 checkpoint；
- checkpoint 同一运行目录下的 `params/agent.yaml`；
- checkpoint 中网络输入、输出维度。

它构造一个最小 `_MockEnv` 让 `rsl_rl.OnPolicyRunner` 还原网络，然后返回可直接调用的推理函数 `policy(obs_tensor)`。这个 mock 环境不进行物理仿真，只是满足 rsl_rl 加载器的接口要求。

### 3.5 雷达：`lidar.py`

文件：`custom_envs/mujoco/lidar.py`

当前代码的默认扫描配置是：

| 参数 | 当前值 |
| --- | --- |
| 安装位置 | `base_link` 局部坐标 `(0.30, 0.0, 0.55)` m |
| 垂直视场 | `-7°` 到 `52°` |
| 垂直通道 | 40 |
| 水平分辨率 | 1° |
| 水平方向数 | 360 |
| 每帧射线 | `40 × 360 = 14400` |
| 最大距离 | 70 m |

当前构造函数已经使用 14,400 条射线/帧，对应的关键默认值是：

```python
channels=40
horizontal_resolution_deg=1.0
```

雷达使用 `mujoco.mj_multiRay()` 一次批量投射整帧射线。机器人自身碰撞几何被移到专用 ray group 并从查询中排除，避免地图里出现机器人自己的腿、轮和机械臂。

`scan()` 返回的是 **世界坐标系** 数据：

- `origin_w`：雷达真实世界位置；
- `directions_w`：全部世界系射线方向；
- `ranges`：全部射线距离，未命中为负；
- `hits_w`：只保留有效命中的世界系三维点；
- `geom_ids`：每条射线命中的 MuJoCo geom id。

### 3.6 栅格融合：`occupancy_grid.py`

文件：`custom_envs/utils/occupancy_grid.py`

内部地图是二维 log-odds 数组：

```text
grid[row, col]
row <-> 世界 Y
col <-> 世界 X
origin = (世界最小 X, 世界最小 Y)
```

坐标转换是：

```python
col = floor((world_x - origin_x) / resolution)
row = floor((world_y - origin_y) / resolution)
```

每帧融合分为两步：

1. 从真实雷达原点到每个击中点做 Bresenham 射线，将经过的格子更新为“更可能空闲”；
2. 只把高度高于 `obstacle_min_z` 的击中端点更新为“更可能占用”。

这样地面点可以帮助清出自由区域，但不会把整个地面标成障碍物。

地图物理范围由下式决定：

```text
宽度（米） = width_cells × resolution
高度（米） = height_cells × resolution
```

建图入口目前用一个 `--grid_size` 同时作为宽和高。默认 `400 × 0.05 m = 20 m`，即 20 m × 20 m。

如果地图要覆盖世界坐标 `[0, 30] × [0, 30]`，可以使用：

```bash
--grid_size 600 --resolution 0.05 --origin_x 0 --origin_y 0
```

增大栅格数会增加内存、融合和导出开销；增大 `resolution` 会降低精度。门框等窄结构场景一般优先保持 0.05 m，再增加 `grid_size`。

### 3.7 地图文件分别是什么

假设 `--map_name map_house_mujoco`，将得到：

| 文件 | 内容 | 谁使用 |
| --- | --- | --- |
| `map_house_mujoco.npz` | log-odds 栅格、origin、resolution、width/height、命中计数 | 项目自己的查看/调试工具；Nav2 不直接读取 |
| `map_house_mujoco_cloud.npy` | 体素降采样后的 `(N,3)` 世界系雷达击中点 | `view_3d.py`，用于检查雷达是否打在正确位置 |
| `map_house_mujoco_nav2.pgm` | 黑/白/灰二维占用图 | Nav2 `map_server` 通过 YAML 间接读取 |
| `map_house_mujoco_nav2.yaml` | PGM 文件名、分辨率、地图 origin、阈值 | 终端 1 的 `map:=...` 参数 |

检查雷达定位是否正确时，优先看 `_cloud.npy`，因为它保留三维世界坐标；再看 NPZ/PGM 是否正确压成二维障碍。

### 3.8 PGM 为什么需要上下翻转

文件：`custom_envs/scripts/navigation/convert_map.py`

项目内部 `row=0` 是世界最小 Y；图像文件的第 0 行显示在顶部。导出 PGM 时因此执行一次 `np.flipud()`。

只翻转图像内容，不交换 X/Y，也不修改 YAML origin。因此导航目标仍然按正常的 `(world_x, world_y)` 输入，不能交换 row/col。

---

## 4. ROS 2 基础概念，以及它们在本项目中的对应物

ROS 2 在本项目里是 **进程之间的通信层和组件运行框架**。它让 Nav2、MoveIt 和桥接程序可以在不同进程中交换标准消息。

### 4.1 Node：节点

节点是一个有名字的 ROS 2 参与者，可以发布/订阅话题、提供/调用服务、提供/调用 action。

本项目常见节点：

```text
/map_server
/planner_server
/controller_server
/bt_navigator
/velocity_smoother
/mujoco_moveit_bridge
/move_group
/moveit_robot_state_publisher
```

一个系统进程可以包含一个或多个节点。节点名不等于 Python 文件名。

### 4.2 Topic：话题

话题适合连续、异步、流式数据。发布者不等待订阅者返回。

本项目主要话题：

| 话题 | 数据 | 发布者 | 订阅者 |
| --- | --- | --- | --- |
| `/map` | 静态 `OccupancyGrid` | `map_server` | global/local costmap |
| `/tf_static` | `map -> odom` | ROS bridge | Nav2、TF consumers |
| `/tf` | `odom -> base_link`，以及 MoveIt 机器人 link TF | ROS bridge、robot_state_publisher | Nav2、MoveIt |
| `/odom` | 机器人位姿和速度 | ROS bridge | Nav2 controller、velocity smoother 等 |
| `/cmd_vel_nav` | 未平滑的底盘速度 | controller/behavior server | velocity smoother |
| `/cmd_vel` | 平滑后的最终速度 | velocity smoother | ROS bridge |
| `/joint_states` | arm/gripper 当前关节角 | MoveIt bridge | move_group、robot_state_publisher |

### 4.3 Service：服务

服务是一次“请求—响应”，适合较短的计算。

MoveIt bridge 调用的主要服务：

```text
/compute_ik
/plan_kinematic_path
/compute_cartesian_path
/apply_planning_scene
```

主程序发一个规划请求，等待 MoveIt 返回成功/失败和轨迹。

### 4.4 Action：动作

Action 用于耗时、可反馈、可取消的任务。Nav2 的“去某个目标点”不是一个普通 topic，而是 `NavigateToPose` action。

`BasicNavigator.goToPose()` 做的事情是向 `/navigate_to_pose` action server 发送目标。随后可以轮询完成状态或取消任务。

### 4.5 TF：带时间的坐标系关系

Nav2 必须知道机器人在地图哪里。本项目没有运行 AMCL，而是直接把 MuJoCo 真值发布到 TF：

```text
map --静态单位变换--> odom --MuJoCo实时位姿--> base_link
```

因为 `map -> odom` 是单位变换，所以 MuJoCo 世界坐标、ROS `map` 坐标和 `odom` 坐标的原点/方向被视为一致。

这意味着：

- 场景中物体的 MuJoCo `(x,y)`；
- 建图文件的世界 `(x,y)`；
- Nav2 goal 的 `(x,y)`；

必须采用同一套坐标。

### 4.6 Lifecycle：生命周期节点

Nav2 server 是生命周期节点，典型状态为：

```text
unconfigured -> inactive -> active -> finalized
```

`lifecycle_manager_navigation` 负责启动时统一 configure/activate。只有 `active` 后，bridge 才报告 ready 并允许主程序发目标。

终端 2 退出时只销毁它自己创建的 `BasicNavigator` 客户端节点，不会再对终端 1 的整个 Nav2 栈调用 lifecycle shutdown。因此正常退出终端 2 后，终端 1 仍保持 `active`，可以直接再次运行终端 2。

### 4.7 DDS：ROS 2 的发现和传输

ROS 2 不需要 ROS 1 的 `roscore`。节点通过 DDS 自动发现彼此。两个终端必须使用兼容的 RMW/DDS 配置并处于同一个 `ROS_DOMAIN_ID`，否则看不到彼此的话题和服务。

---

## 5. Nav2 在本项目中怎样工作

### 5.1 终端 1 启动了哪些组件

文件：`custom_envs/launch/nav2_isaac.launch.py`

名字中保留了 `isaac`，但它现在同时用于 IsaacLab 和 MuJoCo；这是历史命名，不代表 MuJoCo 会启动 Isaac Sim。

该 launch 启动：

| 组件 | 作用 |
| --- | --- |
| `map_server` | 读取 `map:=...yaml` 和 PGM，发布 `/map` |
| `planner_server` | 在 global costmap 上用 Navfn/A* 计算全局路径 |
| `controller_server` | 用 Regulated Pure Pursuit 跟踪路径，输出 `cmd_vel_nav` |
| `behavior_server` | Spin、BackUp、Wait 等恢复行为，也输出到 `cmd_vel_nav` |
| `velocity_smoother` | 对 `cmd_vel_nav` 做速度/加速度约束，输出最终 `/cmd_vel` |
| `smoother_server` | 提供几何路径平滑服务；当前默认 BT 未显式调用 `SmoothPath` |
| `bt_navigator` | 执行 NavigateToPose 行为树，协调规划、控制和恢复 |
| `waypoint_follower` | 多航点功能，本主流程通常不使用 |
| `lifecycle_manager_navigation` | 自动激活上述生命周期节点 |

注意两个名字容易混淆：

- `smoother_server` 平滑的是 **路径几何形状**；
- `velocity_smoother` 平滑的是 **速度命令**。

当前默认 NavigateToPose 行为树会周期性重新规划，但没有显式调用 `smoother_server`。启动它不等于全局路径一定被几何平滑。

### 5.2 Nav2 的地图从哪里来

真正决定 Nav2 地图的是终端 1：

```bash
map:=$(pwd)/custom_envs/maps/map_house_mujoco_nav2.yaml
```

`navigate_mujoco.py` 和 `navigate_mujoco_moveit.py` 的 `--map` 参数仅保留旧命令兼容；脚本会检查文件是否存在，但 **不会把它加载到 Nav2**。

地图流向：

```text
YAML -> map_server -> /map -> StaticLayer -> global/local costmap
```

目前 global 和 local costmap 都只配置了 `static_layer + inflation_layer`，没有运行激光 `obstacle_layer`。因此在线导航时不会把 MuJoCo 雷达持续发给 Nav2：

- 墙和门框来自离线静态地图；
- 移动物体不会自动更新到 costmap；
- 局部 costmap 虽是 rolling window，但其障碍内容仍来自静态地图。

### 5.3 机器人定位从哪里来

文件：

- `custom_envs/utils/ros2_bridge.py`
- `custom_envs/utils/ros2_bridge_process.py`

终端 2 每个仿真循环读取 MuJoCo 真值，然后发送一行 JSON：

```json
{"type":"pose","pos":[x,y,z],"quat":[w,x,y,z],"lin_vel":[...],"ang_vel":[...]}
```

Python 3.10 bridge 子进程把它发布为：

- `/tf` 中的 `odom -> base_link`；
- `/odom` 中的位姿和速度。

Nav2 因而不需要 AMCL 粒子滤波，也不存在激光定位误差；这里使用的是仿真 ground truth。

### 5.4 从 goal 到机器狗运动的完整闭环

```text
1. 主程序 bridge.send_goal(x, y)
2. JSON 到 ros2_bridge_process.py
3. BasicNavigator.goToPose() 发送 NavigateToPose action
4. bt_navigator 请求 planner_server 生成全局路径
5. controller_server 用 TF 找到当前 base_link，跟踪路径
6. controller_server 发布 /cmd_vel_nav
7. velocity_smoother 发布 /cmd_vel
8. bridge 订阅 /cmd_vel，将 vx/wz 用 JSON 返回主程序
9. 主程序把 vx/wz 填入 57 维行走观测
10. rsl_rl policy 输出 16 维腿/轮动作
11. MuJoCoEnv.step() 用 PD/速度控制执行
12. 新的 MuJoCo 位姿再次发布给 Nav2，形成闭环
```

因此日志里看到 Nav2 `cmd_vel vx=0.5`，不代表 MuJoCo 轮子被直接赋值 0.5。它只是行走策略的目标速度输入；机器人实际速度取决于策略、物理接触和惯性。

### 5.5 `nav2_params.yaml` 控制哪些行为

文件：`custom_envs/config/nav2_params.yaml`

主要分组：

- `map_server`：地图 YAML；launch 会用命令行 `map:=` 覆盖它；
- `global_costmap`：全局路径使用的静态地图、footprint、膨胀；
- `local_costmap`：控制器和碰撞预测使用的机器人周围窗口；
- `planner_server`：Navfn A*、是否允许未知区、目标容差；
- `controller_server`：RPP 前瞻距离、曲率减速、碰撞预测、目标/进度检查；
- `velocity_smoother`：最终速度和加速度限制；
- `behavior_server`：恢复行为的速度、碰撞向前模拟时间；
- `lifecycle_manager`：哪些节点要自动激活。

`footprint` 是机器人硬碰撞外廓；`inflation_radius` 是障碍附近的软代价扩张区。增大膨胀并不会物理增大 MuJoCo 机器人，但会让规划更倾向于离障碍远一些，过大也可能把门完全封死。

---

## 6. MoveIt 在本项目中怎样工作

### 6.1 MoveIt 为什么也放在终端 1

文件：`custom_envs/launch/nav2_moveit_mujoco.launch.py`

它同时 include：

```text
nav2_isaac.launch.py
moveit_mujoco.launch.py
```

所以终端 1 同时提供底盘导航 server 和机械臂规划 server。

### 6.2 `moveit_mujoco.launch.py` 启动什么

文件：`custom_envs/launch/moveit_mujoco.launch.py`

它启动：

- `move_group`：MoveIt 的 IK、规划、碰撞场景服务端；
- `moveit_robot_state_publisher`：根据 URDF 和 `/joint_states` 发布机械臂 link TF。

关键配置是：

```text
allow_trajectory_execution = false
```

本项目没有启动 `ros2_control`，MoveIt 不拥有控制器。MoveIt 只返回轨迹数据。

### 6.3 MoveIt 使用的机器人模型

| 文件 | 作用 |
| --- | --- |
| `custom_envs/assets/m20_piper_single/M20_Piper_moveit.urdf` | MoveIt 的机器人 link、joint、visual、collision 和 `grasp_tcp` 模型 |
| `custom_envs/config/moveit/M20_Piper_moveit.srdf` | `piper_arm` 规划组、gripper 组、home 状态、允许忽略的相邻 link 碰撞 |
| `custom_envs/config/moveit/kinematics.yaml` | KDL IK 插件和超时/搜索参数 |
| `custom_envs/config/moveit/joint_limits.yaml` | MoveIt 规划使用的速度/加速度限制 |
| `custom_envs/config/moveit/ompl_planning.yaml` | OMPL 与 RRTConnect 配置 |

MuJoCo 的 MJCF 和 MoveIt 的 URDF 是两份不同格式的模型。项目用 `verify_moveit_fk.py` 对比相同关节角下的 FK，避免两份模型几何不一致。

### 6.4 为什么需要第二个 bridge

文件：

- `custom_envs/utils/moveit_bridge.py`：MuJoCo 主进程一侧；
- `custom_envs/utils/moveit_bridge_process.py`：ROS 2 Python 3.10 一侧。

主进程持续发送 arm/gripper 当前关节角；子进程以 50 Hz 发布 `/joint_states`。规划请求则使用 JSON RPC 包装：

```text
主进程方法                  ROS 2 service
compute_ik()             -> /compute_ik
plan_to_pose()/joint()   -> /plan_kinematic_path
compute_cartesian_path() -> /compute_cartesian_path
apply/attach/detach      -> /apply_planning_scene
```

MoveIt 返回 `JointTrajectory` 后，bridge 转成普通 JSON。主程序按 joint name 排序出 joint1..joint6，并在相邻轨迹点之间插值加密，最后每个仿真 step 调用 `env.set_arm_target()`。

### 6.5 抓取候选怎样进入 MoveIt

完整抓取几何链路：

```text
wrist_cam RGB/depth
    -> 点云（camera optical frame）
    -> grasp_worker.py / AnyGrasp
    -> 掌心位置、抓取旋转、宽度、深度、分数
    -> mujoco_moveit_frames.py
    -> camera optical -> base_link -> grasp_tcp
    -> MoveIt IK + 碰撞检查 + OMPL pre-grasp path
    -> MoveIt Cartesian near-grasp path
    -> mujoco_local_servo.py 完成最后 1~2 cm
    -> CLOSE 接触与恒力夹持
```

`mujoco_moveit_frames.py` 在扫描时保存实际 MuJoCo 相机变换，避免机械臂已经移动后再用错误的“当前相机位姿”解释旧深度图。

`mujoco_local_servo.py` 用 MuJoCo 当前模型的 Jacobian 做阻尼最小二乘控制，完成 MoveIt 停下后到接触点之间很短的一段。这样最后接触阶段使用的运动学与物理仿真是同一份 MuJoCo 模型。

### 6.6 MoveIt PlanningScene 的当前限制

`moveit_bridge_process.py` 的 `default_scene_objects()` 目前硬编码：

- 原始 `scene.xml` 的 table1；
- 原始 `scene.xml` 的 table2；
- 一个 40 m × 40 m 的 floor。

它没有自动解析任意 MJCF，也没有完整镜像 `scene_house.xml` 的墙、门框和家具。因此使用 House 场景时：

- Nav2 会按照 House 静态地图避开门和墙；
- MoveIt 的机械臂规划场景却仍是原始双桌近似；
- 机械臂路径是否与 House 家具碰撞，最终仍由 MuJoCo 物理暴露出来。

这是当前实现的重要边界。若以后希望 House 抓取也有准确的机械臂碰撞规划，应把 PlanningScene 几何改成按场景配置加载，或从 MJCF 提取静态 geom 后转换成 MoveIt `CollisionObject`。

---

## 7. `navigate_mujoco_moveit.py` 的完整状态机

文件：`custom_envs/scripts/navigation/navigate_mujoco_moveit.py`

推荐使用这个入口完成“导航→抓取→再导航→放置”。主循环每帧都做三件公共工作：

1. 从 MuJoCo 读取真值状态；
2. 更新 Nav2 bridge 的位姿和 MoveIt bridge 的关节状态；
3. 根据当前状态决定底盘速度、机械臂目标、夹爪目标，然后只调用一次 `env.step()`。

状态流：

```text
NAV
 -> ALIGN_YAW_1
 -> PAN_XY
 -> ARM_INIT
 -> SCAN
 -> GRASP_PLAN
 -> MOVEIT_PRE_GRASP
 -> MOVEIT_APPROACH
 -> LOCAL_REACH
 -> CLOSE
 -> LIFT
 -> PAN_NEG_X
 -> NAV2_DEST
 -> ALIGN_YAW_2
 -> PAN_DES_X
 -> ROTATE
 -> PUT_DOWN
 -> DONE
```

各状态作用：

| 状态 | 作用 |
| --- | --- |
| `NAV` | 让 Nav2 到第一个目标附近 |
| `ALIGN_YAW_1` | 用项目自己的闭环把底盘朝向精确对齐到抓取方向 |
| `PAN_XY` | 利用全向行走策略做最终二维位置微调 |
| `ARM_INIT` | 固定底盘，机械臂到侧向扫描姿态，打开夹爪 |
| `SCAN` | 累积腕部相机 RGB/depth，保存扫描时坐标变换 |
| `GRASP_PLAN` | AnyGrasp 推理、去平面和候选过滤 |
| `MOVEIT_PRE_GRASP` | 对候选做 IK/碰撞/OMPL 规划并执行到预抓取位姿 |
| `MOVEIT_APPROACH` | 执行 MoveIt 笛卡尔路径，停在接触前 |
| `LOCAL_REACH` | MuJoCo Jacobian 伺服最后短距离，并监控接触/力 |
| `CLOSE` | 两指接触检测、慢速闭合、切换恒力夹持 |
| `LIFT` | 附着物体碰撞盒、笛卡尔抬升、机械臂收回，监控滑落 |
| `PAN_NEG_X` | 搬运前先侧移离开桌面 |
| `NAV2_DEST` | 向目的地发送第二个 Nav2 goal |
| `ALIGN_YAW_2` | 到目的地后对齐放置朝向 |
| `PAN_DES_X` | 精调放置位置 |
| `ROTATE` | 机械臂转到放置侧 |
| `PUT_DOWN` | 放置并结束 |

Nav2 只覆盖 `NAV` 和 `NAV2_DEST`。`ALIGN_*`、`PAN_*` 是主状态机自己的精调控制，因为 Nav2 的到达容差和最终姿态不能满足抓取/放置所需的厘米级相对位置。

抓取状态中使用 MuJoCo equality weld 固定底盘，而不是每帧强行改根位姿。这样碰撞约束仍由物理求解器处理。

### 非 MoveIt 入口

文件：`custom_envs/scripts/navigation/navigate_mujoco.py`

它同样使用 Nav2 和 AnyGrasp，但机械臂部分使用 `arm_ik_mujoco.py` 中的自定义 IK/几何逻辑，不调用 MoveIt。它是保留的旧方案和对照实现；新的完整流程优先使用 `navigate_mujoco_moveit.py`。

---

## 8. 为什么要两个终端

两个终端不是 ROS 2 的硬性规定，而是本项目有意采用的进程职责分离。

### 8.1 终端 1 是“长期服务端”

它启动：

- Nav2 的地图、规划、控制、行为树和生命周期节点；
- MoveIt 版本中还启动 `move_group` 和 `robot_state_publisher`。

这些服务应在 MuJoCo 主程序之前完成配置并进入 `active`。

### 8.2 终端 2 是“仿真和任务客户端”

它拥有：

- MuJoCo 物理状态；
- PyTorch/rsl_rl 行走策略；
- 完整任务状态机；
- 向 Nav2 发 goal 的客户端；
- 向 MoveIt 发规划请求的客户端；
- AnyGrasp 推理子进程。

终端 2 可以结束并重新运行，而不需要每次关闭终端 1。终端 2 退出时如果仍有 Nav2 goal，会先取消自己的活动任务，再销毁自己的客户端节点；终端 1 的 Nav2 lifecycle 节点仍保持 active。

### 8.3 Python 3.11 与 ROS Humble Python 3.10 的边界

MuJoCo/PyTorch 主环境可能运行 conda Python 3.11，而 Ubuntu 上安装的 ROS 2 Humble `rclpy` C 扩展是为系统 Python 3.10 编译的，不能安全地直接 import 到 3.11。

因此两个 bridge 都采用：

```text
conda Python 主程序
    -> subprocess.Popen('/usr/bin/python3.10', ...)
    -> 换行分隔 JSON stdin/stdout
    -> Python 3.10 子进程 import rclpy
```

这就是 `ros2_bridge.py`/`ros2_bridge_process.py` 和 `moveit_bridge.py`/`moveit_bridge_process.py` 成对存在的原因。

---

## 9. `source /opt/ros/humble/setup.bash` 到底做了什么

`source` 是让当前 shell 执行这个脚本，并把它设置的环境变量保留在当前终端。它不会启动 ROS 节点，也不会启动所谓“ROS master”。

它主要设置或扩展：

- `PATH`：找到 `ros2` 和 ROS 可执行程序；
- `AMENT_PREFIX_PATH`：找到已安装 ROS package；
- `CMAKE_PREFIX_PATH`：构建/查找 package；
- `PYTHONPATH`：找到 ROS Python 模块；
- `LD_LIBRARY_PATH`：找到 `rclpy`、RMW、消息 typesupport 等原生库；
- DDS/RMW 相关资源和 package 索引环境。

为什么两个终端都要 source：

- 终端 1 直接运行 `ros2 launch`，显然必须找到 ROS 2 和 Nav2/MoveIt package；
- 终端 2 虽然主 Python 不直接 import `rclpy`，但它创建的 `/usr/bin/python3.10` bridge 子进程会继承当前环境，需要找到 RMW、typesupport、package index 和 MoveIt/Nav2 资源。

每个新终端都有独立环境，因此在终端 1 source 不会自动影响终端 2。

如果另有自己编译的 ROS workspace，还要继续 source 对应 overlay，例如：

```bash
source /opt/ros/humble/setup.bash
source /path/to/your_ws/install/setup.bash
```

后 source 的 workspace 会叠加并可覆盖同名 package。

---

## 10. House 场景从建图到执行的推荐命令

以下命令都在仓库根目录 `/home/mojie/taskdog` 执行。

### 10.1 建图

建图不需要 ROS 2 终端：

```bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/teleop_mapping_mujoco.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --scene custom_envs/mujoco/scene_house.xml \
    --robot_x 1.6 \
    --robot_y 1.6 \
    --map_name map_house_mujoco \
    --origin_x 0 \
    --origin_y 0 \
    --grid_size 400 \
    --resolution 0.05 \
    --render
```

这里地图覆盖 `[0,20) × [0,20)` m。若新场景更大，要一起调整 `grid_size` 和 origin，确保场景全部位于覆盖范围内。

### 10.2 终端 1：启动 Nav2 + MoveIt

```bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog

ros2 launch custom_envs/launch/nav2_moveit_mujoco.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_house_mujoco_nav2.yaml
```

### 10.3 终端 2：运行完整任务

```bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_mujoco_moveit.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --scene custom_envs/mujoco/scene_house.xml \
    --robot_x 1.6 \
    --robot_y 1.6 \
    --goal 4.35 3.65 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --object cube \
    --destination 12.0 12.0 \
    --render
```

注意：checkpoint 路径应以本机实际安装为准；上面使用的是本项目当前常用路径。

建图与执行必须保持一致：

- 同一个 `--scene`；
- 一致的世界坐标约定；
- 机器人初始 `(x,y,yaw)` 与地图坐标匹配；
- 终端 1 加载该场景对应的 YAML；
- goal 和 destination 落在地图覆盖范围及自由区域内。

`--map` 可以省略。给终端 2 传 NPZ 不会替代终端 1 的 YAML。

---

## 11. 相关文件关系总表

### 11.1 MuJoCo 核心

| 文件 | 作用 |
| --- | --- |
| `custom_envs/mujoco/env.py` | MuJoCo 模型加载、状态读取、腿轮/机械臂/夹爪控制、相机和底盘锁 |
| `custom_envs/mujoco/obs_builder.py` | 将 MuJoCo 状态和速度目标拼成训练时一致的 57 维策略观测 |
| `custom_envs/mujoco/policy_loader.py` | 从 rsl_rl checkpoint 和 agent.yaml 恢复推理策略 |
| `custom_envs/mujoco/lidar.py` | `mj_multiRay` 批量雷达、自身几何过滤、世界坐标击中点 |
| `custom_envs/mujoco/teleop_viewer.py` | 建图专用 GLFW viewer，正确处理按下/松开并隔离渲染快捷键 |
| `custom_envs/mujoco/scene.xml` | 原始双桌 MuJoCo 场景、机器人、机械臂、物体、相机、actuator 和 weld |
| `custom_envs/mujoco/scene_house.xml` | House 场景版本，包含房屋墙体/门/家具和机器人 |
| `custom_envs/mujoco/scene_hidden.xml`、`scene_show.xml` | 场景显示/隐藏或调试变体，不是主流程默认入口 |

### 11.2 建图与地图工具

| 文件 | 作用 |
| --- | --- |
| `custom_envs/scripts/navigation/teleop_mapping_mujoco.py` | MuJoCo 建图主入口 |
| `custom_envs/utils/occupancy_grid.py` | log-odds 栅格、ray cast、坐标转换、NPZ 读写 |
| `custom_envs/scripts/navigation/convert_map.py` | NPZ 转 Nav2 PGM/YAML，处理图像 Y 翻转 |
| `custom_envs/scripts/navigation/view_3d.py` | 在浏览器中查看 `_cloud.npy` 三维点云 |
| `custom_envs/scripts/navigation/viz_map.py` | 将 NPZ 占用格或点云渲染为 3D 图 |

### 11.3 Nav2 与 ROS 2 bridge

| 文件 | 作用 |
| --- | --- |
| `custom_envs/launch/nav2_isaac.launch.py` | 启动 map_server、Nav2 servers、velocity smoother 和 lifecycle manager；历史文件名 |
| `custom_envs/config/nav2_params.yaml` | 地图、costmap、footprint、规划器、控制器、恢复和速度平滑参数 |
| `custom_envs/utils/ros2_bridge.py` | MuJoCo/conda Python 侧的 bridge 包装和 JSON 队列 |
| `custom_envs/utils/ros2_bridge_process.py` | 系统 Python 3.10 ROS 节点；发布 TF/odom，订阅 cmd_vel，发送 Nav2 action |

### 11.4 MoveIt 与抓取

| 文件 | 作用 |
| --- | --- |
| `custom_envs/launch/moveit_mujoco.launch.py` | 启动 planning-only move_group 和 robot_state_publisher |
| `custom_envs/launch/nav2_moveit_mujoco.launch.py` | 组合启动 Nav2 和 MoveIt，作为完整任务的终端 1 入口 |
| `custom_envs/utils/moveit_bridge.py` | MuJoCo 侧 MoveIt JSON RPC 包装、轨迹抽取与加密 |
| `custom_envs/utils/moveit_bridge_process.py` | Python 3.10 ROS 节点；发布 joint_states，调用 MoveIt services，维护 PlanningScene |
| `custom_envs/scripts/navigation/grasp_worker.py` | 独立 AnyGrasp 推理进程，通过 `/tmp/*.npz` 接收点云并返回候选 |
| `custom_envs/scripts/navigation/mujoco_moveit_frames.py` | camera optical、base、world、gripper_base、grasp_tcp 坐标变换的唯一实现 |
| `custom_envs/scripts/navigation/mujoco_local_servo.py` | 最后短距离 MuJoCo Jacobian 伺服和接触辅助函数 |
| `custom_envs/scripts/navigation/navigate_mujoco_moveit.py` | 推荐的完整 Nav2 + AnyGrasp + MoveIt 状态机 |
| `custom_envs/assets/m20_piper_single/M20_Piper_moveit.urdf` | MoveIt 机器人模型 |
| `custom_envs/config/moveit/M20_Piper_moveit.srdf` | MoveIt planning group/self-collision 语义 |
| `custom_envs/config/moveit/kinematics.yaml` | KDL IK 配置 |
| `custom_envs/config/moveit/joint_limits.yaml` | MoveIt 关节速度/加速度限制 |
| `custom_envs/config/moveit/ompl_planning.yaml` | OMPL/RRTConnect 规划配置 |

### 11.5 旧方案与诊断工具

| 文件 | 作用 |
| --- | --- |
| `custom_envs/scripts/navigation/navigate_mujoco.py` | 不使用 MoveIt 的旧完整流程，使用自定义机械臂 IK |
| `custom_envs/utils/arm_ik_mujoco.py` | 旧流程的 IK、FK 和抓取坐标工具 |
| `custom_envs/scripts/navigation/navigate_mujoco_diagnostics.py` | 主状态机的只读诊断和调试图片输出 |
| `custom_envs/scripts/navigation/verify_moveit_frames.py` | 不运动机器人，验证相机光学坐标与 TCP 轴约定 |
| `custom_envs/scripts/navigation/verify_moveit_fk.py` | 不依赖 ROS，对比 MuJoCo 与 MoveIt URDF 的 FK |
| `custom_envs/scripts/navigation/moveit_smoke_test.py` | MoveIt 启动后测试 joint state、IK 和 planning service，不执行机械臂 |
| `custom_envs/scripts/navigation/grasp_stability_test.py` | 单独测试夹持、接触与抬升稳定性 |
| `custom_envs/README.md` | MuJoCo 与 IsaacLab 的用户命令总览 |
| `custom_envs/README_MOVEIT_INTEGRATION.md` | MoveIt 集成的设计说明和已知限制 |

`preview_scene.py`、`teleop_mapping.py`、`navigate_to_goal_nav2*.py` 等文件属于 IsaacLab 路线，不是 MuJoCo 主链路。

---

## 12. 常用观察和排错命令

### 12.1 检查节点是否存在

```bash
ros2 node list | grep -E 'map_server|planner_server|controller_server|bt_navigator|move_group'
```

### 12.2 确认真正加载的地图

```bash
ros2 param get /map_server yaml_filename
ros2 topic echo /map nav_msgs/msg/OccupancyGrid --once --field info
```

检查 `width`、`height`、`resolution` 和 `origin` 是否对应新地图。

### 12.3 检查 TF

```bash
ros2 run tf2_ros tf2_echo map base_link
```

终端 2 运行后应连续看到变换；若没有，Nav2 无法定位机器人。

### 12.4 检查速度话题

```bash
ros2 topic echo /cmd_vel
ros2 topic info /cmd_vel -v
```

`/cmd_vel_nav` 是 controller/behavior 的原始输出，`/cmd_vel` 是平滑后 bridge 真正订阅的输出。

### 12.5 检查 Nav2 生命周期

```bash
ros2 lifecycle get /bt_navigator
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
```

正常运行应为 `active [3]`。终端 2 正常退出后这些状态仍应保持 active。

### 12.6 检查 MoveIt

```bash
ros2 service list | grep -E 'compute_ik|plan_kinematic_path|compute_cartesian_path|apply_planning_scene'
ros2 topic echo /joint_states --once
```

也可以运行：

```bash
python custom_envs/scripts/navigation/verify_moveit_frames.py
python custom_envs/scripts/navigation/verify_moveit_fk.py
python custom_envs/scripts/navigation/moveit_smoke_test.py
```

### 12.7 查看地图和雷达点云

```bash
python custom_envs/scripts/navigation/view_3d.py \
    custom_envs/maps/map_house_mujoco_cloud.npy --no_ground

python custom_envs/scripts/navigation/viz_map.py \
    custom_envs/maps/map_house_mujoco.npz
```

---

## 13. 最容易混淆的结论

1. 建图阶段是纯 MuJoCo，不需要 Nav2。
2. 导航阶段不会在线运行这套 MuJoCo 雷达；Nav2 使用离线 PGM/YAML 静态地图。
3. 终端 1 的 `map:=...yaml` 才决定 Nav2 地图；终端 2 的 `--map` 不决定地图。
4. Nav2 输出的是目标速度，不是 MuJoCo 轮子控制量；行走策略位于两者之间。
5. 本项目没有 AMCL；MuJoCo 真值通过 TF 直接充当定位结果。
6. MoveIt 只规划，不执行；机械臂轨迹最终由 MuJoCo PD 执行。
7. `/joint_states` 是 MuJoCo 当前关节状态的镜像，不是 MoveIt 发出的控制命令。
8. MoveIt URDF、SRDF、PlanningScene 和 MuJoCo MJCF 必须保持几何一致，但它们目前仍是两套模型。
9. House 的 Nav2 地图可以正确加载，但 MoveIt 默认碰撞场景还没有自动变成 House 几何。
10. `source setup.bash` 只是在当前 shell 配置环境；它不启动任何节点。
11. ROS 2 无需 `roscore`，终端之间通过 DDS 发现。
12. 终端 2 正常退出后，终端 1 的 Nav2/MoveIt 服务仍在，可以直接再次运行任务。
