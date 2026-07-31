# add_navigation.md — M20 Pro 点对点导航完整方案（SLAM + 经典路径规划）

> **状态**: 方案设计阶段（尚未实现）
> **目标**: 键盘遥控建图 → 机器狗自主导航到环境中的目标物体。
> **前置条件**: LiDAR 已加装（见 add_lidar.md），locomotion 策略已训练（见 project.md）

---

## 目录

1. [总体流程](#1-总体流程)
2. [核心概念速查](#2-核心概念速查)
3. [建图阶段：SLAM 在仿真中的简化](#3-建图阶段slam-在仿真中的简化)
4. [导航阶段：全局规划 + 局部规划 + 运控](#4-导航阶段全局规划--局部规划--运控)
5. [目标物体：在环境中放置可被 LiDAR 探测的球](#5-目标物体在环境中放置可被-lidar-探测的球)
6. [项目代码文件结构](#6-项目代码文件结构)
7. [每个文件详细说明](#7-每个文件详细说明)
8. [交互流程](#8-交互流程)
9. [分阶段实现路线图](#9-分阶段实现路线图)
10. [SLAM 能解决点对点导航吗？](#10-slam-能解决点对点导航吗)
11. [参考文献](#11-参考文献)

---

## 1. 总体流程

```
阶段一：建图（人工遥控）                          阶段二：导航（自主）
─────────────────────────                      ────────────────────

  键盘(↑↓←→ ZX) → 速度指令                       地图上已有的目标物体位置
       │                                              │
       ▼                                              ▼
  M20 运动策略                                全局规划器 (A*)
  (已有训练的模型)                             在已建好的地图上搜索最短路径
       │                                        输出: 路径点列表 [p1, p2, ..., pN]
       ▼                                              │
  LiDAR 扫描 + 里程计(真值)                           ▼
       │                                        局部规划器 (Pure Pursuit)
       ▼                                        跟踪路径点 + 实时避障
  SLAM 建图                                       输出: 速度指令 (vx, vy, ωz)
  (Occupancy Grid Mapping)                             │
  累积 LiDAR 点云到 2D 网格                           ▼
       │                                        M20 运动策略
       ▼                                        执行关节动作
  保存地图 (2D numpy array)
```

**只在仿真中需要的东西**：
- 里程计 = Isaac Sim 直接提供机器人的世界坐标真值（`robot.data.root_pos_w`），没有漂移
- 全局定位 = 同上的真值位置，不需要 AMCL 粒子滤波

这两个简化使「SLAM」退化为「Mapping」——只需建图，不需定位。

---

## 2. 核心概念速查

| 术语 | 含义 | 在我们的方案中 |
|------|------|--------------|
| **占据栅格地图** (Occupancy Grid Map) | 把 2D 平面切成小格子，每格记录「此处有障碍物」的概率 | 建图阶段的输出，导航阶段的输入 |
| **A\*** | 在图/网格上找最短路径的经典搜索算法 | 全局规划器 |
| **Pure Pursuit** | 最简单的路径跟踪算法：在路径前方找一个「追踪点」，持续朝它走 | 局部规划器 |
| **里程计** (Odometry) | 机器人推算自己走了多远、转了多少度 | 仿真中直接用真值 `root_pos_w` |
| **SLAM** | 同时定位与建图。因为不知道自己在哪，所以要一边建图一边推算自己在地图上的位置 | 仿真中简化为纯建图（定位有真值） |

---

## 3. 建图阶段：SLAM 在仿真中的简化

### 3.1 SLAM 到底是什么？（零基础版）

SLAM = **S**imultaneous **L**ocalization **A**nd **M**apping，翻译过来是「同时定位与建图」。它解决的是机器人学中最经典的鸡生蛋问题：

```
要建图 → 需要知道自己在哪 → 要定位 → 需要一张地图 → 要建图 → ...
    ↑                                                                   │
    └──────────────────── 循环依赖 ────────────────────────────┘
```

**定位 (Localization)**：机器人在环境中的 (x, y, 朝向)。比如「我在房间的 (3.2m, 5.1m) 处，面朝东北」。

**建图 (Mapping)**：环境中的障碍物分布。比如「(2.0, 3.0) 处有一堵墙，(5.1, 1.8) 处有一个箱子」。

**为什么会同时出问题**：如果没有地图，机器人看到一个墙角，它不知道自己在房间的哪个墙角——两个墙角长得一样。如果没有准确的位姿，机器人把一个障碍物扫到地图上时，因为这个位姿有误差，障碍物会被画到错误的位置，导致地图变形。

真机上的 SLAM 本质上是一个**概率估计问题**——机器人对自己位置和地图上的每个格子都持有一个「不确定度」，每收到一帧新的 LiDAR 数据，就更新这些不确定度。经过足够多次观测后，位置和地图都收敛到"最合理"的状态。

**SLAM 的三大学派**（只需要知道名字和区别）：

| 学派 | 核心数据结构 | 代表算法 | 适用场景 |
|------|-------------|---------|---------|
| **滤波** (Filter-based) | 高斯分布追踪当前位姿 + 每个路标 | EKF-SLAM, FastSLAM | 小场景, 路标稀疏 |
| **图优化** (Graph-based) | 位姿图 (节点=位姿, 边=约束) | Cartographer, GTSAM | 大场景, 这是目前主流 |
| **粒子滤波** (Particle-based) | 几千个粒子, 每个粒子=一种可能的位置 | GMapping, AMCL | 2D LiDAR, 计算快 |

现代机器人（包括 Nav2）绝大多数用**图优化 + 粒子滤波**组合：Cartographer 做 SLAM 建图（图优化），AMCL 做纯定位（粒子滤波）。

### 3.2 为什么仿真中 SLAM 极其简单

真机的 SLAM 难题在于定位：

```
真机：里程计有漂移 → 需要回环检测 + 图优化来修正轨迹 → 复杂
仿真：里程计是物理引擎的真值 → 零漂移 → 不需要修正 → 极其简单
```

所以在仿真中，SLAM 退化为**纯建图**（Mapping-only）。我们只做建图这一步——把 LiDAR 每一帧扫描到的点，用已知真值位姿投影到世界坐标系下的 2D 网格中，累积起来。这是整个 SLAM 问题中最简单的一部分。

### 3.3 建图算法（Occupancy Grid Mapping）

```
输入:
    第 t 帧 LiDAR 扫描: 2880 个 hits (世界坐标)
    第 t 帧机器人位姿: (x, y, yaw) ← 从 Isaac Sim 直接读

输出:
    一张 2D 占据栅格地图, 比如 200×200 格, 每格 0.1m, 对应 20m×20m

算法 (逐帧):
    ① 把 2880 个 hits 从世界坐标转为地图坐标:
         grid_x = (hit_x - map_origin_x) / resolution
         grid_y = (hit_y - map_origin_y) / resolution

    ② 从机器人位置到每个 hit 做射线 (Bresenham 算法):
         射线经过的格子 → 空闲 (free, -1)
         射线终点的格子 → 占据 (occupied, +1)

    ③ 每个格子的值 = Σ(occupied - free) 累积:
         正值大 → 大概率有障碍物 (多次被 LiDAR 扫到)
         负值大 → 大概率空闲 (多条射线穿过没碰到东西)
         0 附近 → 未知区域 (没被扫到过)

    ④ 持续累积每一帧, 机器人走过的地方越多, 地图越完整
```

**Bresenham 射线追踪** 是计算机图形学中的经典算法, 在 2D 网格上画线, 告诉你一条线穿过了哪些格子。Occupancy Grid Mapping 标准做法, 每个 SLAM 库都内置了这个。

### 3.3 建图时的交互

```bash
python custom_envs/scripts/navigation/teleop_mapping.py --task=Flat-Deeprobotics-M20Pro-Lidar-v0

操作:
    ↑↓←→ 前进/后退/平移  Z/X 转向
    M → 保存地图到 custom_envs/maps/my_map.npz
    ESC → 退出

显示:
    Isaac Sim 窗口: 机器人在环境中走动
    终端实时刷新: 地图的 ASCII 可视化(或一个 opencv 窗口)
```

用户在 `teleop_mapping.py` 中遥控机器人走遍环境, 过程中 LiDAR 扫描实时累积到占据栅格地图。走完之后按 M 保存地图。

---

## 4. 导航阶段：全局规划 + 局部规划 + 运控

### 4.1 总体架构

```
                    已建好的全局代价地图
                    (200×200, 0.1m/grid)
                           │
            ┌──────────────┼──────────────┐
            │               │              │
       用户指定目标     全局规划器 A*    地图实时更新
       (地图坐标)         搜索最短路径     (可选)
            │               │
            │         路径 = [p1,p2,...,pN]
            │               │
            └───────┬───────┘
                    │
               ┌────▼────┐
               │ 局部规划器│  ← 每 50ms 执行一次
               │Pure Pursuit│
               │           │
               │  输入: 路径点│
               │       当前位姿│
               │       LiDAR  │
               │  输出: vx,vy,ω│
               └────┬────┘
                    │
               ┌────▼────┐
               │ 运控层   │  ← 我们已有的 M20 locomotion 策略
               │          │
               │  输入: vx,vy,ω 速度指令│
               │  输出: 16 维关节动作  │
               └────┬────┘
                    │
                机器人
```

### 4.2 全局规划器：A*

**做什么**: 在 2D 占据栅格地图上，从当前位置到目标位置，找到一条不穿过障碍物的最短路径。

**输入/输出**:
```
输入:
    起点: (row_start, col_start)    ← 机器人在栅格地图上的坐标
    终点: (row_goal,  col_goal)     ← 用户指定的点在栅格地图上的坐标
    地图: [200, 200]                ← numpy array, 0=空闲, 1=障碍, -1=未知

输出:
    路径 = [(r1,c1), (r2,c2), ..., (rN,cN)]
    (从起点到终点的一系列相邻格子, 每个格子 0.1m×0.1m)
```

**A* 算法简述**:

A* 维护一个「待探索节点」集合和一个「已探索节点」集合。从起点开始，每次从待探索中挑出「当前代价 + 预估剩余代价」最小的节点扩展——把它的邻居加入待探索。预估剩余代价用曼哈顿距离。重复直到到达终点或待探索为空。最后从终点回溯到起点就得到路径。

这是一个经典的图搜索算法，每个 SLAM/导航库都有实现，也可以用 `scikit-image` 或手写 50 行。

**为什么是 A* 而不是 Dijkstra**: Dijkstra 不依赖启发式，遍历所有节点；A* 用曼哈顿距离指引搜索方向，大幅减少探索量。在 2D 栅格上 A* 是最优选。

### 4.3 局部规划器：Pure Pursuit

**做什么**: 在路径上找一个「追踪点」——距离机器人前方 `lookahead_distance` 处的一个路径点——然后持续朝这个点走。

**基本原理（零基础版）**：

Pure Pursuit 的核心思想来自一个非常日常的直觉：**你走路或骑车时，眼睛总是盯着前方几米的路面，而不是脚下。你自然地朝着那个「注视点」走，路是弯的你就会转弯。**

把这个直觉翻译成几何计算：

```
                    追踪点 (lookahead point)
                   /
                  /  路径上的一个点，在机器人前方 l 米处
                 /
                /
               /  l = lookahead distance
              /
             /
        ────●──── 前进方向
        robot

    dy = 追踪点在机器人局部坐标系中的横向偏移 (正=左边, 负=右边)
    l  = 前瞻距离

    机器人局部坐标系:
         ↑ y (左)
         │
    ─────●───── → x (前)
         │
```

当追踪点在机器人的正前方 (dy ≈ 0)，说明路径是直的，ωz = 0，机器人直走。当追踪点在左边 (dy > 0)，说明路径在向左弯，ωz > 0，机器人左转。dy 越大 = 弯越急 = 角速度越大。

**曲率公式的直觉**：curvature = 2·dy / l²。这个公式来自一个圆轨道的几何关系——用机器人当前位置和追踪点可以确定唯一一个相切圆。圆的半径 = l²/(2·dy)，曲率 = 1/半径 = 2·dy/l²。dy 越大，弯越急，曲率越大，角速度越大。

**前瞻距离的选择**：l 过小 → 机器人反应快但不稳（追着近处一个点跑，路径上一个小凸起就会让方向盘来回打）。l 过大 → 机器人走得稳但在急弯处会切角（"抄近路"，路径拐弯的地方机器人会提前转，轨迹偏离路径）。一般设 l = max(0.5, 速度×0.5)，速度快了自动看远点。

**为什么不用 DWB/TEB**: DWB 需要在速度空间采样几百条轨迹然后打分。Pure Pursuit 不需要采样——它直接从路径的几何关系算出速度和角速度：

```
算法 (每 50ms 执行一次):

  ① 找追踪点: 在路径上找到离机器人距离 ≈ lookahead 的那个点
        lookahead = max(0.5, current_velocity * 0.5)
        (速度越快, 看得越远; 最低看 0.5m)

  ② 把追踪点坐标从世界坐标转成机器人局部坐标: (dx_body, dy_body)

  ③ 计算曲率: curvature = 2 * dy_body / lookahead²

  ④ 计算速度指令:
        vx = min(target_speed, max_speed_obstacle)
            (target_speed 由曲率决定: 弯越急越慢)
        ωz = vx * curvature
        vy = 0  (四足轮式机器人不侧移)

  ⑤ 如果前方有动态障碍物 (LiDAR 检测到):
        紧急刹车或绕行
```

**Pure Pursuit 的优势**:
- 极简: 纯几何计算，不需要采样、不需要优化
- 计算快: 每步 < 0.1ms
- 可解释: 追踪点的位置直接告诉你「它在往哪走」
- 不足: 不考虑机器人的动力学约束（最大转向角、加减速限制等），但对轮足混合机器人影响较小

### 4.4 速度指令到关节动作

这个「运控层」复用我们已训练好的 locomotion 策略, 或用更简单的 PD 控制器:

**方式 A — 复用已训练的 M20 策略**:
```
obs = (vx_cmd, vy_cmd, ωz_cmd, joint_pos, joint_vel, ...)
action_16d = trained_policy(obs)
直接发给机器人
```
策略已经学会了「往这个方向走, 以这个速度走」对应的最佳关节动作。

**方式 B — 简化的 PD 控制器**（如果策略表现不稳定）:
```
vx_cmd → 前腿/后腿的髋关节俯仰角度
ωz_cmd → 左右腿差速 (左腿加速/右腿减速)
```

方式 A 更自然——策略已经在训练中学会了速度跟踪，直接用就行。

---

## 5. 目标物体：在环境中放置可被 LiDAR 探测的球

### 5.1 为什么需要实体目标物

如果环境中只有一个平面，LiDAR 扫出来的地图就是一张空白（全灰色未知或全白色空闲）。建图的意义在于「地图上有些东西」，这样导航才有目标可去。

用一个红色小球放在环境中，模拟未来需要导航到的目标物体（比如一个包裹、一个工具箱）。这个球 **必须在 LiDAR 的 mesh_prim_paths 中被声明**，否则 LiDAR 看不到它，地图上就不会有它的位置。同时它必须在建图前就存在于场景中，这样 LiDAR 在建图过程中能扫到它，地图上就会出现一个「障碍物斑块」——这个斑块的位置就是导航的目标点。

### 5.2 怎么在环境中放一个小球

在环境配置的 `__post_init__` 中添加一个 `AssetBaseCfg` 即可。以 IsaacLab 内置的 `IsaacSphere` 为例：

```python
# 在 lidar_flat_env_cfg.py (或对应 env) 的 __post_init__ 中:

from isaaclab.sim import spawners

@configclass
class DeeproboticsM20ProNavFlatEnvCfg(DeeproboticsM20ProLidarFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 放置目标小球
        self.scene.target_sphere = AssetBaseCfg(
            prim_path="/World/target_ball",
            spawn=spawners.SphereCfg(
                radius=0.15,          # 半径 15cm, 约一个苹果大小
                visual_material=spawners.MdlFileCfg(
                    mdl_path="omniverse://.../Red.mdl"  # 红色材质
                ),
                collision_props=spawners.RigidBodyPropertiesCfg(
                    rigid_body_enabled=False,  # 不需要物理模拟, 只是静态物体
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(5.0, 2.0, 0.3),  # 球的位置: x=5m, y=2m, 离地 0.3m
            ),
        )

        # ★ 关键: 把球的 prim 路径加入 LiDAR 的扫描目标列表
        self.scene.mid360_lidar.mesh_prim_paths = [
            "/World/ground",
            "/World/target_ball",   # ← LiDAR 能看到这个球!
        ]
```

**`mesh_prim_paths` 的作用**：RayCaster 只会对列表中的 prim 做射线碰撞检测。默认只有 `["/World/ground"]`，意味着 LiDAR 只看地面不看别的。加了 `/World/target_ball` 之后，LiDAR 发出的射线如果碰到球，就会返回击中点坐标——球的表面上会出现几十个 LiDAR 点。

### 5.3 球在建好的地图上长什么样

真实 LiDAR 扫过去的效果以 ASCII 示意如下。地图是 0.05m 分辨率的 200×200 格（10m×10m 范围）：

```
       ← 宽度 10m →
  ↑    .....................................
  │    .....███████████......................
  │    ....██████████████....................
 高度  ....████████████████..................  ← 球的投影 ≈ 0.3m 直径
 10m   ....██████████████....................
  │    .....███████████......................
  │    .....................................
  ↓    .....................................
              ↑ 球在 (x=5.0, y=2.0) 处
```

由于球是直径 30cm 的物体，在 5cm 分辨率的地图上占大约 6×6 = 36 个格子（圆形轮廓）。这个斑块的**中心坐标**就是球的真实位置，可以直接用作导航的目标点。

### 5.4 导航时怎么获取球的位置

有两种方式，对应不同的自动化程度：

**方式 A — 手动指定（当前阶段）**：

建图之后，用户在地图上肉眼看到球的斑块，手动估算其世界坐标，传给导航脚本：

```bash
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --map=custom_envs/maps/room_01.npz \
    --goal 5.0 2.0   # ← 手动指定球的位置
```

**方式 B — 自动从仿真状态读取（后续接入物体检测时）**：

```python
# 在仿真中, 球是已知的 USD prim, 可以直接读它的位置
from omni.isaac.core.utils.prims import get_prim_at_path
ball_prim = get_prim_at_path("/World/target_ball")
ball_pos = ball_prim.GetAttribute("xformOp:translate").Get()
# ball_pos = (5.0, 2.0, 0.3) → 取 (x, y) 作为导航目标

# 未来替换为: 用 YOLO/SAM 等视觉模型检测真机环境中的物体,
# 然后从深度相机或 LiDAR 获取其 3D 位置
```

当前阶段用方式 A（手动），因为我们的目标只是让建图有意义。后续接入物体检测时，只需把方式 A 的手动坐标替换为方式 B 的检测结果，导航算法本身不需要改。

### 5.5 放置位置的设计考量

位置不要随机——否则建图时你不知道球在哪。设置一个固定值，比如 `(5.0, 2.0, 0.3)`：

```
机器人起点 (0, 0) ────────────── 球 (5, 2)
                │
                │   用户遥控机器人走一圈
                │   把地面 + 周围的障碍物 + 球都扫进地图
                │
                └── 地图上球的位置是一个明显的斑块
```

如果后续需要多个目标物（比如 3 个不同颜色的球），可以在场景中放置多个 sphere，分别命名 `/World/target_ball_red`、`/World/target_ball_blue` 等。每个球都加入 `mesh_prim_paths`。

### 5.6 对文件结构的影响

新增一个文件来管理目标物体：

```
custom_envs/
└── utils/
    └── target_spawner.py           ★ 新建 (统一管理目标物的生成和位置读取)
```

这个文件封装两个函数：

```python
def spawn_target_sphere(scene, pos, radius=0.15):
    """在场景中放置一个目标小球。返回球的 prim_path。"""
    ...

def get_target_position():
    """从仿真中读取球的 3D 位置。返回 (x, y, z)。"""
    ...
```

已有的文件需要修改一行的有：

| 文件 | 修改 |
|------|------|
| `lidar_flat_env_cfg.py` | 调用 `spawn_target_sphere()` 放置球 |
| `lidar_rough_env_cfg.py` | 同上 |
| 两个 env 中的 `mid360_lidar.mesh_prim_paths` | 追加 `"/World/target_ball"` |
| `navigate_to_goal.py` | 增加 `--goal auto` 选项，自动读球位置 |

---

## 6. 项目代码文件结构

```
custom_envs/
├── utils/
│   ├── lidar_pattern.py              ← 已有 (LiDAR 参数)
│   ├── lidar_observation.py          ← 已有 (点云降采样)
│   ├── target_spawner.py             ★ 新建 (目标物体生成 + 位置读取)
│   ├── occupancy_grid.py             ★ 新建 (占据栅格地图 + Bresenham 射线)
│   ├── astar_planner.py              ★ 新建 (A* 全局路径规划)
│   ├── pure_pursuit.py               ★ 新建 (Pure Pursuit 局部控制器)
│   └── nav_utils.py                  ★ 新建 (地图坐标转换等工具函数)
│
├── tasks/deeprobotics_m20_pro/
│   ├── lidar_flat_env_cfg.py         ○ 修改 (添加 target_sphere + 更新 mesh_prim_paths)
│   └── lidar_rough_env_cfg.py        ○ 修改 (同上)
│
├── scripts/
│   └── navigation/
│       ├── teleop_mapping.py            ★ 新建 (键盘遥控 + 实时建图)
│       └── navigate_to_goal.py         ★ 新建 (加载地图 + 导航到目标点)
│
└── maps/                                ← (保存的 .npz 地图文件)
```

**共新建 7 个文件，修改 2 个已有文件。所有新功能在 custom_envs/utils/ 和 custom_envs/scripts/navigation/ 下，不触动 rl_training 上游代码。**

---

## 7. 每个文件详细说明

### 7.1 `custom_envs/utils/target_spawner.py` — 目标物体生成 (新建)

**核心函数**: `spawn_target_sphere()`, `get_target_position()`

```python
def spawn_target_sphere(scene, pos=(5.0, 2.0, 0.3), radius=0.15):
    """在场景中放置一个被 LiDAR 可见的目标小球。
    
    pos: (x, y, z) 世界坐标，默认 (5, 2, 0.3)
    radius: 球半径，默认 15cm
    返回: prim_path 字符串
    """
    scene.target_sphere = AssetBaseCfg(
        prim_path="/World/target_ball",
        spawn=spawners.SphereCfg(
            radius=radius,
            visual_material=spawners.MdlFileCfg(
                mdl_path=RED_MATERIAL_PATH  # 红色，区别于环境
            ),
            collision_props=spawners.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )
    return "/World/target_ball"


def get_target_position():
    """从仿真中读取目标球的当前 3D 位置。
    
    返回: (x, y, z) 世界坐标。
    仅在仿真运行时可调用 (需要 SimulationApp 已启动)。
    未来替换为视觉检测结果。
    """
    from omni.isaac.core.utils.prims import get_prim_at_path
    ball_prim = get_prim_at_path("/World/target_ball")
    pos = ball_prim.GetAttribute("xformOp:translate").Get()
    return (pos[0], pos[1], pos[2])
```

### 7.2 `custom_envs/utils/occupancy_grid.py` — 占据栅格地图

**核心类**: `OccupancyGrid`

```python
class OccupancyGrid:
    def __init__(self, width, height, resolution=0.1):
        self.grid = np.zeros((height, width))  # 0=未知
        self.resolution = resolution
        self.origin = (0.0, 0.0)  # 地图原点在世界坐标系的 (x, y)

    def update(self, robot_pos, lidar_hits_w):
        """用一帧 LiDAR 扫描更新地图。
        robot_pos: (x, y, yaw) 世界坐标系
        lidar_hits_w: [B, 3] 世界坐标 (未击中的已过滤)
        """
        for hit in lidar_hits_w:
            # Bresenham 射线: 从 robot 到 hit
            # 穿过的格子 → -1 (free)
            # hit 所在的格子 → +1 (occupied)

    def world_to_grid(self, wx, wy):
        """世界坐标 → 网格坐标"""

    def grid_to_world(self, gx, gy):
        """网格坐标 → 世界坐标"""

    def save(self, path):
        """保存为 .npz"""

    def load(self, path):
        """从 .npz 加载"""
```

**Bresenham 射线算法的关键**:
```
grid 坐标 (x1, y1) → (x2, y2) 的直线上所有 grid cell:

while True:
    plot(x, y)
    按误差累积选择 x+1 还是 y+1
    if (x, y) == (x2, y2): break
```
这个算法在手写数字时钟的直线绘制中用了几十年，极其高效。

### 7.3 `custom_envs/utils/astar_planner.py` — A* 全局路径规划

**核心函数**: `astar_plan(grid, start, goal)`

```python
def astar_plan(grid, start, goal):
    """
    grid:  [H, W] numpy array
           0  = 空闲 (可通行)
           1  = 障碍 (不可通行)
           -1 = 未知 (默认不可通行, 或给较低代价)
    start: (row, col) 在 grid 上的坐标
    goal:  (row, col)

    返回:
        path: [(r1,c1), ..., (rN,cN)] 或 None (找不到路径)
    """
    # 标准 A* 实现, 启发式 = 曼哈顿距离
    # 开放列表 = 优先队列 (按 f = g + h 排序)
    # g = 从起点走到当前节点的实际代价
    # h = 从当前节点到终点的预估代价 (= 曼哈顿距离)
    # 每次从开放列表取 f 最小的节点扩展
```

**为什么用曼哈顿距离而非欧氏距离**: 在栅格地图上，机器人只能沿格子移动（上下左右 + 对角），曼哈顿距离更准确地反映了实际走格子的步数。且曼哈顿距离永远是实际代价的下界（admissible），保证 A* 的最优性。

### 7.4 `custom_envs/utils/pure_pursuit.py` — Pure Pursuit 局部控制器

**核心类**: `PurePursuitController`

```python
class PurePursuitController:
    def __init__(self, lookahead_min=0.5, lookahead_ratio=0.5):
        self.lookahead_min = lookahead_min
        self.lookahead_ratio = lookahead_ratio

    def compute_velocity(self, path, robot_pos, current_vel, lidar_data=None):
        """
        path:       [(x1,y1), ...] 全局路径点 (世界坐标)
        robot_pos:  (x, y, yaw) 当前位姿
        current_vel: (vx, vy, ωz) 当前速度

        返回:
            (vx, ωz) 速度指令
        """
        # ① 计算 lookahead distance
        lookahead = max(self.lookahead_min, current_vel[0] * self.lookahead_ratio)

        # ② 在 path 上找到离机器人距离 ≈ lookahead 的点
        target = self._find_lookahead_point(path, robot_pos, lookahead)

        # ③ 把 target 转到机器人局部坐标系
        dx_body, dy_body = self._world_to_body(target, robot_pos)

        # ④ 曲率 → 角速度
        curvature = 2.0 * dy_body / (lookahead ** 2)
        vx = self._compute_linear_vel(curvature)
        ωz = vx * curvature

        # ⑤ (可选) LiDAR 紧急避障
        if lidar_data is not None and self._is_obstacle_ahead(lidar_data):
            vx *= 0.3  # 减速
            ωz = self._compute_evasive_turn(lidar_data)  # 绕行

        return (vx, ωz)
```

### 7.5 `custom_envs/utils/nav_utils.py` — 导航工具函数

```python
def world_to_grid(wx, wy, origin, resolution):
    """世界坐标 → 地图网格坐标"""

def grid_to_world(gx, gy, origin, resolution):
    """地图网格坐标 → 世界坐标"""

def smooth_path(path, window_size=3):
    """对 A* 路径做简单平滑 (滑动平均) → 路径更顺滑"""

def is_goal_reached(robot_pos, goal_pos, threshold=0.3):
    """判断是否到达目标 (0.3m 内)"""
```

### 7.6 `scripts/navigation/teleop_mapping.py` — 键盘遥控 + 实时建图

```python
"""键盘遥控建图脚本。

用法:
    python custom_envs/scripts/navigation/teleop_mapping.py \
        --task=Flat-Deeprobotics-M20Pro-Lidar-v0

操作:
    ↑↓←→ 前进/后退/平移  Z/X 转向
    M     → 保存地图到 custom_envs/maps/
    ESC   → 退出

技术细节:
    - 用 play.py 的 --keyboard 模式驱动 M20
    - 每帧从 scene["mid360_lidar"].data.ray_hits_w 读取 LiDAR 数据
    - 从 robot.data.root_pos_w 读取真值位姿
    - 每 5 帧更新一次地图 (降低频率，建图不需要 10Hz)
    - 实时在 opencv 窗口中显示地图
"""

class TeleopMapper:
    def __init__(self):
        self.grid = OccupancyGrid(width=400, height=400, resolution=0.05)
        # 400 × 0.05 = 20m × 20m

    def step(self, env):
        # ① 读 LiDAR 数据
        lidar_hits = env.unwrapped.scene["mid360_lidar"].data.ray_hits_w[0]

        # ② 读机器人真值位姿
        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
        robot_yaw = euler_from_quat(env.unwrapped.scene["robot"].data.root_quat_w[0])

        # ③ 更新地图 (每 5 帧一次)
        if self.frame_count % 5 == 0:
            self.grid.update(
                (robot_pos[0], robot_pos[1], robot_yaw),
                lidar_hits[lidar_hits_isfinite]
            )

        # ④ 可视化 (opencv imshow)
        # Map auto-saved to disk periodically; no live GUI visualisation
```

### 7.7 `scripts/navigation/navigate_to_goal.py` — 加载地图 + 导航到目标点

```python
"""加载保存的地图, 让机器人导航到指定目标点。

用法:
    python custom_envs/scripts/navigation/navigate_to_goal.py \
        --task=Flat-Deeprobotics-M20Pro-Lidar-v0 \
        --map=custom_envs/maps/my_map.npz \
        --goal 8.0 3.0    ← 目标点的世界坐标

技术细节:
    - 加载保存的地图
    - A* 规划全局路径
    - 循环: Pure Pursuit 跟踪路径 → 速度指令 → locomotion 策略 → 仿真步进
    - 到达目标 (< 0.3m) 或超时 (120s) 或卡住 (> 10s 无进展) → 退出
"""

class Navigator:
    def __init__(self, map_path, goal_pos):
        self.grid = OccupancyGrid.load(map_path)
        self.goal = goal_pos
        self.planner = PurePursuitController()
        self.path = None

    def replan(self, robot_pos):
        """在当前位姿下重新规划到目标的路径"""
        start = self.grid.world_to_grid(robot_pos[0], robot_pos[1])
        goal = self.grid.world_to_grid(self.goal[0], self.goal[1])
        self.path = astar_plan(self.grid.grid, start, goal)

    def step(self, env):
        # ① 每 50 步重新规划一次 (应对动态障碍物)
        if self.step_count % 50 == 0:
            self.replan(robot_pos)

        # ② Pure Pursuit 计算速度指令
        vx, ωz = self.planner.compute_velocity(self.path, robot_pos, current_vel)

        # ③ 发给 locomotion 策略执行
        ...
```

---

## 8. 交互流程

### 8.1 建图会话

```
$ conda activate env_isaaclab
$ cd /home/mojie/taskdog/deps/rl_training

$ python custom_envs/scripts/navigation/teleop_mapping.py \
      --task=Flat-Deeprobotics-M20Pro-Lidar-v0

   [Isaac Sim 窗口打开, 机器人在地面上]
   [终端显示] 实时建图: 空的灰色格子(未知), 逐渐被黑色(障碍物)和白色(空闲)填充]

   ↑ 前进 3 米...
   → 右转, 前进 2 米...
   ↓ 后退到角落...
   (走了一圈, 整个环境的墙壁/障碍物都扫到了)

   M → [地图保存到 /home/mojie/taskdog/custom_envs/maps/room_01.npz]
   ESC → [退出]
```

### 8.2 导航会话

```
$ python custom_envs/scripts/navigation/navigate_to_goal.py \
      --task=Flat-Deeprobotics-M20Pro-Lidar-v0 \
      --map=custom_envs/maps/room_01.npz \
      --goal 8.0 3.0

   [Isaac Sim 窗口打开, 机器人站在地图上的 (0, 0) 位置]
   [终端显示] 地图 + A* 规划的路径 + 机器人位置

   [INFO] Path planned: 42 waypoints, length = 9.3m
   [INFO] Navigating to (8.00, 3.00)...

   [机器人开始沿路径走...]
   [路径上遇到动态障碍物, 局部规划器减速绕行...]
   [到达目标 0.28m 内]

   [INFO] Goal reached!
```

---

## 9. 分阶段实现路线图

### 阶段一：占据栅格地图 + 可视化 (1-2 天)

```
[ ] 1. 实现 OccupancyGrid (含 Bresenham 射线)
[ ] 2. 实现 world_to_grid / grid_to_world 坐标转换
[ ] 3. 用静态 LiDAR 数据验证: 生成一张地图, 用 matplotlib 画出来
```

### 阶段二：A* 全局规划器 (1 天)

```
[ ] 4. 实现 astar_plan()
[ ] 5. 在随机生成的地图上测试: 规划从 (0,0) 到 (10,10) 的路径, 可视化
```

### 阶段三：Pure Pursuit + 键盘遥控建图 (2 天)

```
[ ] 6. 实现 PurePursuitController
[ ] 7. 实现 teleop_mapping.py (键盘 + LiDAR → 建图)
[ ] 8. 在仿真中实际走一圈, 保存地图, 验证地图质量
```

### 阶段四：完整导航 + 测试 (2-3 天)

```
[ ] 9. 实现 navigate_to_goal.py
[ ] 10. 在简单开放环境测试: A→B 能否到达
[ ] 11. 在有障碍物的环境测试: 全局规划绕开障碍物
[ ] 12. 加入动态障碍物、紧急避障逻辑
```

---

## 10. SLAM 能解决点对点导航吗？

**能。实际上 Nav2 的点对点导航就是靠 SLAM + A\* + DWB 这套经典方案解决的。**

SLAM 解决的是「我在哪 + 环境长什么样」。有了这两样，导航就简化为在地图上搜索 + 跟踪路径。

具体到仿真中：

| 问题 | 仿真的答案 | 真机的答案 |
|------|-----------|-----------|
| 我在哪？ | Isaac Sim 直接给真值 (无漂移) | 里程计 + 回环检测 + 图优化 |
| 环境长什么样？ | LiDAR 扫描累积到网格 (纯 Mapping) | 同左，但位姿有误差，需要图优化修正 |
| 怎么去目标？ | A* + Pure Pursuit | 同左 |

仿真中 SLAM 退化为纯建图，因为定位有真值。但建图——把多帧 LiDAR 扫描融合为一张完整的 2D 地图——和真机的流程完全一致。只是少了「定位修正」这一步。

---

## 11. 参考文献

以下按**理解本工程所需的技术栈**排序，优先级用 ★ 表示（★★★ = 必读，★★ = 建议读，★ = 选读）。

---

### ★★★ 1. S. Thrun, W. Burgard, D. Fox — *Probabilistic Robotics* (MIT Press, 2005)

**简介**: 机器人学概率方法的圣经。前 6 章覆盖了本工程用到的全部理论基础：贝叶斯滤波、卡尔曼滤波、粒子滤波、占据栅格地图建图、SLAM 的数学形式化。第 9 章专门讲 Occupancy Grid Mapping（和我们 §3 的建图算法完全对应）。不需要读完全书——第 1-2 章（概率基础）+ 第 4 章（粒子滤波）+ 第 9 章（占据栅格）就够了。

**与本工程的关系**: 直接解释了我们建图算法的每一步：为什么用 log-odds 表示格子占据概率、Bresenham 射线追踪的数学依据、逆传感器模型 (inverse sensor model) 的推导。

**获取**: 各大学图书馆有电子版，Google 可搜到 PDF。

---

### ★★★ 2. Nav2 官方文档 (https://docs.nav2.org/)

**简介**: Nav2 的全套使用指南。Get Started → Concepts → Configuration Guide → Tutorials 四步走。Concepts 页面用非常通俗的语言解释了 costmap、planner、controller、behavior tree 各自的职责和数据流。Configuration Guide 详细列出了每个模块的可调参数和推荐值。

**与本工程的关系**: 本工程的架构（全局规划 + 局部规划 + 运控）直接参考了 Nav2 的分层设计。虽然代码是用 Python 在 IsaacLab 中实现的，但模块的职责划分和接口设计与 Nav2 对齐。

---

### ★★ 3. R. C. Coulter — "Implementation of the Pure Pursuit Path Tracking Algorithm" (CMU Technical Report, 1992)

**简介**: 这篇只有 8 页的技术报告介绍了 Pure Pursuit 的完整实现公式，包括 lookahead 距离的自适应选择、曲率计算、以及在离散路径点上的插值方法。比论文更偏向工程实现——它直接给出可照抄的伪代码。

**与本工程的关系**: §4.3 的 Pure Pursuit 实现直接参考了这篇报告中的公式和 lookahead 自适应策略。如果你要调试路径跟踪的平滑度或响应速度，这篇报告的第 3-5 节是最佳参考。

**获取**: CMU 官网可直接下载 PDF（搜标题即可）。

---

### ★★ 4. P. Hart, N. Nilsson, B. Raphael — "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (IEEE Trans. SSC, 1968)

**简介**: A* 算法原论文，4 页。定义了 A* 的两个核心概念：g(n) = 从起点到节点 n 的实际代价，h(n) = 从 n 到终点的启发式估计代价，以及 f(n) = g(n) + h(n)。证明了只要 h(n) 不超过实际剩余代价（admissible），A* 保证找到最优路径。

**与本工程的关系**: §4.2 的全局规划器用的就是 A*。这篇论文解释了为什么用曼哈顿距离作为 h(n) 是正确的，以及为什么 A* 比 Dijkstra 快。

**获取**: Google Scholar 搜标题，免费。

---

### ★ 5. W. Hess, D. Kohler, H. Rapp, D. Andor — "Real-Time Loop Closure in 2D LIDAR SLAM" (ICRA, 2016)

**简介**: Google Cartographer 的论文。介绍了图优化 SLAM 的完整 pipeline：局部子图构建（scan-to-submap matching）、回环检测（branch-and-bound 搜索）、全局图优化（SPA）。这是目前 ROS2 生态中最常用的 2D LiDAR SLAM 算法。

**与本工程的关系**: 仿真中我们不需要 Cartographer（因为定位有真值），但如果你以后要把这套导航方案部署到真机，Cartographer 就是建图的首选。读这篇论文理解「回环检测」和「图优化」的概念——这是真机 SLAM 中最核心的两个模块。

**获取**: arXiv: https://arxiv.org/abs/1610.03958

---

### ★ 6. D. Fox, W. Burgard, S. Thrun — "The Dynamic Window Approach to Collision Avoidance" (IEEE Robotics & Automation, 1997)

**简介**: DWB 局部规划器的原论文。DWB 把局部导航形式化为一个优化问题：在速度空间中采样几十到几百个候选 (v, ω)，对每个候选模拟一条短轨迹，用目标朝向、障碍物距离、速度大小三个因素打分，选最高分的执行。

**与本工程的关系**: 我们选了 Pure Pursuit 而不是 DWB，因为 Pure Pursuit 更简单。但读这篇论文能帮你理解「局部规划器到底在解决什么优化问题」——DWB 是 Nav2 的默认局部规划器，理解它有助于理解整个 Nav2 的技术全景。

**获取**: Google Scholar 搜标题，免费。

---

## 12. 完整使用方法

### 12.1 前提条件

```bash
# 确认环境
conda activate env_isaaclab
cd /home/mojie/taskdog/deps/rl_training

# 确认 LiDAR 环境可用
python scripts/tools/list_envs.py
# 期望看到: Flat-Deeprobotics-M20Pro-Lidar-v0
```

### 12.2 teleop_mapping.py — 完整命令行参数

```bash
python custom_envs/scripts/navigation/teleop_mapping.py \
    [--task TASK] [--grid_size N] [--resolution R] [--map_name NAME] \
    [--num_envs N] [--load_run RUN] [--checkpoint CKPT] [--policy_task TASK2]
```

| 参数 | 是否必填 | 默认值 | 含义 |
|------|---------|--------|------|
| `--task` | 否 | `Flat-Deeprobotics-M20Pro-Lidar-v0` | 仿真环境 ID。LiDAR 环境提供点云数据用于建图 |
| `--grid_size` | 否 | 400 | 占据栅格地图的边长（格子数）。400×0.05m=20m 覆盖范围 |
| `--resolution` | 否 | 0.05 | 每个格子的物理尺寸（米）。越小地图越精细，但内存越大 |
| `--map_name` | 否 | `my_map` | 保存的地图文件名（不含 `.npz` 扩展名）。保存路径为 `custom_envs/maps/<map_name>.npz` |
| `--num_envs` | 否 | 1 | 并行环境数。建图必须为 1 |
| `--load_run` | 否 | `None`（自动最新） | 指定运控模型的 run 目录名。不传则自动找 `logs/rsl_rl/<experiment>/` 下最新的 |
| `--checkpoint` | 否 | `None`（自动最新） | 指定运控模型的 checkpoint 文件名，如 `model_4999.pt`。不传则自动找数字最大的 |
| `--policy_task` | 否 | `None`（同 `--task`） | 运控模型对应的训练任务 ID。例：`--policy_task=Flat-Deeprobotics-M20-v0` 使用 M20 flat 模型控制 LiDAR 环境中的机器人 |

**键盘操作**：

```
↑↓    前进 / 后退
←→    左移 / 右移
Z / X  左转 / 右转
M      保存地图
ESC    退出
```

**地图保存位置**：`custom_envs/maps/<map_name>.npz`，自动存档为 `<map_name>_auto.npz`（每 30 秒）

### 12.3 navigate_to_goal.py — 完整命令行参数

```bash
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --map PATH --goal X Y \
    [--task TASK] [--target_speed S] [--num_envs N] \
    [--load_run RUN] [--checkpoint CKPT] [--policy_task TASK2]
```

| 参数 | 是否必填 | 默认值 | 含义 |
|------|---------|--------|------|
| `--map` | **必填** | — | 建图阶段保存的 `.npz` 地图文件路径 |
| `--goal` | **必填** | — | 导航目标点的世界坐标 (x, y)，如 `--goal 5.0 2.0` |
| `--task` | 否 | `Flat-Deeprobotics-M20Pro-Lidar-v0` | 仿真环境 ID |
| `--target_speed` | 否 | 0.8 | 巡航速度 (m/s)，0.5~1.5 之间为宜。弯道自动减速 |
| `--num_envs` | 否 | 1 | 并行环境数，导航必须为 1 |
| `--load_run` | 否 | `None`（自动最新） | 运控模型的 run 目录名 |
| `--checkpoint` | 否 | `None`（自动最新） | 运控模型的 checkpoint 文件名 |
| `--policy_task` | 否 | `None`（同 `--task`） | 运控模型对应的训练任务 ID |

### 12.4 典型用法示例

```bash
cd /home/mojie/taskdog

# ===== 建图 =====
# ① 默认配置：LiDAR 环境 + 自动最新 LiDAR 运控模型
python custom_envs/scripts/navigation/teleop_mapping.py

# ② 指定运控模型为 M20 flat（不需要 LiDAR 训练过的模型也能走）
python custom_envs/scripts/navigation/teleop_mapping.py \
    --task=Flat-Deeprobotics-M20Pro-Lidar-v0 \
    --policy_task=Flat-Deeprobotics-M20-v0 \
    --load_run=2026-07-18_10-57-32 \
    --checkpoint=model_4999.pt

# ③ 精细地图（0.02m 分辨率, 20m×20m = 1000×1000 格）
python custom_envs/scripts/navigation/teleop_mapping.py \
    --grid_size=1000 --resolution=0.02 --map_name=detailed

# ===== 导航 =====
# ④ 加载地图, 导航到指定世界坐标
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --map=custom_envs/maps/my_map.npz --goal 5.0 2.0

# ⑤ 使用 M20 flat 运控模型 + 指定 checkpoint
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --map=custom_envs/maps/my_map.npz --goal 5.0 2.0 \
    --policy_task=Flat-Deeprobotics-M20-v0 \
    --load_run=2026-07-18_10-57-32 --checkpoint=model_4999.pt
```

### 12.5 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'custom_envs'` | `.pth` 文件失效 | 检查 `site-packages/taskdog.pth` 是否存在 |
| `size mismatch for actor.0.weight` | 运控模型和环境的观测维度不匹配 | 加 `--policy_task=Flat-Deeprobotics-M20-v0`（非 LiDAR 模型） |
| A* 返回 `None`（无路径） | 地图上起点或终点被标记为障碍物 | 确保建图时走过起点附近 |
| 机器人无法转向 | 按键不对 | 转向键是 Z/X，不是 Q/E 或方向键 |
| 地图全灰 | LiDAR 未更新 | 确认环境是 `-Lidar-v0` |
