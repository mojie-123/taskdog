# add_navigation.md — M20 Pro 点对点导航：建图 + A* + Pure Pursuit

> **状态**: 已完成并验证通过
> **目标**: 键盘遥控建图 → 机器狗自主导航到桌子上的香蕉

---

## 1. 总体流程

```
阶段一：建图（人工遥控）                    阶段二：导航（自主）
─────────────────────────                  ────────────────────

  键盘(WASD/QE) → 速度指令                      地图上已有的桌子+香蕉位置
       │                                              │
       ▼                                              ▼
  M20 运动策略(已有训练模型)                     A* 在膨胀占据栅格上搜索最短路径
       │                                        输出: 路径点列表 [p1, p2, ..., pN]
       ▼                                              │
  LiDAR 扫描 + 真值里程计                              ▼
       │                                        Pure Pursuit 路径跟踪
       ▼                                        输出: (vx, ωz) 速度指令
  2D 占据栅格建图                                      │
  ├─ 所有射线 → Bresenham 标记沿途空闲                ▼
  └─ 高处射线(z>0.15) → mark_elevated 投影占据    locomotion 策略执行关节动作
       │                                              │
       ▼                                              ▼
  保存 my_map.npz                                  机器人移动
  (同时保存 my_map_cloud.npy 3D点云供可视化)
```

---

## 2. 建图：OccupancyGrid

### 2.1 核心数据结构

`OccupancyGrid` (custom_envs/utils/occupancy_grid.py)：一个用 log-odds 存储占据概率的 2D 网格。

- 默认 400×400 格，分辨率 0.05m，覆盖 20m×20m（原点 set 为 (-10, -10) 使范围覆盖 -10m 到 10m）
- 每格：log(p_occ / p_free)。>0 = 占据，<0 = 空闲，0 = 未知
- 通过 `get_binary_map()` 输出 0/1 二值地图（log-odds > 0 → 1，否则 0）
- 通过 `get_inflated_binary_map(robot_radius=0.4)` 输出膨胀后的二值地图（给 A* 使用，确保路径与障碍物保持安全距离）

### 2.2 空地标记：所有射线做 Bresenham 射线追踪

`grid.update(robot_pose, lidar_hits_w)` 对每一帧的 LiDAR 命中点，从机器人位置到命中点做 Bresenham 直线遍历，射线沿途的格子标记为"空闲"（log-odds -= 0.4）。**终点不做标记**——地面射线的终点只是地面，高处射线的终点由 mark_elevated 单独处理。

### 2.3 障碍物标记：高处命中投影

`grid.mark_elevated(hits_xy, min_hits=1)` 解决了一个关键问题：

**问题**：桌子/香蕉在 0.8m 高处，大量地面射线从桌子下方穿过，在 2D 投影上把桌子位置的格子反复标记为"空闲"，淹没了少量高处命中点的"占据"证据。结果：桌子在地图上"透明"。

**解决**：高处命中点（|z|>0.15）直接投影到 2D 网格，用持久计数器累积。超过阈值的格子直接设为最大占据值（+10.0），不被空闲证据淹没。

```
地面射线 → Bresenham 沿途标记空闲，终点不标
高处射线 → Bresenham 沿途标记空闲 + mark_elevated 投影占据
最终：(5,5) 格子同时有"下方空闲"和"上方占据" → 占据获胜
```

### 2.4 障碍物膨胀

`get_inflated_binary_map(robot_radius=0.4)` 用 `scipy.ndimage.binary_dilation` 把占据格向外膨胀 0.4m（M20 半宽 0.22m + 行走余量）。A* 使用膨胀地图规划，确保路径与障碍物保持安全距离。

---

## 3. 导航

### 3.1 A* 全局规划

`astar_plan(grid, start, goal)` (custom_envs/utils/astar_planner.py)：标准 A* 实现。
- 8 连通邻居（含对角），对角移动代价 √2
- 启发式：octile distance（8 连通最优启发式）
- 输入：膨胀后的二值地图 + 起点/终点网格坐标
- 输出：路径点列表 [(r1,c1), ...] 或 None（无路径）
- 重规划前会把起点周围 0.3m 区域的格子清空（避免机器人走入膨胀区后 A* 认为起点被阻挡）

### 3.2 Pure Pursuit 路径跟踪

`PurePursuitController.compute_velocity(path, robot_pos)` (custom_envs/utils/pure_pursuit.py)：
1. 计算 lookahead 距离：max(0.5m, current_vx × 0.5)
2. 在路径上找到距机器人 ≥ lookahead 的第一个点
3. 把该点转换到机器人局部坐标系 → 得 dy_body（横向偏移）
4. 曲率 = 2×dy_body / lookahead²
5. omega = max(current_vx, target_speed×0.3) × curvature（确保静止时也能转向）
6. vx = target_speed / (1 + |omega|×0.5)（弯急减速）

### 3.3 速度指令注入

`navigate_to_goal.py` 不通过 ObsTerm 覆盖观测——而是在每步**直接修改观测字典中的 policy 张量**：

```python
p_obs = obs["policy"].clone()  # 必须 clone：原张量是 inference tensor 不可修改
p_obs[0, 6] = vx      # velocity_commands 在 policy 观测中的索引 6
p_obs[0, 7] = 0.0     # vy 始终为 0
p_obs[0, 8] = omega   # 转向速度
obs["policy"] = p_obs
```

---

## 4. LiDAR 传感器

机器狗背上装有仿 Livox Mid-360 LiDAR（MultiMeshRayCaster，2880 条射线，360°×59° FOV）。

- **安装位置**：base_link 上方，(0.30, 0, 0.55) 偏移
- **射线检测**：使用 `MultiMeshRayCaster`（非基本 RayCaster）——它用 `raycast_dynamic_meshes` warp 内核在物体局部空间做检测，解决了 float32 大坐标精度问题
- **扫描目标**：`/World/ground`（静态）、`{ENV_REGEX_NS}/Shop_Table` + `{ENV_REGEX_NS}/banana`（跟踪网格变换）
- **观测**：LiDAR 传感器存在于场景中可供脚本读取，但不在策略的 57 维观测空间内

---

## 5. 环境配置

5 个 M20 Pro 环境（全部 57 维观测，16 维动作）：

| ID | 地形 | LiDAR 传感器 | 场景 |
|----|------|-------------|------|
| Flat-Deeprobotics-M20Pro-v0 | 平地 | - | - |
| Rough-Deeprobotics-M20Pro-v0 | 崎岖 | - | - |
| Flat-Deeprobotics-M20Pro-Lidar-v0 | 平地 | ✅ | 桌子+香蕉 |
| Rough-Deeprobotics-M20Pro-Lidar-v0 | 崎岖 | ✅ | - |
| Flat-Deeprobotics-M20Pro-Piper-v0 | 平地 | ✅ | 桌子+香蕉+Piper臂 |

---

## 6. 运行命令

### 建图

```bash
cd /home/mojie/taskdog/custom_envs
python scripts/navigation/teleop_mapping.py \
    --task Flat-Deeprobotics-M20Pro-Lidar-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt

# 键盘：WASD 移动，QE 转向，M 保存地图，ESC 退出
```

### 导航

```bash
cd /home/mojie/taskdog/custom_envs
python scripts/navigation/navigate_to_goal.py \
    --task Flat-Deeprobotics-M20Pro-Lidar-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map maps/my_map.npz \
    --goal 5 5 \
    --target_speed 0.5
```

### 3D 点云查看

```bash
python scripts/navigation/view_3d.py maps/my_map_cloud.npy --no_ground --point_size 5
# 打开浏览器交互式 3D 图
```

---

## 7. 文件结构

```
custom_envs/
├── utils/
│   ├── occupancy_grid.py        # 占据栅格地图：Bresenham + mark_elevated + 膨胀
│   ├── astar_planner.py          # A* 路径规划（8连通 + octile启发式）
│   ├── pure_pursuit.py           # Pure Pursuit 路径跟踪器
│   ├── nav_utils.py              # 坐标转换（world↔grid↔body）+ yaw提取
│   ├── lidar_pattern.py          # Livox Mid-360 射线模式（2880条）
│   ├── lidar_observation.py      # KNN 点云降采样（RL训练用）
│   └── target_spawner.py         # 旧版目标生成器（已被 env config 替代）
│
├── tasks/deeprobotics_m20_pro/
│   ├── lidar_flat_env_cfg.py     # LiDAR + 平地 + 桌子 + 香蕉 env 配置
│   ├── lidar_rough_env_cfg.py    # LiDAR + 崎岖地形 env 配置
│   ├── piper_env_cfg.py          # M20 + Piper 臂 env 配置
│   └── __init__.py               # 注册 5 个 Gym 环境
│
├── scripts/navigation/
│   ├── teleop_mapping.py         # 键盘遥控 + LiDAR 实时建图
│   ├── navigate_to_goal.py       # 加载地图 + A* 规划 + Pure Pursuit 导航
│   ├── viz_map.py                # 静态 3D 点云 PNG 渲染
│   └── view_3d.py                # 交互式 3D 点云浏览器查看器
│
├── assets/
│   ├── m20_piper_gui.usda        # GUI 导出的 M20+Piper 组合 USD（备用）
│   ├── m20_piper.usda            # 程序生成的组合 USD（备用）
│   ├── piper_arm.py              # Piper articulation 配置（备用）
│   └── piper/                    # ATEC2026 Piper 模型
│
└── maps/
    ├── my_map.npz                # 占据栅格地图
    ├── my_map_cloud.npy          # 3D 点云
    └── nav_step_*.png            # 导航过程快照
```
