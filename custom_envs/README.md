# custom_envs：建图 → 导航 → 抓取 → 导航

本文以仓库根目录 /home/mojie/taskdog 为工作目录。MuJoCo 和 IsaacLab
现在使用同一套占用栅格坐标及 Nav2 导出规则。

## MuJoCo 直接建图

MuJoCo 可以直接使用 scene.xml 建图，不需要先进入 IsaacLab。机器人运动仍由
现有 rsl_rl 策略控制。

~~~bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/teleop_mapping_mujoco.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --map_name map_whole_mujoco \
    --render
~~~

运动键只在按住时生效，松开立即归零；建图窗口使用独立按键处理，不会触发
MuJoCo viewer 自带的渲染快捷键：

- W/S：前进/后退。
- A/D：左移/右移。
- Q/E：左转/右转。
- Space 或 X：速度清零。
- M：立即保存。
- Esc：保存并退出。

默认输出到 custom_envs/maps/：

- map_whole_mujoco.npz：项目内部占用栅格及元数据。
- map_whole_mujoco_cloud.npy：体素降采样后的三维点云。
- map_whole_mujoco_nav2.pgm/.yaml：Nav2 map server 使用的地图。

建图脚本每 30 秒另存一次 _auto 快照。可用 --scene <scene.xml> 指定其他
MJCF；后续导航也应给 navigate_mujoco.py 或 navigate_mujoco_moveit.py
传入相同的 --scene。

无界面冒烟运行：

~~~bash
python custom_envs/scripts/navigation/teleop_mapping_mujoco.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --map_name mujoco_smoke \
    --max_steps 200 \
    --no_realtime
~~~

## 使用建好的地图导航

终端 1 启动 Nav2。真正使用的地图由这里的 map:=...yaml 决定：

~~~bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog

ros2 launch custom_envs/launch/nav2_isaac.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_whole_mujoco_nav2.yaml
~~~

终端 2 启动 MuJoCo 的“导航 → 抓取 → 导航”流程：

~~~bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_mujoco.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --goal 4.0 5.5 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --object bowl \
    --destination 7.5 5.8 \
    --render
~~~

使用 MoveIt 时，终端 1 改为：

~~~bash
ros2 launch custom_envs/launch/nav2_moveit_mujoco.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_whole_mujoco_nav2.yaml
~~~

终端 2 使用 MoveIt 版本的完整流程：

~~~bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_mujoco_moveit.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --map custom_envs/maps/map_whole.npz \
    --goal 4.0 6.1 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --object cube \
    --destination 7.5 5.8 \
    --render

python custom_envs/scripts/navigation/navigate_mujoco_moveit.py     --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt     --scene custom_envs/mujoco/scene_house.xml     --robot_x 1.6     --robot_y 1.6     --goal 4.35 3.65     --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar     --object cube     --destination 12.0 12.0     --render

~~~

MoveIt 完整说明见 README_MOVEIT_INTEGRATION.md。

四个 Nav2 导航入口里的 --map 都只是兼容旧命令的可选参数：

- navigate_mujoco.py
- navigate_mujoco_moveit.py
- navigate_to_goal_nav2.py
- navigate_to_goal_nav2_whole.py

这些脚本不自行读取 NPZ，也不需要因为 row/col 修复而交换目标坐标。Nav2
只读取终端 1 指定的 YAML/PGM。

## 地图坐标约定

- 内存数组始终是 grid[row, col]。
- row 对应世界坐标 Y，col 对应世界坐标 X。
- origin=(x_min, y_min) 是地图左下角；row=0 对应最小世界 Y。
- PGM 的图像第 0 行在顶部，因此 convert_map.py 只在导出图像时上下翻转；
  YAML origin 不变。

旧实现把自由空间射线的 row/col 反写了，但障碍端点没有反写。修复后的代码仍可
读取旧 NPZ，不过旧图里的自由区域已经转置，建议重新在 MuJoCo 或 IsaacLab 中
建图，不要把旧地图直接作为最终 Nav2 地图。

已有 NPZ 可以单独重新导出为 Nav2 地图：

~~~bash
python custom_envs/scripts/navigation/convert_map.py \
    --map custom_envs/maps/my_map.npz \
    --out custom_envs/maps/my_map_nav2
~~~

## IsaacLab 完整流程

下面保留原 README 中 IsaacLab 的训练、建图、地图查看、传统 A* 导航和
Nav2 导航/抓取用法。命令统一按仓库根目录
/home/mojie/taskdog 执行，只有训练命令需要先进入 deps/rl_training。

### 1. 训练行走策略

~~~bash
cd /home/mojie/taskdog/deps/rl_training

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Flat-Deeprobotics-M20-v0 \
    --headless
~~~

### 2. IsaacLab 建图

不带 Piper 的 LiDAR 场景：

~~~bash
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/teleop_mapping.py \
    --task Flat-Deeprobotics-M20Pro-Lidar-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt
~~~

带 Piper 的场景：

~~~bash
python custom_envs/scripts/navigation/teleop_mapping.py \
    --task Flat-Deeprobotics-M20Pro-Piper-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt
~~~

IsaacLab 建图入口也已改为从真实雷达原点投射射线，其 NPZ、点云及坐标约定
与 MuJoCo 版本一致。

### 3. 查看点云和占用地图

~~~bash
# 三维点云；--no_ground 隐藏地面点
python custom_envs/scripts/navigation/view_3d.py \
    custom_envs/maps/my_map_cloud.npy \
    --no_ground

# 查看 NPZ 占用地图
python custom_envs/scripts/navigation/viz_map.py \
    custom_envs/maps/my_map.npz
~~~

### 4. IsaacLab 传统 A* 导航

M20 Pro LiDAR：

~~~bash
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --task Flat-Deeprobotics-M20Pro-Lidar-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/my_map.npz \
    --goal 9 9 \
    --target_speed 1.0
~~~

M20 Pro + Piper：

~~~bash
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --task Flat-Deeprobotics-M20Pro-Piper-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/my_map.npz \
    --goal 9 9 \
    --target_speed 1.0
~~~

Piper 单桌导航并抓取：

~~~bash
python custom_envs/scripts/navigation/navigate_to_goal.py \
    --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/my_map.npz \
    --goal 4.5 4.95 \
    --target_speed 1.0 \
    --grasp_checkpoint /home/mojie/graspnet-baseline/logs/checkpoint-rs.tar \
    --enable_cameras True \
    2>&1 | tee /tmp/nav_run.log
~~~

### 5. IsaacLab + Nav2

先把 NPZ 转为 Nav2 地图：

~~~bash
python custom_envs/scripts/navigation/convert_map.py \
    --map custom_envs/maps/my_map.npz \
    --out custom_envs/maps/my_map_nav2
~~~

终端 1 在 ROS base 环境启动 Nav2。单桌地图：

~~~bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog

ros2 launch custom_envs/launch/nav2_isaac.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/my_map_nav2.yaml
~~~

双桌完整地图：

~~~bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog

ros2 launch custom_envs/launch/nav2_isaac.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_whole_nav2.yaml
~~~

终端 2 启动 IsaacLab。单桌导航抓取：

~~~bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_to_goal_nav2.py \
    --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/my_map.npz \
    --goal 4.5 5.0 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
~~~

双桌导航抓取：

~~~bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_to_goal_nav2.py \
    --task Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/map_whole.npz \
    --goal 4.5 5.0 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
~~~

双桌“导航 → 抓取 → 再导航 → 放置”完整流程：

~~~bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog

python custom_envs/scripts/navigation/navigate_to_goal_nav2_whole.py \
    --task Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0 \
    --policy_task Flat-Deeprobotics-M20-v0 \
    --load_run 2026-07-18_10-57-32 \
    --checkpoint model_4999.pt \
    --map custom_envs/maps/map_whole.npz \
    --goal 4.5 5.0 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --object banana \
    --destination 7.45 6.0
~~~

上述 Nav2 Python 脚本中的 --map 为兼容原命令而保留；实际供 Nav2 规划的地图
始终是终端 1 的 map:=...yaml。

## 相关代码文件

| 文件 | 作用 |
| --- | --- |
| custom_envs/mujoco/lidar.py | 默认生成完整 Mid-360 规则扫描配置的 14400 条射线（40 通道、水平 1°），批量调用 MuJoCo ray cast，并排除机器人自身碰撞体。 |
| custom_envs/mujoco/policy_loader.py | MuJoCo 建图和两套导航脚本共用的 rsl_rl 策略加载器。 |
| custom_envs/mujoco/teleop_viewer.py | 建图专用 GLFW viewer，提供按下/松开事件并隔离 MuJoCo 内置快捷键。 |
| custom_envs/scripts/navigation/teleop_mapping_mujoco.py | MuJoCo 键盘运动、雷达扫描、栅格融合、点云累计及地图保存入口。 |
| custom_envs/utils/occupancy_grid.py | 统一坐标约定、射线裁剪/融合、障碍累计及 NPZ 序列化。 |
| custom_envs/scripts/navigation/convert_map.py | 将 NPZ 导出为 Nav2 PGM/YAML，并处理图像 Y 轴方向。 |
| custom_envs/mujoco/env.py | 提供场景路径、viewer 键盘回调与无界面运行接口。 |
| custom_envs/scripts/navigation/teleop_mapping.py | IsaacLab 建图入口，与 MuJoCo 共用融合语义。 |
| custom_envs/scripts/navigation/navigate_mujoco.py | MuJoCo + Nav2 + AnyGrasp 全流程；地图由 Nav2 launch 加载。 |
| custom_envs/scripts/navigation/navigate_mujoco_moveit.py | 加入 MoveIt 的 MuJoCo 全流程；地图由 Nav2 launch 加载。 |
| custom_envs/scripts/navigation/navigate_to_goal_nav2.py | IsaacLab + Nav2 单目标流程；不再无效读取 NPZ。 |
| custom_envs/scripts/navigation/navigate_to_goal_nav2_whole.py | IsaacLab + Nav2 抓取/放置完整流程；不再无效读取 NPZ。 |
| tests/custom_envs/ | row/col 回归测试，以及 MuJoCo 雷达到 Nav2 地图的冒烟测试。 |
