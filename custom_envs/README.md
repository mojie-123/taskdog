``` bash
# 训练行走策略  在/home/mojie/taskdog/deps/rl_training下运行
python scripts/reinforcement_learning/rsl_rl/train.py --task=Flat-Deeprobotics-M20-v0 --headless

# 建图
python custom_envs/scripts/navigation/teleop_mapping.py       --task=Flat-Deeprobotics-M20Pro-Lidar-v0       --policy_task=Flat-Deeprobotics-M20-v0       --load_run=2026-07-18_10-57-32       --checkpoint=model_4999.pt

python scripts/navigation/teleop_mapping.py --task Flat-Deeprobotics-M20Pro-Piper-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt



# 生成点云地图（可用html查看）
python scripts/navigation/view_3d.py maps/my_map_cloud.npy --no_ground

# 生成导航用的地图
python scripts/navigation/viz_map.py maps/my_map.npz 


# 导航
python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Lidar-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 9 9 --target_speed 1.0

python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Piper-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 9 9 --target_speed 1.0

python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 4.5 4.95 --target_speed 1.0 --grasp_checkpoint /home/mojie/graspnet-baseline/logs/checkpoint-rs.tar --enable_cameras True 2>&1 | tee /tmp/nav_run.log


# Nav2版本
# 地图格式转换
python3 custom_envs/scripts/navigation/convert_map.py     --map custom_envs/maps/my_map.npz     --out custom_envs/maps/my_map_nav2
# 终端1：就在base环境
source /opt/ros/humble/setup.bash
cd taskdog
ros2 launch custom_envs/launch/nav2_isaac.launch.py     params_file:=$(pwd)/custom_envs/config/nav2_params.yaml     map:=$(pwd)/custom_envs/maps/my_map_nav2.yaml

source /opt/ros/humble/setup.bash
cd taskdog
ros2 launch custom_envs/launch/nav2_isaac.launch.py     params_file:=$(pwd)/custom_envs/config/nav2_params.yaml     map:=$(pwd)/custom_envs/maps/map_whole_nav2.yaml

# 终端2：
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd taskdog
python custom_envs/scripts/navigation/navigate_to_goal_nav2.py     --task      Flat-Deeprobotics-M20Pro-Piper-Single-v0     --policy_task Flat-Deeprobotics-M20-v0     --load_run  2026-07-18_10-57-32     --checkpoint model_4999.pt     --map       custom_envs/maps/my_map.npz     --goal      4.5 5.0     --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar

source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd taskdog
python custom_envs/scripts/navigation/navigate_to_goal_nav2.py     --task      Flat-Deeprobotics-M20Pro-Piper-TwoTables-v0    --policy_task Flat-Deeprobotics-M20-v0     --load_run  2026-07-18_10-57-32     --checkpoint model_4999.pt     --map       custom_envs/maps/map_whole.npz     --goal      4.5 5.0     --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar
```