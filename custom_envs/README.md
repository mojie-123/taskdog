``` bash
# 建图
python custom_envs/scripts/navigation/teleop_mapping.py       --task=Flat-Deeprobotics-M20Pro-Lidar-v0       --policy_task=Flat-Deeprobotics-M20-v0       --load_run=2026-07-18_10-57-32       --checkpoint=model_4999.pt

python scripts/navigation/teleop_mapping.py --task Flat-Deeprobotics-M20Pro-Piper-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt



# 生成点云地图（可用html查看）
python scripts/navigation/view_3d.py maps/my_map_cloud.npy --no_ground


# 导航
python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Lidar-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 9 9 --target_speed 1.0

python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Piper-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 9 9 --target_speed 1.0

python scripts/navigation/navigate_to_goal.py --task Flat-Deeprobotics-M20Pro-Piper-Single-v0 --policy_task Flat-Deeprobotics-M20-v0 --load_run 2026-07-18_10-57-32 --checkpoint model_4999.pt --map maps/my_map.npz --goal 4.5 4.9 --target_speed 1.0 --grasp_checkpoint /home/mojie/graspnet-baseline/logs/checkpoint-rs.tar --enable_cameras True 2>&1 | tee /tmp/nav_run.log




```