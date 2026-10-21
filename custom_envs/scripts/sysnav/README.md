# taskdog SysNav 接入

本目录提供 SysNav 搜索组件和搜索状态机；可执行入口为
`custom_envs/scripts/navigation/navigate_mujoco_sysnav.py`。现有
`navigate_mujoco_n2m.py` 保持不变，新入口在同一个 MuJoCo 环境中先搜索，再把
视觉三维目标位置和 YOLOE/SAM2 掩码交给原 N2M、抓取和搬运流程。

## 数据边界

```text
自然语言指令
  -> vlm.py: SearchInstruction
  -> perception.py: YOLOE prompts（target + anchor）

room_id_map + Nav2 map
  -> room_graph.py: 匿名 RoomNode 和门连接关系
  -> coverage.py: 房间内候选观察位姿

MuJoCo front_realsense RGB-D
  -> perception.py: YOLOE box/class -> SAM2 mask
  -> rgbd.py: mask + depth + camera pose -> 世界坐标物体
  -> scene_memory.py: Room <-> Viewpoint <-> Object 三层表示

三层表示 + 覆盖率 + 视觉证据
  -> search_policy.py:
       选择下一房间 / 相邻房间提前切换 / 目标复核
  -> 主脚本把 room_id 或 viewpoint pose 交给现有 Nav2 控制流程
```

静态地图中只有 `room_0`、`room_1` 等几何 ID。房间语义只能由运行时图像和检测结果写入 `SceneMemory`；组件不会从 `scene_house_N2M.xml` 读取家具或 cube 名称。

## 文件职责

- `config.py` / `custom_envs/config/sysnav.yaml`：路径、阈值与模型配置；API Key 只从环境变量读取。
- `types.py`：模块之间稳定传递的数据类型。
- `vlm.py`：指令拆解、运行时房间描述、选房、提前换房、目标条件复核。
- `perception.py`：YOLOE 开放词汇检测和 SAM2 框提示分割。
- `rgbd.py`：RGB-D 采集与物体掩码的三维定位。
- `room_graph.py`：匿名房间拓扑、最短房间路径和 room id 到门口执行锚点的映射。
- `coverage.py`：基于可通行栅格、相机视场和遮挡的贪心覆盖视点规划。
- `scene_memory.py`：轻量 Room--Viewpoint--Object 三层结构化场景表示。
- `search_policy.py`：把 VLM 离散语义决定与经典覆盖规划组合起来，但不直接控制机器人。
- `runtime.py`：一次性装配上述对象，并把一帧 RGB-D 观测写入三层记忆。
- `search_runner.py`：执行选房、房内覆盖、相邻房间提前切换和目标确认状态机。

## 主脚本调用顺序

1. `build_sysnav_components()`，加载配置、地图、房间图，并创建惰性模型客户端。
2. 用 `QwenVLMClient.parse_instruction()` 得到目标、房间条件和参照物。
3. 用 `instruction.detector_prompts` 设置 YOLOE 文本类别。
4. `SysNavSearchPolicy.choose_next_room()` 返回匿名 `room_id`；用 `execution_anchor()` 映射到门内的地图坐标。
5. 在该房间反复调用 `next_viewpoint()`，由现有 Nav2 到达 `(x, y, yaw)`。
6. 每个视点采集 RGB-D，依次调用 detector、localizer 和 `SceneMemory.add_viewpoint()`。每个 Room 节点保存普通视点与 PEEK 的多视角证据；单张门口图只能产生 `tentative` 语义，至少两张普通视点且多图置信度达到 0.75 后才成为 `stable`。后续高质量多图结论可以修订旧标签，`unknown` 也不会锁死。
7. 若找到目标候选，`verify_candidate()` 先在当前图像中核验目标身份、参照物和空间关系；房间条件不再放进该局部图像提示词，而是由目标三维位置对应的稳定 Room 节点独立判断。局部视觉已经通过但 Room 语义尚未稳定时，候选只进入运行期 pending 缓存，不写入 Object 节点；Room 更新后可直接完成确认，无需再次路过目标。
8. 当前房间覆盖率达到 0.50 且完成至少 3 个普通覆盖视点后，才允许在门口获得相邻房间图像并调用 `consider_early_stop()`；VLM 会联合比较当前房间最近至多 4 个普通覆盖视点与相邻房间的当前 PEEK 图，并同时读取上述共享 Room 节点语义。旧的门口 PEEK 不会替代当前房间证据。每个普通视点后至多执行一次门口观察，实际输入图片记录在 `early_stop_vlm_inputs` 事件中。VLM 只决定是否换房，实际路线仍由房间图和 Nav2 生成。
9. 第一轮 N2M 的 3 个底盘位姿、每个位姿 3 次抓取均失败时，在同一个 MuJoCo 世界中选择与旧观察方向优先相差至少 60 度的新视点，重新检测 cube、采集点云并运行第二轮 N2M。默认最多两轮，可用 `--sysnav-max-n2m-rounds` 调整。

## 设计上的降级行为

- 没有任何运行时语义证据时，按房间图最短代价选择房间。
- API 请求失败时，选房降级为几何代价；带属性/空间约束的目标不会被静默接受。
- 单纯的“找 cube”在检测置信度足够时不额外调用 VLM。
- 已检测到参照物（例如 coffee table）后，覆盖规划器才会提高朝向该位置的视点优先级。

## 运行

终端 1 启动 Nav2 和 MoveIt：

```bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog
ros2 launch custom_envs/launch/nav2_moveit_mujoco.launch.py \
  params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
  map:=$(pwd)/custom_envs/maps/map_house_mujoco_nav2.yaml
```

终端 2 设置 API Key 后运行完整任务；起点和终点仍由命令行指定：

```bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
cd /home/mojie/taskdog
export DASHSCOPE_API_KEY='你的 Key'

python custom_envs/scripts/navigation/navigate_mujoco_sysnav.py \
  --instruction '去客厅茶几上拿 cube' \
  --robot-x 1.6 --robot-y 1.6 --robot-yaw 0.0 \
  --destination 12.0 12.0 \
  --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
  --n2m-checkpoint custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/best.pt \
  --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --scene custom_envs/mujoco/scene_house_N2M.xml \
  --nav2-params custom_envs/config/nav2_params.yaml \
  --teleport-navigation --render
```

`--sysnav-skip-vlm --instruction 'find the cube'` 是离线类名测试模式；此时仍运行
真实覆盖、YOLOE、SAM2 和深度定位，但不执行房间语义排序、提前换房或语义约束复核。
`--sysnav-search-only` 可只验证目标搜索。每次运行会额外输出
`sysnav_search/result.json`、`events.jsonl`、`scene_memory.json` 和逐视点图片。
每轮 N2M 的完整结果分别保存在 `n2m_rounds/round_0001/`、
`n2m_rounds/round_0002/`；根目录的 `summary.json` 和 `n2m_rounds.json`
汇总总轮数、总尝试数以及每轮结果。
