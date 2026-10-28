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
- `coverage.py`：基于可通行栅格生成候选视点；候选收益使用静态地图预测，已完成观察的覆盖范围则由实际 RGB-D 深度射线更新，并在首个深度表面或地图障碍物处截止。
- `scene_memory.py`：轻量 Room--Viewpoint--Object 三层结构化场景表示。
- `view_projection.py`：把长期 Object 的三维包围盒投影回历史视点，并生成 VLM 使用的框选全图与干净局部图。
- `object_verification.py`：按稳定 Object 实例去重并异步执行类别、空间关系和房间条件复核。
- `target_reobservation.py`：为重复出现但图像过小的临时目标，从房间可搜索区和 Nav2 自由栅格中选择近距离复看位姿。
- `search_policy.py`：把 VLM 离散语义决定与经典覆盖规划组合起来，但不直接控制机器人。
- `runtime.py`：一次性装配上述对象，并把一帧 RGB-D 观测写入三层记忆。
- `transit_detection.py`：导航过程中限频采集 RGB-D；观测直接流入 Object 层，不因原始检测框停车。
- `search_runner.py`：执行选房、房内覆盖、相邻房间提前切换以及异步对象确认状态机。

## 主脚本调用顺序

1. `build_sysnav_components()`，加载配置、地图、房间图，并创建惰性模型客户端。
2. 用 `QwenVLMClient.parse_instruction()` 得到目标、房间条件和参照物。
3. 用 `instruction.detector_prompts` 设置 YOLOE 文本类别。
4. 房内搜索期间后台预取下一房间；候选 Room 状态未变化时直接采用 VLM 结果，否则丢弃旧结果。房间耗尽时若预取尚未完成，则立即按房间图旅行代价选择，不停车等待；`execution_anchor()` 再把匿名 `room_id` 映射到门内地图坐标。
5. 在该房间反复调用 `next_viewpoint()`，由现有 Nav2 到达 `(x, y, yaw)`；到达后只把实际 depth 射线经过且未被障碍物截断的目标栅格记为已覆盖，不再直接接受该视点的整片理论可见区域。
6. 普通航点、门口 PEEK 和导航途中的限频 RGB-D 都先经过 detector、localizer，再立即融合进持久 Object 实例。连续导航帧可以更新三维位置和最佳局部图，但同一段导航只算一个语义视角，不会逐帧重复请求 VLM。每个 Room 节点另存普通视点与 PEEK 的多视角证据；房间描述请求使用独立线程池，拍照和本地感知完成后即可继续导航，结果返回后再由导航主线程更新 Room 节点。单张门口图只能产生 `tentative` 语义，至少两张普通视点且多图置信度达到 0.75 后才成为 `stable`。
7. 对达到最小掩码面积的目标/参照物 Object，后台任务先用“框选全图 + 无标注局部图”做一次类别复核；类别通过后，再把最佳近景、该实例所有有效历史视角的三维投影框、当前 Room 标签和完整指令一起交给 VLM，联合判断目标、属性、与参照物的关系及房间条件。查询发出时立刻设置 `is_considered`，因此原始检测不停车，也不存在逐帧重复确认。新的有效规划视角、更好的最佳图或 Room 标签真正改变时才允许该 Object 再次复核；单纯置信度变化不会重开查询。
8. 若目标在三维 Object 层中已被重复稳定定位，但最佳掩码仍小于 VLM 最小面积，系统会暂时中断当前 coverage：从该房间的 `room_searchable_mask ∩ Nav2 free_mask` 中选取朝向目标、默认约 1.25 m 的安全复看位姿。近景仍不足时最多再靠近一次；复看未确认则恢复原导航目标，确认成功则立即结束搜索。单帧小框不会触发这条路径。
9. 参照物使用独立的对象查询；确认后，其三维位置作为覆盖规划的语义兴趣点，提高能看到参照物附近区域的后续航点得分。这里保留原 SysNav 的“先确认 anchor、再用 anchor 引导探索”语义，但底层由当前 Nav2 栅格覆盖器实现，而不是原项目的 TARE 视点 TSP。
10. 当前房间覆盖率达到 0.50 且完成至少 3 个普通覆盖视点后，才允许在门口获得相邻房间图像并异步提交 early-stop 判断；机器人不在门口等待，而是继续当前房间覆盖，肯定结果在下一个安全导航边界触发改道。VLM 会联合比较当前房间最近至多 4 个普通覆盖视点与相邻房间的当前 PEEK 图，并同时读取上述共享 Room 节点语义。旧的门口 PEEK 不会替代当前房间证据。每个普通视点后至多执行一次门口观察，实际输入图片记录在 `early_stop_vlm_inputs` 事件中。VLM 只决定是否换房，实际路线仍由房间图和 Nav2 生成。
11. 第一轮 N2M 的 3 个底盘位姿、每个位姿 3 次抓取均失败时，在同一个 MuJoCo 世界中选择与旧观察方向优先相差至少 60 度的新视点，重新检测 cube、采集点云并运行第二轮 N2M。默认最多两轮，可用 `--sysnav-max-n2m-rounds` 调整。

## 设计上的降级行为

- 没有任何运行时语义证据时，按房间图最短代价选择房间。
- API 请求失败时，选房降级为几何代价；带属性/空间约束的目标不会被静默接受。
- 没有 VLM 的离线类名模式可直接接受 detector 类别；正常 SysNav 模式仍先对稳定 Object 实例做类别复核。
- 参照物（例如 coffee table）通过对象级复核后，覆盖规划器才会提高朝向该位置的视点优先级。

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
