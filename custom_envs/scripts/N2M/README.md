# 六场景 manipulation pose 搜索

这个目录实现当前阶段的独立闭环：在 `scene_house.xml` 的
`living_coffee_table` 上摆放 6 个固定 cube 场景，生成候选底盘操作位姿
`M=(x,y,yaw)`，逐个运行 `ARM_INIT -> SCAN -> GRASP_PLAN ->
MOVEIT_PRE_GRASP -> MOVEIT_APPROACH -> LOCAL_REACH -> CLOSE -> LIFT`，并只把
稳定抬起 cube 的 M 写入成功集。它不启动 Nav2，也不执行 LIFT 之后的任务。

本阶段不修改 `scene_house.xml`：cube 是 freejoint，每次 episode 通过 MuJoCo
API 重置；现有 `wrist_cam` 用于抓取点云。前置 RealSense、导航观察点云和
N2M 网络训练属于下一阶段。

## 六个 cube 场景

| scene id | cube 世界坐标 (m) | 搜索的桌边方向 |
|---|---:|---|
| `corner_lb` | `(5.30, 3.50, 0.475)` | left、bottom |
| `corner_rt` | `(9.70, 3.80, 0.475)` | right、top |
| `long_bottom` | `(7.50, 3.50, 0.475)` | bottom |
| `long_top` | `(7.50, 3.80, 0.475)` | top |
| `short_left` | `(5.30, 3.65, 0.475)` | left |
| `short_right` | `(9.70, 3.65, 0.475)` | right |

角落场景从两条相邻边各搜索一组 M，所以 `--nominal-only --cube-scene all`
会得到 8 个候选（6 个场景，8 个 approach edge），不是遗漏或重复。

## M 的生成方式

默认参考 M 使用一次真实成功抓取时记录的底盘状态：

```text
pos  = [4.791462, 3.558308, 0.563874]
quat = [0.697786, 0.010601, 0.013125, 0.716107]  # MuJoCo wxyz
M    = [4.791462, 3.558308, 1.596763]           # x, y, yaw(rad)
```

默认 reset 同时保留该记录的 `z=0.563874 m`、`roll=1.9251 deg` 和
`pitch=0.1796 deg`，只让候选的 x、y、yaw 随桌边变换。代码先把参考 M 写成
“桌边局部坐标系到机器人底盘”的相对变换，再迁移到目标 cube 对应的桌边。
这一步只提供搜索中心，不会把迁移后的 M 当作成功样本。

每条边默认搜索：

- 法向偏移：`0, -0.05, -0.10, -0.15 m`
- 切向偏移：`0, -0.125, +0.125, -0.25, +0.25 m`
- yaw 偏移：`0, -22.5, +22.5, -45, +45 deg`

法向符号约定为：正值朝桌内，负值远离桌子。参考 M 已经贴近桌边，因此
`0` 只用于复验参考 M，所有新增法向候选都严格 `< 0`，默认不会主动生成更
靠近桌子的 M。

即每条边 100 个候选。候选 0 永远是迁移后的名义 M，便于先做可视化检查。
若原流程日志中实际 `ARM_INIT` 底盘位姿不是默认值，必须通过
`--reference-base X Y YAW` 覆盖；`YAW` 使用弧度。

对每个 M，`joint1` 都重新计算：

```text
joint1_target = wrap(atan2(cube_y-base_y, cube_x-base_x) - base_yaw)
```

ARM_INIT 采用“收臂 -> 旋转 joint1 -> 保持 joint1 展臂”三段动作，避免直接
带着展开的机械臂扫过桌面。其余关节使用现有流程的默认展开姿态。

## 推荐运行顺序

命令均在 `/home/mojie/taskdog` 下运行，并使用已经跑通原脚本的
`env_isaaclab` 环境。

### 1. 只看 cube 与候选 M

```bash
conda activate env_isaaclab
python custom_envs/scripts/N2M/find_manipulation_poses.py \
    --mode preview \
    --cube-scene all \
    --nominal-only \
    --render
```

黄方块是 cube；底盘朝向箭头、cube 方位箭头、joint1 目标/实际箭头用于检查
坐标方向。按键：`N`/Enter 进入下一个检查点，Space 暂停/继续仿真，`X` 跳过
当前候选，`Q` 退出。

### 2. 验证动态 ARM_INIT

```bash
python custom_envs/scripts/N2M/find_manipulation_poses.py \
    --mode arm-init \
    --cube-scene all \
    --nominal-only \
    --render \
    --pause-on-stage
```

这一模式执行真实的三段 ARM_INIT，并检查：joint1 误差是否小于阈值、cube
的 MuJoCo instance-segmentation 像素是否出现在 `wrist_cam` 中。这个判据不受
场景灯光明暗影响。RGB、深度和 mask 会写到 `data/.../debug`。
先用此命令确认旋转方向，再开始 AnyGrasp/MoveIt 的长时间搜索。

### 3. 启动 planning-only MoveIt

另开终端：

```bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog
ros2 launch custom_envs/launch/moveit_mujoco.launch.py
```

搜索脚本自己发布 MuJoCo joint state 并请求 IK/规划；这里不需要 Nav2。
旧接口的 `apply_default_scene()` 描述的是另一个双桌场景，而且依赖 `map` 坐标
系，所以本脚本不会调用它。MoveIt 在这里负责 IK、自碰撞和路径连续性；每个
执行步另外用当前 MuJoCo `living_coffee_table` 真值检查机械臂/夹爪碰桌，发生
接触便立即把候选记为 `SAFETY: arm_table_contact`。

### 4. 先可视化回放一个完整候选

```bash
python custom_envs/scripts/N2M/find_manipulation_poses.py \
    --mode collect \
    --cube-scene short_left \
    --nominal-only \
    --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --render \
    --pause-on-stage \
    --data-dir custom_envs/scripts/N2M/data/short_left_debug
```

每个阶段都会停下，按 `N` 继续。搜索脚本不依赖导航策略；机器人直接被放到 M
并用物理 weld 锁住底盘。

### 5. 批量搜索

建议按 scene 分目录运行，先小批量，再扩大网格：

```bash
source /opt/ros/humble/setup.bash
conda activate env_isaaclab
python custom_envs/scripts/N2M/find_manipulation_poses.py \
    --mode collect \
    --cube-scene short_left \
    --max-candidates 15 \
    --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --headless \
    --data-dir custom_envs/scripts/N2M/data/short_left
```

`--headless` 使用离屏 EGL 渲染 RGB-D，不打开 MuJoCo 窗口；它与 `--render`
互斥。两者都不写时默认也是 headless。`--pause-on-stage` 只在 `--render` 下生效。

去掉 `--max-candidates` 会遍历该场景完整网格。通过初始放置和 ARM_INIT 的候选
默认最多执行 5 次完整抓取，累计成功 2 次就写入稳定成功集并立即停止该 M 的
剩余尝试；第一次失败仍会继续，因此 `失败-成功-成功` 也可通过。5次内不足2次
则判为失败。初始碰撞、joint1 越界等确定性失败不重复。
`--target-successes-per-edge` 默认是 `0`，即穷举；只有显式传入正数才会在找到
指定数量的稳定 M 后提前结束。中断后用完全相同的候选参数加 `--resume`，会
跳过已有 summary 的候选。

### 6. 回放与结果图

```bash
python custom_envs/scripts/N2M/replay_manipulation_pose.py \
    --candidate-id short_left__left__m0000 \
    --data-dir custom_envs/scripts/N2M/data/short_left \
    --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --stop-after lift \
    --render \
    --pause-on-stage

python custom_envs/scripts/N2M/visualize_success_map.py \
    --scene-id short_left \
    --data-dir custom_envs/scripts/N2M/data/short_left
```

## 成功判据与输出

LIFT 成功必须同时满足：

- CLOSE 后双指建立接触并通过短时稳定性检查；
- 抬升途中没有连续丢失双指接触；
- cube 相对 TCP 的滑移不超过阈值；
- cube 的世界 z 相对 LIFT 起点上升至少 4 cm。

达到条件后 episode 立即结束，不继续原状态机。主要输出：

- `candidates.jsonl`：本次选择的候选全集；
- `preview_checks.jsonl`：初始碰撞和预览结果；
- `arm_init_rollouts.jsonl`：joint1、相机可见性等 ARM_INIT 结果；
- `manipulation_rollouts.jsonl`：每一次完整尝试；
- `candidate_summaries.jsonl`：候选的重复验证汇总；
- `success_poses.jsonl`：最终可用于下一阶段的稳定 M；
- `success_map_<scene>.png`：俯视成功/失败分布。

初始碰撞默认是硬过滤条件：MuJoCo 放置 cube 和机器人后先执行一次前向碰撞
检测；只要机器人子树与 `living_coffee_table` 有接触，该候选就记录为
`BASE_PLACED: initial_robot_table_collision`，不会运行 ARM_INIT、AnyGrasp 或
MoveIt。后续每个执行步还会单独检查机械臂/夹爪子树的碰桌情况。只有调试碰撞
检测逻辑时才应使用 `--allow-initial-table-contact` 绕过第一道过滤。

## 代码结构

- `config.py`：六场景、候选网格和成功阈值；
- `candidate_generation.py`：桌边坐标迁移和候选生成；
- `mujoco_state.py`：可复现 reset、cube/base 放置、物理 weld；
- `manipulation_episode.py`：独立 `ARM_INIT -> LIFT` 执行器；
- `find_manipulation_poses.py`：预览、筛选、重复验证和断点续跑；
- `replay_manipulation_pose.py`：单个 M 回放；
- `replay_n2m_prediction.py`：回放推理 JSON 中的均值或采样 M；
- `retry_n2m_grasp_until_success.py`：固定一个预测 M，在不重置 cube 的前提下每轮重新运行 AnyGrasp 并持续抓取，直到 LIFT 成功或人工退出；
- `adaptive_n2m_recovery.py`：只使用一次初始 N2M 预测；当前 M 多次抓取失败后，在保留 cube 位姿的同时换到初始预测中的下一个 M；
- `evaluate_test_set.py`：对 held-out test observation 批量运行 N2M 和 headless MuJoCo 抓取，逐样本保存结果并生成汇总报告；
- `visualize_success_map.py`：俯视搜索结果图。

### 保留 cube 扰动的连续抓取测试

这个调试脚本只在启动时放置一次机器人和 cube。此后每次失败都会保留 cube
当前的位姿；下一轮先安全收臂、张开夹爪，再重新展开、重新采集腕部 RGB-D、
重新运行 AnyGrasp，并完整执行到 LIFT。默认无限重试，按 `Q` 退出：

```bash
python custom_envs/scripts/N2M/retry_n2m_grasp_until_success.py \
  --prediction custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/quick_test.json \
  --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o100_v1 \
  --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --render
```

增加 `--pause-on-stage` 可在每个阶段人工确认。若要无界面测试，必须用
`--max-attempts N --headless` 设置有限尝试次数。`--sample-index N` 可改为测试
推理 JSON 中的某个采样 M。逐次结果写入 `attempts.jsonl`，其中包含每轮开始/
结束的 cube 位姿、平移和旋转量，并明确记录该轮重新执行了 AnyGrasp。

### 单次 N2M 推理下的自动换 M 恢复

该脚本不会因为 cube 被碰动或翻转而重新运行 N2M。它只读取启动时已有的
prediction JSON 一次，将其中的高斯均值和采样按照初始分布密度排序，并用默认
`8 cm / 10 deg` 阈值去掉几乎相同的 M。每个 M 默认重新运行 AnyGrasp 并尝试
3 次；仍未成功时，先收臂、张开夹爪，再只改变机器狗 base 位姿。cube 仅在程序
启动时放置一次，换 M 前后都会记录并核对 cube 位姿。每个新 M 还会重新执行
MuJoCo 环境碰撞预检，碰撞候选直接跳过：

```bash
python custom_envs/scripts/N2M/adaptive_n2m_recovery.py \
  --prediction custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/quick_test.json \
  --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o100_v1 \
  --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --attempts-per-m 3 \
  --render
```

`--max-m-candidates N` 可限制最多尝试多少个 M；默认 0 表示使用去重后全部候选。
`--min-m-xy` 和 `--min-m-yaw-deg` 控制相邻候选的最小差异。按 `X` 可跳过当前 M，
按 `Q` 退出。无界面运行可将 `--render` 改为 `--headless`。输出目录包含
`candidate_queue.json`、`collision_preflights.jsonl`、`attempts.jsonl`、
`m_summaries.jsonl` 和 `summary.json`；所有尝试均记录 `n2m_inference_count=1`。

### 全测试集 headless 操作评测

`evaluate_test_set.py` 严格读取 `splits/test.json`，当前数据集共有 128 个 test O。
它先只加载一次 `best.pt`，为所有 test O 保存确定性的单次 N2M 预测，然后释放
模型显存；之后逐 O 启动独立的 headless 恢复进程。每个 O 最多验证前三个 M，
每个 M 默认重新运行 AnyGrasp 三次，第一次 LIFT 成功后立即结束当前 O。不同 O
之间重置场景，同一个 O 内则保留失败造成的 cube 扰动。

固定参数的后台入口是：

```bash
mkdir -p /home/mojie/taskdog/custom_envs/scripts/N2M/data/overnight_logs

nohup systemd-inhibit \
  --what=sleep \
  --mode=block \
  --why="N2M held-out test evaluation" \
  bash /home/mojie/taskdog/custom_envs/scripts/N2M/evaluate_test_set_headless.sh \
  > /home/mojie/taskdog/custom_envs/scripts/N2M/data/overnight_logs/test_eval_m3.log \
  2>&1 &

echo $! > /tmp/n2m_test_eval.pid
```

MoveIt planning 服务仍需在另一个终端或后台进程中提前启动。查看进度：

```bash
tail -f /home/mojie/taskdog/custom_envs/scripts/N2M/data/overnight_logs/test_eval_m3.log
```

脚本默认带 `--resume`：中断后再次执行同一命令会跳过已有有效 summary 的 O，
只重试超时、缺失 summary 或尚未执行的 O。连续三个系统级错误会提前停止，避免
MoveIt 服务掉线后把剩余样本全部错误地记成操作失败。

评测目录为
`data/evaluations/d30_o100_k1_test_m3_v1/`。`report.md` 是实时更新的人类可读
报告，`report.json` 保留全部聚合统计，`report_observations.csv` 为逐 O 结果，
`report_by_scene_edge.csv` 按 cube 场景和桌边汇总。报告同时统计前三个 M 的累计
成功率、操作尝试次数、碰撞拒绝数、失败阶段及原生资源退出异常。每个 O 的完整
prediction、attempt JSONL、summary 和终端日志保存在 `predictions/` 与 `runs/`。

## N2M 观察数据与训练阶段

这一阶段使用独立的 `custom_envs/mujoco/scene_house_N2M.xml`。它只在机器人
`base_link` 上增加 `front_realsense`：光轴沿机身 `+X`，图像右/下分别对应
机身 `-Y/-Z`，`fovy=57deg`。MuJoCo 提供理想针孔 RGB-D，不模拟双目匹配噪声。

观察点云和成功 M 被刻意分成两步。正式采集前先针对 `short_left/left`
生成 10 个有效 O；脚本会持续测试完整候选池，而不是只测试 10 个候选：

```bash
conda activate env_isaaclab
python custom_envs/scripts/N2M/data_generation/capture_observations.py \
    --cube-scene short_left \
    --target-valid-per-edge 10 \
    --save-previews \
    --save-rejected-previews \
    --output-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_short_left_pilot_yaw20
```

每个有效 O 在 `previews/accepted/<observation_id>/` 下保存原始 RGB、cube
叠加图、彩色 depth、cube mask、base 系点云俯视/侧视图、O/M 世界坐标图、
六宫格 `qc_panel.png` 和 `metadata.json`。失败候选写入
`rejected_observations.jsonl`；开启 `--save-rejected-previews` 时，每种失败原因
默认额外保存前三个 QC 样例。`capture_summary.json` 汇总每条 approach edge 的
有效数量、短缺和拒绝原因。

试采完成后可立即建立 10 个 O 与 37 个同侧成功 M 的 370 个 pair，并校验所有
点云和坐标标签：

```bash
python custom_envs/scripts/N2M/data_generation/build_pairs.py \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_short_left_pilot_yaw20 \
    --success-manifest custom_envs/scripts/N2M/data/manipulation_pose_dataset/d30_v1/successful_poses.jsonl

python custom_envs/scripts/N2M/data_generation/validate_dataset.py \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_short_left_pilot_yaw20
```

确认试采数据后，可沿用同一套采样和筛选配置，把 `short_left` 从 10 个扩展到
50 个。目标数量和预览开关不属于数据语义配置，因此允许在断点续跑时改变：

```bash
python custom_envs/scripts/N2M/data_generation/capture_observations.py \
    --cube-scene short_left \
    --target-valid-per-edge 50 \
    --save-rejected-previews \
    --resume \
    --output-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_short_left_pilot_yaw20
```

确认 RGB、depth、cube mask 和点云坐标后，再生成六场景观察数据。正式数据集
每条 approach edge 保存 50 个有效 O：

```bash
python custom_envs/scripts/N2M/data_generation/capture_observations.py \
    --cube-scene all \
    --target-valid-per-edge 50 \
    --output-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o50_v1
```

### 夜间顺序采集全部 O

`collect_observations_o50.sh` 会依次处理六个 cube 场景。每条 edge 的前 10 个
有效 O 保存完整预览，随后补到 50 个但不再保存 accepted 预览；最后自动建立
14,550 个 `(O,M)` pair 并运行数据校验。脚本使用固定的 `env_isaaclab` Python，
不需要启动 ROS 或 MoveIt，重复启动会从现有 manifest 继续。

```bash
cd /home/mojie/taskdog
mkdir -p custom_envs/scripts/N2M/data/overnight_logs

nohup systemd-inhibit \
    --what=sleep \
    --mode=block \
    --why="N2M O50 overnight observation collection" \
    bash custom_envs/scripts/N2M/collect_observations_o50.sh \
    > custom_envs/scripts/N2M/data/overnight_logs/observations_o50_launcher.log \
    2>&1 &

echo $! > /tmp/n2m_observations_o50.pid
```

用以下命令查看进度；退出 `tail` 的 `Ctrl-C` 只停止日志查看，不会停止采集：

```bash
tail -f custom_envs/scripts/N2M/data/overnight_logs/observations_o50_launcher.log
ps -p "$(cat /tmp/n2m_observations_o50.pid)" -o pid,etime,stat,cmd
```

默认每条 approach edge 的候选池为距 cube
`1.0/1.2/1.4/1.6/1.8/2.0m`、切向
`-0.6/-0.4/-0.2/0/0.2/0.4/0.6m` 和朝向 cube 后再偏转
`-20/-15/-10/0/10/15/20deg`，合计 294 个候选。前置相机安装在 base
前方 0.42m；实测在 1--2m 范围内使用 `+/-30deg` 会把 cube 推到水平视场外，
因此不作为默认档位。候选会根据已经接受的 O 动态选择，保证最终有效集合在距离、切向和 yaw 上尽量均衡，而不只是让尝试顺序均衡。
脚本依次过滤碰撞、cube 距离超过 2m、可见像素不足 80、投影误差、
裁剪/体素化后不足 8192 点，以及体素化后 cube 点不足 15。中断续跑使用
`--resume`；已拒绝候选默认跳过，只有明确传入 `--retry-rejected` 才重新计算。
脚本用场景 XML 哈希、相机参数和筛选阈值生成配置指纹，防止把不兼容的采集
混入同一目录。

成功 M 整理完成后，可一次或多次传入 manifest 建立 pair：

```bash
python custom_envs/scripts/N2M/data_generation/merge_successful_poses.py

python custom_envs/scripts/N2M/data_generation/build_pairs.py \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o50_v1 \
    --success-manifest custom_envs/scripts/N2M/data/manipulation_pose_dataset/d30_v1/successful_poses.jsonl

python custom_envs/scripts/N2M/data_generation/validate_dataset.py \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o50_v1
```

同一 `observation_id` 的所有 M 强制进入同一 split，防止同一点云泄漏到训练集
和验证集。桌角默认只与相同 approach edge 的 M 配对；只有明确验证局部路线可达
后才使用 `--allow-cross-edge`。

可视化一个点云及其全部 M 标签：

```bash
python custom_envs/scripts/N2M/data_generation/visualize_observation.py \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o50_v1 \
    --observation-id OBSERVATION_ID
```

训练与推理：

```bash
python custom_envs/scripts/N2M/learning/train.py \
    --config custom_envs/scripts/N2M/configs/train_d30.json

python custom_envs/scripts/N2M/learning/infer.py \
    --checkpoint custom_envs/scripts/N2M/data/training/d30_o50_v1/best.pt \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o50_v1 \
    --observation-id OBSERVATION_ID
```

推理后先回放最高权重的高斯均值并只检查 ARM_INIT。脚本会核对 observation
坐标、相对/世界位姿变换，并用完整 MuJoCo 几何过滤机器人与桌子或其他家具的
初始接触：

```bash
python custom_envs/scripts/N2M/replay_n2m_prediction.py \
    --prediction custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/quick_test.json \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o100_v1 \
    --stop-after arm_init \
    --render \
    --pause-on-stage
```

确认 joint1 转向和腕部相机可见性后，另一个终端启动 planning-only MoveIt，
再运行完整抓取。`--attempts 5 --required-successes 2` 可复用数据收集阶段的稳定
成功口径；不传 `--mode-index` 时选择最高权重均值，传 `--sample-index N` 可回放
推理 JSON 中第 N 个样本：

```bash
python custom_envs/scripts/N2M/replay_n2m_prediction.py \
    --prediction custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/quick_test.json \
    --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o100_v1 \
    --stop-after lift \
    --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --attempts 5 \
    --required-successes 2 \
    --headless
```

每次运行在 prediction 同级的 `prediction_replays/` 下保存 `run_config.json`、
`collision_preflight.json`、逐次 `rollouts.jsonl`、`summary.json` 和调试图像。

上游 N2M/PointBERT 的最小模型快照位于 `vendor_n2m/`，运行时不导入
`/home/mojie/N2M`。来源 commit 和本地修改记录在 `vendor_n2m/UPSTREAM.md`。
正式配置使用
`checkpoints/pointbert/PointTransformer_ModelNet8192points.pth` 初始化 PointBERT。
官方权重的首层只接收 XYZ；本项目 loader 会保留三通道预训练权重，并将新增 RGB
通道置零，然后在 N2M 训练中学习颜色特征。加载时要求本地 backbone 参数 100%
覆盖，只允许忽略原始 ModelNet 分类头，避免 `strict=False` 掩盖错误。

## 在线 NAV → N2M → 抓取 → 搬运 → 放置集成入口

`navigation/navigate_mujoco_n2m.py` 是独立入口，不修改已经验证的
`navigate_mujoco_moveit.py`。第一版使用 MuJoCo cube 真值计算距 cube 1.4m
的 Nav2 观察停靠点；停稳并朝向 cube 后，从 `front_realsense`
在线生成与训练一致的 base-frame XYZRGB 点云，运行一次 N2M，通过
MuJoCo 碰撞预检后将底盘直接放置到最高密度 M，并复用已验证的
`ARM_INIT -> LIFT` 执行器。默认在每个 M 重新运行三次 AnyGrasp；三次都失败
就保持 cube 当前状态并换到下一个无碰撞 M。最多尝试三个 M（共九次抓取），
仍失败就退出。成功 LIFT 后先把 cube 重新作为 MoveIt attached object，规划并
执行到原完整状态机的 `ARM_SIDE=(-π/2, 1.5, -1.5, 0, 1.2, 0)` 安全运输姿态；
执行中持续检查双指接触和夹爪开口。只有夹爪在短时间内从持物开口迅速塌缩到
接近闭合，或双指接触连续丢失，才判定 cube 掉落；cube/TCP 相对位移仅记录为
诊断指标。确认机械臂收回后才继续后撤、Nav2 搬运、末端对齐、转臂和松爪放置。

`--goal X Y` 为兼容旧命令保留，但在此新入口中会被忽略；原入口的
`--goal` 行为不变。house 场景的默认机器人初始位姿是已验证无碰撞的
`(x, y, yaw) = (1.6, 1.6, 0)`；也可通过 `--robot-x/--robot-y/--robot-yaw`
覆盖。先在终端 1 启动 Nav2 和 planning-only MoveIt：

```bash
source /opt/ros/humble/setup.bash
cd /home/mojie/taskdog
ros2 launch custom_envs/launch/nav2_moveit_mujoco.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_house_mujoco_nav2.yaml
```

再在 `env_isaaclab` 环境的终端 2 运行：

```bash
python custom_envs/scripts/navigation/navigate_mujoco_n2m.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --n2m-checkpoint custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/best.pt \
    --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --scene custom_envs/mujoco/scene_house_N2M.xml \
    --nav2-params custom_envs/config/nav2_params.yaml \
    --destination 12.0 12.0 \
    --render
```

终端 1 的 `params_file` 与终端 2 的 `--nav2-params` 应指向同一个文件。集成脚本
会读取其中的控制器/平滑器频率、最大线角速度、`velocity_timeout` 以及
`SimpleProgressChecker` 阈值，并按 MuJoCo 的 50 Hz 步长做真实时间同步，避免将
Nav2 的同一条 20 Hz 指令在仿真中无节制重复执行而走出弯曲、振荡轨迹。若底盘
在 YAML 配置的 60 秒内移动不足 0.1m，脚本会明确报告 `Nav2 progress timeout`
并退出，不再以零速度一直等到 `--destination-max-steps`。

默认每次运行的在线点云、图像、N2M prediction、候选队列、碰撞预检、
`lift_retraction_result.json` 和抓取/搬运结果保存到
`data/integrated_runs/<timestamp>/`。成功完整放置时
`summary.json` 的 `outcome` 为 `task_success`；仅 LIFT 成功但搬运失败不会被
误记为完整成功。
