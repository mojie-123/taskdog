# add_lidar.md — M20 Pro 加装 Livox Mid-360 LiDAR 完整实现方案

> **状态**: 方案设计阶段（尚未实现）
> **目标**: 在 M20 Pro 仿真模型的前背位置安装一个 Livox Mid-360 LiDAR 传感器，使仿真中的机器人能「看到」周围环境，而非盲走。

---

## 目录

1. [背景：为什么需要 LiDAR？现在 M20 是怎么「走」的？](#1-背景为什么需要-lidar现在-m20-是怎么走的)
2. [Livox Mid-360 是什么？](#2-livox-mid-360-是什么)
3. [整体思路：在仿真中加一个「虚拟雷达」](#3-整体思路在仿真中加一个虚拟雷达)
4. [需要新建/修改哪些文件](#4-需要新建修改哪些文件)
5. [每个文件详细说明](#5-每个文件详细说明)
6. [一步步实现清单](#6-一步步实现清单)
7. [Q&A](#7-qa)

---

## 1. 背景：为什么需要 LiDAR？现在 M20 是怎么「走」的？

### 1.1 现状：M20 是「盲人」

现在你训练的 `Rough-Deeprobotics-M20-v0` 和 `Flat-Deeprobotics-M20-v0` 环境，机器人**看不到任何东西**：

```
机器人观测 (57 维):
    ├── 自己的关节角度/速度    ←「我的腿在什么位置」
    ├── 自己的姿态/角速度      ←「我有没有歪」
    ├── 速度命令 (vx, vy, ωz)  ←「我要往哪走」
    └── 上一步的动作           ←「我刚才做了什么」

    没有任何关于「前面有没有墙」「脚下是不是坑」的信息！
```

当前 `height_scanner`（高度扫描器）只是用来测脚底下的地形高度，让策略学会「在崎岖地形上保持平衡」。它**不能**看到远处的障碍物。

### 1.2 目标：让机器人「看见」

加上 LiDAR 后，观测变成：

```
机器人观测 (57 + 192 = 249 维):
    ├── 原有的 57 维 (自身状态 + 速度命令)
    └── LiDAR 点云 192 维 ←「我周围有什么东西」
         │
         包含: 64 个点的 (x, y, z) 坐标
         每个点 = 一束激光打到障碍物上反射回来的位置
         点越近 = 障碍物离机器人越近
```

有了这些信息，策略就可以学会：
- 「前方 2 米有堵墙」→ 绕开
- 「右侧有障碍物」→ 往左转
- 「前面有坑/台阶」→ 跳跃或绕行

**这就是从「盲走 locomotion」到「感知导航 perceptive locomotion」的升级。**

---

## 2. Livox Mid-360 是什么？

### 2.1 通俗解释

LiDAR (激光雷达) 就像一个**高速旋转的手电筒**，但它发出的不是可见光而是激光。它向四面八方发射成千上万束激光，激光碰到障碍物反射回来，传感器记录每束激光的往返时间，算出一组 3D 点——**点云 (point cloud)**。

```
LiDAR 发射激光 → 碰到障碍物 → 反射回来 → 记录 (x, y, z) 坐标

每一帧输出 = 一组 3D 点（比如 14,400 个点）
每个点 = 这个位置有一个障碍物表面
```

### 2.2 Livox Mid-360 的关键参数

| 参数 | 数值 | 通俗含义 |
|------|------|----------|
| 水平视场角 | 360° | 能看周围一圈，没有死角 |
| 垂直视场角 | -7° ~ +52° | 能看地面以下 7°（近处）,也能看上方 52°（高处） |
| 等效线数 | 40 线 | 垂直方向有 40 条激光束同时发射，比较密集 |
| 最大测距 | 70 米 | 能看清 70 米内的东西 |
| 帧率 | 10 Hz | 每秒钟扫描 10 次 |
| 外形 | 65×65×60mm, 265g | 很小很轻，适合装在机器人上 |

### 2.3 安装在什么位置

安装在 M20 的身体**前背**（base_link 顶面靠前）：

```
        ┌──────┐
        │LiDAR │ ← 装在这里，x=0.30m 前, z=0.55m 高
   ┌────┴──────┴────┐
   │    base_link   │
   └───────┬────────┘
   前轮   │    后轮
          ↓
       前进方向 x+
```

### 2.4 40 条线怎么产生 14,400 个点？

**40 条线是垂直方向的线数，每条线还要在水平方向上转一圈。**

想象一个苹果：竖着切 40 刀，横着切 360 刀，苹果被切成 40 × 360 = 14,400 个交点——每个交点就是一个激光测距点。

```
LiDAR 的工作方式:

    ┌─────────────────────────────────┐
    │  LiDAR 内部有一个旋转镜片        │
    │                                 │
    │  每一「条」激光 (共 40 条):       │
    │    镜片转 1° → 发一束激光        │
    │    镜片转 2° → 发一束激光        │
    │    镜片转 3° → 发一束激光        │
    │    ...                          │
    │    镜片转 360° → 回到起点        │
    │                                 │
    │  40 条 × 每度 1 束 × 360°       │
    │  = 14,400 束激光 / 帧            │
    │  = 14,400 个 3D 点 / 帧         │
    └─────────────────────────────────┘

参数对应:
    channels=40           → 垂直方向 40 条线 (相当于苹果竖切 40 刀)
    horizontal_res=1.0    → 水平方向每 1° 一束 (相当于苹果横切 360 刀)
    40 × (360/1.0) = 14,400 束射线 = 14,400 个测距点 / 帧
```

**如果 horizontal_res 调到 0.72°呢？** 那就是 40 × (360/0.72) ≈ 20,000 点/帧，更密集。但仿真中为了性能通常设 1.0°（数据少一点但够用，仿真也跑得更快）。

---

## 3. 整体思路：在仿真中加一个「虚拟雷达」

### 3.1 核心工具：IsaacLab 的 RayCaster

IsaacLab 提供了一个叫 **RayCaster** 的传感器工具。你可以把它理解为一个「射线发射器」——从机器人的某个位置向指定方向发射大量射线，射线碰到物体后返回击中点的坐标。

```
RayCaster 配置 = {
    挂在哪:        base_link (机器人的身体)
    偏移量:        (0.3m, 0, 0.55m) ← 身体的上前方
    射线模式:      模仿 Mid-360 (360°水平, 59°垂直, 40条线)
    探测距离:      最多 70 米
    更新频率:      每秒 10 次
    探测目标:      /World/ground (地面和障碍物)
}
```

运行时,`scene["mid360_lidar"].data.ray_hits_w` 会输出一个三维张量 `[环境数, 射线数, 3]`，每行是一个击中点的 (x, y, z) 世界坐标。

**这就是仿真的 LiDAR 数据**，等价于真实 Mid-360 输出的点云。

### 3.2 继承链（关键！）

我们通过**继承**把 LiDAR 加到 M20 Pro 上。不需要修改 `rl_training` 的上游代码：

```
rl_training 官方代码 (不动)
    │
    └─ DeeproboticsM20RoughEnvCfg     ← M20 崎岖地形基础配置
        └─ DeeproboticsM20ProRoughEnvCfg   ← custom_envs (已有)
            └─ DeeproboticsM20ProLidarRoughEnvCfg   ← ★ 新建!
                │
                super().__post_init__() 之后追加:
                ├── self.scene.mid360_lidar = RayCasterCfg(...)    ← 安装雷达
                ├── self.observations.policy.lidar = ObsTerm(...)   ← 加到观测
                └── self.observations.critic.lidar = ObsTerm(...)   ← 加到评估
```

**相当于**: M20 Pro 已经会走了 → 在这个基础上给它装个雷达 → 让它学会用雷达数据来走。

### 3.3 点云数据怎么用

RayCaster 输出的原始数据是 14,400 个三维点，太多太大不能直接喂给神经网络。需要**降采样**：

```
原始 14,400 点 → KNN 取最近的 64 个点 → 展平为一维 [64×3=192]
                                     ↓
                          这 192 个数 = 「离我最近的 64 个障碍物点的坐标」
                          最近的 = 最危险的 = 最有信息量
```

---

## 4. 需要新建/修改哪些文件

```
custom_envs/
├── utils/
│   ├── lidar_pattern.py          ★ 新建
│   └── lidar_observation.py      ★ 新建
│
└── tasks/deeprobotics_m20_pro/
    ├── __init__.py                ○ 修改 (增加 2 个 gym.register)
    ├── lidar_rough_env_cfg.py     ★ 新建
    ├── lidar_flat_env_cfg.py      ★ 新建
    └── agents/
        └── rsl_rl_ppo_lidar_cfg.py ★ 新建
```

**共新建 5 个文件，修改 1 个文件。**

下面逐个说明每个文件写什么、为什么写。

---

## 5. 每个文件详细说明

### 5.1 `custom_envs/utils/lidar_pattern.py` — 定义「雷达长什么样」

这个文件只做一件事：告诉 RayCaster「我的 LiDAR 有多少条线、看多宽、看多远」。

```python
from isaaclab.sensors.ray_caster import patterns

def get_mid360_lidar_pattern():
    return patterns.LidarPatternCfg(
        channels=40,                            # 40 条激光线
        vertical_fov_range=(-7.0, 52.0),        # 垂直看 -7° 到 +52°
        horizontal_fov_range=(-180.0, 180.0),   # 水平看 360° 一圈
        horizontal_res=1.0,                     # 水平每 1° 发一条射线
    )
```

**为什么**: 这就是仿真的「雷达规格书」。训练策略时，RayCaster 按照这个模式发射射线，输出匹配真实 Mid-360 视野的点云。

### 5.2 `custom_envs/utils/lidar_observation.py` — 把原始点云压缩成神经网络能吃的格式

```python
import torch

def lidar_point_cloud_downsample(ray_hits_w, num_points=64):
    """从几万个点中挑出离机器人最近的 64 个点。"""
    # ray_hits_w: [N, B, 3] → N 个环境, B 条射线, 3=xyz
    # 未击中的射线值是 inf, 需要替换成一个大数
    # 按距离排序,取前 num_points 个
    # 展平: [N, 64*3] = [N, 192]
    ...
```

**为什么**: 神经网络（MLP）要求输入是固定长度的向量。原始点云的点数不固定（有的方向没打到东西），且 14,400 点太大。通过取最近的 64 个点，得到固定 192 维向量——「最近的点 = 最需要躲避的障碍物」。

### 5.3 `custom_envs/tasks/deeprobotics_m20_pro/lidar_rough_env_cfg.py` — 把雷达装到机器人上

这是**最核心的文件**。它继承已有的 M20 Pro 崎岖地形配置，在 `super().__post_init__()` 加载完所有默认设置之后，追加 LiDAR。

```python
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from custom_envs.tasks.deeprobotics_m20_pro.rough_env_cfg import DeeproboticsM20ProRoughEnvCfg
from custom_envs.utils.lidar_pattern import get_mid360_lidar_pattern
from custom_envs.utils.lidar_observation import lidar_point_cloud_downsample


@configclass
class DeeproboticsM20ProLidarRoughEnvCfg(DeeproboticsM20ProRoughEnvCfg):

    def __post_init__(self):
        # 第一步: 加载父类的全部配置 (M20 模型、地形、奖励、观测...)
        super().__post_init__()

        # 第二步: 在场景中添加 LiDAR 传感器
        self.scene.mid360_lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            # ↑ 挂载点: base_link (机器人的身体),{ENV_REGEX_NS} 会自动匹配每个并行环境
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.30, 0.0, 0.55),
                # ↑ 在 base_link 坐标系中的位置: 前 30cm, 高 55cm
            ),
            ray_alignment="base",
            # ↑ 射线跟随机器人的完整姿态 (不是只跟偏航角)
            pattern_cfg=get_mid360_lidar_pattern(),
            # ↑ 雷达规格: 40 线, 360°×59°
            max_distance=70.0,
            # ↑ 最远看 70 米
            update_period=0.1,
            # ↑ 每秒更新 10 次
            debug_vis=True,
            # ↑ 开发阶段可视化射线 (正式训练时关掉以提升性能)
            mesh_prim_paths=["/World/ground"],
            # ↑ 射线探测的目标: 地面 (后续可加入障碍物)
        )

        # 第三步: 把 LiDAR 数据加入观测
        self.observations.policy.lidar = ObsTerm(
            func=lidar_point_cloud_downsample,
            params={"num_points": 64},
        )
        self.observations.critic.lidar = ObsTerm(
            func=lidar_point_cloud_downsample,
            params={"num_points": 64},
        )

        # 第四步: 清理零权重奖励
        self.disable_zero_weight_rewards()
```

**关键参数解释**:

| 参数 | 值 | 通俗意思 |
|------|-----|----------|
| `prim_path` | `{ENV_REGEX_NS}/Robot/base_link` | 雷达装在机器人的 base_link 上，`{ENV_REGEX_NS}` 是一个自动替换的变量，训练时有 4096 个环境并行，它自动变成 `/World/envs/env_0/Robot/base_link`、`/World/envs/env_1/...` 等等 |
| `offset.pos` | `(0.30, 0.0, 0.55)` | 在 base_link 的坐标系下，雷达位于：x=0.3m（前），y=0（正中间），z=0.55m（身高 55cm） |
| `ray_alignment` | `"base"` | 雷达射线跟随机器人身体的旋转（俯仰/翻滚/偏航都跟），这样机器人歪了雷达也能看到正确的方向 |
| `max_distance` | `70.0` | 超过 70 米的点被认为是「没打中东西」 |
| `update_period` | `0.1` | 每 0.1 秒更新一次，相当于 10Hz |
| `debug_vis` | `True` | 在 Isaac Sim 窗口中渲染出彩色射线，方便确认雷达位置对不对 |
| `mesh_prim_paths` | `["/World/ground"]` | 目前只探测地面，后续添加障碍物后需要更新这个列表 |

### 5.4 `custom_envs/tasks/deeprobotics_m20_pro/lidar_flat_env_cfg.py` — 平坦地形版本

```python
from isaaclab.utils import configclass
from custom_envs.tasks.deeprobotics_m20_pro.lidar_rough_env_cfg import (
    DeeproboticsM20ProLidarRoughEnvCfg,
)

@configclass
class DeeproboticsM20ProLidarFlatEnvCfg(DeeproboticsM20ProLidarRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 把地形改成平面 (覆盖 rough 的崎岖地形)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.disable_zero_weight_rewards()
```

**为什么需要**: 训练通常先在平坦地形上验证管线畅通，再上崎岖地形。这个文件把雷达环境的地形切换为平面。

### 5.5 `custom_envs/tasks/deeprobotics_m20_pro/agents/rsl_rl_ppo_lidar_cfg.py` — 为更大观测调整神经网络

**为什么需要专门的文件**: 原来的观测是 57 维，加了 LiDAR 变为 57+192=249 维。更大的输入需要更大的网络来消化：

```python
@configclass
class DeeproboticsM20ProRoughLidarPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20pro_lidar_rough"   # ← 日志目录名
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 512, 256, 128],    # 比原来 [512,256,128] 多一层
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-3, clip_param=0.2, entropy_coef=0.003,
        gamma=0.99, lam=0.95, ...
    )
```

**变化**: `actor_hidden_dims` 从 `[512, 256, 128]` 增加到 `[512, 512, 256, 128]`，多了一层 512 的隐藏层来更好地处理高维 LiDAR 输入。

### 5.6 `custom_envs/tasks/deeprobotics_m20_pro/__init__.py` — 注册新环境

在这个文件的**末尾追加**两段 `gym.register()`，让训练脚本能通过 `--task` 找到我们的新环境：

```python
# 在已有的 Flat-Deeprobotics-M20Pro-v0 和 Rough-Deeprobotics-M20Pro-v0 后面追加:

gym.register(
    id="Rough-Deeprobotics-M20Pro-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_rough_env_cfg:DeeproboticsM20ProLidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lidar_cfg:DeeproboticsM20ProRoughLidarPPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20Pro-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lidar_flat_env_cfg:DeeproboticsM20ProLidarFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_lidar_cfg:DeeproboticsM20ProFlatLidarPPORunnerCfg",
    },
)
```

注册完成后，`list_envs.py` 就能看到这两个新环境，`train.py --task=Flat-Deeprobotics-M20Pro-Lidar-v0` 就能启动训练。

---

## 6. 一步步实现清单

```
[ ] 1. 创建 custom_envs/utils/lidar_pattern.py
        → 写 get_mid360_lidar_pattern() 函数
        → 验证: 脚本 import 不报错

[ ] 2. 创建 custom_envs/utils/lidar_observation.py
        → 写 lidar_point_cloud_downsample() 函数
        → 验证: 用 torch.randn 造假数据测试输出 shape

[ ] 3. 创建 custom_envs/tasks/.../lidar_rough_env_cfg.py
        → 继承 DeeproboticsM20ProRoughEnvCfg
        → 在 __post_init__ 中添加:
            self.scene.mid360_lidar = RayCasterCfg(...)
            self.observations.policy.lidar = ObsTerm(...)
            self.observations.critic.lidar = ObsTerm(...)

[ ] 4. 创建 custom_envs/tasks/.../lidar_flat_env_cfg.py
        → 继承 lidar_rough, 覆盖地形为平面

[ ] 5. 创建 custom_envs/tasks/.../agents/rsl_rl_ppo_lidar_cfg.py
        → 写 PPO 配置 (更大网络)

[ ] 6. 修改 custom_envs/tasks/.../__init__.py
        → 追加两个 gym.register()

[ ] 7. 验证环境注册
        → python scripts/tools/list_envs.py
        → 期望看到 Flat/Rough-Deeprobotics-M20Pro-Lidar-v0

[ ] 8. 可视化验证 (debug_vis=True)
        → python scripts/.../play.py --task=Flat-Deeprobotics-M20Pro-Lidar-v0
        → 在 Isaac Sim 窗口中确认:
            射线从 base_link 上方发出
            射线程 360° 覆盖
            击中地面时显示彩色点

[ ] 9. 数据验证
        → 在 play.py 中加一行 numpy.save("lidar.npy", ray_hits_w)
        → 用 Python 脚本加载并检查: 不全是 inf, 有合理的 xyz 范围

[ ] 10. 短期训练测试
         → python scripts/.../train.py --task=Flat-Deeprobotics-M20Pro-Lidar-v0
            --headless --num_envs=1024 --max_iterations=200
         → 检查: 不报错, model_200.pt 正常保存

[ ] 11. 完整训练 + 策略评估
```

---

## 7. Q&A

### Q1: 加装 LiDAR 不需要添加 URDF 文件吗？

**不需要。**

你可能在之前的 project.md 中看到 `deeprobotics.py` 中的 `DEEPROBOTICS_M20_CFG` 是通过 USD 文件路径加载机器人模型的：

```python
DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/M20/M20_usd/M20.usd",
        ...
    ),
)
```

这里的 `M20.usd` 文件定义了机器人的**物理身体**——每个 link 的形状、质量、关节连接关系。URDF 是这些 USD 文件的「原始图纸」。

**但传感器不在这个范畴内。** 在 IsaacLab 中，传感器是通过 Python 配置**运行时动态挂载**的，不是写在 URDF/USD 模型中的静态零件。

类比理解：

```
URDF/USD 文件 = 机器人的「骨骼、肌肉」  → 出厂时就定了
RayCasterCfg  = 机器人的「外设/配件」  → 运行时装上去
```

具体来说，`RayCasterCfg` 中的 `prim_path="{ENV_REGEX_NS}/Robot/base_link"` 这个参数，就是告诉 IsaacLab「在运行的时候，把这个虚拟雷达挂到 base_link 这个 prim 上」。这个挂载动作是**纯代码层面**的，不涉及任何文件修改：

```python
# 这句话就是在「安装」雷达，不需要改任何 URDF/USD
self.scene.mid360_lidar = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",  # ← 挂到机器人的 base_link 上
    offset=RayCasterCfg.OffsetCfg(
        pos=(0.30, 0.0, 0.55),                   # ← 前背位置
    ),
    ...
)
```

**为什么这样设计**: 这恰恰是 IsaacLab 的优势——你可以用纯 Python 代码给任何机器人加装传感器，不需要懂 USD/URDF 的复杂格式。同样的代码，改一下 `prim_path` 就可以给 Go2、ANYmal、甚至你自己设计的机器人装上 LiDAR。

**如果真机部署怎么办？** 真机的 Mid-360 是通过物理支架安装的，和仿真无关。训练的时候仿真用 `RayCasterCfg` 产生「仿真 LiDAR 数据」，部署的时候真机通过 ROS2 话题读真实 LiDAR 数据，两者格式对齐即可。
