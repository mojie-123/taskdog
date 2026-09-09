# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse   # Python 标准库提供的命令行参数解析工具：把字符串参数解析成python变量，让代码能够直接使用
import sys
import os

from isaaclab.app import AppLauncher   # Isaac Lab 提供的 Isaac Sim 仿真器启动器

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # /home/mojie/taskdog/deps/rl_training/scripts/reinforcement_learning
import cli_args   # cli_args.py

# 1、add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")   # 创建一个【参数解析器】对象
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")   # 任务名
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)   # 算法配置类入口点名称。默认值是 `"rsl_rl_cfg_entry_point"`
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")   # 训练轮数
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)   # 多GPU训练

# 2、append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)   # cli_args.add_rsl_rl_args(parser) 注册的 RSL-RL 专属参数
"""
包含：
--experiment_name   实验目录名
--run_name          运行后缀名
--resume            是否从 checkpoint 继续
--load_run          要恢复的 run 目录名
--checkpoint        要恢复的 .pt 文件名
--logger            日志后端（tensorboard/wandb/neptune）
--log_project_name  wandb/neptune 项目名
"""
# 3、append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)   # AppLauncher.add_app_launcher_args(parser) 注册 Isaac Sim 参数，如--headless、--device cuda:0、--livestream等
args_cli, hydra_args = parser.parse_known_args()   # `args_cli`：argparse 认识的参数，存入 Namespace 对象。`hydra_args`：argparse 不认识的参数，原样保留为字符串列表。

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True   # isaac sim参数：在启动isaac sim时是否需要初始化相机渲染管线

# clear out sys.argv for Hydra。`@hydra_task_config` 装饰器内部调用了 `hydra.main()`，Hydra会重新从 `sys.argv` 读取参数
sys.argv = [sys.argv[0]] + hydra_args   # 把 `sys.argv` 重置，只保留 Hydra 的参数（sys.argv[0]是脚本名train.py）（Hydra参数例如hydra_args = ["env.seed=42"]，即Hydra是一种风格）

# launch omniverse app
app_launcher = AppLauncher(args_cli)   # 传入解析好的 CLI 参数
simulation_app = app_launcher.app   # 获取仿真器 App 句柄

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
# 用于屏蔽掉一些无用的警告信息，不写入日志
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata   # 用于查询已安装包的元数据（版本号、作者、依赖等）
import platform
from packaging import version

# RSL-RL 版本检测（>=3.0.1 才继续）check minimum supported rsl-rl version
# 版本太低会导致OnPolicyRunner接口不兼容
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")   # 查询rsl-rl-lib的已安装包的版本号，返回字符串
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]   # ./isaaclab.sh -p -m pip install rsl-rl-lib==3.0.1
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
# 两个核心能力：
# - 全局注册表：`gym.register()` / `gym.make()` / `gym.spec()`
# - 环境接口规范：所有 RL 环境必须实现 `reset()` / `step()` / `close()`

import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,   # 多智能体环境基类
    DirectMARLEnvCfg,   # 多智能体环境配置基类
    DirectRLEnvCfg,   # 直接 RL 环境配置基类
    ManagerBasedRLEnvCfg,   # manager-based RL 环境配置基类
    multi_agent_to_single_agent,   # MARL→单智能体转换函数
)
from isaaclab.utils.dict import print_dict   # 打印调试用
from isaaclab.utils.io import dump_yaml   # 用于每次训练开始前把本次训练用到的完整配置保存到日志目录（params/env.yaml和params/agent.yaml）
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  
# RslRlOnPolicyRunnerCfg：
# |
# +-- Runner 自身的运行参数（训练循环的控制旋钮）
# |   seed = 42                    随机种子，保证可复现
# |   device = "cuda:0"            用哪块 GPU
# |   num_steps_per_env = 24       每次 rollout 每个环境走多少步
# |                                4096个环境 x 24步 = 98304 条经验/次更新
# |   max_iterations = 5000        总共更新多少次
# |   save_interval = 50           每50次更新保存一个 checkpoint
# |   experiment_name = "deeprobotics_m20_rough"  日志目录名
# |   logger = "tensorboard"       用什么工具记录曲线
# |   resume = False               是否断点续训
# |   load_run = ".*"              断点续训时加载哪个 run
# |   load_checkpoint = "model_.*.pt"  加载哪个 checkpoint 文件
# |
# +-- policy：神经网络的结构
# |   actor_hidden_dims = [512, 256, 128]   Actor 网络的隐藏层
# |   critic_hidden_dims = [512, 256, 128]  Critic 网络的隐藏层
# |   activation = "elu"                    激活函数
# |   init_noise_std = 1.0                  初始探索噪声标准差
# |   actor_obs_normalization = True         是否对输入做归一化
# |
# +-- algorithm：PPO 算法的超参数
#     num_learning_epochs = 5     每次 rollout 后用数据训多少个 epoch
#     num_mini_batches = 4        把数据切成多少个 mini-batch
#     learning_rate = 1e-3        学习率
#     gamma = 0.99               折扣因子（未来奖励打折率）
#     lam = 0.95                 GAE lambda
#     entropy_coef = 0.01        熵正则系数（鼓励探索）
#     clip_param = 0.2           PPO 裁剪参数 epsilon
#     desired_kl = 0.01          目标 KL 散度（自适应学习率）

# RslRlVecEnvWrapper把 IsaacLab 的 Gym 环境包装成 RSL-RL 的 `VecEnv` 接口
    # IsaacLab 的 `env.step()` 返回：
    #       (obs_dict, reward, terminated, truncated, extras)  # Gym 5-tuple
    # RSL-RL 的 `OnPolicyRunner` 期望的接口是：
    #       (TensorDict(obs), reward, dones, extras)  # RSL-RL 4-tuple

    # RslRlVecEnvWrapper 做的三件事（源码可见）：
    # def step(self, actions):
    #     # ① 动作裁剪
    #     if self.clip_actions is not None:
    #         actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
    #     # ② 调用真实环境，转换返回格式
    #     obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    #     dones = (terminated | truncated).to(dtype=torch.long)  # 合并 terminated+truncated
        
    #     # ③ 处理无限时域任务的 time_out 信息
    #     if not self.unwrapped.cfg.is_finite_horizon:
    #         extras["time_outs"] = truncated
        
    #     return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg    # 这个函数在 IsaacLab 2.4+ 才存在，用于把旧版 RSL-RL 配置自动迁移到新版
except ImportError:
    # Fallback for IsaacLab <2.4 where this function doesn't exist
    def handle_deprecated_rsl_rl_cfg(cfg, installed_version):
        return cfg
from isaaclab_tasks.utils import get_checkpoint_path   # 根据三个参数拼出 `.pt` 模型文件的完整绝对路径，支持正则表达式匹配
from isaaclab_tasks.utils.hydra import hydra_task_config

# 触发gym.register()
import rl_training.tasks  # noqa: F401
import custom_envs.tasks  # noqa: F401 (register M20Pro environments)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

#               env_cfg                       agent_cfg
#               (ManagerBasedRLEnvCfg)        (RslRlOnPolicyRunnerCfg)
# 描述对象       仿真环境                       训练过程
# 被谁用         ManagerBasedRLEnv 实现类       OnPolicyRunner 训练器
# 控制什么       地形/机器人/奖励/观测            网络结构/PPO超参/日志/checkpoint
# 典型字段       num_envs, terrain_type,        max_iterations, learning_rate,
#                reward_weights, obs_dim        actor_hidden_dims, gamma




#在main执行前完成：
# 1、从 Gym 注册表加载环境配置：调用 `load_cfg_from_registry(task_name, "env_cfg_entry_point")` 加载环境配置类（如 `RoughEnvCfg`）并实例化。
# 2、加载 agent 配置：调用 `load_cfg_from_registry(task_name, args_cli.agent)` 加载算法配置类（如 `RoughPPORunnerCfg`）并实例化
# 3、注册到 Hydra ConfigStore：把实例化的配置对象转成字典，存入 Hydra 的配置数据库
    # {
    #   "env": {
    #     "scene": {"num_envs": 4096, ...},
    #     "rewards": {"track_lin_vel": {"weight": 1.0}, ...},
    #     "observations": {...},
    #     ...
    #   },
    #   "agent": {
    #     "seed": 42,
    #     "max_iterations": 5000,
    #     "policy": {"actor_hidden_dims": [512, 256, 128], ...},
    #     "algorithm": {"learning_rate": 0.001, ...},
    #     ...
    #   }
    # }
# 4、用 `@hydra.main()` 包装并调用：Hydra 解析剩余的 `sys.argv`（即之前保留的 `hydra_args`），允许用命令行 `key=value` 方式覆盖任意配置字段，然后把最终的 `env_cfg` 和 `agent_cfg` 对象注入到 `main()` 的两个参数中
#   ConfigStore 内部：
#   {
#     "Rough-Deeprobotics-M20Pro-v0": { "env": {...}, "agent": {...} }  ← 刚存进去的
#   }



# e.g.
# # 用户执行命令
# python train.py --task=Rough-Deeprobotics-M20Pro-v0 --headless env.scene.num_envs=2048 agent.algorithm.learning_rate=0.0003

# # argparse 吃掉：--task, --headless
# # hydra_args 剩下：["env.scene.num_envs=2048", "agent.algorithm.learning_rate=0.0003"]
# # sys.argv 被替换成：["train.py", "env.scene.num_envs=2048", "agent.algorithm.learning_rate=0.0003"]

# # Hydra 读取 sys.argv：
# # ConfigStore 里的默认值：num_envs=4096, learning_rate=0.001
# # 命令行覆盖：num_envs=2048, learning_rate=0.0003
# # 最终 hydra_env_cfg["env"]["scene"]["num_envs"] = 2048    ← 被命令行覆盖
# # 最终 hydra_env_cfg["agent"]["algorithm"]["learning_rate"] = 0.0003  ← 被命令行覆盖
# 然后：
# env_cfg.from_dict(hydra_env_cfg["env"])    # 把合并后的值写回 env_cfg 对象
# agent_cfg.from_dict(hydra_env_cfg["agent"]) # 把合并后的值写回 agent_cfg 对象

# 整个脚本的完整时间线：
# 执行 python train.py --task=Rough-M20Pro-v0 --headless env.scene.num_envs=2048
# │
# ├─① argparse 解析
# │   args_cli.task = "Rough-Deeprobotics-M20Pro-v0"
# │   args_cli.agent = "rsl_rl_cfg_entry_point"   （默认值）
# │   hydra_args = ["env.scene.num_envs=2048"]
# │   sys.argv 被替换为 ["train.py", "env.scene.num_envs=2048"]
# │
# ├─② AppLauncher 启动 Isaac Sim（此时才能 import isaaclab 相关库）
# │
# ├─③ 触发 gym.register()（import rl_training.tasks, import custom_envs.tasks）
# │   gym.envs.registry 里现在有 "Rough-Deeprobotics-M20Pro-v0"
# │
# ├─④ Python 读到 @hydra_task_config(...)
# │   等价于：main = hydra_task_config("Rough-M20Pro-v0", "rsl_rl_cfg_entry_point")(main)
# │   此时 main 这个名字指向 wrapper 函数
# │   （注意：只是「包装」，还没有执行任何配置加载逻辑！）
# │
# ├─⑤ 执行到 if __name__ == "__main__": main()
# │   实际调用 wrapper()
# │
# │   wrapper 内部：
# │   ├─⑤-A register_task_to_hydra()
# │   │   ├─ load_cfg_from_registry("Rough-M20Pro-v0", "env_cfg_entry_point")
# │   │   │   ├─ gym.spec("Rough-M20Pro-v0").kwargs["env_cfg_entry_point"]
# │   │   │   │   → "custom_envs.tasks...rough_env_cfg:DeeproboticsM20ProRoughEnvCfg"
# │   │   │   ├─ importlib.import_module("custom_envs.tasks...rough_env_cfg")
# │   │   │   └─ DeeproboticsM20ProRoughEnvCfg()  → env_cfg（默认值）
# │   │   ├─ load_cfg_from_registry("Rough-M20Pro-v0", "rsl_rl_cfg_entry_point")
# │   │   │   └─ M20ProRoughPPORunnerCfg()  → agent_cfg（默认值）
# │   │   ├─ env_cfg.to_dict() + agent_cfg.to_dict() → cfg_dict
# │   │   └─ ConfigStore.store("Rough-M20Pro-v0", cfg_dict)  存入 Hydra 数据库
# │   │
# │   └─⑤-B hydra_main() 被调用
# │       ├─ Hydra 读取 sys.argv：["env.scene.num_envs=2048"]
# │       ├─ 从 ConfigStore 取出默认值，用 CLI 参数覆盖
# │       │   num_envs: 4096 → 2048（被覆盖）
# │       ├─ env_cfg.from_dict(合并后的env配置)  → env_cfg 更新
# │       ├─ agent_cfg.from_dict(合并后的agent配置) → agent_cfg 更新
# │       └─ func(env_cfg, agent_cfg)  ← 调用你写的真正的 main()
# │
# └─⑥ main(env_cfg, agent_cfg) 执行
#     env_cfg.scene.num_envs == 2048   ← 命令行改的，生效了
#     env_cfg.rewards...               ← 代码里的默认值
#     agent_cfg.algorithm.learning_rate == 0.001  ← 没改，是默认值
#     ...
#     gym.make(args_cli.task, cfg=env_cfg)  → 创建 2048 个并行环境

@hydra_task_config(args_cli.task, args_cli.agent)   # 等价于main = hydra_task_config(args_cli.task, args_cli.agent)(main)。
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)   # CLI 参数覆盖 Hydra 配置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)   # 若配置是旧格式（policy字段），转换成新格式（actor/critic字段）

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # 确定日志目录specify directory for logging experiments
    # 最终路径：logs/rsl_rl/{experiment_name}/{timestamp}_{run_name}/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # 创建issaaclab环境create isaac environment
    # 查 `gym.envs.registry` 字典，找到注册时的 `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
    # 动态导入 `isaaclab.envs` 模块，取出 `ManagerBasedRLEnv` 类
    # 用 `cfg=env_cfg` 实例化这个类 → 创建 N 个并行仿真环境
    # 返回一个标准 Gym 接口的 `env` 对象
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果是多智能体环境，转换为单智能体convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 在 gym.make() 之前，先记录 resume_path。save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL环境包装wrap around environment for rsl-rl
    # 把 Gym 的 `(obs, reward, terminated, truncated, info)` 元组转换为 RSL-RL `OnPolicyRunner` 期望的格式
    # 对网络输出的动作做 clip（防止极端值破坏仿真）
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 PPO Runner 。convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    #   agent_cfg 对象的一生：

    #   [rsl_rl_ppo_cfg.py]          [装饰器内部]              [main() 里]
    #   实例化对象                   对象→字典→ConfigStore       对象→字典
    #        │                            │                        │
    #        ▼                            ▼                        ▼
    #   agent_cfg 对象    ──►    Hydra 用字典处理参数覆盖    agent_cfg.to_dict()
    #   (含所有默认值)            字典是临时中间状态           给 OnPolicyRunner
    #                             处理完还原成对象
    #                                    │
    #                             main(env_cfg, agent_cfg)
    #                                    │
    #                             对象（不是字典）

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)   # 创建 runner 后加载权重

    # 训练前的配置保存dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # 开始训练run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # 训练过程中会自动保存模型：
    # logs/rsl_rl/{experiment_name}/{timestamp}/
    # params/
    #     env.yaml         <- 环境配置快照（训练前 dump）
    #     agent.yaml       <- 算法配置快照（训练前 dump）
    # model_100.pt         <- 第100次迭代的 checkpoint
    # model_200.pt
    # ...
    # model_{N}.pt         <- 最终 checkpoint
    # `.pt` 文件里保存的是 PyTorch 的 `state_dict`，包含 actor 和 critic 网络的所有权重


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()