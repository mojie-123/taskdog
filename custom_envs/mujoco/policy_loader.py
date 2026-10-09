"""MuJoCo 建图和导航入口共用的 rsl_rl 行走策略加载器。

``OnPolicyRunner`` 原本期望一个训练环境；这里的 ``_MockEnv`` 只提供网络维度和
最小接口，使 checkpoint 能被官方 runner 正确还原。它不推进仿真，真实观测由
``obs_builder.py`` 构造，真实执行由 ``MuJoCoEnv.step`` 完成。
"""

import copy
import os

import torch
import yaml


def load_locomotion_policy(policy_path: str, device: str = "cuda"):
    """Load an rsl_rl checkpoint and return ``policy(obs_tensor)``.

    The network dimensions are inferred from the checkpoint and the runner
    configuration is loaded from ``params/agent.yaml`` next to the model.
    """
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise ImportError("请安装 rsl_rl") from exc

    policy_path = os.path.abspath(policy_path)
    run_dir = os.path.dirname(policy_path)
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"找不到策略 checkpoint: {policy_path}")
    if not os.path.isfile(agent_yaml):
        raise FileNotFoundError(
            f"找不到 agent.yaml: {agent_yaml}\n"
            "请确保 params/agent.yaml 与 checkpoint 在同一 run 目录下。"
        )
    with open(agent_yaml, encoding="utf-8") as stream:
        train_cfg = yaml.safe_load(stream)

    checkpoint = torch.load(
        policy_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    num_actor_obs = int(state_dict["actor.0.weight"].shape[1])
    num_critic_obs = int(state_dict["critic.0.weight"].shape[1])
    actor_last_key = max(
        (key for key in state_dict
         if key.startswith("actor.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    num_actions = int(state_dict[actor_last_key].shape[0])

    class _MockEnv:
        num_envs = 1
        max_episode_length = 1000
        episode_length_buf = torch.zeros(1, dtype=torch.long)
        cfg = {}

        def __init__(self):
            self.num_actions = num_actions
            self.device = device

        def get_observations(self):
            return {
                "policy": torch.zeros(1, num_actor_obs, device=device),
                "critic": torch.zeros(1, num_critic_obs, device=device),
            }

        def step(self, actions):
            del actions
            observations = self.get_observations()
            zeros = torch.zeros(1, device=device)
            return observations, zeros, zeros, {}

    runner = OnPolicyRunner(
        _MockEnv(),
        copy.deepcopy(train_cfg),
        log_dir=None,
        device=device,
    )
    # rsl_rl loads the checkpoint a second time inside runner.load().  Pass an
    # explicit mapping so CPU-only/headless runs can open CUDA-trained models.
    runner.load(policy_path, load_optimizer=False, map_location=device)
    raw_policy = runner.get_inference_policy(device=device)

    def policy(obs_tensor):
        return raw_policy({"policy": obs_tensor})

    print(
        f"[MJ] Policy loaded from {policy_path} "
        f"(actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, "
        f"actions={num_actions})",
        flush=True,
    )
    return policy
