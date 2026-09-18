"""Custom IsaacLab environments for DeepRobotics M20 Pro.

This package provides custom RL training environments for the M20 Pro robot dog.
It inherits from the official rl_training Deeprobotics M20 configurations,
allowing customization without modifying upstream code.
"""

try:
    from . import tasks  # noqa: F401
except (ModuleNotFoundError, ImportError):
    pass  # MuJoCo 运行环境下无 isaaclab_tasks，忽略
