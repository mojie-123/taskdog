"""FloAff-style data generation and learning for TaskDog.

The package is intentionally separate from :mod:`custom_envs.scripts.N2M` so
that dataset experiments cannot change the validated N2M/SysNav pipelines.
"""

from .config import DataGenerationConfig, load_data_generation_config

__all__ = ["DataGenerationConfig", "load_data_generation_config"]
