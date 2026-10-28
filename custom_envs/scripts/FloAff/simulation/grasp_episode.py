"""Thin adapter over the validated N2M manipulation episode.

Keeping this adapter small is deliberate: M labels must reflect the same grasp
pipeline used by N2M/SysNav, rather than a copied implementation that silently
drifts over time.
"""

from __future__ import annotations

from pathlib import Path

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.N2M.config import EpisodeConfig
from custom_envs.scripts.N2M.manipulation_episode import ManipulationEpisodeRunner


class FloAffGraspRunner:
    def __init__(
        self,
        config: DataGenerationConfig,
        *,
        grasp_checkpoint: str,
        render: bool = False,
        debug_dir: Path | None = None,
    ):
        episode = EpisodeConfig(
            object_name=config.scene.target_body,
            robot_root_z=config.scene.robot_root_z,
            robot_root_roll=0.0,
            robot_root_pitch=0.0,
            semantic_target_mask=True,
            dynamic_support_surface=True,
            moveit_insertion=0.01,
        )
        self.runner = ManipulationEpisodeRunner(
            Path(config.scene.scene_xml),
            episode,
            grasp_checkpoint=grasp_checkpoint,
            render=render,
            debug_dir=debug_dir,
        )

    def run(self, candidate: dict, attempt: int):
        return self.runner.run_episode(candidate, attempt=attempt, stop_after="lift")

    def close(self) -> None:
        self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
