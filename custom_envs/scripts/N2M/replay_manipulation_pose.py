#!/usr/bin/env python
"""Replay one manifest candidate or one saved successful manipulation pose."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.config import DEFAULT_DATA_DIR, DEFAULT_SCENE, EpisodeConfig
from custom_envs.scripts.N2M.interactive import InteractiveController
from custom_envs.scripts.N2M.manifests import read_jsonl
from custom_envs.scripts.N2M.manipulation_episode import ManipulationEpisodeRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--grasp-checkpoint", default=None)
    parser.add_argument("--stop-after", choices=["arm_init", "lift"], default="lift")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--render", dest="render", action="store_true",
        help="open the interactive MuJoCo viewer",
    )
    display_group.add_argument(
        "--headless", dest="render", action="store_false",
        help="run with off-screen EGL rendering and no viewer (default)",
    )
    parser.set_defaults(render=False)
    parser.add_argument("--pause-on-stage", action="store_true")
    args = parser.parse_args()

    candidates = {
        row["candidate_id"]: row
        for row in read_jsonl(args.data_dir / "candidates.jsonl")
    }
    if args.candidate_id not in candidates:
        raise SystemExit(f"candidate not found: {args.candidate_id}")
    if args.stop_after == "lift" and not args.grasp_checkpoint:
        raise SystemExit("--grasp-checkpoint is required for a full replay")

    controller = InteractiveController()
    runner = ManipulationEpisodeRunner(
        args.scene,
        EpisodeConfig(),
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        pause_on_stage=args.pause_on_stage,
        interactive=controller,
        debug_dir=args.data_dir / "replay_debug",
    )
    try:
        result = runner.run_episode(
            candidates[args.candidate_id], attempt=0, stop_after=args.stop_after
        )
        print(result.to_dict())
        if args.render and not controller.quit:
            controller.wait_for_advance(runner.env, "replay finished")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
