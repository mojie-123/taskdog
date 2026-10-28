#!/usr/bin/env python
"""Preview, ARM_INIT-check, or collect feasible manipulation base poses.

生成候选 M
→ 把机器狗和 cube 放到候选状态
→ 执行 ARM_INIT / 抓取 / LIFT
→ 多次重复验证
→ 保存稳定成功的 M

它支持三种运行模式：
- preview
  只把机器人放到候选 M，检查初始位置及碰撞，方便在 MuJoCo 中查看，不执行抓取。
- arm-init
  执行到 ARM_INIT 为止，检查：
  - 底盘位置是否安全；
  - 机械臂是否能转向目标；
  - 腕部相机和机械臂初始状态是否满足要求。
- collect
    执行完整的：
    ARM_INIT
    → YOLOE + SAM2
    → AnyGrasp
    → MoveIt
    → LOCAL_REACH
    → CLOSE
    → LIFT
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from custom_envs.scripts.N2M.candidate_generation import generate_candidates, select_candidates
from custom_envs.scripts.N2M.config import (
    CUBE_SCENES,
    DEFAULT_DATA_DIR,
    DEFAULT_REFERENCE_BASE,
    DEFAULT_SCENE,
    CandidateGrid,
    EpisodeConfig,
)
from custom_envs.scripts.N2M.interactive import InteractiveController
from custom_envs.scripts.N2M.manifests import (
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)
from custom_envs.scripts.N2M.manipulation_episode import ManipulationEpisodeRunner   # 抓取执行器
from custom_envs.scripts.N2M.visualize_success_map import build_plot


def _utc_now() -> str:   # 返回带时区的 UTC 时间字符串如2026-10-23T08:35:12.123456+00:00
    return datetime.now(timezone.utc).isoformat()


def _parse_scene_ids(value: str):   # 解析要用于生成数据的场景
    if value == "all":
        return list(CUBE_SCENES)
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(CUBE_SCENES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scene ids: {unknown}")
    return values


def _grid_from_args(args) -> CandidateGrid:   # 从标准位置开始生成扰动
    if args.nominal_only:
        return CandidateGrid([0.0], [0.0], [0.0])
    return CandidateGrid(
        normal_offsets=list(args.normal_offsets),
        tangent_offsets=list(args.tangent_offsets),
        yaw_offsets_deg=list(args.yaw_offsets_deg),
    )


def _latest_summaries(path: Path):   # 读取candidate_summaries.jsonl，建立latest[candidate_id] = row（若有同一个candidate_id出现多次，则后面的记录会覆盖前面的记录）
    latest = {}
    for row in read_jsonl(path):
        latest[row["candidate_id"]] = row
    return latest


def _rebuild_success_file(data_dir: Path):   # 根据最新summary生成success_poses.jsonl（稳定成功 M 集合）
    latest = _latest_summaries(data_dir / "candidate_summaries.jsonl")
    successes = []
    for row in latest.values():
        if not row.get("stable_success", False):
            continue
        successes.append({
            "candidate_id": row["candidate_id"],
            "scene_id": row["scene_id"],
            "approach_edge": row.get("approach_edge"),
            "cube_pose": row["cube_pose"],
            "base_pose": row["base_pose"],   # 最重要的是这个
            "joint1_target": row.get("joint1_target"),
            "success_count": row["success_count"],
            "attempt_count": row["attempt_count"],
            "max_attempts": row.get("max_attempts"),
            "required_successes": row.get("required_successes"),
            "success_rate": row["success_rate"],
            "decision": row.get("decision"),
        })
    successes.sort(key=lambda row: (row["scene_id"], row["candidate_id"]))
    write_jsonl(data_dir / "success_poses.jsonl", successes)
    return successes


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search feasible base poses for the six fixed coffee-table cube scenes."
    )
    parser.add_argument("--mode", choices=["preview", "arm-init", "collect"], default="preview")   # 3种运行模式，收集数据时用collect模式
    parser.add_argument("--cube-scene", default="short_left",
                        help="one id, comma-separated ids, or 'all'")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)   # 各种数据和json/jsonl的保存目录
    parser.add_argument("--reference-base", nargs=3, type=float,
                        default=list(DEFAULT_REFERENCE_BASE), metavar=("X", "Y", "YAW"),
                        help="actual base pose at the known-good short_left ARM_INIT")
    parser.add_argument("--normal-offsets", nargs="+", type=float,
                        default=CandidateGrid().normal_offsets)
    parser.add_argument("--tangent-offsets", nargs="+", type=float,
                        default=CandidateGrid().tangent_offsets)
    parser.add_argument("--yaw-offsets-deg", nargs="+", type=float,
                        default=CandidateGrid().yaw_offsets_deg)
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
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
    parser.add_argument("--grasp-checkpoint", default=None)
    parser.add_argument("--confirm-repeats", type=int, default=5)
    parser.add_argument("--confirm-min-success", type=int, default=2)
    parser.add_argument(
        "--target-successes-per-edge", type=int, default=0,
        help="stop an approach-edge search after this many stable poses; 0 is exhaustive",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wrist-min-object-pixels", "--wrist-min-yellow-pixels",
        dest="wrist_min_object_pixels", type=int, default=20,
        help="minimum target-object segmentation pixels in wrist_cam; the old "
             "yellow-pixels spelling is retained as a command-line alias",
    )
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--reject-initial-table-contact",
        dest="reject_initial_table_contact",
        action="store_true",
        help="reject colliding base poses before ARM_INIT (default)",
    )
    collision_group.add_argument(
        "--allow-initial-table-contact",
        dest="reject_initial_table_contact",
        action="store_false",
        help="diagnostic only: retain colliding base poses",
    )
    parser.set_defaults(reject_initial_table_contact=True)
    return parser


def main():
    args = build_parser().parse_args()
    scene_ids = _parse_scene_ids(args.cube_scene)
    grid = _grid_from_args(args)
    if any(float(offset) > 0.0 for offset in grid.normal_offsets):
        raise SystemExit(
            "--normal-offsets must all be <= 0; positive values move into the table"
        )   # 禁止向桌内采样
    candidates = generate_candidates(args.reference_base, scene_ids, grid)
    candidates = select_candidates(candidates, args.candidate_id, args.max_candidates)   # 生成并筛选候选
    if not candidates:
        raise SystemExit("No candidates selected")
    if args.mode == "collect" and not args.grasp_checkpoint:
        raise SystemExit("--grasp-checkpoint is required in collect mode")
    if args.confirm_repeats < 1:
        raise SystemExit("--confirm-repeats must be >= 1")
    if args.confirm_min_success < 1:
        raise SystemExit("--confirm-min-success must be >= 1")
    if args.confirm_min_success > args.confirm_repeats:
        raise SystemExit("--confirm-min-success cannot exceed --confirm-repeats")
    if args.target_successes_per_edge < 0:
        raise SystemExit("--target-successes-per-edge must be >= 0")

    config = EpisodeConfig(
        wrist_min_object_pixels=args.wrist_min_object_pixels,
        max_episode_steps=args.max_episode_steps,
        reject_initial_table_contact=args.reject_initial_table_contact,
    )   # 配置

    args.data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.data_dir / "candidates.jsonl", candidates)   # 保存候选和本次运行配置
    write_json(
        args.data_dir / "run_config.json",
        {
            "created_at": _utc_now(),
            "mode": args.mode,
            "scene_path": str(args.scene.resolve()),
            "scene_ids": scene_ids,
            "reference_base": list(args.reference_base),
            "candidate_grid": asdict(grid),   # 扰动网格
            "candidate_count": len(candidates),
            "confirm_repeats": args.confirm_repeats,   # 重复验证标准
            "confirm_min_success": args.confirm_min_success,
            "target_successes_per_edge": args.target_successes_per_edge,
            "episode_config": config.to_dict(),
        },
    )
    print(
        f"[N2M] mode={args.mode} scenes={scene_ids} candidates={len(candidates)} "
        f"data={args.data_dir}",
        flush=True,
    )

    if (args.pause_on_stage or args.mode == "preview") and not args.render:
        print("[N2M] NOTE: --pause-on-stage/preview is most useful with --render", flush=True)

    controls = InteractiveController()   # 创建控制器
    runner = ManipulationEpisodeRunner(
        args.scene,
        config,
        grasp_checkpoint=args.grasp_checkpoint,
        render=args.render,
        pause_on_stage=args.pause_on_stage,
        interactive=controls,
        debug_dir=args.data_dir / "debug",
    )   # 创建执行器

    try:
        if args.mode == "preview":
            for candidate in candidates:
                if controls.quit:
                    break
                row = runner.preview_candidate(candidate)
                append_jsonl(args.data_dir / "preview_checks.jsonl", {
                    **row, "created_at": _utc_now(),
                })
            return

        if args.mode == "arm-init":
            for candidate in candidates:
                if controls.quit:
                    break
                result = runner.run_episode(candidate, attempt=0, stop_after="arm_init")
                row = result.to_dict()
                row.update({"phase": "arm_init_check", "created_at": _utc_now()})
                append_jsonl(args.data_dir / "arm_init_rollouts.jsonl", row)
                print(
                    f"[N2M] {candidate['candidate_id']} ARM_INIT "
                    f"{'OK' if result.failure_reason is None else result.failure_reason}",
                    flush=True,
                )
            return

        summary_path = args.data_dir / "candidate_summaries.jsonl"
        already = _latest_summaries(summary_path) if args.resume else {}
        stable_by_edge = defaultdict(int)   # 统计每条桌边已有多少成功 M
        for row in already.values():
            if row.get("stable_success", False):
                stable_by_edge[(row["scene_id"], row.get("approach_edge"))] += 1
        for candidate in candidates:   # 逐个处理候选 M
            if controls.quit:
                break
            cid = candidate["candidate_id"]
            edge_key = (candidate["scene_id"], candidate["approach_edge"])
            if cid in already:   # 已经总结过的候选
                print(f"[N2M] resume: skip summarized {cid}", flush=True)
                continue
            target_count = int(args.target_successes_per_edge)
            if target_count > 0 and stable_by_edge[edge_key] >= target_count:   # 桌边已有足够多稳定成功 M
                print(
                    f"[N2M] target reached for {edge_key[0]}/{edge_key[1]}: "
                    f"{stable_by_edge[edge_key]} stable poses; skip {cid}",
                    flush=True,
                )
                continue

            results = []
            screen = runner.run_episode(candidate, attempt=0, stop_after="lift")   # 每个候选首先执行一次完整抓取
            results.append(screen)   # 测试结果加入results
            row = screen.to_dict()
            row.update({"phase": "screen", "created_at": _utc_now()})
            append_jsonl(args.data_dir / "manipulation_rollouts.jsonl", row)   # 保存这一次尝试

            # Placement/ARM_INIT failures are deterministic geometry/configuration
            # rejections.  Once a candidate reaches the perception/manipulation
            # chain, run all repeats even if the first attempt fails: otherwise
            # a fail-success-success candidate could never reach the success threshold.
            deterministic_failure = screen.failure_stage in {
                "BASE_PLACED", "ARM_INIT", "OPERATOR", "INTERNAL",
            }   # 确定性失败：机器人初始位置与桌子碰撞；机械臂姿态、关节范围等不满足要求；操作人员主动终止；程序内部异常
            max_attempts = max(1, int(args.confirm_repeats))   # 最多尝试次数
            required = max(1, int(args.confirm_min_success))   # 至少成功次数
            if not deterministic_failure:
                for attempt in range(1, max_attempts):   # 尝试剩下的次数
                    if sum(int(item.success is True) for item in results) >= required:   # 达到成功次数
                        break
                    result = runner.run_episode(candidate, attempt=attempt, stop_after="lift")
                    results.append(result)
                    row = result.to_dict()
                    row.update({"phase": "confirm", "created_at": _utc_now()})
                    append_jsonl(args.data_dir / "manipulation_rollouts.jsonl", row)

            success_count = sum(int(result.success is True) for result in results)   # 记录成功次数
            stable = bool(success_count >= required)
            if deterministic_failure:
                decision = "deterministic_rejection"
            elif stable and len(results) < max_attempts:
                decision = "success_threshold_reached_early"
            elif stable:
                decision = "success_threshold_reached_at_limit"
            else:
                decision = "attempt_limit_exhausted"
            summary = {
                "candidate_id": cid,
                "scene_id": candidate["scene_id"],
                "approach_edge": candidate["approach_edge"],
                "cube_pose": candidate["cube_pose"],
                "base_pose": candidate["base_pose"],
                "joint1_target": candidate.get("joint1_target"),
                "attempt_count": len(results),
                "max_attempts": max_attempts,
                "success_count": success_count,
                "required_successes": required,
                "success_rate": success_count / max(len(results), 1),
                "stable_success": stable,
                "decision": decision,
                "screen_failure_stage": screen.failure_stage,   # 这里只记录第一次尝试失败的情况
                "screen_failure_reason": screen.failure_reason,
                "created_at": _utc_now(),
            }
            append_jsonl(summary_path, summary)   # 写入summary
            # Keep the consumable success set crash-safe even if the operator
            # stops a long search before the final plotting pass.
            _rebuild_success_file(args.data_dir)   # 每完成一个候选就重建成功 M 文件success_poses.jsonl
            if stable:
                stable_by_edge[edge_key] += 1
            print(
                f"[N2M] {cid}: stable={stable} success={success_count}/{len(results)} "
                f"screen_failure={screen.failure_stage}:{screen.failure_reason}",
                flush=True,
            )

        # 全部候选结束后
        successes = _rebuild_success_file(args.data_dir)   # 再次重建成功文件
        by_scene = defaultdict(int)
        for row in successes:   # 按场景统计成功数
            by_scene[row["scene_id"]] += 1
        for scene_id in scene_ids:
            try:
                output = build_plot(scene_id, args.data_dir)
                print(f"[N2M] success map: {output}", flush=True)
            except Exception as exc:
                print(f"[N2M] success-map warning for {scene_id}: {exc}", flush=True)
        print(f"[N2M] stable successes by scene: {dict(by_scene)}", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
