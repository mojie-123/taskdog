#!/usr/bin/env python3
"""Generate screened NAV-end observations and RGB-D point clouds in MuJoCo.

The collector keeps testing a stratified proposal pool until every selected
``scene/approach_edge`` has the requested number of *valid* observations.  It
also writes a durable rejection manifest and optional visual quality-control
panels so a small pilot can be audited before full collection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from custom_envs.scripts.N2M.config import (
    CUBE_SCENES,
    DEFAULT_ROBOT_ROOT_Z,
    PROJECT_ROOT,
)
from custom_envs.scripts.N2M.data_generation.camera_geometry import (
    camera_points_to_frame,
    crop_point_mask,
    depth_to_optical_points,
    intrinsics_from_mujoco,
    voxel_downsample_indices,
)
from custom_envs.scripts.N2M.data_generation.observation_preview import (
    save_observation_preview,
)
from custom_envs.scripts.N2M.data_generation.observation_sampling import (
    propose_observation_poses,
    stratified_order,
)
from custom_envs.scripts.N2M.manifests import (
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_SCENE = PROJECT_ROOT / "custom_envs" / "mujoco" / "scene_house_N2M.xml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "custom_envs" / "scripts" / "N2M" / "data" / "observations_d30_v1"
)
DEFAULT_SUCCESS_MANIFEST = (
    PROJECT_ROOT
    / "custom_envs" / "scripts" / "N2M" / "data"
    / "manipulation_pose_dataset" / "d30_v1" / "successful_poses.jsonl"
)


def _float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PROPOSAL_GRID_FIELDS = ("ranges", "tangent_offsets", "yaw_offsets_deg")


def _proposal_grid_extension_compatible(previous: dict, requested: dict) -> bool:
    """Return whether ``requested`` only adds candidates to the old grid.

    Accepted observations and deterministic rejection results remain valid when
    the proposal grid is enlarged without changing camera, scene, filtering or
    placement settings.  This deliberately does not permit removing old grid
    values or changing any other functional setting.
    """

    previous_without_grid = dict(previous)
    requested_without_grid = dict(requested)
    for field in _PROPOSAL_GRID_FIELDS:
        old_values = [float(value) for value in previous_without_grid.pop(field, [])]
        new_values = [float(value) for value in requested_without_grid.pop(field, [])]
        if any(
            not any(abs(old - new) <= 1e-9 for new in new_values)
            for old in old_values
        ):
            return False
    return previous_without_grid == requested_without_grid


def _observation_id(proposal: dict) -> str:
    proposal_key = json.dumps(proposal["proposal"], sort_keys=True, separators=(",", ":"))
    proposal_hash = hashlib.sha1(proposal_key.encode("utf-8")).hexdigest()[:10]
    return f"{proposal['scene_id']}__{proposal['approach_edge']}__o{proposal_hash}"


def _subtree_body_ids(model, root_body_name: str) -> set[int]:
    import mujoco

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root < 0:
        raise ValueError(f"body not found: {root_body_name}")
    result: set[int] = set()
    for body_id in range(model.nbody):
        current = int(body_id)
        while current > 0:
            if current == root:
                result.add(int(body_id))
                break
            current = int(model.body_parentid[current])
    result.add(int(root))
    return result


def _geom_ids_for_bodies(model, body_ids: set[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(np.asarray(model.geom_bodyid), list(body_ids)))


def _segmentation_mask(segmentation: np.ndarray, geom_ids: Iterable[int]) -> np.ndarray:
    import mujoco

    geom_ids = np.asarray(list(geom_ids), dtype=np.int32)
    return np.isin(segmentation[..., 0], geom_ids) & (
        segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )


def _render_rgb_depth_segmentation(env, camera_name: str):
    import mujoco

    renderer = mujoco.Renderer(
        env.model, height=env._cam_h, width=env._cam_w  # existing environment contract
    )
    try:
        renderer.update_scene(env.data, camera=camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.data, camera=camera_name)
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    if segmentation.shape != (*depth.shape, 2):
        raise RuntimeError(f"unexpected MuJoCo segmentation shape: {segmentation.shape}")
    return rgb, depth, segmentation


def _robot_environment_contacts(env, robot_body_ids: set[int]) -> list[str]:
    """Reject contacts except robot self-contact and the expected floor contact."""
    import mujoco

    hits: set[str] = set()
    model, data = env.model, env.data
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        robot1, robot2 = body1 in robot_body_ids, body2 in robot_body_ids
        if robot1 == robot2:
            continue
        other_geom = geom2 if robot1 else geom1
        other_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
            or f"geom#{other_geom}"
        )
        if other_name == "floor":
            continue
        robot_geom = geom1 if robot1 else geom2
        robot_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            or f"geom#{robot_geom}"
        )
        hits.add(f"{robot_name}<->{other_name}")
    return sorted(hits)


def _success_poses_by_scene_edge(path: Path | None) -> dict[tuple[str, str], list[list[float]]]:
    result: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    if path is None or not Path(path).exists():
        return result
    for row in read_jsonl(path):
        if "scene_id" in row and "approach_edge" in row and "base_pose" in row:
            result[(row["scene_id"], row["approach_edge"])].append(
                [float(value) for value in row["base_pose"][:3]]
            )
    return result


def _functional_config(args, intrinsics) -> dict:
    """Return only fields that affect accepted data, not run/output policy."""
    return {
        "schema_version": 2,
        "scene_xml": str(args.scene_xml.resolve()),
        "scene_xml_sha256": _sha256(args.scene_xml.resolve()),
        "camera": args.camera,
        "width": int(args.width),
        "height": int(args.height),
        "camera_intrinsics": intrinsics.to_dict(),
        "ranges": [float(value) for value in args.ranges],
        "tangent_offsets": [float(value) for value in args.tangent_offsets],
        "yaw_offsets_deg": [float(value) for value in args.yaw_offsets_deg],
        "seed": int(args.seed),
        "min_cube_pixels": int(args.min_cube_pixels),
        "min_cube_points": int(args.min_cube_points),
        "max_cube_distance": float(args.max_cube_distance),
        "max_depth": float(args.max_depth),
        "voxel_size": float(args.voxel_size),
        "max_cube_centroid_error": float(args.max_cube_centroid_error),
        "minimum_point_count": int(args.minimum_point_count),
        "crop_min": [float(value) for value in args.crop_min],
        "crop_max": [float(value) for value in args.crop_max],
        "robot_root_z": float(args.robot_root_z),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cube-scene", nargs="+", default=["all"], choices=["all", *CUBE_SCENES]
    )
    parser.add_argument("--camera", default="front_realsense")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ranges", type=_float_list, default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    parser.add_argument(
        "--tangent-offsets",
        type=_float_list,
        default=[-0.60, -0.40, -0.20, 0.0, 0.20, 0.40, 0.60],
    )
    parser.add_argument(
        "--yaw-offsets-deg",
        type=_float_list,
        # With the camera mounted 0.42 m ahead of the base, a 30-degree base
        # offset puts the cube beyond the 69-degree horizontal FOV throughout
        # the 1--2 m proposal range.  Keep denser, empirically visible angles.
        default=[-20.0, -15.0, -10.0, 0.0, 10.0, 15.0, 20.0],
    )
    parser.add_argument("--target-valid-per-edge", type=int, default=100)
    parser.add_argument(
        "--max-observations-per-scene", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cube-pixels", type=int, default=80)
    parser.add_argument("--min-cube-points", type=int, default=15)
    parser.add_argument("--max-cube-distance", type=float, default=2.0)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.010)
    parser.add_argument("--max-cube-centroid-error", type=float, default=0.06)
    parser.add_argument("--minimum-point-count", type=int, default=8192)
    parser.add_argument("--crop-min", type=_float_list, default=[-1.0, -2.0, -1.0])
    parser.add_argument("--crop-max", type=_float_list, default=[2.5, 2.0, 1.5])
    parser.add_argument("--robot-root-z", type=float, default=DEFAULT_ROBOT_ROOT_Z)
    parser.add_argument(
        "--success-manifest",
        type=Path,
        default=DEFAULT_SUCCESS_MANIFEST,
        help="successful M manifest used only in preview world maps",
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--save-rejected-previews", action="store_true")
    parser.add_argument("--max-rejected-previews-per-reason", type=int, default=3)
    parser.add_argument("--render", action="store_true", help="open MuJoCo viewer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-proposal-grid-extension",
        action="store_true",
        help=(
            "when resuming, allow ranges/tangent/yaw grids to be strict supersets "
            "of the saved grid while requiring every other capture setting to match"
        ),
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="re-evaluate proposals already present in rejected_observations.jsonl",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="return success even if a proposal pool cannot reach its valid target",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_observations_per_scene is not None:
        args.target_valid_per_edge = args.max_observations_per_scene
        print(
            "[N2M][WARN] --max-observations-per-scene is deprecated; "
            "interpreting it as --target-valid-per-edge"
        )
    if args.target_valid_per_edge <= 0:
        raise ValueError("--target-valid-per-edge must be positive")
    if args.minimum_point_count <= 0 or args.min_cube_points <= 0:
        raise ValueError("point-count thresholds must be positive")
    if not args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if len(args.crop_min) != 3 or len(args.crop_max) != 3:
        raise ValueError("--crop-min and --crop-max require exactly three values")
    scene_ids = list(CUBE_SCENES) if args.cube_scene == ["all"] else list(dict.fromkeys(args.cube_scene))
    output_dir = args.output_dir.resolve()
    observations_dir = output_dir / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "observations.jsonl"
    rejected_path = output_dir / "rejected_observations.jsonl"
    config_path = output_dir / "capture_config.json"
    if not args.resume and (manifest_path.exists() or rejected_path.exists()):
        raise FileExistsError(
            f"{output_dir} already has capture manifests; pass --resume or choose another output"
        )

    accepted_rows = read_jsonl(manifest_path)
    accepted_ids = {row["observation_id"] for row in accepted_rows}
    rejected_rows = {row["observation_id"]: row for row in read_jsonl(rejected_path)}
    accepted_by_key: Counter = Counter(
        (row["scene_id"], row["approach_edge"]) for row in accepted_rows
    )
    successes = _success_poses_by_scene_edge(args.success_manifest)

    from custom_envs.mujoco.env import MuJoCoEnv
    from custom_envs.scripts.N2M.mujoco_state import place_episode_state
    import mujoco

    env = MuJoCoEnv(
        camera_width=args.width,
        camera_height=args.height,
        render=args.render,
        scene_path=str(args.scene_xml.resolve()),
    )
    try:
        intrinsics = intrinsics_from_mujoco(env.model, args.camera, args.width, args.height)
        functional_config = _functional_config(args, intrinsics)
        fingerprint = _stable_hash(functional_config)
        if args.resume and config_path.exists():
            previous = read_json(config_path)
            previous_fingerprint = previous.get("functional_fingerprint")
            if previous_fingerprint != fingerprint:
                extension_is_safe = (
                    args.allow_proposal_grid_extension
                    and _proposal_grid_extension_compatible(
                        previous.get("functional_config", {}), functional_config
                    )
                )
                if not extension_is_safe:
                    raise ValueError(
                        "capture configuration differs from the existing dataset; "
                        "use the original settings or a new --output-dir. A proposal-only "
                        "superset can be resumed with --allow-proposal-grid-extension.\n"
                        f"existing={previous_fingerprint}\nrequested={fingerprint}"
                    )
                print(
                    "[N2M] Extending the saved observation proposal grid; existing "
                    "accepted/rejected observations remain valid."
                )
        elif args.resume and (accepted_rows or rejected_rows):
            raise ValueError(f"{output_dir} has manifests but no compatible capture_config.json")
        write_json(
            config_path,
            {
                "schema_version": 2,
                "functional_fingerprint": fingerprint,
                "functional_config": functional_config,
                "last_request": {
                    "cube_scenes": scene_ids,
                    "target_valid_per_edge": int(args.target_valid_per_edge),
                    "save_previews": bool(args.save_previews),
                    "save_rejected_previews": bool(args.save_rejected_previews),
                },
            },
        )

        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if min(camera_id, base_id, cube_id) < 0:
            raise RuntimeError("scene must define front camera, base_link and cube")
        robot_bodies = _subtree_body_ids(env.model, "base_link")
        robot_geoms = _geom_ids_for_bodies(env.model, robot_bodies)
        cube_geoms = _geom_ids_for_bodies(env.model, _subtree_body_ids(env.model, "cube"))

        proposal_groups: dict[tuple[str, str], list[dict]] = {}
        for scene_id in scene_ids:
            rows = propose_observation_poses(
                scene_id, args.ranges, args.tangent_offsets, args.yaw_offsets_deg
            )
            for edge_index, edge in enumerate(CUBE_SCENES[scene_id].approach_edges):
                edge_rows = [row for row in rows if row["approach_edge"] == edge]
                proposal_groups[(scene_id, edge)] = stratified_order(
                    edge_rows,
                    args.seed + list(CUBE_SCENES).index(scene_id) * 17 + edge_index,
                )

        new_accepted = 0
        new_rejected = 0
        rejected_preview_counts: Counter = Counter()
        rejected_preview_root = output_dir / "previews" / "rejected"
        if rejected_preview_root.exists():
            for reason_dir in rejected_preview_root.iterdir():
                if reason_dir.is_dir():
                    rejected_preview_counts[reason_dir.name] = sum(
                        1 for path in reason_dir.glob("*/qc_panel.png") if path.is_file()
                    )

        def save_preview(
            directory: Path,
            *,
            observation_id: str,
            status: str,
            reason: str | None,
            scene_id: str,
            edge: str,
            base_pose,
            cube_pose,
            metrics: dict,
            rgb=None,
            depth=None,
            cube_mask=None,
            xyz=None,
            point_rgb=None,
            cube_center=None,
            cube_centroid=None,
            metadata=None,
        ) -> None:
            save_observation_preview(
                directory,
                observation_id=observation_id,
                status=status,
                reason=reason,
                rgb=rgb,
                depth=depth,
                cube_mask=cube_mask,
                xyz=xyz,
                point_rgb=point_rgb,
                cube_center_base=cube_center,
                cube_centroid_base=cube_centroid,
                observation_pose_world=base_pose,
                cube_pose_world=cube_pose,
                manipulation_poses_world=successes.get((scene_id, edge), []),
                metrics=metrics,
                preview_max_depth=args.max_depth,
            )
            write_json(directory / "metadata.json", metadata or metrics)

        for (scene_id, edge), proposals in proposal_groups.items():
            key = (scene_id, edge)
            scene = CUBE_SCENES[scene_id]
            cube_pose = [*scene.cube_xyz, 1.0, 0.0, 0.0, 0.0]
            parameter_keys = ("range_m", "tangent_m", "yaw_offset_deg")
            accepted_parameter_counts = {name: Counter() for name in parameter_keys}
            for existing in accepted_rows:
                if (existing["scene_id"], existing["approach_edge"]) != key:
                    continue
                for name in parameter_keys:
                    accepted_parameter_counts[name][existing["proposal"][name]] += 1
            remaining = []
            for proposal in proposals:
                observation_id = _observation_id(proposal)
                if observation_id in accepted_ids:
                    continue
                if observation_id in rejected_rows and not args.retry_rejected:
                    continue
                remaining.append(proposal)
            print(
                f"[N2M][EDGE_START] {scene_id}/{edge} "
                f"valid={accepted_by_key[key]}/{args.target_valid_per_edge} "
                f"pool={len(proposals)}"
            )
            while remaining and accepted_by_key[key] < args.target_valid_per_edge:
                def balance_score(index: int) -> tuple[int, int, int]:
                    proposal_values = remaining[index]["proposal"]
                    used = [
                        accepted_parameter_counts[name][proposal_values[name]]
                        for name in parameter_keys
                    ]
                    # The precomputed stratified order is the final tie-breaker.
                    return max(used), sum(used), index

                proposal = remaining.pop(
                    min(range(len(remaining)), key=balance_score)
                )
                observation_id = _observation_id(proposal)

                base_pose = proposal["observation_pose_world"]
                metrics: dict = {}
                rgb = depth = cube_mask = None
                xyz_base = rgb_points = None
                cube_center_base = cube_centroid_base = None
                place_episode_state(
                    env,
                    cube_pose,
                    base_pose,
                    robot_root_z=args.robot_root_z,
                    robot_root_roll=0.0,
                    robot_root_pitch=0.0,
                )

                reason = None
                details: dict = {}
                contacts = _robot_environment_contacts(env, robot_bodies)
                if contacts:
                    reason = "collision"
                    details["contacts"] = contacts
                else:
                    rgb, depth, segmentation = _render_rgb_depth_segmentation(env, args.camera)
                    cube_mask = _segmentation_mask(segmentation, cube_geoms)
                    robot_mask = _segmentation_mask(segmentation, robot_geoms)
                    cube_pixels = int(cube_mask.sum())
                    camera_position = env.data.cam_xpos[camera_id].astype(np.float64).copy()
                    cube_position = env.data.xpos[cube_id].astype(np.float64).copy()
                    cube_distance = float(np.linalg.norm(cube_position - camera_position))
                    metrics.update(
                        cube_visible_pixels=cube_pixels,
                        cube_camera_distance_m=round(cube_distance, 5),
                    )
                    if cube_distance > args.max_cube_distance:
                        reason = "cube_too_far"
                    elif cube_pixels < args.min_cube_pixels:
                        reason = "cube_too_few_pixels"
                    else:
                        valid = (
                            (~robot_mask)
                            & np.isfinite(depth)
                            & (depth > 0.0)
                            & (depth <= args.max_depth)
                        )
                        optical_points, flat_indices = depth_to_optical_points(
                            depth, intrinsics, valid
                        )
                        rgb_points = rgb.reshape(-1, 3)[flat_indices]
                        cube_flags = cube_mask.reshape(-1)[flat_indices]
                        xyz_base = camera_points_to_frame(
                            optical_points,
                            camera_position,
                            env.data.cam_xmat[camera_id],
                            env.data.xpos[base_id],
                            env.data.xmat[base_id],
                        )
                        cube_optical, _ = depth_to_optical_points(
                            depth,
                            intrinsics,
                            cube_mask
                            & np.isfinite(depth)
                            & (depth > 0.0)
                            & (depth <= args.max_depth),
                        )
                        cube_center_base = (
                            cube_position - env.data.xpos[base_id]
                        ) @ env.data.xmat[base_id].reshape(3, 3)
                        if len(cube_optical) == 0:
                            reason = "cube_depth_unavailable"
                        else:
                            cube_points_base = camera_points_to_frame(
                                cube_optical,
                                camera_position,
                                env.data.cam_xmat[camera_id],
                                env.data.xpos[base_id],
                                env.data.xmat[base_id],
                            )
                            cube_centroid_base = np.median(cube_points_base, axis=0)
                            cube_centroid_error = float(
                                np.linalg.norm(cube_centroid_base - cube_center_base)
                            )
                            metrics["cube_centroid_error_m"] = round(cube_centroid_error, 5)
                            if cube_centroid_error > args.max_cube_centroid_error:
                                reason = "cube_centroid_error"
                            else:
                                keep = crop_point_mask(xyz_base, args.crop_min, args.crop_max)
                                xyz_base = xyz_base[keep]
                                rgb_points = rgb_points[keep]
                                cube_flags = cube_flags[keep]
                                keep = voxel_downsample_indices(xyz_base, args.voxel_size)
                                xyz_base = xyz_base[keep]
                                rgb_points = rgb_points[keep]
                                cube_flags = cube_flags[keep]
                                point_count = int(len(xyz_base))
                                cube_point_count = int(cube_flags.sum())
                                metrics.update(
                                    point_count=point_count,
                                    cube_points_after_voxel=cube_point_count,
                                )
                                if point_count < args.minimum_point_count:
                                    reason = "too_few_cloud_points"
                                elif cube_point_count < args.min_cube_points:
                                    reason = "too_few_cube_points_after_voxel"

                if reason is not None:
                    rejection_row = {
                        "schema_version": 2,
                        "observation_id": observation_id,
                        "scene_id": scene_id,
                        "approach_edge": edge,
                        "observation_pose_world": [float(value) for value in base_pose],
                        "cube_pose_world": [float(value) for value in cube_pose],
                        "proposal": proposal["proposal"],
                        "reason": reason,
                        "metrics": metrics,
                        "details": details,
                    }
                    save_rejected_preview = (
                        args.save_rejected_previews
                        and rejected_preview_counts[reason]
                        < args.max_rejected_previews_per_reason
                    )
                    if save_rejected_preview:
                        preview_relative = (
                            Path("previews") / "rejected" / reason / observation_id
                        )
                        rejection_row["preview_dir"] = str(preview_relative)
                    rejected_rows[observation_id] = rejection_row
                    write_jsonl(
                        rejected_path,
                        sorted(rejected_rows.values(), key=lambda row: row["observation_id"]),
                    )
                    new_rejected += 1
                    if save_rejected_preview:
                        preview_dir = output_dir / preview_relative
                        save_preview(
                            preview_dir,
                            observation_id=observation_id,
                            status="REJECTED",
                            reason=reason,
                            scene_id=scene_id,
                            edge=edge,
                            base_pose=base_pose,
                            cube_pose=cube_pose,
                            metrics=metrics,
                            rgb=rgb,
                            depth=depth,
                            cube_mask=cube_mask,
                            xyz=xyz_base,
                            point_rgb=rgb_points,
                            cube_center=cube_center_base,
                            cube_centroid=cube_centroid_base,
                            metadata=rejection_row,
                        )
                        rejected_preview_counts[reason] += 1
                    print(
                        f"[N2M][OBS_REJECT] {observation_id} reason={reason} metrics={metrics}"
                    )
                    continue

                cloud_relative = Path("observations") / f"{observation_id}.npz"
                np.savez_compressed(
                    output_dir / cloud_relative,
                    xyz=xyz_base.astype(np.float32),
                    rgb=rgb_points.astype(np.uint8),
                )
                camera_position = env.data.cam_xpos[camera_id].astype(np.float64).copy()
                row = {
                    "schema_version": 2,
                    "observation_id": observation_id,
                    "scene_id": scene_id,
                    "approach_edge": edge,
                    "cloud_file": str(cloud_relative),
                    "observation_pose_world": [float(value) for value in base_pose],
                    "cube_pose_world": [float(value) for value in cube_pose],
                    "proposal": proposal["proposal"],
                    "cube_visible_pixels": int(metrics["cube_visible_pixels"]),
                    "cube_camera_distance_m": float(metrics["cube_camera_distance_m"]),
                    "cube_center_base": cube_center_base.tolist(),
                    "cube_visible_centroid_base": cube_centroid_base.tolist(),
                    "cube_centroid_error_m": float(metrics["cube_centroid_error_m"]),
                    "point_count": int(metrics["point_count"]),
                    "cube_points_after_voxel": int(metrics["cube_points_after_voxel"]),
                    "camera_name": args.camera,
                    "camera_intrinsics": intrinsics.to_dict(),
                    "camera_pose_world": {
                        "position": camera_position.tolist(),
                        "rotation": env.data.cam_xmat[camera_id].reshape(3, 3).tolist(),
                    },
                }
                if args.save_previews:
                    preview_relative = Path("previews") / "accepted" / observation_id
                    row["preview_dir"] = str(preview_relative)
                    save_preview(
                        output_dir / preview_relative,
                        observation_id=observation_id,
                        status="ACCEPTED",
                        reason=None,
                        scene_id=scene_id,
                        edge=edge,
                        base_pose=base_pose,
                        cube_pose=cube_pose,
                        metrics=metrics,
                        rgb=rgb,
                        depth=depth,
                        cube_mask=cube_mask,
                        xyz=xyz_base,
                        point_rgb=rgb_points,
                        cube_center=cube_center_base,
                        cube_centroid=cube_centroid_base,
                        metadata=row,
                    )
                append_jsonl(manifest_path, row)
                accepted_rows.append(row)
                accepted_ids.add(observation_id)
                accepted_by_key[key] += 1
                for name in parameter_keys:
                    accepted_parameter_counts[name][proposal["proposal"][name]] += 1
                new_accepted += 1
                if observation_id in rejected_rows:
                    del rejected_rows[observation_id]
                    write_jsonl(
                        rejected_path,
                        sorted(rejected_rows.values(), key=lambda item: item["observation_id"]),
                    )
                print(
                    f"[N2M][OBS_OK] {observation_id} "
                    f"valid={accepted_by_key[key]}/{args.target_valid_per_edge} "
                    f"points={metrics['point_count']} cube_points={metrics['cube_points_after_voxel']} "
                    f"pixels={metrics['cube_visible_pixels']} "
                    f"distance={metrics['cube_camera_distance_m']:.3f}m"
                )

        selected_keys = set(proposal_groups)
        rejection_reason_by_key: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for row in rejected_rows.values():
            row_key = (row["scene_id"], row["approach_edge"])
            if row_key in selected_keys:
                rejection_reason_by_key[row_key][row["reason"]] += 1
        per_edge = {}
        shortfalls = []
        for key, proposals in proposal_groups.items():
            scene_id, edge = key
            valid_count = int(accepted_by_key[key])
            shortfall = max(0, int(args.target_valid_per_edge) - valid_count)
            if shortfall:
                shortfalls.append(
                    {"scene_id": scene_id, "approach_edge": edge, "missing": shortfall}
                )
            per_edge[f"{scene_id}/{edge}"] = {
                "candidate_count": len(proposals),
                "valid_count": valid_count,
                "target_valid_count": int(args.target_valid_per_edge),
                "shortfall": shortfall,
                "evaluated_count": valid_count + sum(rejection_reason_by_key[key].values()),
                "unevaluated_count": max(
                    0,
                    len(proposals) - valid_count - sum(rejection_reason_by_key[key].values()),
                ),
                "rejection_reasons": dict(sorted(rejection_reason_by_key[key].items())),
            }
        summary = {
            "schema_version": 2,
            "functional_fingerprint": fingerprint,
            "accepted_total": len(accepted_rows),
            "rejected_total": len(rejected_rows),
            "accepted_this_run": new_accepted,
            "rejected_this_run": new_rejected,
            "per_edge": per_edge,
            "shortfalls": shortfalls,
        }
        write_json(output_dir / "capture_summary.json", summary)
        print(
            f"[N2M] observation capture complete: accepted_this_run={new_accepted} "
            f"rejected_this_run={new_rejected} shortfalls={len(shortfalls)}"
        )
        if shortfalls and not args.allow_shortfall:
            print(f"[N2M][SHORTFALL] {shortfalls}")
            return 2
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
