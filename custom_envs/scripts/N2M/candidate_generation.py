"""Generate deterministic manipulation-base candidates from one reference pose."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from custom_envs.scripts.N2M.config import (
    CUBE_SCENES,
    EDGES,
    REFERENCE_EDGE_ID,
    REFERENCE_SCENE_ID,
    CandidateGrid,
)
from custom_envs.scripts.N2M.geometry import (
    joint1_towards_cube,
    matrix_to_se2,
    se2_to_matrix,
    wrap_angle,
)


TABLE_BOUNDS = {
    "x_min": 5.0,
    "x_max": 10.0,
    "y_min": 3.2,
    "y_max": 4.1,
}   # 定义茶几边界


def edge_projection(cube_xyz: Iterable[float], edge_id: str) -> np.ndarray:   # 找到 cube 在指定桌边上的投影点。这个点会成为对应桌边局部坐标系的原点
    cube = np.asarray(cube_xyz, dtype=np.float64)
    if edge_id == "left":   # left   → (5.0, 3.6)
        return np.array([TABLE_BOUNDS["x_min"], cube[1]], dtype=np.float64)
    if edge_id == "right":   # right  → (10.0, 3.6)
        return np.array([TABLE_BOUNDS["x_max"], cube[1]], dtype=np.float64)
    if edge_id == "bottom":   # bottom → (7.5, 3.2)
        return np.array([cube[0], TABLE_BOUNDS["y_min"]], dtype=np.float64)
    if edge_id == "top":   # top    → (7.5, 4.1)
        return np.array([cube[0], TABLE_BOUNDS["y_max"]], dtype=np.float64)
    raise KeyError(f"Unknown edge: {edge_id}")


def edge_frame_world(cube_xyz: Iterable[float], edge_id: str) -> np.ndarray:   # 根据投影点建立一个桌边局部坐标系，局部 +X 指向桌子外侧，返回世界坐标系到该桌边坐标系的 4×4 位姿矩阵 T_world_edge
    """Return ^W T_E with +x_E pointing out of the table.

    +y_E is chosen as z cross +x_E, so the frame remains right handed.
    """
    origin = edge_projection(cube_xyz, edge_id)
    outward = np.asarray(EDGES[edge_id].outward_xy, dtype=np.float64)
    yaw = math.atan2(outward[1], outward[0])
    return se2_to_matrix((origin[0], origin[1], yaw))


def reference_pose_in_edge(reference_base: Sequence[float]) -> np.ndarray:   # 把已知成功的参考 M，从世界坐标转换到参考桌边局部坐标系
    ref_scene = CUBE_SCENES[REFERENCE_SCENE_ID]
    T_w_edge = edge_frame_world(ref_scene.cube_xyz, REFERENCE_EDGE_ID)
    return np.linalg.inv(T_w_edge) @ se2_to_matrix(reference_base)


def nominal_pose_for_edge(
    reference_base: Sequence[float],
    cube_xyz: Sequence[float],
    edge_id: str,
) -> np.ndarray:   # 把上面得到的“参考桌边到成功 M 的相对关系”，迁移到新的 cube 和桌边，得到各种场景的参考位姿（搜索中心）
    T_edge_m = reference_pose_in_edge(reference_base)
    return matrix_to_se2(edge_frame_world(cube_xyz, edge_id) @ T_edge_m)


def _candidate_id(scene_id: str, edge_id: str, index: int) -> str:   # 为每个候选生成唯一名称
    return f"{scene_id}__{edge_id}__m{index:04d}"


def generate_candidates(
    reference_base: Sequence[float],
    scene_ids: Sequence[str] | None = None,
    grid: CandidateGrid | None = None,
) -> List[dict]:
    """
    计算迁移后的名义 M；
    沿桌边法向增加不同距离偏移；
    沿桌边切向增加左右偏移；
    增加不同 yaw 偏移；
    计算该候选下朝向 cube 所需的 joint1_target；
    保存候选位姿及其完整来源信息。
    """
    grid = grid or CandidateGrid()
    if any(float(offset) > 0.0 for offset in grid.normal_offsets):
        raise ValueError(
            "normal offsets must be <= 0: zero is the reference M and "
            "negative values move away from the table"
        )
    if scene_ids is None:
        scene_ids = list(CUBE_SCENES)

    candidates: List[dict] = []
    for scene_id in scene_ids:
        if scene_id not in CUBE_SCENES:
            raise KeyError(f"Unknown cube scene {scene_id!r}")
        scene = CUBE_SCENES[scene_id]
        scene_index = 0
        for edge_id in scene.approach_edges:
            nominal = nominal_pose_for_edge(reference_base, scene.cube_xyz, edge_id)   # 生成M
            outward = np.asarray(EDGES[edge_id].outward_xy, dtype=np.float64)
            inward = -outward
            tangent = np.array([-outward[1], outward[0]], dtype=np.float64)

            # Keep local variants adjacent: nominal first, then outward/inward
            # alternatives for the same tangent/yaw.  This makes a bounded
            # --max-candidates search useful and reaches collision-free
            # outward poses early.
            for dt in grid.tangent_offsets:
                for dyaw_deg in grid.yaw_offsets_deg:
                    for dn in grid.normal_offsets:
                        # User-facing normal convention: positive points into
                        # the tabletop, negative points away from it.
                        xy = nominal[:2] + float(dn) * inward + float(dt) * tangent
                        yaw = wrap_angle(nominal[2] + math.radians(float(dyaw_deg)))
                        base_pose = np.array([xy[0], xy[1], yaw], dtype=np.float64)
                        j1 = joint1_towards_cube(base_pose, scene.cube_xyz)
                        candidates.append({
                            "candidate_id": _candidate_id(scene_id, edge_id, scene_index),   # 如short_left__left__m0003
                            "scene_id": scene_id,   # 如short_left
                            "category": scene.category,   # 如corner
                            "approach_edge": edge_id,   # 这个候选是从哪条桌边接近
                            "cube_pose": list(scene.cube_xyz) + [1.0, 0.0, 0.0, 0.0],
                            "base_pose": base_pose.tolist(),   # 机器狗底盘的目标位姿
                            "joint1_target": float(j1),
                            "source": "reference_edge_transfer",   # 候选生成方式。当前都是由参考M沿边迁移得到的
                            "reference_scene_id": REFERENCE_SCENE_ID,   # 参考成功 M 原本所属的场景
                            "reference_edge_id": REFERENCE_EDGE_ID,   # 参考成功 M 原本对应的桌边
                            "reference_base_pose": [float(x) for x in reference_base],   # 已知成功的参考底盘位姿
                            "nominal_base_pose": nominal.tolist(),   # 迁移到当前 cube 和桌边后，还没有添加网格扰动的名义 M
                            "perturbation": {
                                "normal_m": float(dn),   # 法向偏移
                                "tangent_m": float(dt),   # 切向偏移
                                "yaw_deg": float(dyaw_deg),   # 朝向偏移
                            },
                            "grid": asdict(grid),   # 本次候选搜索使用的完整网格配置，包括所有法向、切向和 yaw 偏移值
                        })
                        scene_index += 1
    return candidates


def select_candidates(
    candidates: Sequence[dict],
    candidate_id: str | None = None,
    max_candidates: int | None = None,
) -> List[dict]:
    """
    指定 candidate_id：只保留对应候选；
    指定 max_candidates：只保留前 N 个；
    都不指定：返回全部候选。
    """
    selected = list(candidates)
    if candidate_id:
        selected = [row for row in selected if row["candidate_id"] == candidate_id]
        if not selected:
            raise KeyError(f"Candidate not found: {candidate_id}")
    if max_candidates is not None:
        selected = selected[:max(0, int(max_candidates))]
    return selected
