"""Repeatable MuJoCo reset, free-body placement and physical base locking.
可重复地重置 MuJoCo；
放置带 freejoint 的物体；
通过物理约束锁定底盘。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from custom_envs.scripts.N2M.geometry import rpy_to_quat_wxyz


def set_freejoint_pose(env, joint_name: str, position, quaternion_wxyz) -> None:   # 用来把cube放到指定位置
    import mujoco

    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)   # 把cube当成一个free joint来操作
    if jid < 0:
        raise ValueError(f"MuJoCo freejoint not found: {joint_name}")
    if int(env.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):   # 确认是free joint
        raise ValueError(f"Joint {joint_name!r} is not a freejoint")

    qadr = int(env.model.jnt_qposadr[jid])   # qpos起始地址
    dadr = int(env.model.jnt_dofadr[jid])   # qvel起始地址
    env.data.qpos[qadr:qadr + 3] = np.asarray(position, dtype=np.float64)   # [x, y, z]
    env.data.qpos[qadr + 3:qadr + 7] = np.asarray(quaternion_wxyz, dtype=np.float64)    # qw, qx, qy, qz
    env.data.qvel[dadr:dadr + 6] = 0.0   # 速度为0
    mujoco.mj_forward(env.model, env.data)


def place_episode_state(
    env,
    cube_pose,
    base_pose,
    robot_root_z: float = 0.563874,
    robot_root_roll: float = 0.0,
    robot_root_pitch: float = 0.0,
) -> None:   # 建立一次候选 M 抓取实验的初始状态
    cube_pose = np.asarray(cube_pose, dtype=np.float64)   # [cube_x, cube_y, cube_z, qw, qx, qy, qz]
    base_pose = np.asarray(base_pose, dtype=np.float64)   # [robot_x, robot_y, robot_yaw]

    env.reset()   # 重置环境
    set_freejoint_pose(
        env,
        "cube_freejoint",
        cube_pose[:3],
        cube_pose[3:7] if cube_pose.size >= 7 else (1.0, 0.0, 0.0, 0.0),
    )   # 重置cube
    env.set_root_pose(
        np.array([base_pose[0], base_pose[1], float(robot_root_z)], dtype=np.float64),
        rpy_to_quat_wxyz(robot_root_roll, robot_root_pitch, base_pose[2]),
    )   # 重置robot
    import mujoco
    mujoco.mj_forward(env.model, env.data)


def _body_is_in_subtree(model, body_id: int, root_body_id: int) -> bool:   # 判断接触 body 是否属于整个 base_link 子树
    cur = int(body_id)
    root = int(root_body_id)
    while cur > 0:
        if cur == root:
            return True
        cur = int(model.body_parentid[cur])
    return cur == root


def robot_table_contacts(env, robot_root_body: str = "base_link") -> list[str]:   # 机器人某个子树是否与 living_coffee_table 接触
    """Return current robot-subtree contacts with the coffee table.

    ``base_link`` checks the whole robot at placement time.  Passing ``link1``
    restricts the check to the arm/gripper subtree during execution, so known
    leg-vs-table contacts at the legacy reference M do not hide a new arm hit.
    """
    import mujoco

    model, data = env.model, env.data
    robot_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body
    )
    table_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "living_coffee_table"
    )
    if robot_bid < 0 or table_bid < 0:   # 返回“没有接触”
        return []

    hits = {}
    for index in range(int(data.ncon)):   # 遍历当前所有接触，data.ncon 是当前状态下的接触数量
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        pair_is_hit = (
            (_body_is_in_subtree(model, b1, robot_bid) and _body_is_in_subtree(model, b2, table_bid))
            or
            (_body_is_in_subtree(model, b2, robot_bid) and _body_is_in_subtree(model, b1, table_bid))
        )   # 判断是不是机器人与茶几的接触
        if pair_is_hit:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom#{g1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom#{g2}"
            body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body#{b1}"
            body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body#{b2}"
            key = f"{body1}/{n1}<->{body2}/{n2}"
            penetration_mm = max(0.0, -float(contact.dist) * 1000.0)
            hits[key] = max(hits.get(key, 0.0), penetration_mm)
    return [f"{key} penetration={hits[key]:.2f}mm" for key in sorted(hits)]


def robot_environment_contacts(
    env,
    robot_root_body: str = "base_link",
    ignored_environment_geoms: tuple[str, ...] = ("floor",),   # 默认忽略地面
) -> list[str]:
    """Return robot contacts with non-ignored environment geometry.

    Self-contact is deliberately excluded.  This is stricter than
    :func:`robot_table_contacts` and is intended as a placement preflight for
    model-predicted base poses, which are not guaranteed to stay near the
    coffee table or away from other furniture.

    检查：
    机器人子树
    与
    所有非机器人 geom
    之间的接触
    """
    import mujoco

    model, data = env.model, env.data
    robot_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_root_body
    )
    if robot_bid < 0:
        raise ValueError(f"body not found: {robot_root_body}")
    ignored = set(ignored_environment_geoms)
    hits: dict[str, float] = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        robot1 = _body_is_in_subtree(model, b1, robot_bid)
        robot2 = _body_is_in_subtree(model, b2, robot_bid)
        if robot1 == robot2:
            continue
        robot_geom, environment_geom = (g1, g2) if robot1 else (g2, g1)
        environment_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom
            )
            or f"geom#{environment_geom}"
        )
        if environment_name in ignored:
            continue
        robot_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            or f"geom#{robot_geom}"
        )
        key = f"{robot_name}<->{environment_name}"
        penetration_mm = max(0.0, -float(contact.dist) * 1000.0)
        hits[key] = max(hits.get(key, 0.0), penetration_mm)
    return [f"{key} penetration={hits[key]:.2f}mm" for key in sorted(hits)]


@dataclass
class BaseWeld:
    env: object

    def __post_init__(self):
        import mujoco

        model = self.env.model
        self.base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.anchor_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_lock_anchor"
        )   # 查找 mocap 锚点
        self.eqid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, "base_lock_weld"
        )   # 查找 weld 约束
        if min(self.base_bid, self.anchor_bid, self.eqid) < 0:   # 如果场景中缺少任意一个，就无法使用底盘锁
            raise RuntimeError("Scene is missing base_lock_anchor/base_lock_weld")
        self.mocapid = int(model.body_mocapid[self.anchor_bid])   # mocap ID
        if self.mocapid < 0:
            raise RuntimeError("base_lock_anchor must be a mocap body")
        self.active = False
        self.reference_pos = None
        self.reference_quat = None

    def enable(self) -> None:   # 把机器人根节点固定在当前世界位姿
        import mujoco

        data = self.env.data
        pos = data.xpos[self.base_bid].astype(np.float64).copy()
        quat = data.xquat[self.base_bid].astype(np.float64).copy()
        data.mocap_pos[self.mocapid] = pos   # 把锚点移动到底盘当前位置
        data.mocap_quat[self.mocapid] = quat
        data.eq_active[self.eqid] = 1   # 激活 equality constraint。从这一刻起，每个 MuJoCo 物理步都会求解这个 weld 约束
        mujoco.mj_forward(self.env.model, data)   # 更新mujoco状态
        self.reference_pos = pos   # 记录参考位姿
        self.reference_quat = quat
        self.active = True

    def disable(self) -> None:   # 关闭 weld，使底盘重新成为自由状态
        import mujoco

        self.env.data.eq_active[self.eqid] = 0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.active = False

    def drift(self) -> Tuple[float, float]:   # 计算锁定后底盘偏离参考位姿多少
        if not self.active or self.reference_pos is None:
            return 0.0, 0.0
        pos = self.env.data.xpos[self.base_bid].astype(np.float64)
        quat = self.env.data.xquat[self.base_bid].astype(np.float64)
        position_m = float(np.linalg.norm(pos - self.reference_pos))
        dot = float(np.clip(abs(np.dot(quat, self.reference_quat)), 0.0, 1.0))
        rotation_rad = float(2.0 * np.arccos(dot))
        return position_m, rotation_rad
