"""Generic reset and target/base placement for the FloAff collector."""

from __future__ import annotations

import math

import numpy as np


def yaw_to_quaternion_wxyz(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(float(yaw) / 2.0), 0.0, 0.0, math.sin(float(yaw) / 2.0)],
        dtype=np.float64,
    )


def set_freejoint_pose(env, joint_name: str, pose_world) -> None:
    import mujoco

    pose = np.asarray(pose_world, dtype=np.float64).reshape(7)
    joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"freejoint not found: {joint_name}")
    if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"joint {joint_name!r} is not a freejoint")
    qpos_address = int(env.model.jnt_qposadr[joint_id])
    dof_address = int(env.model.jnt_dofadr[joint_id])
    env.data.qpos[qpos_address : qpos_address + 7] = pose
    env.data.qvel[dof_address : dof_address + 6] = 0.0


def place_observation_state(
    env,
    target_pose_world,
    base_pose_world,
    *,
    target_freejoint: str,
    robot_root_z: float,
) -> None:
    import mujoco

    target_pose = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
    base_pose = np.asarray(base_pose_world, dtype=np.float64).reshape(3)
    env.reset()
    set_freejoint_pose(env, target_freejoint, target_pose)
    env.set_root_pose(
        np.array([base_pose[0], base_pose[1], float(robot_root_z)], dtype=np.float64),
        yaw_to_quaternion_wxyz(base_pose[2]),
    )
    mujoco.mj_forward(env.model, env.data)
