"""Collision and camera-geometry checks for proposed observations."""

from __future__ import annotations

import math

import numpy as np

from custom_envs.scripts.FloAff.geometry import angular_distance_deg, angle_of, observation_metrics


def body_subtree_ids(model, root_body_name: str) -> set[int]:
    import mujoco

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root < 0:
        raise ValueError(f"body not found: {root_body_name}")
    result: set[int] = set()
    for body_id in range(int(model.nbody)):
        current = body_id
        while current > 0:
            if current == root:
                result.add(body_id)
                break
            current = int(model.body_parentid[current])
    result.add(int(root))
    return result


def robot_environment_contacts(
    env,
    robot_root_body: str = "base_link",
    ignored_environment_geoms: tuple[str, ...] = ("floor",),
) -> list[str]:
    import mujoco

    robot_bodies = body_subtree_ids(env.model, robot_root_body)
    ignored = set(ignored_environment_geoms)
    hits: dict[str, float] = {}
    for index in range(int(env.data.ncon)):
        contact = env.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(env.model.geom_bodyid[geom1])
        body2 = int(env.model.geom_bodyid[geom2])
        robot1, robot2 = body1 in robot_bodies, body2 in robot_bodies
        if robot1 == robot2:
            continue
        robot_geom, environment_geom = (geom1, geom2) if robot1 else (geom2, geom1)
        environment_name = (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom)
            or f"geom#{environment_geom}"
        )
        if environment_name in ignored:
            continue
        robot_name = (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            or f"geom#{robot_geom}"
        )
        key = f"{robot_name}<->{environment_name}"
        hits[key] = max(hits.get(key, 0.0), max(0.0, -float(contact.dist)))
    return [f"{name} penetration={1000.0 * hits[name]:.2f}mm" for name in sorted(hits)]


def actual_camera_metrics(env, camera_name: str, target_body: str, outward_xy) -> dict[str, float]:
    import mujoco

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    if camera_id < 0 or target_id < 0:
        raise ValueError("camera or target body not found")
    camera_xy = env.data.cam_xpos[camera_id, :2].astype(np.float64)
    target_xy = env.data.xpos[target_id, :2].astype(np.float64)
    distance, alpha = observation_metrics(camera_xy, target_xy, outward_xy)
    camera_rotation = env.data.cam_xmat[camera_id].reshape(3, 3)
    # MuJoCo camera looks along local -Z.
    forward_world = -camera_rotation[:, 2]
    bearing_world = target_xy - camera_xy
    bearing_yaw = angle_of(bearing_world)
    forward_yaw = math.atan2(float(forward_world[1]), float(forward_world[0]))
    return {
        "camera_range_xy_m": distance,
        "alpha_deg": math.degrees(alpha),
        "target_bearing_offset_deg": angular_distance_deg(forward_yaw, bearing_yaw),
    }
