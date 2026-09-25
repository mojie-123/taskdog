#!/usr/bin/env python
"""
verify_moveit_fk.py

Compare MuJoCo forward kinematics against the MoveIt URDF using exactly the
same current joint1..joint6 values.  This script does NOT call MoveIt or ROS 2.

Run from the taskdog repository root:

    conda activate env_isaaclab
    python custom_envs/scripts/navigation/verify_moveit_fk.py

Expected result:
- every common link should have very small position/rotation error
- grasp_tcp should also agree (MuJoCo gripper_base + 0.1358 m along local +Z)

If an error first appears at a particular link, inspect that URDF joint's
origin/axis against scene.xml.
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)

from custom_envs.mujoco.env import MuJoCoEnv


URDF_PATH = os.path.join(
    _ROOT, "custom_envs", "assets", "m20_piper_single", "M20_Piper_moveit.urdf"
)

ARM_NAMES = [f"joint{i}" for i in range(1, 7)]
COMMON_LINKS = [
    "arm_base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_base",
]


def rpy_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    # URDF origin rpy uses fixed-axis roll-pitch-yaw:
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return Rz @ Ry @ Rx


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def load_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints_by_child = {}
    limits = {}

    for j in root.findall("joint"):
        name = j.attrib["name"]
        typ = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]

        origin = j.find("origin")
        xyz = parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
        rpy = parse_xyz(origin.attrib.get("rpy") if origin is not None else None)
        axis_node = j.find("axis")
        axis = parse_xyz(
            axis_node.attrib.get("xyz") if axis_node is not None else None,
            default=(1.0, 0.0, 0.0),
        )

        lim = j.find("limit")
        if lim is not None:
            lower = float(lim.attrib.get("lower", "-inf"))
            upper = float(lim.attrib.get("upper", "inf"))
            limits[name] = (lower, upper)

        joints_by_child[child] = {
            "name": name,
            "type": typ,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }

    return joints_by_child, limits


def urdf_link_transform(link_name, joints_by_child, qmap):
    """Return base_link -> link transform."""
    chain = []
    cur = link_name
    while cur != "base_link":
        if cur not in joints_by_child:
            raise RuntimeError(
                f"Cannot trace '{link_name}' to base_link: no parent joint for '{cur}'"
            )
        j = joints_by_child[cur]
        chain.append(j)
        cur = j["parent"]
    chain.reverse()

    T = np.eye(4, dtype=np.float64)
    for j in chain:
        T_origin = T_from_Rp(rpy_matrix(j["rpy"]), j["xyz"])
        T = T @ T_origin

        typ = j["type"]
        q = float(qmap.get(j["name"], 0.0))
        if typ in ("revolute", "continuous"):
            T = T @ T_from_Rp(axis_angle_matrix(j["axis"], q), np.zeros(3))
        elif typ == "prismatic":
            T = T @ T_from_Rp(np.eye(3), j["axis"] * q)
        elif typ == "fixed":
            pass
        else:
            raise RuntimeError(f"Unsupported URDF joint type: {typ}")

    return T


def mujoco_body_T_world(env, body_name):
    import mujoco

    bid = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if bid < 0:
        raise RuntimeError(f"MuJoCo body '{body_name}' not found")

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(env.data.xpos[bid], dtype=np.float64)
    T[:3, :3] = np.asarray(env.data.xmat[bid], dtype=np.float64).reshape(3, 3)
    return T


def inv_T(T):
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def rot_err_rad(Ra, Rb):
    dR = Ra.T @ Rb
    c = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(c))


def main():
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"MoveIt URDF not found: {URDF_PATH}")

    env = MuJoCoEnv(render=False)
    try:
        rs = env.get_robot_state()
        jpos = rs["joint_pos"]
        arm = jpos[np.asarray(env._arm_all_idx, dtype=np.int32)].astype(np.float64)
        qmap = dict(zip(ARM_NAMES, arm.tolist()))

        joints_by_child, limits = load_urdf(URDF_PATH)

        print("MoveIt URDF:", URDF_PATH)
        print("Current MuJoCo arm q:")
        for name, q in qmap.items():
            lim = limits.get(name)
            if lim is None:
                print(f"  {name}: {q:+.8f} rad  [no URDF limit]")
            else:
                lo, hi = lim
                ok = (q >= lo - 1e-9) and (q <= hi + 1e-9)
                print(
                    f"  {name}: {q:+.8f} rad  "
                    f"limit=[{lo:+.6f}, {hi:+.6f}]  "
                    f"{'OK' if ok else '*** OUT OF LIMIT ***'}"
                )

        T_w_base = mujoco_body_T_world(env, "base_link")
        T_base_w = inv_T(T_w_base)

        print("\nPer-link FK comparison (base_link frame):")
        first_bad = None

        for link in COMMON_LINKS:
            T_urdf = urdf_link_transform(link, joints_by_child, qmap)
            T_mj = T_base_w @ mujoco_body_T_world(env, link)

            pe = float(np.linalg.norm(T_urdf[:3, 3] - T_mj[:3, 3]))
            re = rot_err_rad(T_urdf[:3, :3], T_mj[:3, :3])

            status = "PASS" if pe < 1e-5 and re < 1e-5 else "FAIL"
            print(
                f"[{status}] {link:16s} "
                f"pos_err={pe:.9e} m  "
                f"rot_err={math.degrees(re):.9e} deg"
            )

            if status == "FAIL" and first_bad is None:
                first_bad = link
                print("       URDF p =", np.round(T_urdf[:3, 3], 9))
                print("       MJ   p =", np.round(T_mj[:3, 3], 9))
                print("       URDF R =\n", np.round(T_urdf[:3, :3], 9))
                print("       MJ   R =\n", np.round(T_mj[:3, :3], 9))

        # grasp_tcp has no MuJoCo body.  Construct it from physical gripper_base:
        # TCP = gripper_base + 0.1358 m along local +Z, same orientation.
        T_urdf_tcp = urdf_link_transform("grasp_tcp", joints_by_child, qmap)
        T_mj_gb = T_base_w @ mujoco_body_T_world(env, "gripper_base")
        T_gb_tcp = np.eye(4, dtype=np.float64)
        T_gb_tcp[2, 3] = 0.1358
        T_mj_tcp = T_mj_gb @ T_gb_tcp

        pe = float(np.linalg.norm(T_urdf_tcp[:3, 3] - T_mj_tcp[:3, 3]))
        re = rot_err_rad(T_urdf_tcp[:3, :3], T_mj_tcp[:3, :3])
        status = "PASS" if pe < 1e-5 and re < 1e-5 else "FAIL"

        print(
            f"[{status}] {'grasp_tcp':16s} "
            f"pos_err={pe:.9e} m  "
            f"rot_err={math.degrees(re):.9e} deg"
        )
        print("       MuJoCo TCP position =", np.round(T_mj_tcp[:3, 3], 9))
        print("       URDF   TCP position =", np.round(T_urdf_tcp[:3, 3], 9))

        print("\nConclusion:")
        if first_bad is None and status == "PASS":
            print(
                "  PASS: MuJoCo and MoveIt URDF forward kinematics agree for "
                "the current joint state."
            )
            print(
                "  If MoveIt IK still fails for this exact TCP pose, investigate "
                "the MoveIt IK plugin/request configuration rather than frame geometry."
            )
        else:
            print(
                "  FAIL: MuJoCo and MoveIt URDF kinematics do not agree."
            )
            if first_bad is not None:
                print(
                    f"  First mismatch appears at '{first_bad}'. "
                    "Compare that joint origin/axis in the MoveIt URDF against scene.xml."
                )

    finally:
        env.close()


if __name__ == "__main__":
    main()
