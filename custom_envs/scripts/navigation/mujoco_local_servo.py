"""Short-range MuJoCo-native Cartesian servo for the final grasp approach.

This deliberately does not use IKPy.  It linearizes the *actual* MuJoCo model
around the current state with mj_jacBody(), so the same kinematic model that is
being simulated is also used for the final 1--2 cm correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from mujoco_moveit_frames import Pose, pose_error


@dataclass
class ServoDiagnostics:
    pos_error_m: float
    rot_error_rad: float
    dq_max_rad: float
    condition: float


class MuJoCoGripperBaseServo:
    def __init__(
        self,
        env,
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        body_name="gripper_base",
        damping=2e-3,
        max_joint_step=(0.010, 0.012, 0.012, 0.012, 0.012, 0.015),
        position_gain=1.0,
        orientation_gain=0.6,
    ):
        import mujoco

        self.env = env
        self.mj = mujoco
        self.body_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise RuntimeError(f"MuJoCo body not found: {body_name}")

        self.joint_names = tuple(joint_names)
        self.joint_ids = np.array([
            mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.joint_names
        ], dtype=np.int32)
        if np.any(self.joint_ids < 0):
            raise RuntimeError("one or more arm joints are missing from the MuJoCo model")

        self.qpos_adr = np.array([
            int(env._model.jnt_qposadr[j]) for j in self.joint_ids
        ], dtype=np.int32)
        self.dof_adr = np.array([
            int(env._model.jnt_dofadr[j]) for j in self.joint_ids
        ], dtype=np.int32)
        self.lo = np.array([env._model.jnt_range[j, 0] for j in self.joint_ids], dtype=np.float64)
        self.hi = np.array([env._model.jnt_range[j, 1] for j in self.joint_ids], dtype=np.float64)

        self.damping = float(damping)
        self.max_joint_step = np.asarray(max_joint_step, dtype=np.float64).reshape(6)
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)

    def current_pose_world(self) -> Pose:
        p = self.env._data.xpos[self.body_id].astype(np.float64).copy()
        R = self.env._data.xmat[self.body_id].reshape(3, 3).astype(np.float64).copy()
        return Pose(p, R)

    def arm_q(self) -> np.ndarray:
        return self.env._data.qpos[self.qpos_adr].astype(np.float64).copy()

    def step_world(self, target_world: Pose) -> Tuple[np.ndarray, ServoDiagnostics]:
        """One DLS servo iteration. Returns a new joint-position target."""
        cur = self.current_pose_world()
        ep, er = pose_error(cur, target_world)

        jacp = np.zeros((3, self.env._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.env._model.nv), dtype=np.float64)
        self.mj.mj_jacBody(self.env._model, self.env._data, jacp, jacr, self.body_id)
        J = np.vstack([
            jacp[:, self.dof_adr],
            jacr[:, self.dof_adr],
        ])

        e = np.concatenate([
            self.position_gain * ep,
            self.orientation_gain * er,
        ])

        lam2 = self.damping * self.damping
        A = J @ J.T + lam2 * np.eye(6, dtype=np.float64)
        try:
            dq = J.T @ np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            dq = J.T @ np.linalg.pinv(A) @ e

        # Scale the *whole* increment rather than clipping each joint separately;
        # this preserves the local Cartesian direction better.
        ratio = np.max(np.abs(dq) / np.maximum(self.max_joint_step, 1e-9))
        if ratio > 1.0:
            dq = dq / ratio

        q_now = self.arm_q()
        q_cmd = np.clip(q_now + dq, self.lo, self.hi)

        try:
            cond = float(np.linalg.cond(J @ J.T))
        except Exception:
            cond = float("inf")

        diag = ServoDiagnostics(
            pos_error_m=float(np.linalg.norm(ep)),
            rot_error_rad=float(np.linalg.norm(er)),
            dq_max_rad=float(np.max(np.abs(dq))),
            condition=cond,
        )
        return q_cmd, diag


def robot_table_contact(env) -> Tuple[bool, str]:
    """Detect arm/gripper contacts with table1/table2.

    Returns (hit, description).  Contacts involving the mobile base/legs are not
    considered here because grasp-stage base locking can leave unrelated support
    contacts in the scene.
    """
    import mujoco

    arm_bodies = {
        "arm_base_link", "link1", "link2", "link3", "link4", "link5",
        "link6", "gripper_base", "link7", "link8",
    }
    table_bodies = {"table1", "table2"}
    m, d = env._model, env._data

    for ci in range(int(d.ncon)):
        c = d.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if ((n1 in arm_bodies and n2 in table_bodies) or
                (n2 in arm_bodies and n1 in table_bodies)):
            return True, f"{n1}<->{n2}"
    return False, ""


def finger_object_contact(env, object_name: str) -> Dict[str, bool]:
    """Return whether link7/link8 currently contact the requested object body."""
    import mujoco

    out = {"link7": False, "link8": False}
    m, d = env._model, env._data
    obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if obj < 0:
        return out
    finger_ids = {
        n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in out
    }
    for ci in range(int(d.ncon)):
        c = d.contact[ci]
        b1 = int(m.geom_bodyid[int(c.geom1)])
        b2 = int(m.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        for name, bid in finger_ids.items():
            if bid in (b1, b2):
                out[name] = True
    return out
