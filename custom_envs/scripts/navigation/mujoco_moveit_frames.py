"""Coordinate conversions shared by the MuJoCo + AnyGrasp + MoveIt pipeline.

The single source of truth used here is the actual MuJoCo model at SCAN time:

    base_link -> gripper_base -> wrist_camera_optical_frame

AnyGrasp's translation / rotation are already expressed in the optical frame
(x right, y down, z forward).  joint7 is intentionally *not* part of the high
level transform chain.  joint7 is a finger prismatic joint; the legacy IKPy
code only used its frame because the old FK chain happened to terminate there.

Frame conventions
-----------------
- MuJoCo quaternions in the project: (w, x, y, z)
- ROS geometry_msgs quaternions:     (x, y, z, w)
- grasp_tcp: fixed at +0.1358 m along gripper_base +Z, same orientation as
  gripper_base.  This matches the old geometry where the joint7 origin was
  located 0.1358 m in front of gripper_base, but without inheriting joint7's
  Rx(+90 deg) frame rotation.
- AnyGrasp rotation columns used by the current SDK pipeline:
    col 0 = approach
    col 1 = closing
  The fixed right multiplication Ry(+90 deg) maps those axes to Piper TCP axes:
    AnyGrasp +X -> TCP +Z (approach)
    AnyGrasp +Y -> TCP +Y (closing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

import numpy as np


CAMERA_OFFSET_POS_GB = np.array([-0.05, 0.0, 0.06], dtype=np.float64)
# optical frame -> gripper_base frame: image +X(right)->GB -Y,
# image +Y(down)->GB +X, image +Z(forward)->GB +Z.
CAMERA_OPTICAL_R_GB = np.array(
    [[0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

GRASP_TCP_OFFSET_GB = np.array([0.0, 0.0, 0.1358], dtype=np.float64)

# Right-multiply AnyGrasp grasp orientation by this matrix so the resulting
# frame uses Piper TCP axes: +Z approach, +Y closing.
ANYGRASP_TO_TCP_AXES = np.array(
    [[0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)  # Ry(+90 deg)


@dataclass(frozen=True)
class Pose:
    """Rigid pose represented as position + 3x3 rotation."""

    position: np.ndarray
    rotation: np.ndarray

    def matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(self.rotation, dtype=np.float64)
        T[:3, 3] = np.asarray(self.position, dtype=np.float64)
        return T


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Return the nearest proper rotation matrix using SVD."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(R)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def sanitize_anygrasp_rotation(R_cam: np.ndarray) -> np.ndarray:
    """Repair small numerical error / det=-1 reflection from AnyGrasp output.

    The current project assumes col0 (approach) and col1 (closing) are the
    physically meaningful axes.  If det<0, rebuild col2 as approach x closing
    before a final SO(3) projection so those two directions are preserved.
    """
    R = np.asarray(R_cam, dtype=np.float64).reshape(3, 3).copy()
    a = R[:, 0]
    c = R[:, 1]
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(c) < 1e-8:
        raise ValueError("degenerate AnyGrasp rotation")
    a = a / np.linalg.norm(a)
    c = c - a * float(a @ c)
    if np.linalg.norm(c) < 1e-8:
        raise ValueError("AnyGrasp approach/closing axes are nearly parallel")
    c = c / np.linalg.norm(c)
    z = np.cross(a, c)
    R_rebuilt = np.column_stack([a, c, z])
    return _project_to_so3(R_rebuilt)


def make_T(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = map(float, np.asarray(q, dtype=np.float64).reshape(4))
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ROS quaternion (x,y,z,w)."""
    R = _project_to_so3(R)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def pose_to_ros_dict(pose: Pose) -> Dict[str, list]:
    return {
        "position": np.asarray(pose.position, dtype=np.float64).tolist(),
        "quaternion_xyzw": rot_to_quat_xyzw(pose.rotation).tolist(),
    }


def ros_dict_to_pose(data: Dict[str, list]) -> Pose:
    p = np.asarray(data["position"], dtype=np.float64)
    x, y, z, w = map(float, data["quaternion_xyzw"])
    # xyzw -> wxyz helper
    R = quat_wxyz_to_rot(np.array([w, x, y, z], dtype=np.float64))
    return Pose(p, R)


def mujoco_body_pose_world(env, body_name: str) -> Pose:
    """Read a body's *actual* MuJoCo world pose, not FK reconstructed elsewhere."""
    import mujoco

    bid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    p = env._data.xpos[bid].astype(np.float64).copy()
    R = env._data.xmat[bid].reshape(3, 3).astype(np.float64).copy()
    return Pose(p, R)




def mujoco_camera_optical_pose_world(env, camera_name: str = "wrist_cam") -> Pose:
    """Read the rendered camera's optical-frame pose directly from MuJoCo.

    MuJoCo camera frame convention is +X right, +Y up, -Z forward.  The depth
    point cloud / AnyGrasp optical convention used by this project is +X right,
    +Y down, +Z forward, hence R_camBody_optical = diag(1,-1,-1).
    """
    import mujoco

    cid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cid < 0:
        raise ValueError(f"MuJoCo camera not found: {camera_name}")
    p = env._data.cam_xpos[cid].astype(np.float64).copy()
    R_w_cam_body = env._data.cam_xmat[cid].reshape(3, 3).astype(np.float64).copy()
    R_cam_body_optical = np.diag([1.0, -1.0, -1.0])
    R_w_optical = R_w_cam_body @ R_cam_body_optical
    return Pose(p, R_w_optical)


def validate_camera_optical_transform(
    env,
    camera_name: str = "wrist_cam",
    position_tolerance_m: float = 5e-5,
    rotation_tolerance_rad: float = 5e-5,
) -> Dict[str, float]:
    """Cross-check the hard-coded camera mount against MuJoCo runtime truth.

    This is intentionally a fail-fast assertion: if scene.xml camera mounting is
    changed later, AnyGrasp targets must not silently be transformed with stale
    extrinsics.
    """
    gb = mujoco_body_pose_world(env, "gripper_base")
    T_w_pred = gb.matrix() @ T_gripper_base_camera_optical()
    actual = mujoco_camera_optical_pose_world(env, camera_name)

    pos_err = float(np.linalg.norm(T_w_pred[:3, 3] - actual.position))
    R_delta = T_w_pred[:3, :3].T @ actual.rotation
    cosang = float(np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0))
    rot_err = float(math.acos(cosang))

    if pos_err > float(position_tolerance_m) or rot_err > float(rotation_tolerance_rad):
        raise RuntimeError(
            "camera optical transform mismatch: "
            f"position_error={pos_err:.6g}m rotation_error={rot_err:.6g}rad. "
            "Refuse to run MoveIt grasping with inconsistent camera extrinsics."
        )
    return {"position_error_m": pos_err, "rotation_error_rad": rot_err}

def T_gripper_base_camera_optical() -> np.ndarray:
    """^GB T_Copt from scene.xml camera mount + optical convention."""
    return make_T(CAMERA_OFFSET_POS_GB, CAMERA_OPTICAL_R_GB)


def T_gripper_base_grasp_tcp() -> np.ndarray:
    """^GB T_TCP.  TCP keeps gripper_base orientation."""
    return make_T(GRASP_TCP_OFFSET_GB, np.eye(3, dtype=np.float64))


def capture_scan_transforms(env) -> Dict[str, np.ndarray]:
    """Capture immutable transforms at the exact SCAN pose.

    Returned values are safe to keep after the arm starts moving.  This avoids
    accidentally transforming an AnyGrasp result using the *current* camera
    pose instead of the pose at which the depth image was captured.
    """
    T_w_b = mujoco_body_pose_world(env, "base_link").matrix()
    T_w_gb = mujoco_body_pose_world(env, "gripper_base").matrix()
    T_b_w = inv_T(T_w_b)
    T_b_gb = T_b_w @ T_w_gb
    T_b_cam = T_b_gb @ T_gripper_base_camera_optical()
    return {
        "T_world_base": T_w_b,
        "T_base_world": T_b_w,
        "T_world_gripper_base": T_w_gb,
        "T_base_gripper_base": T_b_gb,
        "T_base_camera_optical": T_b_cam,
    }


def transform_points(T_dst_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    P = np.asarray(points_src, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(T_dst_src, dtype=np.float64)[:3, :3]
    p = np.asarray(T_dst_src, dtype=np.float64)[:3, 3]
    return P @ R.T + p


def anygrasp_candidate_to_tcp(
    T_base_camera_optical: np.ndarray,
    palm_cam: np.ndarray,
    grasp_R_cam: np.ndarray,
    depth: float,
    insertion: float = 0.005,
    pregrasp_distance: float = 0.115,
    local_reach_distance: float = 0.015,
) -> Dict[str, object]:
    """Convert one AnyGrasp candidate into MoveIt TCP target poses.

    Returns all quantities in base_link unless the key explicitly says camera.
    The contact target is the grasp_tcp origin.  It is placed at

        palm + (depth + insertion) * approach

    so insertion is explicit instead of being hidden in combinations of the
    legacy 0.1358 / 0.2458 constants.
    """
    T_b_c = np.asarray(T_base_camera_optical, dtype=np.float64).reshape(4, 4)
    R_b_c = T_b_c[:3, :3]
    p_b_c = T_b_c[:3, 3]

    R_c_q = sanitize_anygrasp_rotation(grasp_R_cam)
    palm_c = np.asarray(palm_cam, dtype=np.float64).reshape(3)
    approach_c = R_c_q[:, 0]

    contact_c = palm_c + (float(depth) + float(insertion)) * approach_c
    approach_b = R_b_c @ approach_c
    approach_b /= max(np.linalg.norm(approach_b), 1e-12)

    contact_b = R_b_c @ contact_c + p_b_c
    palm_b = R_b_c @ palm_c + p_b_c

    # Desired TCP/gripper_base orientation.  TCP +Z is approach, TCP +Y closing.
    R_tcp_b = _project_to_so3(R_b_c @ R_c_q @ ANYGRASP_TO_TCP_AXES)

    pre_b = contact_b - float(pregrasp_distance) * approach_b
    near_b = contact_b - float(local_reach_distance) * approach_b

    return {
        "palm_base": palm_b,
        "contact_tcp_pose_base": Pose(contact_b, R_tcp_b),
        "pregrasp_tcp_pose_base": Pose(pre_b, R_tcp_b),
        "near_tcp_pose_base": Pose(near_b, R_tcp_b),
        "approach_base": approach_b,
        "closing_base": R_tcp_b[:, 1].copy(),
        "depth": float(depth),
        "insertion": float(insertion),
        "pregrasp_distance": float(pregrasp_distance),
        "local_reach_distance": float(local_reach_distance),
    }


def tcp_pose_to_gripper_base_pose(tcp_pose: Pose) -> Pose:
    """Convert a desired grasp_tcp pose to the desired gripper_base pose."""
    R = np.asarray(tcp_pose.rotation, dtype=np.float64)
    p_tcp = np.asarray(tcp_pose.position, dtype=np.float64)
    p_gb = p_tcp - R @ GRASP_TCP_OFFSET_GB
    return Pose(p_gb, R.copy())


def pose_error(current: Pose, target: Pose) -> Tuple[np.ndarray, np.ndarray]:
    """Return world/base-frame position error and small-angle rotation vector.

    rotvec is expressed in the same parent frame as the input rotation matrices.
    """
    ep = np.asarray(target.position) - np.asarray(current.position)
    R_err = np.asarray(target.rotation) @ np.asarray(current.rotation).T
    er = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=np.float64)
    return ep, er
