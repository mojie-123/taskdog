#!/usr/bin/env python
"""Fail-fast coordinate-frame diagnostics for MuJoCo + AnyGrasp + MoveIt.

Run from the taskdog repository root:

    python custom_envs/scripts/navigation/verify_moveit_frames.py

This script does not move the robot.  It checks:
1) scene.xml's actual wrist_cam pose vs the optical-frame transform used by the
   MoveIt integration;
2) AnyGrasp-axis -> grasp_tcp-axis mapping;
3) grasp_tcp -> gripper_base 0.1358 m geometry.
"""

import math
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)

from custom_envs.mujoco.env import MuJoCoEnv
from mujoco_moveit_frames import (
    ANYGRASP_TO_TCP_AXES,
    CAMERA_OPTICAL_R_GB,
    GRASP_TCP_OFFSET_GB,
    Pose,
    anygrasp_candidate_to_tcp,
    capture_scan_transforms,
    tcp_pose_to_gripper_base_pose,
    validate_camera_optical_transform,
)


def angle_between(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def main():
    env = MuJoCoEnv(render=False)
    try:
        diag = validate_camera_optical_transform(env)
        print("[PASS] MuJoCo wrist_cam optical transform")
        print(f"       position error = {diag['position_error_m']*1e6:.3f} um")
        print(f"       rotation error = {math.degrees(diag['rotation_error_rad']):.9f} deg")
        print("       optical +X(right) -> gripper_base", CAMERA_OPTICAL_R_GB[:, 0])
        print("       optical +Y(down)  -> gripper_base", CAMERA_OPTICAL_R_GB[:, 1])
        print("       optical +Z(front) -> gripper_base", CAMERA_OPTICAL_R_GB[:, 2])

        # Synthetic AnyGrasp frame: identity means approach=cam +X, closing=cam +Y.
        tf = capture_scan_transforms(env)
        cand = anygrasp_candidate_to_tcp(
            tf["T_base_camera_optical"],
            palm_cam=np.array([0.0, 0.0, 0.5]),
            grasp_R_cam=np.eye(3),
            depth=0.05,
            insertion=0.005,
            pregrasp_distance=0.115,
            local_reach_distance=0.015,
        )
        R_tcp = cand["contact_tcp_pose_base"].rotation
        approach_b = cand["approach_base"]
        closing_b = cand["closing_base"]
        assert angle_between(R_tcp[:, 2], approach_b) < 1e-6
        assert angle_between(R_tcp[:, 1], closing_b) < 1e-6
        print("[PASS] AnyGrasp axes -> grasp_tcp axes")
        print("       TCP +Z aligns with AnyGrasp approach")
        print("       TCP +Y aligns with AnyGrasp closing")

        tcp = Pose(np.array([0.2, -0.1, 0.8]), R_tcp)
        gb = tcp_pose_to_gripper_base_pose(tcp)
        reconstructed = gb.position + gb.rotation @ GRASP_TCP_OFFSET_GB
        err = float(np.linalg.norm(reconstructed - tcp.position))
        assert err < 1e-12
        print("[PASS] grasp_tcp <-> gripper_base geometry")
        print(f"       TCP offset = {GRASP_TCP_OFFSET_GB.tolist()} m; round-trip error={err:.3e} m")

        # Pure algebra sanity check for the fixed axis mapping.
        assert np.allclose(ANYGRASP_TO_TCP_AXES[:, 2], [1.0, 0.0, 0.0])
        assert np.allclose(ANYGRASP_TO_TCP_AXES[:, 1], [0.0, 1.0, 0.0])
        print("[PASS] fixed AnyGrasp->TCP axis permutation")
        print("\nAll frame checks passed. Safe to proceed to MoveIt planning smoke test.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
