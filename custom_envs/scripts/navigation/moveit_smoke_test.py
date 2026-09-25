#!/usr/bin/env python
"""Small MoveIt/MuJoCo integration smoke test; does not move the simulated robot.

Start `moveit_mujoco.launch.py` (or the combined Nav2+MoveIt launch) first, then:

    python custom_envs/scripts/navigation/moveit_smoke_test.py

Checks that the new robot model, KDL IK, /joint_states bridge and planning service
agree on the *current* MuJoCo grasp_tcp pose.
"""

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "utils"))

from custom_envs.mujoco.env import MuJoCoEnv
from custom_envs.utils.moveit_bridge import MoveItROS2Bridge
from mujoco_moveit_frames import (
    Pose,
    T_gripper_base_grasp_tcp,
    inv_T,
    mujoco_body_pose_world,
    pose_to_ros_dict,
    validate_camera_optical_transform,
)

ARM = [f"joint{i}" for i in range(1, 7)]
GRIP = ["joint7", "joint8"]


def main():
    env = MuJoCoEnv(render=False)
    bridge = MoveItROS2Bridge(request_timeout=10.0)
    try:
        cam = validate_camera_optical_transform(env)
        print(f"[PASS] camera transform: pos={cam['position_error_m']:.3e}m "
              f"rot={cam['rotation_error_rad']:.3e}rad")

        rs = env.get_robot_state()
        jpos = rs["joint_pos"]
        arm = jpos[np.asarray(env._arm_all_idx, dtype=np.int32)].astype(np.float64)
        grip = jpos[np.asarray(env._grip_all_idx, dtype=np.int32)].astype(np.float64)
        names = ARM + GRIP
        positions = np.concatenate([arm, grip])

        T_w_b = mujoco_body_pose_world(env, "base_link").matrix()
        T_w_gb = mujoco_body_pose_world(env, "gripper_base").matrix()
        T_b_tcp = inv_T(T_w_b) @ T_w_gb @ T_gripper_base_grasp_tcp()
        target = pose_to_ros_dict(Pose(T_b_tcp[:3, 3], T_b_tcp[:3, :3]))

        bridge.start(timeout=35.0)
        bridge.update_joint_state(names, positions)
        time.sleep(0.25)

        ik = bridge.compute_ik(target, names, positions, avoid_collisions=True, timeout_sec=3.0, timeout=6.0)
        if not ik.get("ok", False):
            raise RuntimeError(f"current-pose IK failed: {ik}")
        print("[PASS] MoveIt IK for current MuJoCo grasp_tcp pose")

        plan = bridge.plan_to_pose(
            target, names, positions,
            position_tolerance=0.003,
            orientation_tolerance=0.03,
            allowed_planning_time=2.0,
            timeout=8.0,
        )
        if not plan.get("ok", False):
            raise RuntimeError(f"current-pose planning failed: {plan}")
        n = len(plan.get("trajectory", {}).get("points", []))
        print(f"[PASS] MoveIt planning service (trajectory points={n})")
        print("\nMoveIt/MuJoCo smoke test passed.")
    finally:
        try:
            bridge.stop()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
