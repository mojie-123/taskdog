"""Piper arm inverse kinematics using ikpy.

Chain: base_link -> arm_base_link -> link1..link6 -> gripper_base
Joint limits from SOURCE_M20_Piper.urdf:
  joint1: [-2.618, 2.618]  z-axis
  joint2: [0, 3.14]        z-axis
  joint3: [-2.443, 2.443]  z-axis
  joint4: [-2.618, 2.618]  z-axis
  joint5: [-2.618, 2.618]  z-axis
  joint6: [-2.094, 2.094]  z-axis
"""

import os
import numpy as np

_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "assets", "m20_piper_single", "SOURCE_M20_Piper.urdf"
)

_chain = None


def _build_chain():
    """Build ikpy chain for Piper arm (joints 1-6, base -> gripper_base)."""
    import ikpy.chain
    chain = ikpy.chain.Chain.from_urdf_file(
        os.path.abspath(_URDF_PATH),
        base_elements=["base_link", "arm_base_link"],
        active_links_mask=[
            False,  # arm_base_link (fixed base)
            True,   # joint1
            True,   # joint2
            True,   # joint3
            True,   # joint4
            True,   # joint5
            True,   # joint6
            False,  # gripper_base (fixed ee)
        ],
        name="piper_arm",
    )
    return chain


def get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


def solve(target_pos, target_rot=None, initial_angles=None):
    """Solve IK for Piper arm.

    Parameters
    ----------
    target_pos : (3,) float
        Target position in arm_base_link frame.
    target_rot : (3,3) float or None
        Target rotation matrix. None = ignore orientation.
    initial_angles : (6,) float or None
        Initial joint angles. Defaults to zeros.

    Returns
    -------
    joint_angles : (6,) float
        Angles for joint1..joint6 in radians.
    """
    chain = get_chain()

    target_frame = np.eye(4)
    target_frame[:3, 3] = target_pos
    if target_rot is not None:
        target_frame[:3, :3] = target_rot

    if initial_angles is None:
        q0 = [0.0] * len(chain.links)
    else:
        q0 = [0.0] + list(initial_angles) + [0.0]

    result = chain.inverse_kinematics(
        target_frame,
        initial_position=q0,
        orientation_mode="all" if target_rot is not None else None,
    )
    # strip base and ee placeholders
    return np.array(result[1:-1], dtype=np.float32)


def fk(joint_angles):
    """Forward kinematics. Returns (4,4) transform in arm_base_link frame."""
    chain = get_chain()
    q = [0.0] + list(joint_angles) + [0.0]
    return chain.forward_kinematics(q)


def quat_to_rot(quat_wxyz):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),  1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),  2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])
    return R


def world_pos_to_arm_frame(world_pos, robot_pos_w, robot_quat_w):
    """Convert world-frame position to Piper arm_base_link frame.

    Parameters
    ----------
    world_pos    : (3,) target position in world frame
    robot_pos_w  : (3,) robot base position in world frame
    robot_quat_w : (4,) robot quaternion [w, x, y, z]

    Returns
    -------
    pos_in_arm : (3,) position in arm_base_link frame
    """
    # arm_base_link offset from base_link: z+0.0888 (URDF base_to_arm)
    ARM_BASE_OFFSET = np.array([0.0, 0.0, 0.0888])
    R = quat_to_rot(robot_quat_w)
    arm_base_w = robot_pos_w + R @ ARM_BASE_OFFSET
    pos_in_arm = R.T @ (world_pos - arm_base_w)
    return pos_in_arm


if __name__ == "__main__":
    print("[arm_ik] Building chain...")
    chain = get_chain()
    print(f"[arm_ik] Links: {[l.name for l in chain.links]}")
    fk0 = fk(np.zeros(6))
    print(f"[arm_ik] FK at zero: {fk0[:3, 3]}")
    angles = solve(fk0[:3, 3])
    print(f"[arm_ik] IK result: {np.round(angles, 3)}")
    fk1 = fk(angles)
    print(f"[arm_ik] IK check: {np.round(fk1[:3, 3], 4)} (target {np.round(fk0[:3, 3], 4)})")
    print("[arm_ik] Self-test passed!")
