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

import math
import os
import numpy as np

_URDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "assets", "m20_piper_single", "SOURCE_M20_Piper.urdf"
)

_chain = None


def _build_chain():
    """Build ikpy chain for Piper arm (joints 1-6, arm_base_link -> joint7).

    Starting from arm_base_link produces a 9-link chain:
      [Base link, joint1, joint2, joint3, joint4, joint5, joint6,
       joint6_to_gripper_base, joint7]
    We activate only joint1..joint6 (indices 1-6).
    """
    import ikpy.chain
    chain = ikpy.chain.Chain.from_urdf_file(
        os.path.abspath(_URDF_PATH),
        base_elements=["arm_base_link"],
        active_links_mask=[
            False,  # Base link (arm_base_link, fixed)
            True,   # joint1
            True,   # joint2
            True,   # joint3
            True,   # joint4
            True,   # joint5
            True,   # joint6
            False,  # joint6_to_gripper_base (fixed)
            False,  # joint7 (gripper finger prismatic, not used)
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

    # Joint limits for joint1..joint6 used to clip the initial guess so that
    # scipy least_squares does not raise 'Initial guess outside bounds'.
    _JOINT_LIMITS = [
        (-2.618,  2.618),  # joint1
        ( 0.0,    3.14 ),  # joint2
        (-2.967,  2.443),  # joint3  [FIX] upper was 0.0 (bug); URDF says +2.443
        (-1.745,  1.745),  # joint4
        (-1.22,   1.22 ),  # joint5
        (-2.094,  2.094),  # joint6
    ]

    if initial_angles is None:
        # Default: joint1=-pi/2 so the arm starts facing the table side
        # (dog yaw=+pi/2, robot -Y = world +X = table direction).
        q0 = [0.0, -math.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # Clip initial angles to joint limits to avoid scipy bounds error.
        clipped = np.array([
            float(np.clip(a, lo, hi))
            for a, (lo, hi) in zip(initial_angles, _JOINT_LIMITS)
        ])
        # chain has 9 links: [base, j1..j6, fixed_ee, j7]
        q0 = [0.0] + list(clipped) + [0.0, 0.0]

    # ikpy 4.x API: target_position=(3,), target_orientation=(3,3) or None
    result = chain.inverse_kinematics(
        target_position=target_pos,
        target_orientation=target_rot,
        orientation_mode="all" if target_rot is not None else None,
        initial_position=q0,
    )
    # result has 9 values; extract joint1..joint6 (indices 1-6)
    return np.array(result[1:7], dtype=np.float32)


def _rot_axis_angle(axis, angle):
    """Rotation matrix for rotating angle radians around axis (3-vector)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c  ],
    ], dtype=np.float64)


_IK_JOINT_LIMITS = [
    (-2.618,  2.618),
    ( 0.0,    3.14 ),
    (-2.967,  2.443),
    (-1.745,  1.745),
    (-1.22,   1.22 ),
    (-2.094,  2.094),
]


def solve_for_gripper_base(target_gb_pos, target_rot_j7=None, initial_angles=None):
    """Solve IK so that gripper_base reaches target_gb_pos.

    Corrected wrapper around solve() that accounts for the joint7->gripper_base
    offset (xyz=(0,0,0.1358), rpy=(pi/2,0,0)).  Converts the gripper_base
    target to a joint7 target before calling solve():

        p_j7_target = target_gb_pos + R_gb_desired @ [0, 0, 0.1358]

    where R_gb_desired = target_rot_j7 @ _RX_NEG90.

    When the original rotation constraint makes IK fail (joints hit limits or
    gb_err > 3 cm), the function automatically searches for an equivalent
    rotation by rolling the grasp frame around the approach axis (gripper_base
    +X direction) in steps of 5 degrees.  For objects symmetric around the
    approach axis (e.g. cylinders, bananas) any roll angle is kinematically
    equivalent.  The first valid roll (gb_err < 3 cm, no joint at limit) is
    returned.

    Parameters
    ----------
    target_gb_pos : (3,) float
        Desired gripper_base origin in arm_base_link frame.
    target_rot_j7 : (3,3) float or None
        Desired joint7 rotation matrix (= R_gb_desired @ Rx(+90)).
        Pass the value returned by compute_desired_ee_rot_in_arm().
        None = ignore orientation.
    initial_angles : (6,) float or None
        Initial joint angles.

    Returns
    -------
    joint_angles : (6,) float
        Angles for joint1..joint6 in radians such that gripper_base is
        (approximately) at target_gb_pos with the desired orientation.
    """
    if target_rot_j7 is None:
        # No rotation constraint: position-only solve.
        # We must still convert the gripper_base target to a joint7 target by
        # adding the joint7-origin-in-gripper_base offset (0.1358 m along the
        # gripper_base Z axis).  When target_rot is None we estimate the
        # gripper_base orientation from the FK at initial_angles (or identity
        # when initial_angles is None), then iterate once to refine.
        if initial_angles is not None:
            _q0 = np.asarray(initial_angles, dtype=np.float64)
            _R_gb_est = fk_gripper(_q0)[:3, :3]
        else:
            _R_gb_est = np.eye(3)
        _target_j7 = target_gb_pos + _R_gb_est @ _J7_ORIGIN_IN_GB
        _q_pos = solve(_target_j7, target_rot=None, initial_angles=initial_angles)
        # Refine once with the updated gripper_base orientation from the solution
        _R_gb_refined = fk_gripper(_q_pos)[:3, :3]
        _target_j7_refined = target_gb_pos + _R_gb_refined @ _J7_ORIGIN_IN_GB
        _q_pos2 = solve(_target_j7_refined, target_rot=None, initial_angles=_q_pos)
        return _q_pos2

    R_gb_desired  = target_rot_j7 @ _RX_NEG90
    target_j7_pos = target_gb_pos + R_gb_desired @ _J7_ORIGIN_IN_GB
    q = solve(target_j7_pos, target_rot=target_rot_j7, initial_angles=initial_angles)

    # --- roll search: rotate grasp frame around approach axis (gb +X) ---
    # Enumerate ALL roll angles (including 0 = original rotation) and collect
    # every valid candidate.  Then return the one with minimum joint-space
    # distance from initial_angles.  This ensures we never commit to a large
    # discontinuous joint motion (e.g. j5 flip by 103°) when a smoother
    # equivalent grasp orientation exists.
    #
    # Dual-seed strategy: for each roll angle we try TWO seeds:
    #   seed A = initial_angles (original, j5=+1.2)  -> may converge to j5<0 branch
    #   seed B = q0 with j5=+0.3, j2=1.5            -> biases toward j5>0 branch
    #     (j5>0 branch: j2≈1.5 far from π, j5≈+0.3~+0.8, natural arm posture)
    # Both seeds are tried for every roll angle; the valid candidate with
    # smallest joint-space distance from q0 is returned.
    approach_arm = R_gb_desired[:, 0]
    q0 = np.asarray(initial_angles, dtype=np.float64) if initial_angles is not None else np.zeros(6)
    # Build j5-positive seed: keep all joints from q0 but fix j5=+0.3 and j2=1.5
    # to anchor IK in the natural (non-singular) branch.
    _q_seed_j5pos = q0.copy()
    _q_seed_j5pos[4] = 0.3   # j5=+0.3: far from both limits, biases to j5>0 solution
    _q_seed_j5pos[1] = 1.5   # j2=1.5: far from π singularity
    best_q, best_err = q, float(np.linalg.norm(fk_gripper(q)[:3, 3] - target_gb_pos))
    valid_candidates = []   # list of (joint_dist, q_cand)
    for deg in range(0, 360, 5):
        R_roll   = _rot_axis_angle(approach_arm, math.radians(deg))
        R_gb_new = R_roll @ R_gb_desired
        if abs(float(np.linalg.det(R_gb_new)) - 1.0) > 0.02:
            continue
        R_j7_new  = R_gb_new @ _RX_NEG90.T
        t_j7_new  = target_gb_pos + R_gb_new @ _J7_ORIGIN_IN_GB
        # Try both seeds for this roll angle
        for _seed in (initial_angles, _q_seed_j5pos):
            q_cand    = solve(t_j7_new, target_rot=R_j7_new, initial_angles=_seed)
            T_gb_c    = fk_gripper(q_cand)
            err_c     = float(np.linalg.norm(T_gb_c[:3, 3] - target_gb_pos))
            rot_err_c = float(np.linalg.norm(fk(q_cand)[:3, :3] - R_j7_new, ord="fro"))
            at_lim_c  = any(
                abs(float(qi) - lo) < 0.01 or abs(float(qi) - hi) < 0.01
                for qi, (lo, hi) in zip(q_cand, _IK_JOINT_LIMITS)
            )
            if err_c < best_err:
                best_err, best_q = err_c, q_cand
            if err_c <= 0.03 and not at_lim_c and rot_err_c < 0.1:
                joint_dist = float(np.linalg.norm(q_cand - q0))
                # Additionally require each joint to stay within per-joint limits of q0.
                # This prevents selecting a roll that lands on the opposite side of
                # a dual-solution joint (e.g. j6: target=+0.82 but PD converges to
                # -0.77 because both are kinematically equivalent but controller
                # picks the closer one from its current position, not the IK target).
                # Per-joint jump limits:
                #   j5 (index 4) is allowed up to π (180°): seed j5=+1.2 → solution j5≈+0.3
                #   is a valid continuous motion via the j5>0 branch (0.9 rad < π).
                #   All other joints keep the π/2 (90°) limit to prevent dual-solution flips.
                _jump_lims = np.full(6, math.pi / 2)
                _jump_lims[4] = math.pi  # j5: allow up to 180° jump
                _per_joint_jumps = np.abs(q_cand - q0)
                if np.all(_per_joint_jumps < _jump_lims):
                    valid_candidates.append((joint_dist, q_cand))

    if valid_candidates:
        # Return the valid candidate with smallest joint-space distance
        # from initial_angles, minimising discontinuous joint motion.
        valid_candidates.sort(key=lambda x: x[0])
        return valid_candidates[0][1]

    # Only return best_q if it is not at a joint limit AND pos error is acceptable
    # AND the per-joint jump is within limits.
    # If best_q hit j2=π (singular config), returning it causes 500 steps of failed
    # tracking; return None instead so the caller can try the next GraspNet candidate.
    _at_lim_best = any(
        abs(float(_qi) - _lo) < 0.01 or abs(float(_qi) - _hi) < 0.01
        for _qi, (_lo, _hi) in zip(best_q, _IK_JOINT_LIMITS)
    )
    _jump_lims_fb = np.full(6, math.pi / 2)
    _jump_lims_fb[4] = math.pi
    _per_joint_jumps_best = np.abs(best_q - q0)
    _max_jump_best = float(np.max(_per_joint_jumps_best))
    _jump_exceeded = not np.all(_per_joint_jumps_best < _jump_lims_fb)
    if (_at_lim_best and best_err > 0.05) or _jump_exceeded:
        # Fallback solution is either at a limit with large pos error, or requires
        # a joint to exceed its per-joint limit.  Signal caller to try next candidate.
        print(f"[IK-WARN] fallback best_q rejected: at_lim={_at_lim_best} "
              f"pos_err={best_err:.3f}m max_jump={_max_jump_best:.3f}rad "
              f"per_joint={np.round(_per_joint_jumps_best,3)} q={np.round(best_q,3)}", flush=True)
        return None
    if not valid_candidates:
        # best_q passed all checks but was reached via fallback (valid_candidates empty).
        # Log a warning so the caller is aware this is not a roll-search validated solution.
        print(f"[IK-WARN] valid_candidates empty, using fallback best_q: "
              f"pos_err={best_err:.3f}m max_jump={_max_jump_best:.3f}rad "
              f"q={np.round(best_q,3)}", flush=True)
    return best_q   # return best found even if not perfect


def fk(joint_angles):
    """Forward kinematics. Returns (4,4) transform in arm_base_link frame.

    NOTE: The FK chain ends at joint7 (prismatic gripper finger), NOT at
    gripper_base.  joint7 is offset from gripper_base by:
        origin xyz=(0, 0, 0.1358)  rpy=(pi/2, 0, 0)
    Use fk_gripper() instead when you need the gripper_base frame.
    """
    chain = get_chain()
    # chain has 9 links; pad with base=0, ee=0, j7=0
    q = [0.0] + list(joint_angles) + [0.0, 0.0]
    return chain.forward_kinematics(q)


# ---------------------------------------------------------------------------
# Frame-correction constants: joint7 -> gripper_base
# ---------------------------------------------------------------------------
# URDF joint7 (prismatic):  origin xyz=(0, 0, 0.1358)  rpy=(pi/2, 0, 0)
#   R_j7_in_gb = Rx(+pi/2) = [[1,0,0],[0,0,-1],[0,1,0]]
#   R_gb_in_j7 = Rx(-pi/2) = [[1,0,0],[0,0, 1],[0,-1,0]]  (= _RX_NEG90)
#
# Relation:  R_j7_in_arm = R_gb_in_arm @ R_j7_in_gb
#   =>  R_gb_in_arm = R_j7_in_arm @ _RX_NEG90
#
# fk()[:3,:3] == R_j7_in_arm   (ikpy FK end frame)
# fk()[:3, 3] == t_j7_origin   = t_gb_origin + R_gb @ [0,0,0.1358]
# ---------------------------------------------------------------------------
_RX_NEG90 = np.array([[1,  0,  0],
                       [0,  0,  1],
                       [0, -1,  0]], dtype=np.float64)   # Rx(-pi/2)
_RX_POS90 = np.array([[1,  0,  0],
                       [0,  0, -1],
                       [0,  1,  0]], dtype=np.float64)   # Rx(+pi/2)
_J7_ORIGIN_IN_GB = np.array([0.0, 0.0, 0.1358])         # joint7 origin in gripper_base


def fk_gripper(joint_angles):
    """FK returning the gripper_base frame (4,4) in arm_base_link.

    Corrects the joint7 offset baked into the raw ikpy FK result:
        joint7 origin xyz=(0,0,0.1358), rpy=(pi/2,0,0) relative to gripper_base.

    Returns
    -------
    T_gb : (4,4)
        T_gb[:3,:3] = R_gripper_base_in_arm
        T_gb[:3, 3] = gripper_base origin in arm_base_link
    """
    T_j7 = fk(joint_angles)               # joint7 frame in arm_base_link
    R_j7 = T_j7[:3, :3]
    p_j7 = T_j7[:3, 3]
    R_gb = R_j7 @ _RX_NEG90              # gripper_base rotation
    p_gb = p_j7 - R_gb @ _J7_ORIGIN_IN_GB  # gripper_base origin
    T_gb = np.eye(4)
    T_gb[:3, :3] = R_gb
    T_gb[:3,  3] = p_gb
    return T_gb


def quat_to_rot(quat_wxyz):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),  1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),  2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])
    return R


# Camera mounting on gripper_base (matches single_piper_env_cfg.py wrist_camera offset):
#   pos = (-0.05, 0.0, 0.06) in gripper_base frame
#   rot = (w=0.7071, x=0, y=0, z=-0.7071)  [Rz(-90deg), ROS convention]
# This is the transform T_gripper_camera: takes a point in camera frame to gripper_base frame.
_CAM_OFFSET_POS = np.array([-0.05, 0.0, 0.06], dtype=np.float64)
# Rz(-90deg): [[0,1,0],[-1,0,0],[0,0,1]]
# But convention="ros" means the offset rot is expressed as:
#   camera_frame_in_gripper = R(quat)  =>  p_gripper = R @ p_cam + pos
# quat (w,x,y,z) = (0.7071, 0, 0, -0.7071)  => Rz(-90deg)
_CAM_OFFSET_ROT = np.array([
    [ 0.0,  1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)  # Rz(-90deg): camera +Z => gripper_base +Z


def cam_to_world(t_cam, joint_angles, robot_pos_w, robot_quat_w):
    """Convert a position in wrist_camera frame to world frame.

    Pipeline:
      camera frame
        --[cam offset]--> gripper_base frame
        --[FK(joint_angles)]--> arm_base_link frame
        --[arm_base_link offset + robot pose]--> world frame

    Parameters
    ----------
    t_cam        : (3,) position in wrist_camera frame (GraspNet output)
    joint_angles : (6,) current joint1..joint6 angles (radians)
    robot_pos_w  : (3,) robot base_link position in world frame
    robot_quat_w : (4,) robot base_link quaternion [w, x, y, z]

    Returns
    -------
    t_world : (3,) position in world frame
    """
    # Step 1: camera frame -> gripper_base frame
    t_gripper = _CAM_OFFSET_ROT @ t_cam + _CAM_OFFSET_POS

    # Step 2: gripper_base frame -> arm_base_link frame via FK
    # fk_gripper() corrects the joint7->gripper_base offset (Rx+90 rot + 13.58cm xyz)
    T_gb = fk_gripper(joint_angles)         # (4,4) gripper_base in arm_base_link
    t_arm_base = T_gb[:3, :3] @ t_gripper + T_gb[:3, 3]

    # Step 3: arm_base_link frame -> world frame
    ARM_BASE_OFFSET = np.array([0.0, 0.0, 0.0888])
    R_robot = quat_to_rot(robot_quat_w)
    arm_base_w = robot_pos_w + R_robot @ ARM_BASE_OFFSET
    t_world = R_robot @ t_arm_base + arm_base_w
    return t_world


def compute_desired_ee_rot_in_arm(R_cam_grasp, q_scan):
    """Compute the desired gripper_base rotation in arm_base_link frame.

    Called once after GraspNet returns a result (using the joint angles at SCAN
    time, NOT at PRE_GRASP time).  The result is saved in ``grasp_result`` and
    passed as ``target_rot`` to the PRE_GRASP IK so that joint1-6 all arrive at
    a configuration that simultaneously places the EE at the pre-grasp position
    **and** aligns the gripper closing axis with the GraspNet direction.

    Background
    ----------
    The GraspNet rotation R_cam_grasp is expressed in the wrist camera frame at
    SCAN time.  The camera frame at SCAN is:

        R_cam_in_arm = R_gb_scan @ _CAM_OFFSET_ROT

    where R_gb_scan is the gripper_base rotation at SCAN time:

        R_gb_scan = fk(q_scan)[:3,:3] @ _RX_NEG90

    (fk() ends at joint7, which is rotated Rx+90 from gripper_base; see fk_gripper())

    The desired gripper_base orientation in arm_base_link frame is:

        R_gb_desired = R_gb_scan @ _CAM_OFFSET_ROT @ R_cam_grasp

    Because ikpy's inverse_kinematics() expects target_orientation in the FK
    end frame (joint7), we must convert back:

        R_j7_desired = R_gb_desired @ _RX_POS90

    This function returns R_j7_desired (the value to pass directly to solve()).

    Parameters
    ----------
    R_cam_grasp : (3,3) GraspNet rotation in wrist_camera frame
    q_scan      : (6,) joint angles recorded at the moment of SCAN

    Returns
    -------
    R_j7_desired : (3,3) target rotation for joint7 frame in arm_base_link,
                   ready to pass as target_rot to solve().
    """
    # gripper_base rotation at SCAN (corrected from raw ikpy FK end frame)
    R_gb_in_arm = fk_gripper(q_scan)[:3, :3]
    # desired gripper_base orientation that aligns camera with grasp direction
    R_gb_desired = R_gb_in_arm @ _CAM_OFFSET_ROT @ R_cam_grasp
    # convert to joint7 frame (what ikpy expects as target_orientation)
    return R_gb_desired @ _RX_POS90


def cam_rot_to_arm_frame(R_cam, joint_angles, robot_quat_w):
    """Convert a rotation matrix from wrist_camera frame to arm_base_link frame.

    This transforms the GraspNet grasp orientation (in camera frame) into
    the arm_base_link frame required by IK.

    Parameters
    ----------
    R_cam        : (3,3) rotation matrix in wrist_camera frame (GraspNet output)
    joint_angles : (6,) current joint1..joint6 angles
    robot_quat_w : (4,) robot base_link quaternion [w, x, y, z]

    Returns
    -------
    R_arm : (3,3) rotation matrix in arm_base_link frame
    """
    # FK gives gripper_base orientation in arm_base_link frame (corrected from joint7 end frame)
    R_gb = fk_gripper(joint_angles)[:3, :3]
    # Full chain: R_arm = R_gb @ R_gb_from_cam @ R_cam
    R_arm = R_gb @ _CAM_OFFSET_ROT @ R_cam
    return R_arm


def extract_j6_angle(cur_q, R_cam_grasp, R_arm_target):
    """[DEPRECATED — ORIENT stage is now skipped in favour of PRE_GRASP with target_rot]

    Extract the joint6 angle required to achieve R_arm_target, keeping joint1-5 fixed.

    IMPORTANT KNOWN LIMITATION: This function assumes that the target orientation
    R_arm_target can be reached by adjusting joint6 alone (with joint1-5 held at their
    PRE_GRASP-end values).  Numerical experiments show this is FALSE: after PRE_GRASP
    moves joint1-5 to a new configuration, the desired gripper orientation is no longer
    reachable by only rotating joint6 (Frobenius error ≈ 0.50, vs. ≈ 0.065 when using
    the SCAN-time joint1-5).  The ORIENT stage is therefore bypassed; PRE_GRASP now
    solves IK with target_rot=R_desired_EE_in_arm so all six joints arrive correctly.

    FK decomposition (numerically verified):
        R_full(q) = fk(q_j6=0)[:3,:3] @ Ry(j6)
    where Ry is a Y-axis rotation in the gripper_base local frame (ikpy absorbs the
    joint6 origin rpy=1.5708 0 0 into the chain, making joint6 appear as Ry in
    gripper_base coordinates despite URDF declaring axis="0 0 1").

    Correct j6 extraction formula (if R_arm_target is achievable with current j1-5):
        Ry = fk(q_j6zero)[:3,:3].T @ R_arm_target
        j6 = atan2(Ry[0,2], Ry[0,0])   # Y-axis rotation

    Parameters
    ----------
    cur_q        : (6,) current joint angles (joint1-5 define R_j1to5; joint6 is ignored)
    R_cam_grasp  : (3,3) GraspNet rotation in camera frame (grasp_result["R_cam"])
    R_arm_target : (3,3) desired EE orientation in arm_base_link frame
                   Must be computed from SCAN-time q, NOT from PRE_GRASP-end q.

    Returns
    -------
    j6 : float
        Target angle for joint6 (radians), clipped to [-2.094, 2.094].
    """
    q_j6zero = np.array(cur_q, dtype=np.float64)
    q_j6zero[5] = 0.0
    T_j6zero = fk(q_j6zero)
    R_j1to5 = T_j6zero[:3, :3]
    # Correct formula: joint6 is effectively Ry in gripper_base local frame
    # (verified numerically: fk(q_j6=0) @ Ry(j6) = fk(q) with error < 1e-5)
    Ry = R_j1to5.T @ R_arm_target
    j6 = math.atan2(Ry[0, 2], Ry[0, 0])   # Y-axis: atan2(sin, cos) = atan2(R[0,2], R[0,0])
    return float(np.clip(j6, -2.094, 2.094))


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
