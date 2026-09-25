# MuJoCo + AnyGrasp + MoveIt integration (new-files-only)

This bundle is intentionally **additive**: it does not replace or edit the existing
`navigate_mujoco.py`, `ros2_bridge.py`, `ros2_bridge_process.py`, `arm_ik_mujoco.py`,
`nav2_isaac.launch.py`, or the existing URDF/MJCF files.

## New files

```text
custom_envs/
├── assets/m20_piper_single/
│   └── M20_Piper_moveit.urdf
├── config/moveit/
│   ├── M20_Piper_moveit.srdf
│   ├── joint_limits.yaml
│   ├── kinematics.yaml
│   └── ompl_planning.yaml
├── launch/
│   ├── moveit_mujoco.launch.py
│   └── nav2_moveit_mujoco.launch.py
├── scripts/navigation/
│   ├── moveit_smoke_test.py
│   ├── mujoco_local_servo.py
│   ├── mujoco_moveit_frames.py
│   ├── navigate_mujoco_moveit.py
│   └── verify_moveit_frames.py
└── utils/
    ├── moveit_bridge.py
    └── moveit_bridge_process.py
```

## Architecture

```text
MuJoCo /joint state
       │
       ├──────────────> MoveIt 2 (planning only)
       │                   │
       │                   ├─ IK + collision check
       │                   ├─ OMPL pre-grasp path
       │                   └─ Cartesian approach path
       │                           │
       └<──── joint path ──────────┘
                 │
        env.set_arm_target()
                 │
        existing MuJoCo PD
```

MoveIt does **not** own the actuator/controller in this first version. This keeps the
existing MuJoCo PD and gripper/contact behavior unchanged.

Grasp pipeline:

```text
SCAN
 -> AnyGrasp candidates
 -> camera optical -> grasp_tcp/base_link conversion
 -> MoveIt candidate IK + collision + full path check
 -> MoveIt pre-grasp plan
 -> MoveIt Cartesian approach (stop short)
 -> short MuJoCo Jacobian LOCAL_REACH
 -> existing CLOSE
 -> existing LIFT / navigation / placement
```

## Coordinate-frame contract

AnyGrasp input and output are in the depth-camera optical frame:

```text
+x = image right
+y = image down
+z = camera forward
```

The fixed optical transform used by the code is:

```text
camera optical +X -> gripper_base -Y
camera optical +Y -> gripper_base +X
camera optical +Z -> gripper_base +Z
```

`joint7` is **not** part of the high-level transform chain. A new fixed `grasp_tcp`
is placed at `gripper_base +Z * 0.1358 m` with the **same orientation** as
`gripper_base`.

The main script snapshots the actual MuJoCo `base_link -> gripper_base` transform at
SCAN time. It also verifies the fixed camera transform against MuJoCo's live
`cam_xpos/cam_xmat` before navigation starts. A mismatch is treated as a fatal error.

For each AnyGrasp candidate:

```text
contact = palm + (depth + insertion) * approach
pre     = contact - pregrasp_distance * approach
near    = contact - local_reach_distance * approach
```

Defaults preserve the current project's effective geometry:

```text
insertion            = 0.005 m
pregrasp_distance    = 0.115 m
local_reach_distance = 0.015 m
```

## MoveIt requirement

The ROS 2 Humble environment used by terminal 1/ROS subprocesses must contain MoveIt 2
with `move_group`, OMPL, and the KDL kinematics plugin. In a normal Humble binary
installation this is typically provided by the MoveIt packages. Make sure the same
terminal environment sources ROS 2 (and any MoveIt workspace if MoveIt was built from
source).

A quick check is:

```bash
source /opt/ros/humble/setup.bash
ros2 pkg prefix moveit_ros_move_group
ros2 pkg prefix moveit_kinematics
```

## Recommended first run

### 1. Copy/extract these files at the taskdog repository root

The paths in this bundle already mirror the project tree.

### 2. Coordinate-only check

This does not move the robot:

```bash
python custom_envs/scripts/navigation/verify_moveit_frames.py
```

All four checks should print `[PASS]`.

### 3. Start Nav2 + MoveIt

```bash
source /opt/ros/humble/setup.bash

ros2 launch custom_envs/launch/nav2_moveit_mujoco.launch.py \
    params_file:=$(pwd)/custom_envs/config/nav2_params.yaml \
    map:=$(pwd)/custom_envs/maps/map_whole_nav2.yaml
```

You should see `move_group` start in addition to the existing Nav2 nodes.

### 4. Optional MoveIt smoke test

In another terminal, with the project Python environment active:

```bash
python custom_envs/scripts/navigation/moveit_smoke_test.py
```

This does not command the arm. It asks MoveIt to solve IK and a plan for MuJoCo's
current `grasp_tcp` pose.

### 5. Run the new full pipeline

```bash
python custom_envs/scripts/navigation/navigate_mujoco_moveit.py \
    --policy_path deps/rl_training/logs/rsl_rl/deeprobotics_m20_flat/2026-07-18_10-57-32/model_4999.pt \
    --map custom_envs/maps/map_whole.npz \
    --goal 4.0 6.0 \
    --grasp_checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
    --object cube \
    --destination 7.5 5.8 \
    --render
```

## What to look for in the first full run

Expected key logs:

```text
[FRAME] wrist_cam optical transform verified against MuJoCo
[FRAME] SCAN camera optical in base: ...
[SM] AnyGrasp candidates for MoveIt: ...
[MOVEIT] candidate[...] ...
[MOVEIT] SELECT candidate[...] ...
[MOVEIT] pre-grasp reached -> MOVEIT_APPROACH
[MOVEIT] near-grasp reached -> LOCAL_REACH
[LOCAL_REACH] ... -> CLOSE
```

If MoveIt rejects every candidate, do **not** immediately loosen all tolerances. First
inspect the first candidate's printed `approach_base`, `approach_world`, IK error code,
and planning error code; this distinguishes frame errors, reachability, self-collision,
and table collision.

## First-version limitations

- MoveIt collision world currently models table1, table2, and the floor. It does not yet
  mirror every free object in `scene.xml`.
- The MoveIt URDF models the quadruped `base_link` collision envelope and the Piper arm,
  but not every articulated leg/wheel link. This is sufficient for the first fixed-base
  arm-planning integration, but can be expanded later.
- The target object is intentionally not inserted as a hard collision obstacle during
  the grasp approach; otherwise MoveIt would prevent the gripper from entering the
  grasp region.
- MoveIt trajectory time stamps are not used for execution in this first version.
  The returned joint-space geometry is densified and followed by the existing MuJoCo
  position-PD loop.
- Placement/ROTATE and most LIFT behavior are intentionally kept from the existing
  pipeline. The first integration boundary ends at `CLOSE`.
