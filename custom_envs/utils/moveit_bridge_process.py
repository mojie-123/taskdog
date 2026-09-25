#!/usr/bin/env python3.10
"""ROS 2 Humble subprocess used by moveit_bridge.py.

The process exposes a tiny JSON RPC surface around MoveIt move_group services:
- /compute_ik
- /plan_kinematic_path
- /compute_cartesian_path
- /apply_planning_scene
and continuously republishes the latest MuJoCo joint state on /joint_states.

All logs go to stderr. stdout is reserved for one-line JSON replies.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback

# Allow this process to be spawned from a conda parent without losing ROS paths.
for _p in (
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def main():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.duration import Duration

        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import Pose, PoseStamped
        from shape_msgs.msg import SolidPrimitive
        from moveit_msgs.msg import (
            BoundingVolume,
            CollisionObject,
            Constraints,
            MoveItErrorCodes,
            OrientationConstraint,
            PlanningScene,
            PositionConstraint,
            RobotState,
        )
        from moveit_msgs.srv import (
            ApplyPlanningScene,
            GetCartesianPath,
            GetMotionPlan,
            GetPositionIK,
        )
    except Exception as exc:
        log(f"[MoveItProc] FATAL import error: {exc}")
        traceback.print_exc(file=sys.stderr)
        emit({"init_failed": True, "error": str(exc)})
        return

    rclpy.init()
    node = Node("mujoco_moveit_bridge")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    joint_pub = node.create_publisher(JointState, "/joint_states", 20)
    ik_cli = node.create_client(GetPositionIK, "/compute_ik")
    plan_cli = node.create_client(GetMotionPlan, "/plan_kinematic_path")
    cart_cli = node.create_client(GetCartesianPath, "/compute_cartesian_path")
    scene_cli = node.create_client(ApplyPlanningScene, "/apply_planning_scene")

    clients = {
        "compute_ik": ik_cli,
        "plan_kinematic_path": plan_cli,
        "compute_cartesian_path": cart_cli,
        "apply_planning_scene": scene_cli,
    }
    log("[MoveItProc] waiting for move_group services...")
    deadline = time.monotonic() + 35.0
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        missing = [name for name, cli in clients.items() if not cli.service_is_ready()]
        if not missing:
            break
        time.sleep(0.05)
    else:
        missing = [name for name, cli in clients.items() if not cli.service_is_ready()]
        emit({"init_failed": True, "error": f"MoveIt services unavailable: {missing}"})
        log(f"[MoveItProc] services unavailable: {missing}")
        rclpy.shutdown()
        return

    incoming = queue.Queue()
    stop_flag = threading.Event()
    latest_joint = {"names": [], "positions": []}
    latest_lock = threading.Lock()

    def stdin_reader():
        for line in sys.stdin:
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            if msg.get("type") == "stop":
                stop_flag.set()
                return
            if msg.get("type") == "joint_state":
                with latest_lock:
                    latest_joint["names"] = list(msg.get("names", []))
                    latest_joint["positions"] = list(msg.get("positions", []))
            else:
                incoming.put(msg)

    threading.Thread(target=stdin_reader, daemon=True).start()

    def publish_joint_state():
        with latest_lock:
            names = list(latest_joint["names"])
            pos = list(latest_joint["positions"])
        if not names or len(names) != len(pos):
            return
        js = JointState()
        js.header.stamp = node.get_clock().now().to_msg()
        js.name = names
        js.position = [float(x) for x in pos]
        js.velocity = [0.0] * len(names)
        joint_pub.publish(js)

    def pose_msg(data):
        p = Pose()
        xyz = data["position"]
        q = data["quaternion_xyzw"]
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
        return p

    def robot_state(names, positions):
        rs = RobotState()
        rs.joint_state.header.stamp = node.get_clock().now().to_msg()
        rs.joint_state.name = list(names)
        rs.joint_state.position = [float(x) for x in positions]
        rs.is_diff = False
        return rs

    def wait_future(fut, timeout):
        start = time.monotonic()
        while rclpy.ok() and not fut.done():
            publish_joint_state()
            executor.spin_once(timeout_sec=0.01)
            if time.monotonic() - start > timeout:
                return None
        return fut.result() if fut.done() else None

    def trajectory_to_dict(traj):
        jt = traj.joint_trajectory
        out = {"joint_names": list(jt.joint_names), "points": []}
        for p in jt.points:
            out["points"].append({
                "positions": [float(x) for x in p.positions],
                "velocities": [float(x) for x in p.velocities],
                "accelerations": [float(x) for x in p.accelerations],
                "time_from_start": [int(p.time_from_start.sec), int(p.time_from_start.nanosec)],
            })
        return out

    def moveit_success(code):
        return int(code.val) == int(MoveItErrorCodes.SUCCESS)

    def build_pose_constraints(msg):
        frame = str(msg.get("frame_id", "base_link"))
        link = str(msg.get("link_name", "grasp_tcp"))
        target = pose_msg(msg["pose"])
        pos_tol = float(msg.get("position_tolerance", 0.008))
        rot_tol = float(msg.get("orientation_tolerance", 0.10))

        pc = PositionConstraint()
        pc.header.frame_id = frame
        pc.link_name = link
        pc.weight = 1.0
        bv = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [max(pos_tol, 1e-4)]
        center = Pose()
        center.position = target.position
        center.orientation.w = 1.0
        bv.primitives = [primitive]
        bv.primitive_poses = [center]
        pc.constraint_region = bv

        oc = OrientationConstraint()
        oc.header.frame_id = frame
        oc.link_name = link
        oc.orientation = target.orientation
        oc.absolute_x_axis_tolerance = rot_tol
        oc.absolute_y_axis_tolerance = rot_tol
        oc.absolute_z_axis_tolerance = rot_tol
        oc.weight = 1.0
        # ROTATION_VECTOR avoids Euler-axis singularity when available.
        if hasattr(OrientationConstraint, "ROTATION_VECTOR") and hasattr(oc, "parameterization"):
            oc.parameterization = OrientationConstraint.ROTATION_VECTOR

        c = Constraints()
        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        return c

    def add_box(obj, xyz, size_xyz):
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(v) for v in size_xyz]
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.w = 1.0
        obj.primitives.append(prim)
        obj.primitive_poses.append(p)

    def default_scene_objects():
        objects = []

        table1 = CollisionObject()
        table1.header.frame_id = "map"
        table1.id = "mujoco_table1"
        add_box(table1, [5.0, 5.0, 0.6213], [0.6468, 2.7254, 0.08])
        for dx in (0.2834, -0.2834):
            for dy in (1.3227, -1.3227):
                add_box(table1, [5.0 + dx, 5.0 + dy, 0.2907], [0.08, 0.08, 0.5814])
        table1.operation = CollisionObject.ADD
        objects.append(table1)

        table2 = CollisionObject()
        table2.header.frame_id = "map"
        table2.id = "mujoco_table2"
        add_box(table2, [8.0, 6.0, 0.6213], [0.6468, 0.9084, 0.08])
        for dx in (0.2834, -0.2834):
            for dy in (0.4142, -0.4142):
                add_box(table2, [8.0 + dx, 6.0 + dy, 0.2907], [0.08, 0.08, 0.5814])
        table2.operation = CollisionObject.ADD
        objects.append(table2)

        floor = CollisionObject()
        floor.header.frame_id = "map"
        floor.id = "mujoco_floor"
        add_box(floor, [0.0, 0.0, -0.05], [40.0, 40.0, 0.10])
        floor.operation = CollisionObject.ADD
        objects.append(floor)
        return objects

    def handle(msg):
        rid = msg.get("id", "")
        kind = msg.get("type", "")
        try:
            if kind == "apply_default_scene":
                req = ApplyPlanningScene.Request()
                ps = PlanningScene()
                ps.is_diff = True
                ps.world.collision_objects = default_scene_objects()
                req.scene = ps
                res = wait_future(scene_cli.call_async(req), 5.0)
                ok = bool(res and res.success)
                return {"id": rid, "ok": ok, "type": kind,
                        "error": "" if ok else "apply_planning_scene failed"}

            if kind == "compute_ik":
                req = GetPositionIK.Request()
                ik = req.ik_request
                ik.group_name = str(msg.get("group_name", "piper_arm"))
                ik.ik_link_name = str(msg.get("link_name", "grasp_tcp"))
                ik.pose_stamped = PoseStamped()
                ik.pose_stamped.header.frame_id = str(msg.get("frame_id", "base_link"))
                ik.pose_stamped.header.stamp = node.get_clock().now().to_msg()
                ik.pose_stamped.pose = pose_msg(msg["pose"])
                ik.robot_state = robot_state(msg["start_joint_names"], msg["start_joint_positions"])
                ik.avoid_collisions = bool(msg.get("avoid_collisions", True))
                t = float(msg.get("ik_timeout_sec", 0.15))
                ik.timeout = Duration(seconds=t).to_msg()
                res = wait_future(ik_cli.call_async(req), max(2.0, t + 1.0))
                if res is None:
                    return {"id": rid, "ok": False, "type": kind, "error": "IK timeout"}
                ok = moveit_success(res.error_code)
                sol = res.solution.joint_state
                return {
                    "id": rid, "ok": ok, "type": kind,
                    "error_code": int(res.error_code.val),
                    "solution_names": list(sol.name),
                    "solution_positions": [float(x) for x in sol.position],
                }

            if kind == "plan_to_pose":
                req = GetMotionPlan.Request()
                mpr = req.motion_plan_request
                mpr.group_name = str(msg.get("group_name", "piper_arm"))
                mpr.start_state = robot_state(msg["start_joint_names"], msg["start_joint_positions"])
                mpr.goal_constraints = [build_pose_constraints(msg)]
                mpr.num_planning_attempts = int(msg.get("planning_attempts", 3))
                mpr.allowed_planning_time = float(msg.get("allowed_planning_time", 2.5))
                mpr.planner_id = str(msg.get("planner_id", "RRTConnectkConfigDefault"))
                if hasattr(mpr, "pipeline_id"):
                    mpr.pipeline_id = "ompl"
                if hasattr(mpr, "max_velocity_scaling_factor"):
                    mpr.max_velocity_scaling_factor = float(msg.get("velocity_scaling", 0.25))
                if hasattr(mpr, "max_acceleration_scaling_factor"):
                    mpr.max_acceleration_scaling_factor = float(msg.get("acceleration_scaling", 0.25))

                timeout = max(4.0, float(mpr.allowed_planning_time) + 2.0)
                res = wait_future(plan_cli.call_async(req), timeout)
                if res is None:
                    return {"id": rid, "ok": False, "type": kind, "error": "planning timeout"}
                ans = res.motion_plan_response
                ok = moveit_success(ans.error_code)
                return {
                    "id": rid, "ok": ok, "type": kind,
                    "error_code": int(ans.error_code.val),
                    "planning_time": float(ans.planning_time),
                    "trajectory": trajectory_to_dict(ans.trajectory) if ok else {},
                }

            if kind == "cartesian_path":
                req = GetCartesianPath.Request()
                req.header.frame_id = str(msg.get("frame_id", "base_link"))
                req.header.stamp = node.get_clock().now().to_msg()
                req.start_state = robot_state(msg["start_joint_names"], msg["start_joint_positions"])
                req.group_name = str(msg.get("group_name", "piper_arm"))
                req.link_name = str(msg.get("link_name", "grasp_tcp"))
                req.waypoints = [pose_msg(p) for p in msg.get("poses", [])]
                req.max_step = float(msg.get("max_step", 0.005))
                req.jump_threshold = float(msg.get("jump_threshold", 0.0))
                if hasattr(req, "prismatic_jump_threshold"):
                    req.prismatic_jump_threshold = 0.0
                if hasattr(req, "revolute_jump_threshold"):
                    req.revolute_jump_threshold = 0.0
                req.avoid_collisions = bool(msg.get("avoid_collisions", True))
                if hasattr(req, "max_velocity_scaling_factor"):
                    req.max_velocity_scaling_factor = float(msg.get("velocity_scaling", 0.20))
                if hasattr(req, "max_acceleration_scaling_factor"):
                    req.max_acceleration_scaling_factor = float(msg.get("acceleration_scaling", 0.20))

                res = wait_future(cart_cli.call_async(req), 6.0)
                if res is None:
                    return {"id": rid, "ok": False, "type": kind, "error": "cartesian timeout"}
                ok = moveit_success(res.error_code) and float(res.fraction) > 0.0
                return {
                    "id": rid, "ok": ok, "type": kind,
                    "error_code": int(res.error_code.val),
                    "fraction": float(res.fraction),
                    "trajectory": trajectory_to_dict(res.solution) if ok else {},
                }

            return {"id": rid, "ok": False, "type": kind, "error": f"unknown request: {kind}"}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return {"id": rid, "ok": False, "type": kind, "error": repr(exc)}

    emit({"ready": True})
    log("[MoveItProc] ready")
    last_joint_pub = 0.0

    while rclpy.ok() and not stop_flag.is_set():
        now = time.monotonic()
        if now - last_joint_pub >= 0.02:  # 50 Hz
            publish_joint_state()
            last_joint_pub = now

        try:
            msg = incoming.get_nowait()
        except queue.Empty:
            executor.spin_once(timeout_sec=0.002)
            time.sleep(0.001)
            continue

        reply = handle(msg)
        emit(reply)

    try:
        executor.remove_node(node)
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
