#!/usr/bin/env python3.10
"""ROS 2 Bridge subprocess — launched by IsaacROS2Bridge via subprocess.Popen.

This script MUST be executed with /usr/bin/python3.10 so that rclpy
C-extensions (compiled for CPython 3.10) import correctly.

Protocol: newline-delimited JSON on stdin/stdout.
  stdin  (main -> this):
    {"type":"pose","pos":[x,y,z],"quat":[w,x,y,z],
     "lin_vel":[vx,vy,vz],"ang_vel":[wx,wy,wz]}
    {"type":"goal","x":float,"y":float}
    {"type":"stop"}
  stdout (this -> main):
    {"ready":true}                                     -- emitted once on init OK
    {"init_failed":true}                               -- emitted once on import error
    {"vx":float,"omega_z":float,
     "nav_done":bool,"nav_failed":bool}                -- emitted every publish cycle

All diagnostic prints go to stderr (visible in parent terminal).
"""
import json
import os
import sys
import time
import threading
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Ensure ROS 2 Humble Python packages are findable even when this process
# is spawned from a conda environment that may not have sourced setup.bash.
# ---------------------------------------------------------------------------
for _p in [
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _emit(obj: dict) -> None:
    """Write one JSON line to stdout (the parent process reads it)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _eprint(*args, **kw) -> None:
    """Print a diagnostic message to stderr."""
    print(*args, **kw, file=sys.stderr, flush=True)


def main() -> None:
    # -----------------------------------------------------------------------
    # Import ROS / Nav2 packages
    # -----------------------------------------------------------------------
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            QoSProfile, QoSReliabilityPolicy,
            QoSDurabilityPolicy, QoSHistoryPolicy)
        from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
        from nav_msgs.msg import Odometry
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
        from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    except ImportError as exc:
        _eprint(f"[BridgeProc] FATAL import: {exc}")
        traceback.print_exc(file=sys.stderr)
        _emit({"init_failed": True})
        return

    _eprint("[BridgeProc] imports OK, init rclpy...")
    rclpy.init()
    node = Node("isaac_ros2_bridge")
    tf_bcast     = TransformBroadcaster(node)
    static_bcast = StaticTransformBroadcaster(node)

    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST, depth=10)
    odom_pub = node.create_publisher(Odometry, "/odom", qos)

    # Static map->odom identity transform (ground-truth localisation)
    stf = TransformStamped()
    stf.header.stamp    = node.get_clock().now().to_msg()
    stf.header.frame_id = "map"
    stf.child_frame_id  = "odom"
    stf.transform.rotation.w = 1.0
    static_bcast.sendTransform(stf)

    # -----------------------------------------------------------------------
    # Shared state
    # -----------------------------------------------------------------------
    _lock = threading.Lock()
    _pos_w   = np.zeros(3,  dtype=np.float64)
    _quat_w  = np.array([1., 0., 0., 0.], dtype=np.float64)
    _lin_vel = np.zeros(3,  dtype=np.float64)
    _ang_vel = np.zeros(3,  dtype=np.float64)
    _vx          = 0.0
    _omega_z     = 0.0
    _last_cmd_t  = 0.0
    _nav_done    = False
    _nav_failed  = False
    _task_active = False
    _stop        = False
    _pending_goal: list = []
    CMD_TIMEOUT = 0.5

    def _cmd_cb(msg: Twist) -> None:
        nonlocal _vx, _omega_z, _last_cmd_t
        with _lock:
            _vx         = float(msg.linear.x)
            _omega_z    = float(msg.angular.z)
            _last_cmd_t = time.monotonic()
    node.create_subscription(Twist, "/cmd_vel_nav", _cmd_cb, 10)

    # -----------------------------------------------------------------------
    # BasicNavigator — do NOT call waitUntilNav2Active() here because TF
    # (odom->base_link) is only published after the main loop starts.
    # Instead, poll bt_navigator/get_state in a background thread AFTER
    # the main loop has already begun broadcasting TF.
    # -----------------------------------------------------------------------
    navigator = BasicNavigator()
    _nav2_ready = threading.Event()

    def _nav2_poller() -> None:
        """Poll bt_navigator/get_state until active, then emit ready."""
        import time as _time
        from lifecycle_msgs.srv import GetState
        cli = node.create_client(GetState, "bt_navigator/get_state")
        _eprint("[BridgeProc] waiting for bt_navigator/get_state service...")
        while not cli.wait_for_service(timeout_sec=1.0):
            _eprint("[BridgeProc] bt_navigator/get_state not available, retrying...")
        _eprint("[BridgeProc] bt_navigator/get_state available, polling for active...")
        while True:
            future = cli.call_async(GetState.Request())
            # spin until future done
            timeout = 5.0
            start = _time.monotonic()
            while not future.done():
                try:
                    executor.spin_once(timeout_sec=0.05)
                except Exception:
                    pass
                if _time.monotonic() - start > timeout:
                    break
            if future.done():
                state = future.result().current_state.label
                _eprint(f"[BridgeProc] bt_navigator state: {state}")
                if state == "active":
                    _eprint("[BridgeProc] Nav2 active. Sending ready signal.")
                    _emit({"ready": True})
                    _nav2_ready.set()
                    return
            _time.sleep(1.0)

    # -----------------------------------------------------------------------
    # stdin reader thread
    # -----------------------------------------------------------------------
    def _stdin_reader() -> None:
        nonlocal _stop
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
            except json.JSONDecodeError:
                _eprint(f"[BridgeProc] bad JSON: {raw_line!r}")
                continue
            t = msg.get("type", "pose")
            if t == "stop":
                with _lock:
                    _stop = True
                break
            elif t == "goal":
                gx, gy = float(msg["x"]), float(msg["y"])
                _eprint(f"[BridgeProc] goal queued ({gx:.2f},{gy:.2f})")
                with _lock:
                    _pending_goal.clear()
                    _pending_goal.append((gx, gy))
            elif t == "pose":
                with _lock:
                    _pos_w[:]   = msg["pos"]
                    _quat_w[:]  = msg["quat"]
                    _lin_vel[:] = msg["lin_vel"]
                    _ang_vel[:] = msg["ang_vel"]

    stdin_t = threading.Thread(target=_stdin_reader, daemon=True)
    stdin_t.start()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    TF_PERIOD   = 1.0 / 50.0
    ODOM_PERIOD = 1.0 / 50.0
    last_tf = last_odom = 0.0
    _loop_count = 0
    _last_debug_t = 0.0

    # Start Nav2 poller AFTER executor is created (poller uses executor.spin_once)
    # It runs in a daemon thread so it won't block the main TF publish loop.
    poller_t = threading.Thread(target=_nav2_poller, daemon=True)
    poller_t.start()

    while True:
        _loop_count += 1
        with _lock:
            should_stop = _stop
            goal_list   = list(_pending_goal)
            _pending_goal.clear()
        if should_stop:
            _eprint("[BridgeProc] stop received, shutting down.")
            break

        # Process any pending goal
        for gx, gy in goal_list:
            _eprint(f"[BridgeProc] goToPose ({gx:.2f},{gy:.2f})")
            ps = PoseStamped()
            ps.header.stamp    = node.get_clock().now().to_msg()
            ps.header.frame_id = "map"
            ps.pose.position.x = gx
            ps.pose.position.y = gy
            ps.pose.orientation.w = 1.0
            navigator.goToPose(ps)
            with _lock:
                _nav_done    = False
                _nav_failed  = False
                _task_active = True

        # Spin ROS once
        try:
            executor.spin_once(timeout_sec=0.002)
        except Exception:
            pass

        t_now = time.monotonic()
        with _lock:
            pos  = _pos_w.copy()
            quat = _quat_w.copy()
            lv   = _lin_vel.copy()
            av   = _ang_vel.copy()
        ros_now = node.get_clock().now().to_msg()

        # Publish TF at 50 Hz
        if t_now - last_tf >= TF_PERIOD:
            last_tf = t_now
            tf_msg = TransformStamped()
            tf_msg.header.stamp    = ros_now
            tf_msg.header.frame_id = "odom"
            tf_msg.child_frame_id  = "base_link"
            tf_msg.transform.translation.x = float(pos[0])
            tf_msg.transform.translation.y = float(pos[1])
            tf_msg.transform.translation.z = float(pos[2])
            tf_msg.transform.rotation.x = float(quat[1])
            tf_msg.transform.rotation.y = float(quat[2])
            tf_msg.transform.rotation.z = float(quat[3])
            tf_msg.transform.rotation.w = float(quat[0])
            tf_bcast.sendTransform(tf_msg)
            # Debug: print pose every 5 seconds
            if t_now - _last_debug_t >= 5.0:
                _last_debug_t = t_now
                with _lock:
                    _vx_dbg = _vx
                    _oz_dbg = _omega_z
                    _stale_dbg = (time.monotonic() - _last_cmd_t) > CMD_TIMEOUT
                _eprint(f"[BridgeProc] loop={_loop_count} "
                        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                        f"quat_w={quat[0]:.3f} "
                        f"cmd_vel vx={_vx_dbg:.3f} oz={_oz_dbg:.3f} "
                        f"stale={_stale_dbg}")

        # Publish Odometry at 50 Hz
        if t_now - last_odom >= ODOM_PERIOD:
            last_odom = t_now
            om = Odometry()
            om.header.stamp    = ros_now
            om.header.frame_id = "odom"
            om.child_frame_id  = "base_link"
            om.pose.pose.position.x = float(pos[0])
            om.pose.pose.position.y = float(pos[1])
            om.pose.pose.position.z = float(pos[2])
            om.pose.pose.orientation.x = float(quat[1])
            om.pose.pose.orientation.y = float(quat[2])
            om.pose.pose.orientation.z = float(quat[3])
            om.pose.pose.orientation.w = float(quat[0])
            w = float(quat[0]); x = float(quat[1])
            y = float(quat[2]); z = float(quat[3])
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
            ])
            bl = R.T @ lv
            ba = R.T @ av
            om.twist.twist.linear.x  = float(bl[0])
            om.twist.twist.linear.y  = float(bl[1])
            om.twist.twist.linear.z  = float(bl[2])
            om.twist.twist.angular.x = float(ba[0])
            om.twist.twist.angular.y = float(ba[1])
            om.twist.twist.angular.z = float(ba[2])
            odom_pub.publish(om)

        # Poll Nav2 task completion
        with _lock:
            task_active = _task_active
            nav_done    = _nav_done
            nav_failed  = _nav_failed
        if task_active and not nav_done and not nav_failed:
            try:
                if navigator.isTaskComplete():
                    result = navigator.getResult()
                    nd = (result == TaskResult.SUCCEEDED)
                    nf = not nd
                    _eprint(f"[BridgeProc] task done={nd} fail={nf}")
                    with _lock:
                        _nav_done   = nd
                        _nav_failed = nf
                        nav_done    = nd
                        nav_failed  = nf
            except Exception:
                pass

        # Emit cmd_vel + nav status to stdout
        with _lock:
            stale  = (time.monotonic() - _last_cmd_t) > CMD_TIMEOUT
            vx_out = 0.0 if stale else _vx
            oz_out = 0.0 if stale else _omega_z
            nd_out = _nav_done
            nf_out = _nav_failed
        _emit({"vx": vx_out, "omega_z": oz_out,
               "nav_done": nd_out, "nav_failed": nf_out})

        time.sleep(0.001)

    # Clean up
    try:
        navigator.lifecycleShutdown()
    except Exception:
        pass
    try:
        rclpy.shutdown()
    except Exception:
        pass
    _eprint("[BridgeProc] shutdown complete.")


if __name__ == "__main__":
    main()

