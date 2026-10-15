#!/usr/bin/env python3.10
"""ROS 2/Nav2 bridge 子进程，由 ``IsaacROS2Bridge`` 启动。

它在系统 Python 3.10 中运行 rclpy，把 MuJoCo/IsaacLab 主进程的 JSON 转成
标准 ROS 2 接口，并把 Nav2 输出转回 JSON。节点名 ``isaac_ros2_bridge`` 同样是
历史名称，不影响 MuJoCo 使用。

ROS 侧职责：

* 发布静态 ``map -> odom`` 单位变换；
* 以 50 Hz 发布仿真真值 ``odom -> base_link`` TF 和 ``/odom``；
* 用 ``BasicNavigator`` 发送 NavigateToPose action；
* 订阅 velocity_smoother 的最终 ``/cmd_vel``；
* 订阅 PlannerServer 的 ``/plan`` 并转发给主进程；
* 对陈旧速度命令做 watchdog 归零。

This script MUST be executed with /usr/bin/python3.10 so that rclpy
C-extensions (compiled for CPython 3.10) import correctly.

Protocol: newline-delimited JSON on stdin/stdout.
  stdin  (main -> this):
    {"type":"pose","pos":[x,y,z],"quat":[w,x,y,z],
     "lin_vel":[vx,vy,vz],"ang_vel":[wx,wy,wz]}
    {"type":"plan_path","request_id":int,
     "start":[x,y],"goal":[x,y]}
    {"type":"goal","x":float,"y":float}
    {"type":"cancel_goal"}
    {"type":"stop"}
  stdout (this -> main):
    {"ready":true}                                     -- emitted once on init OK
    {"init_failed":true}                               -- emitted once on import error
    {"vx":float,"omega_z":float,
     "nav_done":bool,"nav_failed":bool}                -- emitted every publish cycle
    {"global_path":[[x,y],...],"path_revision":int}     -- emitted on every new /plan
    {"plan_result":{"request_id":int,"success":bool,
     "path":[[x,y],...],"error":str}}                   -- plan-only reply
    {"cancel_complete":int}                              -- cancellation settled

All diagnostic prints go to stderr (visible in parent terminal).
"""
import argparse
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
# 手动添加 ROS 2 Humble 的 Python 包路径
for _p in [
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
    "/opt/ros/humble/lib/python3.10/site-packages",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _emit(obj: dict) -> None:   # python字典 -> json，并写入stdout
    """Write one JSON line to stdout (the parent process reads it)."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _eprint(*args, **kw) -> None:   # 在终端打印输出诊断信息（报错）
    """Print a diagnostic message to stderr."""
    print(*args, **kw, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("TaskDog ROS 2 bridge subprocess")
    parser.add_argument(
        "--cmd-vel-timeout",
        type=float,
        default=0.1,
        help="seconds to retain the last cmd_vel before forcing zero",
    )
    args = parser.parse_args()
    if args.cmd_vel_timeout <= 0.0:
        parser.error("--cmd-vel-timeout must be positive")

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
        from nav_msgs.msg import Odometry, Path
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
        from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
    except ImportError as exc:
        _eprint(f"[BridgeProc] FATAL import: {exc}")
        traceback.print_exc(file=sys.stderr)
        _emit({"init_failed": True})
        return

    _eprint("[BridgeProc] imports OK, init rclpy...")
    rclpy.init()   # 初始化ROS2客户端库
    # 节点名为了兼容已有日志和调试命令保留 isaac 前缀；功能与仿真后端无关。
    node = Node("isaac_ros2_bridge")  # 创建节点
    tf_bcast     = TransformBroadcaster(node)   # 发布坐标变换
    static_bcast = StaticTransformBroadcaster(node)   # 发布里程计数据

    qos = QoSProfile(   # 配置 ROS 2 消息传输的服务质量（QoS），并创建一个用于发布里程计数据（/odom）的发布者
        reliability=QoSReliabilityPolicy.RELIABLE,   # RELIABLE：可靠传输
        durability=QoSDurabilityPolicy.VOLATILE,   # VOLATILE：瞬态（不缓存）
        history=QoSHistoryPolicy.KEEP_LAST, depth=10)   # 保留最新10条
    odom_pub = node.create_publisher(Odometry, "/odom", qos)

    # Static map->odom identity transform (ground-truth localisation)。仿真世界、
    # Nav2 map 和 odom 使用同一坐标；动态 odom->base_link 由下方主循环发布。
    stf = TransformStamped()   # 创建一个用于存储坐标变换信息的消息对象
    stf.header.stamp    = node.get_clock().now().to_msg()   # 设置时间戳为当前 ROS 时间
    stf.header.frame_id = "map"
    stf.child_frame_id  = "odom"   # 描述的是"odom"相对于"map"的坐标变换
    stf.transform.rotation.w = 1.0   # 四元数，无旋转。平移量默认为0
    static_bcast.sendTransform(stf)   # 通过静态变换广播器发布这个变换（永远不变）

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
    _global_path: list = []
    _path_revision = 0
    _stop        = False
    _pending_plan: list = []
    _pending_goal: list = []
    _pending_cancel_id = None
    _cancel_sequence = 0
    CMD_TIMEOUT = float(args.cmd_vel_timeout)
    # 上次发射给父进程的 nav 状态（None = 还没发过，保证首条必发）。
    # 主循环只在本对值变化时才发 nav_done/nav_failed：否则 NAV1 完成后
    # _nav_done=True 会被无限重复发射，堆积在父进程队列里，导致下一个
    # 导航任务（NAV2_DEST）在 send_goal 后立刻读到陈旧 true 而"秒完成"。
    _last_nd_out = None
    _last_nf_out = None
    _last_path_revision_out = -1

    def _cmd_cb(msg: Twist) -> None:
        nonlocal _vx, _omega_z, _last_cmd_t
        with _lock:
            _vx         = float(msg.linear.x)
            _omega_z    = float(msg.angular.z)
            _last_cmd_t = time.monotonic()
    # 只接收 velocity_smoother 的最终输出。controller_server 与
    # behavior_server 都先发布到 /cmd_vel_nav，再由平滑器输出 /cmd_vel。
    node.create_subscription(Twist, "/cmd_vel", _cmd_cb, 10)   # 订阅平滑后的速度指令

    def _plan_cb(msg: Path) -> None:
        nonlocal _global_path, _path_revision
        points = [
            [float(pose.pose.position.x), float(pose.pose.position.y)]
            for pose in msg.poses
        ]
        with _lock:
            _global_path = points
            _path_revision += 1

    # PlannerServer publishes every newly computed global plan here.  Sending
    # the exact ROS plan to the parent lets diagnostic figures show replanning
    # and doorway selection instead of drawing a guessed straight segment.
    node.create_subscription(Path, "/plan", _plan_cb, 10)

    # -----------------------------------------------------------------------
    # BasicNavigator — do NOT call waitUntilNav2Active() here because TF
    # (odom->base_link) is only published after the main loop starts.
    # Instead, poll bt_navigator/get_state in a background thread AFTER
    # the main loop has already begun broadcasting TF.
    # -----------------------------------------------------------------------
    navigator = BasicNavigator()   # 实例化一个 Nav2 的高级导航控制器对象
    _nav2_ready = threading.Event()   # 确保在 Nav2 还没完全启动好之前，主程序不会发送无效的导航请求，避免报错或任务丢失

    def _nav2_poller() -> None:   # 后台线程轮询 bt_navigator 的状态，当 Nav2 完全激活后，向父进程发送 {"ready": true} 信号
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
    def _stdin_reader() -> None:   # 持续读取父进程通过 stdin 发送的 JSON 消息
        nonlocal _stop, _pending_cancel_id, _cancel_sequence
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
            elif t == "plan_path":
                request_id = int(msg["request_id"])
                start_xy = [float(value) for value in msg["start"][:2]]
                goal_xy = [float(value) for value in msg["goal"][:2]]
                _eprint(
                    f"[BridgeProc] path request {request_id} "
                    f"({start_xy[0]:.2f},{start_xy[1]:.2f}) -> "
                    f"({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
                )
                with _lock:
                    _pending_plan.append((request_id, start_xy, goal_xy))
            elif t == "cancel_goal":
                _eprint("[BridgeProc] cancel queued")
                with _lock:
                    _pending_goal.clear()
                    _cancel_sequence += 1
                    _pending_cancel_id = _cancel_sequence
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
            cancel_request_id = _pending_cancel_id
            _pending_cancel_id = None
            plan_list   = list(_pending_plan)
            _pending_plan.clear()
            goal_list   = list(_pending_goal)
            _pending_goal.clear()
        if should_stop:
            _eprint("[BridgeProc] stop received, shutting down.")
            break

        if cancel_request_id is not None:
            with _lock:
                task_active = _task_active
            if task_active:
                try:
                    navigator.cancelTask()
                    _eprint("[BridgeProc] active Nav2 task cancelled")
                except Exception as exc:
                    _eprint(f"[BridgeProc] warning: cancelTask failed: {exc}")
            with _lock:
                _task_active = False
                _nav_done = False
                _nav_failed = False
                _vx = 0.0
                _omega_z = 0.0
            # cancelTask() waits for Nav2's cancellation response. Tell the
            # parent only after the previous action has stopped driving.
            _emit({"cancel_complete": cancel_request_id})

        # ComputePathToPose uses the same planner/costmap as navigation but
        # does not start the controller or recovery behaviors.  Explicitly
        # supplying the start pose also avoids relying on a possibly delayed
        # TF update immediately after a teleport.
        for request_id, start_xy, goal_xy in plan_list:
            with _lock:
                task_active = _task_active
            if task_active:
                _emit(
                    {
                        "plan_result": {
                            "request_id": request_id,
                            "success": False,
                            "path": [],
                            "error": "navigation task is active",
                        }
                    }
                )
                continue
            try:
                stamp = navigator.get_clock().now().to_msg()
                start_pose = PoseStamped()
                start_pose.header.stamp = stamp
                start_pose.header.frame_id = "map"
                start_pose.pose.position.x = float(start_xy[0])
                start_pose.pose.position.y = float(start_xy[1])
                start_pose.pose.orientation.w = 1.0
                goal_pose = PoseStamped()
                goal_pose.header.stamp = stamp
                goal_pose.header.frame_id = "map"
                goal_pose.pose.position.x = float(goal_xy[0])
                goal_pose.pose.position.y = float(goal_xy[1])
                goal_pose.pose.orientation.w = 1.0
                planned = navigator.getPath(
                    start_pose,
                    goal_pose,
                    use_start=True,
                )
                points = [] if planned is None else [
                    [
                        float(pose.pose.position.x),
                        float(pose.pose.position.y),
                    ]
                    for pose in planned.poses
                ]
                success = len(points) >= 2
                _eprint(
                    f"[BridgeProc] path request {request_id} "
                    f"success={success} points={len(points)}"
                )
                _emit(
                    {
                        "plan_result": {
                            "request_id": request_id,
                            "success": success,
                            "path": points,
                            "error": "" if success else "planner returned no path",
                        }
                    }
                )
            except Exception as exc:
                _eprint(
                    f"[BridgeProc] path request {request_id} failed: {exc}"
                )
                _emit(
                    {
                        "plan_result": {
                            "request_id": request_id,
                            "success": False,
                            "path": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    }
                )

        # Process any pending goal
        for gx, gy in goal_list:
            _eprint(f"[BridgeProc] goToPose ({gx:.2f},{gy:.2f})")
            ps = PoseStamped()
            ps.header.stamp    = node.get_clock().now().to_msg()
            ps.header.frame_id = "map"
            ps.pose.position.x = gx
            ps.pose.position.y = gy
            ps.pose.orientation.w = 1.0
            navigator.goToPose(ps)   # 调用 Nav2 的 goToPose 方法，触发路径规划和导航
            with _lock:
                _nav_done    = False
                _nav_failed  = False
                _task_active = True
                _global_path = []
                _path_revision += 1

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
        # 发布TF变换
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
        # 发布里程计
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

        # Emit cmd_vel BEFORE polling task completion.  BasicNavigator's
        # isTaskComplete() internally spins for up to 0.10 s.  Emitting after
        # that blocking call made a command sampled as fresh above cross the
        # 0.10 s CMD_TIMEOUT boundary, so the parent received (0, 0) even while
        # the diagnostic line correctly showed a fresh non-zero command.
        with _lock:
            stale  = (time.monotonic() - _last_cmd_t) > CMD_TIMEOUT
            vx_out = 0.0 if stale else _vx
            oz_out = 0.0 if stale else _omega_z
        _emit({"vx": vx_out, "omega_z": oz_out})

        # Poll Nav2 task completion.  This may block for up to 0.10 s; only
        # navigation status reporting belongs after it.
        with _lock:
            task_active = _task_active
            nav_done    = _nav_done
            nav_failed  = _nav_failed
        if task_active and not nav_done and not nav_failed:
            try:
                if navigator.isTaskComplete():   # 检查导航完成
                    result = navigator.getResult()
                    nd = (result == TaskResult.SUCCEEDED)
                    nf = not nd
                    _eprint(f"[BridgeProc] task done={nd} fail={nf}")
                    with _lock:   # 发送状态到父进程
                        _nav_done   = nd
                        _nav_failed = nf
                        nav_done    = nd
                        nav_failed  = nf
            except Exception:
                pass

        # Emit nav status to stdout.
        with _lock:
            nd_out = _nav_done
            nf_out = _nav_failed
        # nav 状态只在变化时发射：避免上一个任务的 nav_done=true 被无限重复发送，
        # 让父进程下个任务在 send_goal 后读到陈旧状态而秒完成。
        if (nd_out, nf_out) != (_last_nd_out, _last_nf_out):
            _emit({"nav_done": nd_out, "nav_failed": nf_out})
            _last_nd_out, _last_nf_out = nd_out, nf_out

        with _lock:
            path_revision_out = _path_revision
            path_out = list(_global_path)
        if path_revision_out != _last_path_revision_out:
            _emit({
                "global_path": path_out,
                "path_revision": path_revision_out,
            })
            _last_path_revision_out = path_revision_out

        time.sleep(0.001)

    # Clean up only resources owned by this bridge process.  Nav2 itself is
    # launched and managed by the separate terminal-1 launch process, so
    # calling BasicNavigator.lifecycleShutdown() here would incorrectly send
    # SHUTDOWN to every Nav2 lifecycle node and leave that stack "finalized".
    with _lock:
        cancel_active_task = _task_active and not _nav_done and not _nav_failed
    if cancel_active_task:
        try:
            navigator.cancelTask()
        except Exception as exc:
            _eprint(f"[BridgeProc] warning: failed to cancel active Nav2 task: {exc}")
    try:
        navigator.destroyNode()
    except Exception as exc:
        _eprint(f"[BridgeProc] warning: failed to destroy navigator node: {exc}")
    try:
        executor.remove_node(node)
        node.destroy_node()
    except Exception as exc:
        _eprint(f"[BridgeProc] warning: failed to destroy bridge node: {exc}")
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    _eprint("[BridgeProc] shutdown complete.")


if __name__ == "__main__":
    main()
