"""ROS 2/Nav2 bridge 的仿真主进程一侧（通常是 conda Python 3.11）。

类名 ``IsaacROS2Bridge`` 是历史名称；当前 MuJoCo 和 IsaacLab 两套入口都使用
它。ROS 2 Humble 的 rclpy C 扩展针对系统 Python 3.10 编译，因此不能直接加载到
主进程：本模块启动 ``/usr/bin/python3.10 ros2_bridge_process.py``，双方只通过
换行分隔 JSON 通信。

Python 3.11 (MuJoCo/IsaacLab)              Python 3.10 (ROS 2 client)
       │                                              │
       │  IsaacROS2Bridge 类                          │  ros2_bridge_process.py
       │                                              │
       ├─── stdin (JSON) ────────────────────────────>│  发送位姿、目标点
       │                                              │
       │<─── stdout (JSON) ───────────────────────────┤  接收 cmd_vel、导航状态、全局路径
       │                                              │

主进程发送 MuJoCo 真值位姿和 Nav2 goal；子进程返回平滑后的 ``vx/wz``、导航
状态及最新 ``/plan``。终端 1 的 Nav2 server 不属于这个子进程，``stop()`` 只关闭 bridge 客户端，
不会关闭或 finalized 终端 1 的 Nav2 lifecycle 节点。

Public API:
    bridge = IsaacROS2Bridge()
    bridge.start(timeout=30.0)
    # each sim step:
    bridge.update_robot_pose(pos_w, quat_w, lin_vel, ang_vel)
    vx, omega_z = bridge.get_cmd_vel()
    done, failed = bridge.get_nav_status()
    path_xy = bridge.get_global_path()
    # once before the NAV loop:
    bridge.send_goal(x, y)
    bridge.cancel_goal()
    bridge.stop()
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np


class IsaacROS2Bridge:
    """Starts /usr/bin/python3.10 bridge subprocess; talks JSON over stdio."""

    def __init__(
        self,
        cmd_vel_timeout: float = 0.1,
        feedback_timeout: float = 0.35,
    ) -> None:
        """Create the bridge with two deliberately separate watchdogs.

        ``cmd_vel_timeout`` runs in the ROS subprocess and measures the age of
        the real smoothed ``/cmd_vel`` callback.  ``feedback_timeout`` runs here in
        the MuJoCo process and only detects a stalled/dead subprocess.  The
        subprocess currently reports at about 10 Hz while a navigation task is
        active, so using the 0.1 s ROS timeout for both checks makes valid
        commands flicker to zero at the parent side.
        """
        if cmd_vel_timeout <= 0.0:
            raise ValueError("cmd_vel_timeout must be positive")
        if feedback_timeout <= 0.0:
            raise ValueError("feedback_timeout must be positive")
        self._cmd_vel_timeout = float(cmd_vel_timeout)
        self._feedback_timeout = float(feedback_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._stdin_lock = threading.Lock()
        self._feedback_q: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._vx:         float = 0.0
        self._omega_z:    float = 0.0
        self._nav_done:   bool  = False
        self._nav_failed: bool  = False
        self._global_path: list[list[float]] = []
        self._path_revision: int = 0
        self._plan_sequence: int = 0
        self._plan_results: dict[int, dict] = {}
        self._last_feedback_time: float = 0.0
        self._cancel_sequence: int = 0
        self._pending_cancel_id: Optional[int] = None
        self._completed_cancel_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, timeout: float = 30.0) -> None:
        """Launch /usr/bin/python3.10 subprocess and wait for ready signal."""
        if self._proc is not None:
            raise RuntimeError("[IsaacROS2Bridge] bridge is already started")
        bridge_script = os.path.join(
            os.path.dirname(__file__), "ros2_bridge_process.py")

        # Pass current ROS environment (sourced by caller via setup.bash)
        env = os.environ.copy()# 复制当前进程的环境变量（包括 ROS_DOMAIN_ID、AMENT_PREFIX_PATH 等）。确保子进程能够找到 ROS 2 的库和节点。

        self._proc = subprocess.Popen(
            [
                "/usr/bin/python3.10",
                bridge_script,
                "--cmd-vel-timeout",
                str(self._cmd_vel_timeout),
            ],   # 启动 ROS 子进程，并让子进程使用同一速度指令超时
            stdin=subprocess.PIPE,   # 父进程可以写入 stdin
            stdout=subprocess.PIPE,    # 父进程可以读取 stdout
            stderr=sys.stderr,   # bridge stderr -> parent stderr (visible)  子进程的 stderr 直接输出到终端（方便调试）
            env=env,   # 继承环境变量
            bufsize=1,           # line-buffered（每条 JSON 消息立即刷新）
        )
        print(f"[IsaacROS2Bridge] PID={self._proc.pid}, waiting for ready...",
              flush=True)

        # Background thread reads stdout JSON lines
        proc = self._proc
        self._reader_thread = threading.Thread(
            target=self._stdout_reader,
            args=(proc,),
            daemon=True,
        )   # 创建一个守护线程，持续读取子进程的 stdout。
        self._reader_thread.start()

        # Wait for ready signal
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:   # 子进程提前崩溃
                    raise RuntimeError(
                        f"[IsaacROS2Bridge] subprocess exited early "
                        f"(code={proc.returncode})")
                try:   # 尝试从消息队列获取反馈
                    msg = self._feedback_q.get(timeout=0.3)   # 从消息队列获取一条消息，最多等待 0.3 秒
                    if msg.get("init_failed"):   # 消息中是否有"init_failed"
                        raise RuntimeError(
                            "[IsaacROS2Bridge] subprocess init failed; "
                            "check rclpy/Nav2 installation.")
                    if msg.get("ready"):   # 消息中是否有"ready"
                        print("[IsaacROS2Bridge] ready: /tf /odom /cmd_vel",
                              flush=True)
                        return
                except queue.Empty:   # 没收到消息，继续循环
                    pass
            raise RuntimeError(
                f"[IsaacROS2Bridge] not ready within {timeout}s")   # 超时抛出异常
        except BaseException:
            # A partially started child and its stdout reader must not survive
            # a failed start; both own native rclpy/pipe resources.
            self.stop()
            raise

    def stop(self) -> None:   # 关闭本 bridge 子进程，不关闭终端1的 Nav2 栈
        """Stop the child, reap it, and join the stdout reader.

        Merely terminating the child is insufficient: a daemon reader blocked
        on its pipe can otherwise survive until CPython/native-library teardown.
        This method is deliberately idempotent so every exception path may call
        it unconditionally.
        """
        proc = self._proc
        reader = self._reader_thread
        if proc is None:
            self._reader_thread = None
            return

        if proc.poll() is None:
            self._send({"type": "stop"})
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        if reader is not None and reader.is_alive() and proc.stdout is not None:
            # Closing the read end releases a reader even if a descendant kept
            # a duplicate write descriptor alive.
            try:
                proc.stdout.close()
            except Exception:
                pass
            if reader is not threading.current_thread():
                reader.join(timeout=1.0)
        elif proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

        self._proc = None
        self._reader_thread = None
        self._vx = 0.0
        self._omega_z = 0.0
        self._nav_done = False
        self._nav_failed = False
        self._global_path = []
        self._path_revision = 0
        self._plan_sequence = 0
        self._plan_results = {}
        self._cancel_sequence = 0
        self._pending_cancel_id = None
        self._completed_cancel_id = None
        print("[IsaacROS2Bridge] stopped.", flush=True)

    def update_robot_pose(
        self,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        lin_vel: Optional[np.ndarray] = None,
        ang_vel: Optional[np.ndarray] = None,
    ) -> None:
        """Send current robot pose (non-blocking, drop if subprocess not ready)."""
        lv = lin_vel.tolist() if lin_vel is not None else [0., 0., 0.]
        av = ang_vel.tolist() if ang_vel is not None else [0., 0., 0.]
        self._send({
            "type":    "pose",
            "pos":     list(map(float, pos_w)),
            "quat":    list(map(float, quat_w)),
            "lin_vel": list(map(float, lv)),
            "ang_vel": list(map(float, av)),
        })   # 仿真 -> bridge；子进程再发布 /tf 和 /odom 供 Nav2 使用
        self._drain_feedback()

    def send_goal(self, x: float, y: float) -> None:   # 向 Nav2发送目标点坐标，Nav2 接收后触发 A* 路径规划和 Pure Pursuit 跟踪
        """Send a Nav2 navigation goal."""
        if self._pending_cancel_id is not None:
            self.wait_for_cancel(timeout=3.0)
            # Let TF and the costmaps observe the final teleported pose before
            # starting the next behavior tree.
            time.sleep(0.25)
        self._drain_feedback()   # 先清掉队列里上一个任务遗留的 nav_done 消息（否则本任务会读数秒完成）
        self._nav_done   = False
        self._nav_failed = False
        self._global_path = []
        self._send({"type": "goal", "x": float(x), "y": float(y)})

    def plan_path(
        self,
        start_xy,
        goal_xy,
        *,
        timeout: float = 15.0,
    ) -> list[list[float]]:
        """Compute a Nav2 global path without starting navigation.

        The ROS child uses ``ComputePathToPose`` with the supplied start pose,
        so this request neither drives the robot nor depends on a guessed
        straight-line midpoint. An empty list means that the planner completed
        the request but found no path.
        """

        if timeout <= 0.0:
            raise ValueError("plan timeout must be positive")
        if self._pending_cancel_id is not None:
            self.wait_for_cancel(timeout=3.0)
            time.sleep(0.25)

        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        self._plan_sequence += 1
        request_id = self._plan_sequence
        self._plan_results.pop(request_id, None)
        self._send(
            {
                "type": "plan_path",
                "request_id": request_id,
                "start": start.astype(float).tolist(),
                "goal": goal.astype(float).tolist(),
            }
        )

        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self._drain_feedback()
            result = self._plan_results.pop(request_id, None)
            if result is not None:
                if not result.get("success", False):
                    return []
                return [point.copy() for point in result.get("path", [])]
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    "[IsaacROS2Bridge] subprocess stopped while planning path"
                )
            time.sleep(0.01)
        raise RuntimeError(
            f"[IsaacROS2Bridge] Nav2 path request timed out after "
            f"{float(timeout):.1f}s"
        )

    def cancel_goal(self, *, wait: bool = False, timeout: float = 3.0) -> None:
        """Cancel the active Nav2 task without stopping the bridge.

        The next :meth:`send_goal` is a cancellation barrier. Callers that
        need the barrier immediately may pass ``wait=True``.
        """
        self._drain_feedback()
        self._nav_done = False
        self._nav_failed = False
        self._vx = 0.0
        self._omega_z = 0.0
        self._cancel_sequence += 1
        request_id = self._cancel_sequence
        self._pending_cancel_id = request_id
        self._send({"type": "cancel_goal"})
        if wait:
            self.wait_for_cancel(timeout=timeout)

    def wait_for_cancel(self, timeout: float = 3.0) -> None:
        """Wait until the ROS child confirms the latest cancellation."""
        request_id = self._pending_cancel_id
        if request_id is None:
            return
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self._drain_feedback()
            if self._completed_cancel_id == request_id:
                self._pending_cancel_id = None
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    "[IsaacROS2Bridge] subprocess stopped while canceling Nav2 task"
                )
            time.sleep(0.01)
        raise RuntimeError(
            f"[IsaacROS2Bridge] Nav2 cancellation was not acknowledged "
            f"within {float(timeout):.1f}s"
        )

    def get_cmd_vel(self) -> Tuple[float, float]:   # 从Nav2获取速度指令cmd_vel 
        """Return the subprocess-filtered command, or zero if feedback stalls.

        The ROS subprocess has already replaced stale Nav2 commands with zero
        according to ``cmd_vel_timeout``.  This second, longer timeout is only
        a transport/process watchdog.
        """
        self._drain_feedback()
        if ((time.monotonic() - self._last_feedback_time)
                > self._feedback_timeout):
            return 0.0, 0.0
        return self._vx, self._omega_z

    def get_nav_status(self) -> Tuple[bool, bool]:   # 获取导航状态（是否完成/失败）
        """Return (nav_done, nav_failed)."""
        self._drain_feedback()
        return self._nav_done, self._nav_failed

    def get_global_path(self) -> list[list[float]]:
        """Return the newest Nav2 ``/plan`` path as world-frame XY points."""
        self._drain_feedback()
        return [point.copy() for point in self._global_path]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, obj: dict) -> None:   # python字典 -> json字符串，并且写入子进程的stdin
        """Serialize obj as JSON and write one line to subprocess stdin."""
        if self._proc is None or self._proc.stdin is None:   # 子进程未启动或者stdin不可用
            return
        try:
            line = json.dumps(obj, separators=(",", ":")) + "\n"
            with self._stdin_lock:   # 发防止多个线程同时写入stdin导致json混乱
                self._proc.stdin.write(line.encode())
                self._proc.stdin.flush()   # 立即刷新缓冲区，确保消息立即发送给子进程。
        except Exception:
            pass

    def _stdout_reader(self, proc: subprocess.Popen) -> None:   # 后台持续读取消息的线程
        """Daemon thread: read JSON lines from subprocess stdout."""
        if proc.stdout is None:
            return
        try:
            for raw in proc.stdout:   # 持续读子进程的stdout
                try:
                    msg = json.loads(raw.decode().strip())
                    self._feedback_q.put(msg)   # 放入队列
                except Exception:
                    pass
        except (OSError, ValueError):
            # stop() may close the read end to release a stubborn reader.
            pass

    def _drain_feedback(self) -> None:   # 非阻塞地清空反馈队列，并更新缓存的状态变量
        """Non-blocking drain of feedback queue; update cached state."""
        while True:   # 直到没有更新的消息 -> 拿到的是最新的消息
            try:
                msg = self._feedback_q.get_nowait()   # 尝试从队列中取出一条消息
            except queue.Empty:
                break
            if "vx" in msg:
                self._vx      = float(msg["vx"])
                self._omega_z = float(msg["omega_z"])
                self._last_feedback_time = time.monotonic()
            if "nav_done" in msg:
                self._nav_done   = bool(msg["nav_done"])
                self._nav_failed = bool(msg["nav_failed"])
            if "global_path" in msg:
                path = msg.get("global_path") or []
                self._global_path = [
                    [float(point[0]), float(point[1])]
                    for point in path
                    if isinstance(point, (list, tuple)) and len(point) >= 2
                ]
                self._path_revision = int(
                    msg.get("path_revision", self._path_revision + 1)
                )
            if "plan_result" in msg:
                raw_result = msg.get("plan_result") or {}
                request_id = int(raw_result.get("request_id", -1))
                if request_id >= 0:
                    path = raw_result.get("path") or []
                    self._plan_results[request_id] = {
                        "success": bool(raw_result.get("success", False)),
                        "path": [
                            [float(point[0]), float(point[1])]
                            for point in path
                            if isinstance(point, (list, tuple))
                            and len(point) >= 2
                        ],
                        "error": str(raw_result.get("error", "")),
                    }
            if "cancel_complete" in msg:
                self._completed_cancel_id = int(msg["cancel_complete"])

# MuJoCo/IsaacLab -> ROS bridge -> Nav2
# 类型	                        JSON 格式	                                                                            作用
# 位姿更新	            {"type":"pose","pos":[x,y,z],"quat":[w,x,y,z],"lin_vel":[...],"ang_vel":[...]}	            发布真实位姿到 /tf、/odom
# 导航目标	            {"type":"goal","x":5.0,"y":3.0}	                                                            触发 Nav2 导航
# 停止信号	            {"type":"stop"}	                                                                            关闭 bridge 客户端进程

# Nav2 -> ROS bridge -> MuJoCo/IsaacLab
# 类型	                        JSON 格式	                               作用
# 就绪信号	            {"ready":true}	                            表示 Nav2 已启动完成
# 速度指令	            {"vx":0.5,"omega_z":0.3}	                    控制器输出的速度
# 导航状态	            {"nav_done":true,"nav_failed":false}	        导航完成/失败
