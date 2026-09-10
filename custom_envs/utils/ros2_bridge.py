"""ROS 2 Bridge launcher (Python 3.11 side).
在 Python 3.11（Isaac Lab 主进程）和 Python 3.10（Nav2 ROS 2 进程）之间建立通信通道
用子进程 + JSON 通信的方式桥接两个进程（ROS 2 Humble 和 Isaac Lab）
Python 3.11 (Isaac Lab)                    Python 3.10 (Nav2 进程)
       │                                              │
       │  IsaacROS2Bridge 类                          │  ros2_bridge_process.py
       │                                              │
       ├─── stdin (JSON) ────────────────────────────>│  发送位姿、目标点
       │                                              │
       │<─── stdout (JSON) ───────────────────────────┤  接收 cmd_vel、导航状态
       │                                              │

Launches /usr/bin/python3.10 ros2_bridge_process.py as a real subprocess.
Communicates via newline-delimited JSON on stdin/stdout.

Public API:
    bridge = IsaacROS2Bridge()
    bridge.start(timeout=30.0)
    # each sim step:
    bridge.update_robot_pose(pos_w, quat_w, lin_vel, ang_vel)
    vx, omega_z = bridge.get_cmd_vel()
    done, failed = bridge.get_nav_status()
    # once before the NAV loop:
    bridge.send_goal(x, y)
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

    def __init__(self, cmd_vel_timeout: float = 0.5) -> None:
        self._cmd_vel_timeout = cmd_vel_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._stdin_lock = threading.Lock()
        self._feedback_q: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._vx:         float = 0.0
        self._omega_z:    float = 0.0
        self._nav_done:   bool  = False
        self._nav_failed: bool  = False
        self._last_cmd_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, timeout: float = 30.0) -> None:
        """Launch /usr/bin/python3.10 subprocess and wait for ready signal."""
        bridge_script = os.path.join(
            os.path.dirname(__file__), "ros2_bridge_process.py")

        # Pass current ROS environment (sourced by caller via setup.bash)
        env = os.environ.copy()# 复制当前进程的环境变量（包括 ROS_DOMAIN_ID、AMENT_PREFIX_PATH 等）。确保子进程能够找到 ROS 2 的库和节点。

        self._proc = subprocess.Popen(
            ["/usr/bin/python3.10", bridge_script],   # 启动 ros2_bridge_process.py 作为子进程（用python3.10运行）
            stdin=subprocess.PIPE,   # 父进程可以写入 stdin
            stdout=subprocess.PIPE,    # 父进程可以读取 stdout
            stderr=sys.stderr,   # bridge stderr -> parent stderr (visible)  子进程的 stderr 直接输出到终端（方便调试）
            env=env,   # 继承环境变量
            bufsize=1,           # line-buffered（每条 JSON 消息立即刷新）
        )
        print(f"[IsaacROS2Bridge] PID={self._proc.pid}, waiting for ready...",
              flush=True)

        # Background thread reads stdout JSON lines
        self._reader_thread = threading.Thread(
            target=self._stdout_reader, daemon=True)   # 创建一个守护线程，持续读取子进程的 stdout。主程序退出后，守护线程自动被杀死
        self._reader_thread.start()

        # Wait for ready signal
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:   # 子进程提前崩溃
                raise RuntimeError(
                    f"[IsaacROS2Bridge] subprocess exited early "
                    f"(code={self._proc.returncode})")
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

    def stop(self) -> None:   # 关闭 Nav2进程
        """Ask subprocess to shut down gracefully."""
        self._send({"type": "stop"})
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:   # 超时强制终止并抛出异常
                self._proc.terminate()
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
        })   # 发送机器人位姿、速度、加速度。Isaac Sim -> Nav2，Nav2接收后发布到/tf话题供代价地图和规划器使用
        self._drain_feedback()

    def send_goal(self, x: float, y: float) -> None:   # 向 Nav2发送目标点坐标，Nav2 接收后触发 A* 路径规划和 Pure Pursuit 跟踪
        """Send a Nav2 navigation goal."""
        self._nav_done   = False
        self._nav_failed = False
        self._send({"type": "goal", "x": float(x), "y": float(y)})

    def get_cmd_vel(self) -> Tuple[float, float]:   # 从Nav2获取速度指令cmd_vel 
        """Return (vx, omega_z). Returns (0, 0) if stale."""
        self._drain_feedback()
        if (time.monotonic() - self._last_cmd_time) > self._cmd_vel_timeout:   # 超时返回0
            return 0.0, 0.0
        return self._vx, self._omega_z

    def get_nav_status(self) -> Tuple[bool, bool]:   # 获取导航状态（是否完成/失败）
        """Return (nav_done, nav_failed)."""
        self._drain_feedback()
        return self._nav_done, self._nav_failed

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

    def _stdout_reader(self) -> None:   # 后台持续读取消息的线程
        """Daemon thread: read JSON lines from subprocess stdout."""
        for raw in self._proc.stdout:   # 持续读子进程的stdout
            try:
                msg = json.loads(raw.decode().strip())
                self._feedback_q.put(msg)   # 放入队列
            except Exception:
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
                self._last_cmd_time = time.monotonic()
            if "nav_done" in msg:
                self._nav_done   = bool(msg["nav_done"])
                self._nav_failed = bool(msg["nav_failed"])

# IsaacLab -> Nav2
# 类型	                        JSON 格式	                                                                            作用
# 位姿更新	            {"type":"pose","pos":[x,y,z],"quat":[x,y,z,w],"lin_vel":[...],"ang_vel":[...]}	            发布真实位姿到 /tf
# 导航目标	            {"type":"goal","x":5.0,"y":3.0}	                                                            触发 Nav2 导航
# 停止信号	            {"type":"stop"}	                                                                            关闭 Nav2 进程

# Nav2 -> IsaacLab
# 类型	                        JSON 格式	                               作用
# 就绪信号	            {"ready":true}	                            表示 Nav2 已启动完成
# 速度指令	            {"vx":0.5,"omega_z":0.3}	                    控制器输出的速度
# 导航状态	            {"nav_done":true,"nav_failed":false}	        导航完成/失败