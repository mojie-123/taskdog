"""ROS 2 Bridge launcher (Python 3.11 side).

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
        env = os.environ.copy()

        self._proc = subprocess.Popen(
            ["/usr/bin/python3.10", bridge_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # bridge stderr -> parent stderr (visible)
            env=env,
            bufsize=1,           # line-buffered
        )
        print(f"[IsaacROS2Bridge] PID={self._proc.pid}, waiting for ready...",
              flush=True)

        # Background thread reads stdout JSON lines
        self._reader_thread = threading.Thread(
            target=self._stdout_reader, daemon=True)
        self._reader_thread.start()

        # Wait for ready signal
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"[IsaacROS2Bridge] subprocess exited early "
                    f"(code={self._proc.returncode})")
            try:
                msg = self._feedback_q.get(timeout=0.3)
                if msg.get("init_failed"):
                    raise RuntimeError(
                        "[IsaacROS2Bridge] subprocess init failed; "
                        "check rclpy/Nav2 installation.")
                if msg.get("ready"):
                    print("[IsaacROS2Bridge] ready: /tf /odom /cmd_vel",
                          flush=True)
                    return
            except queue.Empty:
                pass
        raise RuntimeError(
            f"[IsaacROS2Bridge] not ready within {timeout}s")

    def stop(self) -> None:
        """Ask subprocess to shut down gracefully."""
        self._send({"type": "stop"})
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
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
        })
        self._drain_feedback()

    def send_goal(self, x: float, y: float) -> None:
        """Send a Nav2 navigation goal."""
        self._nav_done   = False
        self._nav_failed = False
        self._send({"type": "goal", "x": float(x), "y": float(y)})

    def get_cmd_vel(self) -> Tuple[float, float]:
        """Return (vx, omega_z). Returns (0, 0) if stale."""
        self._drain_feedback()
        if (time.monotonic() - self._last_cmd_time) > self._cmd_vel_timeout:
            return 0.0, 0.0
        return self._vx, self._omega_z

    def get_nav_status(self) -> Tuple[bool, bool]:
        """Return (nav_done, nav_failed)."""
        self._drain_feedback()
        return self._nav_done, self._nav_failed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        """Serialize obj as JSON and write one line to subprocess stdin."""
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            line = json.dumps(obj, separators=(",", ":")) + "\n"
            with self._stdin_lock:
                self._proc.stdin.write(line.encode())
                self._proc.stdin.flush()
        except Exception:
            pass

    def _stdout_reader(self) -> None:
        """Daemon thread: read JSON lines from subprocess stdout."""
        for raw in self._proc.stdout:
            try:
                msg = json.loads(raw.decode().strip())
                self._feedback_q.put(msg)
            except Exception:
                pass

    def _drain_feedback(self) -> None:
        """Non-blocking drain of feedback queue; update cached state."""
        while True:
            try:
                msg = self._feedback_q.get_nowait()
            except queue.Empty:
                break
            if "vx" in msg:
                self._vx      = float(msg["vx"])
                self._omega_z = float(msg["omega_z"])
                self._last_cmd_time = time.monotonic()
            if "nav_done" in msg:
                self._nav_done   = bool(msg["nav_done"])
                self._nav_failed = bool(msg["nav_failed"])


