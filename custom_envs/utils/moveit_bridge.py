"""Python-side bridge for MoveIt 2 services.

This module is imported by the MuJoCo process (which may be running in a conda
Python different from ROS 2 Humble's Python).  A separate /usr/bin/python3.10
process owns rclpy and MoveIt message/service types.  Communication is newline
JSON over stdio, matching the pattern already used by ros2_bridge.py.

MoveIt is used as a *planner only*.  Planned JointTrajectory points are returned
to the MuJoCo process, which continues to execute arm position targets through
its existing env.set_arm_target() + PD controller.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from typing import Dict, Iterable, Optional

import numpy as np


class MoveItROS2Bridge:
    def __init__(self, request_timeout: float = 8.0) -> None:
        self.request_timeout = float(request_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._stdin_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._messages: "queue.Queue[dict]" = queue.Queue()
        self._pending: Dict[str, dict] = {}

    def start(self, timeout: float = 30.0) -> None:
        script = os.path.join(os.path.dirname(__file__), "moveit_bridge_process.py")
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            ["/usr/bin/python3.10", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=env,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._stdout_reader, daemon=True)
        self._reader_thread.start()

        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"MoveIt bridge process exited early (code={self._proc.returncode})")
            try:
                msg = self._messages.get(timeout=0.25)
            except queue.Empty:
                continue
            if msg.get("init_failed"):
                raise RuntimeError(
                    "MoveIt bridge init failed; inspect stderr and verify MoveIt 2 Humble is installed")
            if msg.get("ready"):
                print("[MoveItBridge] ready: joint_states + compute_ik + planning + cartesian path", flush=True)
                return
            if "id" in msg:
                self._pending[str(msg["id"])] = msg
        raise RuntimeError(f"MoveIt bridge not ready within {timeout:.1f}s")

    def stop(self) -> None:
        try:
            self._send({"type": "stop"})
        except Exception:
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        self._proc = None

    # ------------------------------------------------------------------
    # State publication
    # ------------------------------------------------------------------
    def update_joint_state(self, names: Iterable[str], positions: Iterable[float]) -> None:
        self._send({
            "type": "joint_state",
            "names": list(names),
            "positions": [float(x) for x in positions],
        })

    # ------------------------------------------------------------------
    # Planning scene
    # ------------------------------------------------------------------
    def apply_default_scene(self, timeout: Optional[float] = None) -> dict:
        """Add the two MuJoCo tables to MoveIt's PlanningScene in the map frame."""
        return self._request({"type": "apply_default_scene"}, timeout=timeout)

    # ------------------------------------------------------------------
    # MoveIt calls
    # ------------------------------------------------------------------
    def compute_ik(
        self,
        pose: dict,
        start_joint_names,
        start_joint_positions,
        frame_id: str = "base_link",
        group_name: str = "piper_arm",
        link_name: str = "grasp_tcp",
        avoid_collisions: bool = True,
        timeout_sec: float = 0.15,
        timeout: Optional[float] = None,
    ) -> dict:
        return self._request({
            "type": "compute_ik",
            "pose": pose,
            "frame_id": frame_id,
            "group_name": group_name,
            "link_name": link_name,
            "start_joint_names": list(start_joint_names),
            "start_joint_positions": [float(x) for x in start_joint_positions],
            "avoid_collisions": bool(avoid_collisions),
            "ik_timeout_sec": float(timeout_sec),
        }, timeout=timeout)

    def plan_to_pose(
        self,
        pose: dict,
        start_joint_names,
        start_joint_positions,
        frame_id: str = "base_link",
        group_name: str = "piper_arm",
        link_name: str = "grasp_tcp",
        position_tolerance: float = 0.008,
        orientation_tolerance: float = 0.10,
        allowed_planning_time: float = 2.5,
        planning_attempts: int = 3,
        velocity_scaling: float = 0.25,
        acceleration_scaling: float = 0.25,
        planner_id: str = "RRTConnectkConfigDefault",
        timeout: Optional[float] = None,
    ) -> dict:
        return self._request({
            "type": "plan_to_pose",
            "pose": pose,
            "frame_id": frame_id,
            "group_name": group_name,
            "link_name": link_name,
            "start_joint_names": list(start_joint_names),
            "start_joint_positions": [float(x) for x in start_joint_positions],
            "position_tolerance": float(position_tolerance),
            "orientation_tolerance": float(orientation_tolerance),
            "allowed_planning_time": float(allowed_planning_time),
            "planning_attempts": int(planning_attempts),
            "velocity_scaling": float(velocity_scaling),
            "acceleration_scaling": float(acceleration_scaling),
            "planner_id": str(planner_id),
        }, timeout=timeout or max(self.request_timeout, allowed_planning_time + 4.0))

    def compute_cartesian_path(
        self,
        poses,
        start_joint_names,
        start_joint_positions,
        frame_id: str = "base_link",
        group_name: str = "piper_arm",
        link_name: str = "grasp_tcp",
        max_step: float = 0.005,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True,
        velocity_scaling: float = 0.20,
        acceleration_scaling: float = 0.20,
        timeout: Optional[float] = None,
    ) -> dict:
        return self._request({
            "type": "cartesian_path",
            "poses": list(poses),
            "frame_id": frame_id,
            "group_name": group_name,
            "link_name": link_name,
            "start_joint_names": list(start_joint_names),
            "start_joint_positions": [float(x) for x in start_joint_positions],
            "max_step": float(max_step),
            "jump_threshold": float(jump_threshold),
            "avoid_collisions": bool(avoid_collisions),
            "velocity_scaling": float(velocity_scaling),
            "acceleration_scaling": float(acceleration_scaling),
        }, timeout=timeout)

    # ------------------------------------------------------------------
    # Joint path helpers used by the MuJoCo executor
    # ------------------------------------------------------------------
    @staticmethod
    def extract_arm_path(trajectory: dict, arm_joint_names) -> np.ndarray:
        """Return Nx6 positions ordered exactly as arm_joint_names."""
        names = list(trajectory.get("joint_names", []))
        points = trajectory.get("points", [])
        if not names or not points:
            return np.zeros((0, len(arm_joint_names)), dtype=np.float64)
        idx = []
        for n in arm_joint_names:
            if n not in names:
                raise ValueError(f"trajectory is missing joint {n}; got {names}")
            idx.append(names.index(n))
        out = []
        for p in points:
            q = np.asarray(p["positions"], dtype=np.float64)
            out.append(q[idx])
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def densify_joint_path(path: np.ndarray, max_joint_delta=None) -> np.ndarray:
        """Densify MoveIt's piecewise-linear joint path for the MuJoCo PD loop."""
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or len(path) == 0:
            return path.reshape(0, path.shape[-1] if path.ndim else 0)
        if max_joint_delta is None:
            max_joint_delta = np.array([0.012, 0.018, 0.018, 0.015, 0.015, 0.018])
        lim = np.asarray(max_joint_delta, dtype=np.float64).reshape(path.shape[1])
        dense = [path[0].copy()]
        for a, b in zip(path[:-1], path[1:]):
            dq = b - a
            n = int(max(1, np.ceil(np.max(np.abs(dq) / np.maximum(lim, 1e-9)))))
            for k in range(1, n + 1):
                dense.append(a + (float(k) / n) * dq)
        return np.asarray(dense, dtype=np.float64)

    # ------------------------------------------------------------------
    # JSON plumbing
    # ------------------------------------------------------------------
    def _request(self, payload: dict, timeout: Optional[float] = None) -> dict:
        rid = uuid.uuid4().hex
        msg = dict(payload)
        msg["id"] = rid
        self._send(msg)
        deadline = time.monotonic() + float(timeout or self.request_timeout)
        while time.monotonic() < deadline:
            if rid in self._pending:
                return self._pending.pop(rid)
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"MoveIt bridge exited while waiting for {payload.get('type')} "
                    f"(code={self._proc.returncode})")
            try:
                incoming = self._messages.get(timeout=0.05)
            except queue.Empty:
                continue
            if "id" in incoming:
                incoming_id = str(incoming["id"])
                if incoming_id == rid:
                    return incoming
                self._pending[incoming_id] = incoming
        raise TimeoutError(f"MoveIt bridge request timed out: {payload.get('type')}")

    def _send(self, obj: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("MoveIt bridge is not running")
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        with self._stdin_lock:
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()

    def _stdout_reader(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for raw in self._proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8").strip())
                self._messages.put(msg)
            except Exception:
                continue
