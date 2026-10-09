"""Keyboard state and passive-viewer debug overlays."""

from __future__ import annotations

import math
import threading
import time
from typing import Iterable

import numpy as np


class InteractiveController:
    """Thread-safe key state for ``mujoco.viewer.launch_passive``.

    Keys are intentionally ASCII-only so this does not depend on a particular
    glfw Python package being installed in the active conda environment.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.paused = False
        self.advance = False
        self.abort_candidate = False
        self.quit = False

    def key_callback(self, keycode: int) -> None:
        try:
            key = chr(int(keycode)).lower()
        except Exception:
            return
        with self._lock:
            if key == " ":
                self.paused = not self.paused
            elif key == "n" or int(keycode) == 257:
                self.advance = True
            elif key == "x":
                self.abort_candidate = True
                self.advance = True
            elif key == "q":
                self.quit = True
                self.advance = True

    def clear_advance(self) -> None:
        with self._lock:
            self.advance = False

    def should_abort(self) -> bool:
        with self._lock:
            return bool(self.abort_candidate or self.quit)

    def clear_candidate_abort(self) -> None:
        with self._lock:
            self.abort_candidate = False

    def wait_for_advance(self, env, prompt: str) -> bool:
        print(
            f"[INSPECT] {prompt} | N/Enter: continue, Space: pause/resume, "
            "X: skip, Q: quit",
            flush=True,
        )
        self.clear_advance()
        while env.viewer_is_running():
            with self._lock:
                if self.advance or self.abort_candidate or self.quit:
                    return not (self.abort_candidate or self.quit)
            if env.viewer is not None:
                env.viewer.sync()
            time.sleep(0.03)
        with self._lock:
            self.quit = True
        return False

    def honor_pause(self, env) -> bool:
        while env.viewer_is_running():
            with self._lock:
                if self.quit or self.abort_candidate:
                    return False
                if not self.paused:
                    return True
            if env.viewer is not None:
                env.viewer.sync()
            time.sleep(0.03)
        return False


class ViewerOverlay:
    """Best-effort user-scene overlay; silently degrades on older MuJoCo."""

    def __init__(self, env):
        self.env = env

    @staticmethod
    def _add_arrow(scene, start, end, color, width=0.012):
        import mujoco

        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(width),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    @staticmethod
    def _add_sphere(scene, position, radius, color):
        import mujoco

        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

    def update(self, cube_xyz, base_pose, joint1_target=None, joint1_actual=None, status="candidate"):
        viewer = self.env.viewer
        if viewer is None or not hasattr(viewer, "user_scn"):
            return
        try:
            base_pose = np.asarray(base_pose, dtype=np.float64)
            cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
            base = np.array([base_pose[0], base_pose[1], 0.72], dtype=np.float64)
            colors = {
                "candidate": (1.0, 0.8, 0.1, 0.95),
                "success": (0.1, 0.9, 0.2, 0.95),
                "failure": (0.95, 0.1, 0.1, 0.95),
            }
            lock = viewer.lock() if hasattr(viewer, "lock") else _NullContext()
            with lock:
                scene = viewer.user_scn
                scene.ngeom = 0
                self._add_sphere(scene, cube_xyz, 0.04, (1.0, 1.0, 0.0, 0.65))

                yaw = float(base_pose[2])
                heading = base + 0.45 * np.array([math.cos(yaw), math.sin(yaw), 0.0])
                self._add_arrow(scene, base, heading, colors.get(status, colors["candidate"]), 0.018)
                self._add_arrow(scene, base, cube_xyz, (1.0, 0.85, 0.0, 0.85), 0.008)

                if joint1_target is not None:
                    target_yaw = yaw + float(joint1_target)
                    end = base + 0.55 * np.array(
                        [math.cos(target_yaw), math.sin(target_yaw), 0.0]
                    )
                    self._add_arrow(scene, base, end, (1.0, 0.9, 0.0, 1.0), 0.014)
                if joint1_actual is not None:
                    actual_yaw = yaw + float(joint1_actual)
                    end = base + 0.48 * np.array(
                        [math.cos(actual_yaw), math.sin(actual_yaw), 0.0]
                    )
                    self._add_arrow(scene, base, end, (0.1, 0.4, 1.0, 1.0), 0.010)
        except Exception as exc:
            print(f"[INSPECT] viewer overlay unavailable: {exc}", flush=True)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
