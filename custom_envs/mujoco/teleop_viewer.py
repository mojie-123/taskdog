"""Small GLFW viewer dedicated to MuJoCo keyboard teleoperation.

The stock passive viewer only exposes key-down events to Python and continues
to process the same keys as its own rendering shortcuts.  This viewer owns its
GLFW callbacks, so teleoperation receives both press and release events and no
MuJoCo shortcut handler sees W/S/A/D/Q/E.
"""

from typing import Callable, Optional

import glfw
import mujoco


KeyCallback = Callable[[object, int, int, int, int], None]


class MuJoCoTeleopViewer:
    """Render a MuJoCo scene without installing stock keyboard shortcuts."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_callback: Optional[KeyCallback] = None,
        width: int = 1280,
        height: int = 720,
        title: str = "MuJoCo mapping teleop",
        track_body_name: str = "base_link",
    ) -> None:
        self.model = model
        self.data = data
        self._key_callback = key_callback
        self._closed = False
        self._last_cursor = (0.0, 0.0)
        self._mouse_left = False
        self._mouse_right = False
        self._mouse_middle = False
        self._held_keys = set()

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_window_focus_callback(self.window, self._on_focus)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)
        if body_id >= 0:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = body_id
            self.camera.lookat[:] = data.xpos[body_id]
        self.camera.distance = 5.5
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0

    def is_running(self) -> bool:
        return (
            not self._closed
            and self.window is not None
            and not glfw.window_should_close(self.window)
        )

    def poll_events(self) -> None:
        if self.is_running():
            glfw.poll_events()

    def render(self) -> None:
        if not self.is_running():
            return
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            self.perturb,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            "Hold: W/S forward  A/D strafe  Q/E turn",
            "Space/X stop  M save  Esc quit",
            self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if getattr(self, "context", None) is not None:
            self.context.free()
        if getattr(self, "window", None) is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.PRESS:
            self._held_keys.add(key)
        elif action == glfw.RELEASE:
            self._held_keys.discard(key)
        if self._key_callback is not None:
            self._key_callback(window, key, scancode, action, mods)

    def _on_focus(self, window, focused) -> None:
        """Release held controls if the window loses focus."""
        if focused or self._key_callback is None:
            return
        for key in tuple(self._held_keys):
            self._key_callback(window, key, 0, glfw.RELEASE, 0)
        self._held_keys.clear()

    def _on_mouse_button(self, window, button, action, mods) -> None:
        del button, action, mods
        self._mouse_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._mouse_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        self._mouse_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos, ypos) -> None:
        if not (self._mouse_left or self._mouse_right or self._mouse_middle):
            self._last_cursor = (xpos, ypos)
            return
        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._mouse_right:
            mouse_action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        elif self._mouse_left:
            mouse_action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        else:
            mouse_action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(
            self.model,
            mouse_action,
            (xpos - last_x) / width,
            (ypos - last_y) / height,
            self.camera,
        )

    def _on_scroll(self, window, x_offset, y_offset) -> None:
        del window, x_offset
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            self.camera,
        )
