import glfw
import numpy as np

from custom_envs.scripts.navigation.teleop_mapping_mujoco import (
    TeleopController,
)
from custom_envs.mujoco.teleop_viewer import MuJoCoTeleopViewer


def _key(controller, key, action):
    controller.on_key(None, key, 0, action, 0)


def test_motion_exists_only_between_press_and_release():
    controller = TeleopController(0.6, 0.5, 0.8)

    _key(controller, glfw.KEY_W, glfw.PRESS)
    np.testing.assert_allclose(controller.snapshot()[0], [0.6, 0.0, 0.0])

    _key(controller, glfw.KEY_W, glfw.RELEASE)
    np.testing.assert_allclose(controller.snapshot()[0], [0.0, 0.0, 0.0])


def test_simultaneous_keys_and_opposites_are_composed():
    controller = TeleopController(0.6, 0.5, 0.8)
    for key in (glfw.KEY_W, glfw.KEY_A, glfw.KEY_Q):
        _key(controller, key, glfw.PRESS)
    np.testing.assert_allclose(controller.snapshot()[0], [0.6, 0.5, 0.8])

    _key(controller, glfw.KEY_S, glfw.PRESS)
    np.testing.assert_allclose(controller.snapshot()[0], [0.0, 0.5, 0.8])
    _key(controller, glfw.KEY_S, glfw.RELEASE)
    np.testing.assert_allclose(controller.snapshot()[0], [0.6, 0.5, 0.8])


def test_stop_save_and_quit_are_one_shot_controls():
    controller = TeleopController(0.6, 0.5, 0.8)
    _key(controller, glfw.KEY_D, glfw.PRESS)
    _key(controller, glfw.KEY_SPACE, glfw.PRESS)
    command, save, quit_requested = controller.snapshot()
    np.testing.assert_allclose(command, [0.0, 0.0, 0.0])
    assert not save
    assert not quit_requested

    _key(controller, glfw.KEY_M, glfw.PRESS)
    assert controller.snapshot()[1]
    assert not controller.snapshot()[1]

    _key(controller, glfw.KEY_ESCAPE, glfw.PRESS)
    assert controller.snapshot()[2]


def test_focus_loss_releases_all_held_movement_keys():
    controller = TeleopController(0.6, 0.5, 0.8)
    for key in (glfw.KEY_W, glfw.KEY_A):
        _key(controller, key, glfw.PRESS)

    viewer = object.__new__(MuJoCoTeleopViewer)
    viewer._held_keys = {glfw.KEY_W, glfw.KEY_A}
    viewer._key_callback = controller.on_key
    viewer._on_focus(None, False)

    np.testing.assert_allclose(controller.snapshot()[0], [0.0, 0.0, 0.0])
    assert not viewer._held_keys
