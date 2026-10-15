from __future__ import annotations

import numpy as np

from custom_envs.mujoco import env as env_module


class _FakeRenderer:
    instances = []

    def __init__(self, model, *, height, width):
        self.model = model
        self.height = height
        self.width = width
        self.depth_enabled = False
        self.close_count = 0
        self.update_calls = []
        type(self).instances.append(self)

    def update_scene(self, data, *, camera):
        self.update_calls.append((data, camera, self.depth_enabled))

    def render(self):
        if self.depth_enabled:
            return np.ones((self.height, self.width), dtype=np.float32)
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def enable_depth_rendering(self):
        self.depth_enabled = True

    def disable_depth_rendering(self):
        self.depth_enabled = False

    def close(self):
        self.close_count += 1


def test_render_camera_reuses_one_renderer_until_environment_close(monkeypatch):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr(env_module.mujoco, "Renderer", _FakeRenderer)
    monkeypatch.setattr(env_module.mujoco, "mj_name2id", lambda *args: 1)

    env = env_module.MuJoCoEnv.__new__(env_module.MuJoCoEnv)
    env._model = object()
    env._data = object()
    env._cam_h = 3
    env._cam_w = 4
    env._camera_renderer = None
    env._viewer = None

    first_rgb, first_depth = env.render_camera("wrist_cam")
    second_rgb, second_depth = env.render_camera("front_cam")

    assert len(_FakeRenderer.instances) == 1
    renderer = _FakeRenderer.instances[0]
    assert first_rgb.shape == second_rgb.shape == (3, 4, 3)
    assert first_depth.shape == second_depth.shape == (3, 4)
    assert renderer.depth_enabled is False
    assert [call[1] for call in renderer.update_calls] == [
        "wrist_cam",
        "wrist_cam",
        "front_cam",
        "front_cam",
    ]

    env.close()
    env.close()
    assert renderer.close_count == 1
    assert env._camera_renderer is None
