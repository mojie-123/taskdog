from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from custom_envs.scripts.navigation import navigate_mujoco_n2m as n2m
from custom_envs.utils.ros2_bridge import IsaacROS2Bridge


def _live_process_stub():
    return SimpleNamespace(poll=lambda: None)


def test_next_goal_waits_for_cancel_acknowledgement(monkeypatch):
    bridge = IsaacROS2Bridge()
    bridge._proc = _live_process_stub()
    sent = []
    monkeypatch.setattr(bridge, "_send", sent.append)
    monkeypatch.setattr("custom_envs.utils.ros2_bridge.time.sleep", lambda _: None)

    bridge.cancel_goal()
    request_id = bridge._pending_cancel_id
    bridge._feedback_q.put({"cancel_complete": request_id})
    bridge.send_goal(3.375, 5.025)

    assert [message["type"] for message in sent] == ["cancel_goal", "goal"]
    assert bridge._pending_cancel_id is None


def test_cancel_barrier_reports_missing_acknowledgement(monkeypatch):
    bridge = IsaacROS2Bridge()
    bridge._proc = _live_process_stub()
    monkeypatch.setattr(bridge, "_send", lambda _: None)
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        "custom_envs.utils.ros2_bridge.time.monotonic",
        lambda: next(ticks),
    )
    monkeypatch.setattr("custom_envs.utils.ros2_bridge.time.sleep", lambda _: None)

    bridge.cancel_goal()
    with pytest.raises(RuntimeError, match="cancellation was not acknowledged"):
        bridge.wait_for_cancel(timeout=0.5)


def test_plan_path_uses_plan_only_protocol_and_returns_matching_result(monkeypatch):
    bridge = IsaacROS2Bridge()
    bridge._proc = _live_process_stub()
    sent = []
    monkeypatch.setattr(bridge, "_send", sent.append)
    bridge._feedback_q.put(
        {
            "plan_result": {
                "request_id": 1,
                "success": True,
                "path": [[3.0, 5.0], [7.0, 6.0], [11.0, 2.0]],
                "error": "",
            }
        }
    )

    path = bridge.plan_path((3.0, 5.0), (11.0, 2.0), timeout=1.0)

    assert path == [[3.0, 5.0], [7.0, 6.0], [11.0, 2.0]]
    assert sent == [
        {
            "type": "plan_path",
            "request_id": 1,
            "start": [3.0, 5.0],
            "goal": [11.0, 2.0],
        }
    ]


def test_teleport_planner_retries_once_before_failing(monkeypatch):
    class FakeEnv:
        dt = 0.02

        def get_robot_state(self):
            return {
                "pos": np.array([0.0, 0.0, 0.58]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "lin_vel": np.zeros(3),
                "ang_vel": np.zeros(3),
                "yaw": 0.0,
            }

        def step(self, unused_action):
            pass

    class FakeBridge:
        def __init__(self):
            self.goals = []
            self.cancel_count = 0

        def update_robot_pose(self, *unused_args):
            pass

        def send_goal(self, x, y):
            self.goals.append((x, y))

        def cancel_goal(self):
            self.cancel_count += 1

        def get_nav_status(self):
            return False, False

        def get_global_path(self):
            return []

    ticks = iter(float(value) for value in range(20))
    monkeypatch.setattr(n2m.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        n2m,
        "RealtimePacer",
        lambda unused_period: SimpleNamespace(wait=lambda: None),
    )

    bridge = FakeBridge()
    with pytest.raises(RuntimeError, match="2 attempts"):
        n2m._follow_nav2_path_by_teleport(
            name="TEST",
            env=FakeEnv(),
            bridge=bridge,
            interactive=SimpleNamespace(quit=False),
            goal_xy=(1.0, 0.0),
            final_yaw=0.0,
            max_steps=8,
            teleport_every_steps=1,
            teleport_spacing_m=0.1,
            plan_timeout_s=1.0,
            weld=SimpleNamespace(active=True),
            plan_retries=1,
        )

    assert bridge.goals == [(1.0, 0.0), (1.0, 0.0)]
    assert bridge.cancel_count == 1
