from __future__ import annotations

import subprocess
import sys
import threading
import unittest

from custom_envs.utils.moveit_bridge import MoveItROS2Bridge
from custom_envs.utils.ros2_bridge import IsaacROS2Bridge


_DUMMY_CHILD = """
import json
import sys

for line in sys.stdin:
    try:
        message = json.loads(line)
    except Exception:
        continue
    if message.get("type") == "stop":
        break
"""


class BridgeCleanupTests(unittest.TestCase):
    def _attach_dummy_child(self, bridge) -> tuple[subprocess.Popen, threading.Thread]:
        proc = subprocess.Popen(
            [sys.executable, "-c", _DUMMY_CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        bridge._proc = proc
        reader = threading.Thread(
            target=bridge._stdout_reader,
            args=(proc,),
            daemon=True,
        )
        bridge._reader_thread = reader
        reader.start()
        return proc, reader

    def _assert_graceful_idempotent_stop(self, bridge) -> None:
        proc, reader = self._attach_dummy_child(bridge)
        bridge.stop()
        self.assertIsNotNone(proc.poll())
        self.assertFalse(reader.is_alive())
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)
        proc.stderr.close()
        self.assertIsNone(bridge._proc)
        self.assertIsNone(bridge._reader_thread)
        bridge.stop()

    def test_nav_bridge_stop_reaps_child_and_joins_reader(self):
        self._assert_graceful_idempotent_stop(IsaacROS2Bridge())

    def test_moveit_bridge_stop_reaps_child_and_joins_reader(self):
        self._assert_graceful_idempotent_stop(MoveItROS2Bridge())


if __name__ == "__main__":
    unittest.main()
