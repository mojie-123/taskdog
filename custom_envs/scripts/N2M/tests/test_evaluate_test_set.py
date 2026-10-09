from __future__ import annotations

import unittest

from custom_envs.scripts.N2M.evaluate_test_set import build_report


def _observation(observation_id: str, scene: str, edge: str) -> dict:
    return {
        "observation_id": observation_id,
        "scene_id": scene,
        "approach_edge": edge,
    }


def _result(
    observation_id: str,
    scene: str,
    edge: str,
    *,
    success: bool,
    rank: int | None,
    status: str = "valid",
) -> dict:
    return {
        "observation_id": observation_id,
        "scene_id": scene,
        "approach_edge": edge,
        "evaluation_status": status,
        "outcome": "lift_success" if success else "candidate_queue_exhausted",
        "success": success,
        "successful_m_number_one_based": rank,
        "attempt_count": 2,
        "attempted_m_count": rank or 3,
        "collision_rejected_m_count": 0,
        "attempt_failure_counts": {"LOCAL_REACH:timeout": 1},
        "last_failure": None if success else "LOCAL_REACH:timeout",
        "abnormal_exit_after_summary": False,
        "wall_time_s": 10.0,
    }


class EvaluationReportTests(unittest.TestCase):
    def test_cumulative_success_and_pending_counts(self):
        observations = [
            _observation("o0", "short_left", "left"),
            _observation("o1", "short_left", "left"),
            _observation("o2", "long_top", "top"),
            _observation("o3", "long_top", "top"),
        ]
        rows = [
            _result("o0", "short_left", "left", success=True, rank=1),
            _result("o1", "short_left", "left", success=True, rank=3),
            _result("o2", "long_top", "top", success=False, rank=None),
        ]
        report = build_report(rows, observations, {"split": "test", "max_m_candidates": 3})
        self.assertEqual(report["planned_observation_count"], 4)
        self.assertEqual(report["evaluated_observation_count"], 3)
        self.assertEqual(report["pending_observation_count"], 1)
        self.assertEqual(report["success_count"], 2)
        self.assertEqual(report["cumulative_success_by_m_number"]["1"]["successes"], 1)
        self.assertEqual(report["cumulative_success_by_m_number"]["2"]["successes"], 1)
        self.assertEqual(report["cumulative_success_by_m_number"]["3"]["successes"], 2)

    def test_invalid_result_is_not_in_valid_rate_denominator(self):
        observations = [
            _observation("o0", "short_left", "left"),
            _observation("o1", "short_left", "left"),
        ]
        rows = [
            _result("o0", "short_left", "left", success=True, rank=1),
            _result(
                "o1", "short_left", "left", success=False, rank=None,
                status="timeout",
            ),
        ]
        report = build_report(rows, observations, {"split": "test", "max_m_candidates": 3})
        self.assertEqual(report["valid_result_count"], 1)
        self.assertEqual(report["invalid_result_count"], 1)
        self.assertEqual(report["success_rate_valid"], 1.0)
        self.assertEqual(report["success_rate_planned"], 0.5)


if __name__ == "__main__":
    unittest.main()
