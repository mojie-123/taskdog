"""Bounded-rate RGB-D target detection while the robot is navigating.

The first implementation deliberately runs in the simulation/navigation
thread.  This keeps MuJoCo rendering, YOLOE and SAM2 on one thread and avoids
stale-frame queues: ``frequency_hz`` is an upper bound, not a promise that a
slow detector will be called concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
from PIL import Image

from custom_envs.scripts.N2M.manifests import append_jsonl, write_json

from .config import TransitDetectionConfig
from .perception import annotate_detection
from .rgbd import RGBDFrame
from .runtime import SysNavComponents
from .types import Detection2D, LocalizedDetection


def _angle_difference(a: float, b: float) -> float:
    return abs(float((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi))


def _localized_dict(item: LocalizedDetection) -> dict:
    return {
        "label": item.detection.label,
        "confidence": float(item.detection.confidence),
        "bbox_xyxy": list(item.detection.bbox_xyxy),
        "position_world": item.position_world.astype(float).tolist(),
        "bbox_min_world": item.bbox_min_world.astype(float).tolist(),
        "bbox_max_world": item.bbox_max_world.astype(float).tolist(),
        "point_count": int(item.point_count),
        "room_id": item.room_id,
    }


@dataclass(frozen=True)
class TransitObservation:
    sequence: int
    kind: str
    navigation_name: str
    robot_pose_world: np.ndarray
    metric_room_id: str | None
    frame: RGBDFrame
    image_path: str
    annotated_image_path: str
    detections: tuple[Detection2D, ...]
    localized_detections: tuple[LocalizedDetection, ...]


@dataclass(frozen=True)
class TransitCandidate:
    observation: TransitObservation
    target: LocalizedDetection


@dataclass(frozen=True)
class ConfirmedTransitCandidate:
    """Two spatially consistent sightings and their fused target estimate."""

    first: TransitCandidate
    confirmation: TransitCandidate
    representative: TransitObservation
    target: LocalizedDetection
    separation_m: float


@dataclass(frozen=True)
class DeferredTransitObservation:
    """One non-confirming view, used only to avoid reviewing that view again."""

    position_world: np.ndarray
    observer_pose_world: np.ndarray
    verdict: str
    reason: str


class ContinuousTransitDetector:
    """Capture, save and infer navigation frames at a bounded rate."""

    def __init__(
        self,
        components: SysNavComponents,
        capture_frame: Callable[[], RGBDFrame],
        output_dir: Path | str,
        target_label: str,
        config: TransitDetectionConfig,
    ):
        self.components = components
        self.capture_frame = capture_frame
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_label = " ".join(str(target_label).strip().lower().split())
        if not self.target_label:
            raise ValueError("target_label must not be empty")
        self.config = config
        self._sequence = 0
        self._last_sample_time = -math.inf
        self._last_sample_pose: np.ndarray | None = None
        self._deferred_observations: list[DeferredTransitObservation] = []
        self._sample_count = 0
        self._candidate_count = 0
        self._confirmed_count = 0
        self._deferred_count = 0
        self._suppressed_count = 0
        self._failure_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _due(self, pose: np.ndarray, now: float, *, force: bool) -> bool:
        if force:
            return True
        if not self.enabled:
            return False
        if now - self._last_sample_time < 1.0 / self.config.frequency_hz:
            return False
        if self._last_sample_pose is None:
            return True
        translation = float(np.linalg.norm(pose[:2] - self._last_sample_pose[:2]))
        yaw = math.degrees(_angle_difference(pose[2], self._last_sample_pose[2]))
        return (
            translation >= self.config.min_translation_m
            or yaw >= self.config.min_yaw_deg
        )

    def _is_deferred_from_similar_view(
        self, position_world: np.ndarray, observer_pose_world: np.ndarray
    ) -> bool:
        """Suppress only a spatial match seen again from essentially the same view."""

        position = np.asarray(position_world, dtype=np.float64).reshape(3).copy()
        observer = np.asarray(observer_pose_world, dtype=np.float64).reshape(3)
        for deferred in self._deferred_observations:
            target_separation = float(
                np.linalg.norm(position[:2] - deferred.position_world[:2])
            )
            if target_separation > self.config.candidate_dedup_radius_m:
                continue
            observer_translation = float(
                np.linalg.norm(observer[:2] - deferred.observer_pose_world[:2])
            )
            observer_yaw = math.degrees(
                _angle_difference(observer[2], deferred.observer_pose_world[2])
            )
            if (
                observer_translation < self.config.reobserve_min_translation_m
                and observer_yaw < self.config.reobserve_min_yaw_deg
            ):
                return True
        return False

    def defer(
        self,
        position_world: np.ndarray,
        observer_pose_world: np.ndarray,
        *,
        verdict: str,
        reason: str,
    ) -> None:
        """Remember a non-confirming view without blacklisting the target region."""

        position = np.asarray(position_world, dtype=np.float64).reshape(3).copy()
        observer = np.asarray(observer_pose_world, dtype=np.float64).reshape(3).copy()
        item = DeferredTransitObservation(
            position_world=position,
            observer_pose_world=observer,
            verdict=str(verdict),
            reason=str(reason),
        )
        self._deferred_observations.append(item)
        self._deferred_count += 1
        append_jsonl(
            self.output_dir / "transit_observations.jsonl",
            {
                "event": "candidate_deferred",
                "position_world": position.tolist(),
                "observer_pose_world": observer.tolist(),
                "verdict": item.verdict,
                "reason": str(reason),
                "policy": "same-view deduplication; a materially new view may retry",
            },
        )
        print(
            f"[SYSNAV][TRANSIT_DEFERRED] verdict={item.verdict} "
            f"position={position.tolist()} observer={observer.tolist()} "
            "retry=after_viewpoint_change",
            flush=True,
        )

    def _save_images(
        self,
        frame: RGBDFrame,
        sequence: int,
        kind: str,
        detections: tuple[Detection2D, ...],
    ) -> tuple[str, str]:
        if not self.config.save_images:
            return "", ""
        stem = f"{kind}_{sequence:06d}"
        raw_path = self.output_dir / "images" / f"{stem}.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame.rgb, mode="RGB").save(raw_path)
        annotated_path = ""
        if detections:
            annotated = np.asarray(frame.rgb).copy()
            for detection in detections:
                annotated = annotate_detection(annotated, detection)
            path = self.output_dir / "annotated" / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotated, mode="RGB").save(path)
            annotated_path = str(path)
        return str(raw_path), annotated_path

    def sample(
        self,
        robot_pose_world: tuple[float, float, float] | np.ndarray,
        *,
        navigation_name: str,
        kind: str = "transit",
        force: bool = False,
        expected_position_world: np.ndarray | None = None,
    ) -> TransitCandidate | None:
        """Take one due sample and return the strongest reviewable target."""

        pose = np.asarray(robot_pose_world, dtype=np.float64).reshape(3)
        now = time.monotonic()
        if not self._due(pose, now, force=force):
            return None
        # Advance the gates before inference. A failed frame therefore cannot
        # create a tight retry loop that freezes locomotion.
        self._last_sample_time = now
        self._last_sample_pose = pose.copy()
        self._sequence += 1
        sequence = self._sequence
        started = time.monotonic()
        try:
            frame = self.capture_frame()
            detections, localized = self.components.infer_observation(frame)
            detections = tuple(detections)
            localized = tuple(localized)
            image_path, annotated_path = self._save_images(
                frame, sequence, kind, detections
            )
        except Exception as exc:
            self._failure_count += 1
            append_jsonl(
                self.output_dir / "transit_observations.jsonl",
                {
                    "event": "sample_failed",
                    "sequence": sequence,
                    "kind": kind,
                    "navigation_name": navigation_name,
                    "robot_pose_world": pose.tolist(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(
                f"[SYSNAV][TRANSIT_DETECTION_FAILED] sequence={sequence} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

        room_id = self.components.room_graph.room_at(float(pose[0]), float(pose[1]))
        observation = TransitObservation(
            sequence=sequence,
            kind=kind,
            navigation_name=str(navigation_name),
            robot_pose_world=pose.copy(),
            metric_room_id=room_id,
            frame=frame,
            image_path=image_path,
            annotated_image_path=annotated_path,
            detections=detections,
            localized_detections=localized,
        )
        target_candidates = [
            item
            for item in localized
            if item.detection.label == self.target_label
            and item.detection.confidence
            >= self.components.config.search.target_accept_confidence
        ]
        suppressed: list[LocalizedDetection] = []
        eligible: list[LocalizedDetection] = []
        for item in target_candidates:
            if self._is_deferred_from_similar_view(item.position_world, pose):
                suppressed.append(item)
            else:
                eligible.append(item)
        self._suppressed_count += len(suppressed)
        if expected_position_world is None:
            eligible.sort(
                key=lambda item: (item.detection.confidence, item.point_count),
                reverse=True,
            )
        else:
            expected = np.asarray(expected_position_world, dtype=np.float64).reshape(3)
            eligible.sort(
                key=lambda item: (
                    float(np.linalg.norm(item.position_world - expected)),
                    -item.detection.confidence,
                    -item.point_count,
                )
            )
        self._sample_count += 1
        append_jsonl(
            self.output_dir / "transit_observations.jsonl",
            {
                "event": "sample",
                "sequence": sequence,
                "kind": kind,
                "navigation_name": navigation_name,
                "robot_pose_world": pose.tolist(),
                "metric_room_id": room_id,
                "image_path": image_path,
                "annotated_image_path": annotated_path,
                "elapsed_s": time.monotonic() - started,
                "detections": [
                    {
                        "label": item.label,
                        "confidence": float(item.confidence),
                        "bbox_xyxy": list(item.bbox_xyxy),
                        "mask_pixels": int(item.mask.sum()),
                    }
                    for item in detections
                ],
                "localized": [_localized_dict(item) for item in localized],
                "same_view_suppressed": len(suppressed),
                "target_candidate": bool(eligible),
            },
        )
        print(
            f"[SYSNAV][TRANSIT_FRAME] sequence={sequence} kind={kind} "
            f"detections={len(detections)} localized={len(localized)} "
            f"candidate={bool(eligible)} same_view_suppressed={len(suppressed)} "
            f"image={image_path}",
            flush=True,
        )
        if not eligible:
            return None
        self._candidate_count += 1
        return TransitCandidate(observation=observation, target=eligible[0])

    def confirm(
        self,
        first: TransitCandidate,
        robot_pose_world: tuple[float, float, float] | np.ndarray,
        *,
        navigation_name: str,
    ) -> ConfirmedTransitCandidate | None:
        """Require a second stationary 3-D detection near the first one."""

        second = self.sample(
            robot_pose_world,
            navigation_name=navigation_name,
            kind="confirmation",
            force=True,
            expected_position_world=first.target.position_world,
        )
        if second is None:
            self.defer(
                first.target.position_world,
                robot_pose_world,
                verdict="not_confirmed",
                reason="stationary confirmation did not detect/localize the target",
            )
            return None
        separation = float(
            np.linalg.norm(second.target.position_world - first.target.position_world)
        )
        room_consistent = (
            first.target.room_id is None
            or second.target.room_id is None
            or first.target.room_id == second.target.room_id
        )
        if separation > self.config.confirmation_distance_m or not room_consistent:
            self.defer(
                first.target.position_world,
                robot_pose_world,
                verdict="not_confirmed",
                reason=(
                    f"confirmation mismatch: separation={separation:.3f}m "
                    f"rooms=({first.target.room_id},{second.target.room_id})"
                ),
            )
            return None

        observations = (first, second)
        representative = max(
            observations,
            key=lambda item: (
                item.target.detection.confidence,
                item.target.point_count,
            ),
        )
        positions = np.stack([item.target.position_world for item in observations])
        minimum = np.minimum(
            first.target.bbox_min_world, second.target.bbox_min_world
        )
        maximum = np.maximum(
            first.target.bbox_max_world, second.target.bbox_max_world
        )
        fused = LocalizedDetection(
            detection=representative.target.detection,
            position_world=np.median(positions, axis=0),
            bbox_min_world=minimum,
            bbox_max_world=maximum,
            point_count=first.target.point_count + second.target.point_count,
            room_id=second.target.room_id or first.target.room_id,
        )
        self._confirmed_count += 1
        append_jsonl(
            self.output_dir / "transit_observations.jsonl",
            {
                "event": "candidate_confirmed",
                "first_sequence": first.observation.sequence,
                "confirmation_sequence": second.observation.sequence,
                "separation_m": separation,
                "fused_target": _localized_dict(fused),
            },
        )
        print(
            f"[SYSNAV][TRANSIT_CONFIRMED] first={first.observation.sequence} "
            f"confirmation={second.observation.sequence} "
            f"separation={separation:.3f}m",
            flush=True,
        )
        return ConfirmedTransitCandidate(
            first=first,
            confirmation=second,
            representative=representative.observation,
            target=fused,
            separation_m=separation,
        )

    def save_summary(self) -> None:
        write_json(
            self.output_dir / "transit_summary.json",
            {
                "schema_version": 2,
                "enabled": self.enabled,
                "target_label": self.target_label,
                "frequency_hz": self.config.frequency_hz,
                "save_images": self.config.save_images,
                "sample_count": self._sample_count,
                "candidate_count": self._candidate_count,
                "confirmed_count": self._confirmed_count,
                "deferred_count": self._deferred_count,
                "same_view_suppressed_count": self._suppressed_count,
                "failure_count": self._failure_count,
                "deferred_observations": [
                    {
                        "position_world": item.position_world.tolist(),
                        "observer_pose_world": item.observer_pose_world.tolist(),
                        "verdict": item.verdict,
                        "reason": item.reason,
                    }
                    for item in self._deferred_observations
                ],
            },
        )
