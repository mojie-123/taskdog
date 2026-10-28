from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from custom_envs.scripts.N2M.data_generation.camera_geometry import CameraIntrinsics
from custom_envs.scripts.map.nav2_map_io import Nav2Map, load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap
from custom_envs.scripts.sysnav.async_vlm import AsyncRoomVLMManager
from custom_envs.scripts.sysnav.config import load_sysnav_config
from custom_envs.scripts.sysnav.coverage import (
    CoveragePlanner,
    CoverageViewpoint,
    RoomCoveragePlan,
)
from custom_envs.scripts.sysnav.object_verification import ObjectVerificationManager
from custom_envs.scripts.sysnav.rgbd import DepthLocalizer, RGBDFrame
from custom_envs.scripts.sysnav.room_graph import AnonymousRoomGraph
from custom_envs.scripts.sysnav.runtime import SysNavComponents, build_sysnav_components
from custom_envs.scripts.sysnav.scene_memory import SceneMemory, ViewpointRecord
from custom_envs.scripts.sysnav.search_policy import SysNavSearchPolicy
from custom_envs.scripts.sysnav.search_runner import (
    RoomImageEvidence,
    SysNavSearchRunner,
)
from custom_envs.scripts.sysnav.target_reobservation import (
    plan_target_reobservation_pose,
)
from custom_envs.scripts.sysnav.transit_detection import (
    ContinuousTransitDetector,
    TransitReobserveRequested,
    TransitTargetFound,
)
from custom_envs.scripts.sysnav.types import (
    Detection2D,
    EarlyStopDecision,
    LocalizedDetection,
    RoomDescription,
    RoomSelection,
    SearchInstruction,
    TargetVerification,
)
from custom_envs.scripts.sysnav.vlm import QwenVLMClient
from custom_envs.scripts.sysnav.view_projection import project_bbox_world


def _components():
    config = load_sysnav_config()
    room_map = RoomMap.load(config.search.room_map_manifest)
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    graph = AnonymousRoomGraph(room_map)
    planner = CoveragePlanner(room_map, nav_map, config.coverage)
    return config, room_map, graph, planner


def _localized(position, confidence=0.8):
    position = np.asarray(position, dtype=np.float64)
    detection = Detection2D(
        label="cube",
        confidence=confidence,
        bbox_xyxy=(1.0, 1.0, 3.0, 3.0),
        mask=np.ones((4, 4), dtype=bool),
    )
    return LocalizedDetection(
        detection=detection,
        position_world=position,
        bbox_min_world=position - 0.05,
        bbox_max_world=position + 0.05,
        point_count=50,
        room_id="room_0",
    )


def test_config_and_instruction_prompts(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.delenv("QWEN_MODEL_LITE", raising=False)
    config = load_sysnav_config()
    config.validate_files(require_models=False)
    instruction = SearchInstruction(
        raw_text="去客厅茶几上拿 Cube",
        target_object=" Cube ",
        room_condition="Living Room",
        spatial_condition="on the coffee table",
        anchor_object="Coffee Table",
    )
    assert instruction.target_object == "cube"
    assert instruction.detector_prompts == ("cube", "coffee table")
    assert instruction.needs_vlm_verification
    assert config.vlm.reasoning_model == "qwen3.6-plus"
    assert config.vlm.fast_model == "qwen3.6-flash"
    assert config.search.early_stop_max_current_images == 4
    assert config.search.room_semantic_max_images == 6
    assert config.search.room_semantic_refresh_min_new_images == 2
    assert config.search.room_semantic_stable_min_ordinary_views == 2
    assert config.search.room_semantic_stable_confidence == 0.75
    assert config.search.object_vlm_min_mask_pixels == 500
    assert config.search.provisional_reobserve_enabled
    assert config.search.provisional_reobserve_min_observations == 3
    assert config.search.provisional_reobserve_min_mask_pixels == 80
    assert config.search.provisional_reobserve_preferred_standoff_m == 1.25
    assert config.search.provisional_reobserve_max_attempts == 2
    assert config.coverage.depth_ray_stride_px == 8
    assert config.transit_detection.enabled
    assert config.transit_detection.frequency_hz == 2.0
    assert config.transit_detection.candidate_dedup_radius_m == 0.30
    assert config.transit_detection.reobserve_min_translation_m == 0.50
    assert config.transit_detection.reobserve_min_yaw_deg == 20.0
    assert config.transit_detection.save_images


def test_vlm_model_environment_overrides(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL", "room-model")
    monkeypatch.setenv("QWEN_MODEL_LITE", "object-model")

    config = load_sysnav_config()

    assert config.vlm.reasoning_model == "room-model"
    assert config.vlm.fast_model == "object-model"


def test_joint_vlm_rejection_keeps_room_verdict_revisable(monkeypatch):
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    monkeypatch.setattr(
        client,
        "_parse",
        lambda *args, **kwargs: SimpleNamespace(
            is_target=False,
            reason="the requested relation is not visible",
        ),
    )
    instruction = SearchInstruction(
        raw_text="go to the coffee table and get the cube",
        target_object="cube",
        room_condition="living room",
        spatial_condition="on",
        anchor_object="coffee table",
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    target = client.verify_spatial_target(
        instruction,
        "cube",
        image,
        (),
        room_label="living room",
    )
    anchor = client.verify_anchor_object(
        instruction,
        "coffee table",
        image,
        room_label="living room",
    )

    for decision in (target, anchor):
        assert not decision.is_target
        assert decision.verdict == "rejected"
        assert decision.visual_verdict == "rejected"
        assert decision.room_verdict == "uncertain"
        assert decision.reason == "the requested relation is not visible"


def test_transit_detector_saves_and_confirms_spatially_consistent_frames(tmp_path):
    config = load_sysnav_config()
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    positions = iter(([2.0, 2.0, 0.5], [2.08, 2.0, 0.5]))

    class Components:
        def __init__(self):
            self.config = SimpleNamespace(search=config.search)
            self.room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            target = _localized(next(positions), confidence=0.15)
            return (target.detection,), (target,)

    detector = ContinuousTransitDetector(
        Components(),
        lambda: frame,
        tmp_path,
        "cube",
        config.transit_detection,
    )
    first = detector.sample(
        (1.0, 1.0, 0.0), navigation_name="SEARCH", force=True
    )
    assert first is not None
    confirmed = detector.confirm(
        first, (1.0, 1.0, 0.0), navigation_name="SEARCH"
    )
    assert confirmed is not None
    assert confirmed.separation_m == pytest.approx(0.08)
    np.testing.assert_allclose(confirmed.target.position_world, [2.04, 2.0, 0.5])
    assert (tmp_path / "images" / "transit_000001.png").is_file()
    assert (tmp_path / "images" / "confirmation_000002.png").is_file()
    assert (tmp_path / "annotated" / "transit_000001.png").is_file()
    detector.save_summary()
    summary = json.loads((tmp_path / "transit_summary.json").read_text())
    assert summary["sample_count"] == 2
    assert summary["candidate_count"] == 2
    assert summary["confirmed_count"] == 1


def test_transit_confirmation_failure_defers_only_the_same_view(tmp_path):
    config = load_sysnav_config()
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    responses = iter((True, False, True, True))

    class Components:
        def __init__(self):
            self.config = SimpleNamespace(search=config.search)
            self.room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            if not next(responses):
                return (), ()
            target = _localized([2.0, 2.0, 0.5], confidence=0.9)
            return (target.detection,), (target,)

    detector = ContinuousTransitDetector(
        Components(),
        lambda: frame,
        tmp_path,
        "cube",
        config.transit_detection,
    )
    original_pose = np.array([1.0, 1.0, 0.0])
    first = detector.sample(original_pose, navigation_name="SEARCH", force=True)
    assert first is not None
    assert detector.confirm(
        first, original_pose, navigation_name="SEARCH"
    ) is None

    # The failed stationary confirmation is not immediately reviewed again
    # from the same pose.
    assert detector.sample(
        original_pose, navigation_name="SEARCH", force=True
    ) is None

    # It remains eligible after a meaningful viewpoint change, even though
    # the estimated target location is exactly the same.
    moved_pose = np.array([1.6, 1.0, 0.0])
    assert detector.sample(
        moved_pose, navigation_name="SEARCH", force=True
    ) is not None

    detector.save_summary()
    summary = json.loads((tmp_path / "transit_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["deferred_count"] == 1
    assert summary["same_view_suppressed_count"] == 1
    assert "rejected_positions" not in summary


def test_transit_streams_observations_without_interrupting_for_raw_candidate(tmp_path):
    config = load_sysnav_config()
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    target = _localized([2.0, 2.0, 0.5], confidence=0.9)

    class Components:
        room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            return (target.detection,), (target,)

    received = []
    detector = ContinuousTransitDetector(
        Components(), lambda: frame, tmp_path, "cube", config.transit_detection
    )
    detector.set_observation_callback(received.append, interrupt_on_candidate=False)

    candidate = detector.sample(
        (1.0, 1.0, 0.0), navigation_name="SEARCH", force=True
    )

    assert candidate is None
    assert len(received) == 1
    assert received[0].localized_detections == (target,)


def test_transit_propagates_stable_target_reobserve_request(tmp_path):
    config = load_sysnav_config()
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    target = _localized([2.0, 2.0, 0.5], confidence=0.9)
    request = TransitReobserveRequested(
        object_id="object_0",
        room_id="room_0",
        position_world=target.position_world,
        best_image_score=200,
        observation_count=3,
    )

    class Components:
        room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            return (target.detection,), (target,)

    detector = ContinuousTransitDetector(
        Components(), lambda: frame, tmp_path, "cube", config.transit_detection
    )

    def request_reobserve(unused_observation):
        raise request

    detector.set_observation_callback(
        request_reobserve,
        interrupt_on_candidate=False,
    )

    with pytest.raises(TransitReobserveRequested) as raised:
        detector.sample((1.0, 1.0, 0.0), navigation_name="SEARCH", force=True)
    assert raised.value is request


def test_transit_stream_keeps_clean_object_evidence_when_debug_saving_is_off(
    tmp_path,
):
    config = load_sysnav_config()
    transit_config = replace(config.transit_detection, save_images=False)
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    target = _localized([2.0, 2.0, 0.5], confidence=0.9)

    class Components:
        room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            return (target.detection,), (target,)

    received = []
    detector = ContinuousTransitDetector(
        Components(), lambda: frame, tmp_path, "cube", transit_config
    )
    detector.set_observation_callback(received.append, interrupt_on_candidate=False)
    detector.sample((1.0, 1.0, 0.0), navigation_name="SEARCH", force=True)

    assert len(received) == 1
    assert received[0].image_path
    assert Path(received[0].image_path).is_file()
    assert received[0].annotated_image_path == ""


def test_transit_raises_only_after_async_object_review_is_accepted(tmp_path):
    config = load_sysnav_config()
    frame = RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(8, 8, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    target = _localized([2.0, 2.0, 0.5], confidence=0.9)

    class Components:
        room_graph = SimpleNamespace(room_at=lambda x, y: "room_0")

        def infer_observation(self, unused_frame):
            return (target.detection,), (target,)

    accepted = iter((None, target))
    detector = ContinuousTransitDetector(
        Components(), lambda: frame, tmp_path, "cube", config.transit_detection
    )
    detector.set_observation_callback(
        lambda unused_observation: None,
        verified_target_provider=lambda: next(accepted),
        interrupt_on_candidate=False,
    )

    with pytest.raises(TransitTargetFound) as raised:
        detector.sample((1.0, 1.0, 0.0), navigation_name="SEARCH", force=True)
    assert raised.value.target is target


def test_provisional_target_reobserve_requires_repeated_small_stable_detection():
    config = load_sysnav_config()
    runner = object.__new__(SysNavSearchRunner)
    runner.components = SimpleNamespace(config=config)
    runner._instruction = SearchInstruction("find cube", "cube")
    runner._verified_target = None
    runner._reobserve_in_progress = False
    runner._reobserve_attempts = {}
    latest = _localized([5.34, 3.65, 0.47], confidence=0.6)
    record = SimpleNamespace(
        object_id="object_1",
        room_id="room_0",
        label="cube",
        confirmed_label="",
        is_considered_strong=False,
        query_in_flight=False,
        type_verdict="",
        observation_count=3,
        best_image_score=258,
        best_confidence=0.6,
        position_world=np.array([5.32, 3.65, 0.47]),
        latest_detection=latest,
    )

    request = runner._provisional_reobserve_request((record,))

    assert request is not None
    assert request.object_id == "object_1"
    assert request.best_image_score == 258
    record.observation_count = 2
    assert runner._provisional_reobserve_request((record,)) is None
    record.observation_count = 3
    record.best_image_score = config.search.object_vlm_min_mask_pixels
    assert runner._provisional_reobserve_request((record,)) is None


def test_target_reobserve_pose_is_searchable_free_and_faces_target():
    config, room_map, _, _ = _components()
    nav_map = load_nav2_map(config.search.nav2_map_yaml)
    target = np.array([5.3204, 3.6485, 0.475])
    current = np.array([7.42, 4.77, 0.0])

    planned = plan_target_reobservation_pose(
        room_map,
        nav_map,
        "room_0",
        target,
        current,
        preferred_standoff_m=1.25,
        min_standoff_m=0.85,
        max_standoff_m=1.65,
    )

    assert planned is not None
    assert nav_map.free_mask[planned.grid_cell]
    assert room_map.searchable_mask("room_0")[planned.grid_cell]
    assert 0.85 <= planned.target_distance_m <= 1.65
    expected_yaw = np.arctan2(
        target[1] - planned.pose[1], target[0] - planned.pose[0]
    )
    assert planned.pose[2] == pytest.approx(expected_yaw)


def test_navigation_resumes_original_goal_after_target_reobserve():
    request = TransitReobserveRequested(
        object_id="object_1",
        room_id="room_0",
        position_world=np.array([5.3, 3.6, 0.5]),
        best_image_score=250,
        observation_count=3,
    )
    calls = []

    class Navigator:
        def move_to(self, pose, *, name):
            calls.append((pose, name))
            if len(calls) == 1:
                raise request
            return {"done": True}

    runner = object.__new__(SysNavSearchRunner)
    runner.navigator = Navigator()
    handled = []
    runner._perform_target_reobservation = handled.append
    runner._record_event = lambda *args, **kwargs: None

    result = runner._navigate_with_transit_detection((1.0, 2.0, 0.3), name="SEARCH")

    assert result == {"done": True}
    assert handled == [request]
    assert calls == [
        ((1.0, 2.0, 0.3), "SEARCH"),
        ((1.0, 2.0, 0.3), "SEARCH"),
    ]


def test_early_stop_vlm_receives_every_selected_room_view():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    captured = {}

    def fake_parse(model, response_format, messages):
        captured["messages"] = messages
        return SimpleNamespace(switch_room=False, reason="current room remains stronger")

    client._parse = fake_parse
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    decision = client.should_switch_room(
        SearchInstruction("find cube", "cube"),
        "room_0",
        (image, image, image),
        "room_1",
        (image,),
    )

    assert not decision.switch_room
    content = captured["messages"][1]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    assert len(image_items) == 4
    text = "\n".join(item["text"] for item in content if item["type"] == "text")
    assert "Current room view 3/3" in text
    assert "Nearby room view 1/1" in text


def test_vlm_target_verification_rejects_identity_before_relation():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    calls = []

    def fake_parse(model, response_format, messages):
        calls.append((response_format.__name__, messages))
        return SimpleNamespace(
            verdict="rejected",
            reason="the outlined region is a wall panel, not a cube",
        )

    client._parse = fake_parse
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    decision = client.verify_target(
        SearchInstruction(
            "去客厅茶几上拿 cube",
            "cube",
            room_condition="客厅",
            spatial_condition="上",
            anchor_object="茶几",
        ),
        "cube",
        image,
        candidate_crop_rgb=image,
    )

    assert not decision.is_target
    assert decision.verdict == "rejected"
    assert decision.reason.startswith("identity rejected:")
    assert len(calls) == 1
    assert calls[0][0] == "_TargetIdentityResult"
    system_prompt = calls[0][1][0]["content"]
    assert "untrusted detector proposal" in system_prompt
    assert "Do not turn missing evidence into rejection" in system_prompt
    image_items = [
        item
        for item in calls[0][1][1]["content"]
        if item["type"] == "image_url"
    ]
    assert len(image_items) == 2


def test_vlm_target_verification_runs_relation_only_after_identity_passes():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    responses = iter(
        (
            SimpleNamespace(
                verdict="confirmed",
                reason="the outlined object is visibly a cube",
            ),
            SimpleNamespace(
                verdict="rejected",
                reason="the outlined cube is not visibly on the table",
            ),
        )
    )
    calls = []

    def fake_parse(model, response_format, messages):
        calls.append((response_format.__name__, messages))
        return next(responses)

    client._parse = fake_parse
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    decision = client.verify_target(
        SearchInstruction(
            "去客厅茶几上拿 cube",
            "cube",
            room_condition="客厅",
            spatial_condition="上",
            anchor_object="茶几",
        ),
        "cube",
        image,
        candidate_crop_rgb=image,
    )

    assert not decision.is_target
    assert decision.verdict == "rejected"
    assert "identity confirmed:" in decision.reason
    assert "constraints rejected:" in decision.reason
    assert [item[0] for item in calls] == [
        "_TargetIdentityResult",
        "_TargetRelationResult",
    ]
    relation_prompt = calls[1][1][0]["content"]
    assert "identity has already been confirmed by Stage 1" in relation_prompt
    assert "Do not re-identify, question, discuss, or contradict" in relation_prompt
    assert "Room type is deliberately evaluated elsewhere" in relation_prompt
    assert "do not infer, discuss, confirm, or reject the room type" in relation_prompt
    assert "do not substitute another" in relation_prompt
    assert "do not rename an arbitrary support surface" in relation_prompt
    assert "Never turn missing evidence" in relation_prompt
    relation_context = calls[1][1][1]["content"][0]["text"]
    assert "room_condition" not in relation_context
    assert "runtime_room_description" not in relation_context


def test_vlm_target_verification_preserves_uncertain_identity():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    calls = []

    def fake_parse(model, response_format, messages):
        calls.append((response_format.__name__, messages))
        return SimpleNamespace(
            verdict="uncertain",
            reason="the distant square may be a cube but depth is not visible",
        )

    client._parse = fake_parse
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    decision = client.verify_target(
        SearchInstruction("find cube", "cube"),
        "cube",
        image,
        candidate_crop_rgb=image,
    )

    assert not decision.is_target
    assert decision.is_uncertain
    assert decision.reason.startswith("identity uncertain:")
    assert len(calls) == 1
    system_prompt = calls[0][1][0]["content"]
    assert "artificial annotation overlay" in system_prompt
    assert "do not require multiple visible faces" in system_prompt
    assert "do not use assumed metric dimensions" in system_prompt
    assert "touches an image boundary" in system_prompt
    assert "never confirm from the crop alone" in system_prompt
    assert "flag is authoritative visibility metadata" in system_prompt
    assert "MANDATORY CLIPPING RULE" in system_prompt


def test_vlm_clipping_guard_downgrades_rejection_to_uncertain():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)

    def fake_parse(model, response_format, messages):
        return SimpleNamespace(
            verdict="rejected",
            reason="the partial patch looks flat rather than cubic",
        )

    client._parse = fake_parse
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    decision = client.verify_target(
        SearchInstruction("find cube", "cube"),
        "cube",
        image,
        candidate_crop_rgb=image,
        candidate_clipped_by_image_edge=True,
    )

    assert not decision.is_target
    assert decision.is_uncertain
    assert "uncertain after clipping guard" in decision.reason
    assert "not a conclusive mismatch" in decision.reason


def test_room_description_treats_detector_labels_as_unverified_hints():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    calls = []

    def fake_parse(model, response_format, messages):
        calls.append(messages)
        return SimpleNamespace(
            room_type="unknown",
            visual_cues=[],
            confidence=0.1,
        )

    client._parse = fake_parse
    client.describe_room("room_0", np.zeros((32, 32, 3), dtype=np.uint8), ["cube"])

    system_prompt = calls[0][0]["content"]
    user_text = calls[0][1]["content"][0]["text"]
    assert "never claim an object is present merely" in system_prompt
    assert "small movable task targets" in system_prompt
    assert "unverified_detector_proposals=['cube']" in user_text


def test_multiview_room_description_jointly_sends_every_view():
    config = load_sysnav_config()
    client = QwenVLMClient(config.vlm)
    calls = []

    def fake_parse(model, response_format, messages):
        calls.append(messages)
        return SimpleNamespace(
            room_type="living room",
            visual_cues=["sofa", "low central table"],
            confidence=0.9,
        )

    client._parse = fake_parse
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = client.describe_room_multiview(
        "room_0",
        (image, image, image),
        observation_kinds=("peek", "ordinary", "ordinary"),
    )

    assert result.room_type == "living room"
    content = calls[0][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 3
    text = "\n".join(item["text"] for item in content if item["type"] == "text")
    assert "same anonymous geometric room" in text
    assert "observation_kind=peek" in text
    assert "observation_kind=ordinary" in text


def test_component_factory_is_lazy_and_does_not_require_api_key():
    components = build_sysnav_components(enable_vlm=True)
    assert set(components.room_graph.nodes) == {"room_0", "room_1", "room_2"}
    assert components.vlm is not None
    assert components.vlm._client is None
    assert components.perception._yolo is None


def test_anonymous_room_graph_resolves_topology_and_anchor():
    _, _, graph, _ = _components()
    assert set(graph.nodes) == {"room_0", "room_1", "room_2"}
    path, distance = graph.shortest_path("room_1", "room_2")
    assert path == ["room_1", "room_2"]
    assert distance > 0.0
    anchor = graph.execution_anchor("room_0", "room_1")
    assert graph.room_at(anchor[0], anchor[1]) == "room_1"


def test_coverage_plan_uses_depth_rays_and_stops_at_map_obstacle(tmp_path):
    config = replace(
        load_sysnav_config().coverage,
        max_view_range_m=10.0,
        depth_ray_stride_px=1,
    )
    shape = (11, 11)
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 5] = True
    nav_map = Nav2Map(
        yaml_path=tmp_path / "map.yaml",
        image_path=tmp_path / "map.pgm",
        image=np.zeros(shape, dtype=np.uint8),
        resolution=1.0,
        origin=(0.0, 0.0),
        free_mask=~occupied,
        occupied_mask=occupied,
        unknown_mask=np.zeros(shape, dtype=bool),
    )
    target_cells = np.asarray(
        [[5, 2], [5, 3], [5, 4], [5, 6], [5, 7], [5, 8]],
        dtype=np.int32,
    )
    target_world = np.asarray(
        [nav_map.grid_to_world(int(row), int(col)) for row, col in target_cells]
    )
    viewpoint = CoverageViewpoint(
        index=0,
        room_id="room_0",
        pose=(1.5, 5.5, 0.0),
        grid_cell=(5, 1),
        # Deliberately claim that every target is theoretically visible. The
        # measured-depth update must still stop at the occupied wall.
        visible_target_indices=np.arange(len(target_cells), dtype=np.int32),
    )
    plan = RoomCoveragePlan(
        "room_0", [viewpoint], target_cells, target_world, config, nav_map
    )
    camera_rotation_world = np.asarray(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    frame = RGBDFrame(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        depth=np.full((1, 1), 9.0, dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(1, 1, 90.0),
        camera_position_world=np.array([1.5, 5.5, 0.8]),
        camera_rotation_world=camera_rotation_world,
        camera_name="test",
        timestamp_s=1.0,
    )

    update = plan.mark_observed(0, frame)

    assert update.theoretical_target_count == 6
    assert update.observed_target_count == 3
    assert update.newly_observed_target_count == 3
    assert plan.covered.tolist() == [True, True, True, False, False, False]
    assert plan.visited_viewpoints == {0}


def test_depth_mask_localizes_without_simulator_segmentation():
    config, room_map, _, _ = _components()
    perception = replace(config.perception, min_localization_points=3)
    frame = RGBDFrame(
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.full((5, 5), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(5, 5, 90.0),
        camera_position_world=np.array([2.0, 2.0, 2.0]),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    detection = Detection2D("cube", 0.9, (1.0, 1.0, 4.0, 4.0), mask)
    localized = DepthLocalizer(perception, room_map).localize(frame, detection)
    assert localized is not None
    np.testing.assert_allclose(localized.position_world, [2.0, 2.0, 0.0], atol=1e-6)
    assert localized.room_id == "room_0"


def test_world_bbox_projects_into_saved_mujoco_camera_view():
    viewpoint = ViewpointRecord(
        viewpoint_id="viewpoint_0",
        room_id="room_0",
        pose_world=(0.0, 0.0, 0.0),
        timestamp_s=1.0,
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_intrinsics={
            "width": 100,
            "height": 100,
            "fx": 50.0,
            "fy": 50.0,
            "cx": 50.0,
            "cy": 50.0,
        },
    )
    projected = project_bbox_world(
        np.array([-0.2, -0.2, -2.2]),
        np.array([0.2, 0.2, -1.8]),
        viewpoint,
    )

    assert projected is not None
    x1, y1, x2, y2 = projected
    assert x1 < 50.0 < x2
    assert y1 < 50.0 < y2


def test_scene_memory_merges_repeated_object_observations(tmp_path):
    config, _, graph, _ = _components()
    memory = SceneMemory(graph, merge_distance_m=config.search.object_merge_distance_m)
    memory.add_viewpoint("room_0", (1.0, 1.0, 0.0), [_localized([2.0, 2.0, 0.8])])
    memory.add_viewpoint("room_0", (1.2, 1.0, 0.0), [_localized([2.1, 2.0, 0.8])])
    assert len(memory.viewpoints) == 2
    assert len(memory.objects) == 1
    record = next(iter(memory.objects.values()))
    assert record.observation_count == 2
    assert memory.rooms["room_0"].observed_labels == ("cube",)
    output = tmp_path / "scene_memory.json"
    memory.save(output)
    assert '"room_0"' in output.read_text(encoding="utf-8")


def test_object_instance_review_is_deduplicated_and_reopens_on_room_label_change(
    tmp_path,
):
    config, _, graph, _ = _components()
    memory = SceneMemory(graph)
    image_path = tmp_path / "view.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:44, 12:44] = True
    detection = Detection2D(
        label="cube",
        confidence=0.9,
        bbox_xyxy=(12.0, 12.0, 44.0, 44.0),
        mask=mask,
    )
    localized = LocalizedDetection(
        detection=detection,
        position_world=np.array([2.0, 2.0, 0.8]),
        bbox_min_world=np.array([1.95, 1.95, 0.75]),
        bbox_max_world=np.array([2.05, 2.05, 0.85]),
        point_count=1024,
        room_id="room_0",
    )
    viewpoint = memory.add_viewpoint(
        "room_0", (1.0, 1.0, 0.0), [localized], image_path=str(image_path)
    )
    record = memory.objects[viewpoint.object_ids[0]]
    memory.update_room_description(
        "room_0", RoomDescription("living room", ("sofa",), 0.6),
        evidence_count=1, ordinary_evidence_count=1,
    )

    class Reasoner:
        def __init__(self):
            self.type_calls = 0
            self.spatial_calls = 0

        def verify_object_type(self, *unused_args, **unused_kwargs):
            self.type_calls += 1
            return TargetVerification(True, "cube", verdict="confirmed")

        def verify_spatial_target(self, *unused_args, **unused_kwargs):
            self.spatial_calls += 1
            return TargetVerification(False, "relation unclear", verdict="rejected")

    reasoner = Reasoner()
    manager = ObjectVerificationManager(
        memory,
        reasoner,
        SearchInstruction("find cube", "cube"),
        min_best_image_pixels=500,
    )
    try:
        assert manager.consider([record]) == [record.object_id]
        first = manager.wait()
        assert not first[0].target_confirmed
        assert manager.consider([record]) == []
        assert reasoner.type_calls == 1
        assert reasoner.spatial_calls == 1

        # Confidence/status-only updates retain the same room label and must
        # not reopen the object's completed query.
        memory.update_room_description(
            "room_0", RoomDescription("living room", ("sofa",), 0.9),
            evidence_count=2, ordinary_evidence_count=2,
        )
        assert manager.consider([record]) == []

        memory.update_room_description(
            "room_0", RoomDescription("kitchen", ("cabinet",), 0.9),
            evidence_count=3, ordinary_evidence_count=2,
        )
        assert manager.consider([record]) == [record.object_id]
        manager.wait()
        assert reasoner.type_calls == 1
        assert reasoner.spatial_calls == 2
    finally:
        manager.close()


def test_room_semantics_are_revisable_and_only_multiview_evidence_is_stable():
    config, _, graph, _ = _components()
    memory = SceneMemory(graph)

    memory.update_room_description(
        "room_0",
        RoomDescription("kitchen", ("cabinet-like wall",), 0.6),
        evidence_count=1,
        ordinary_evidence_count=0,
        evidence_tokens=("peek_1",),
        stable_min_ordinary_views=2,
        stable_confidence=0.75,
    )
    state = memory.rooms["room_0"]
    assert state.semantic_description == "kitchen"
    assert state.semantic_status == "tentative"
    assert state.semantic_revision == 1

    memory.update_room_description(
        "room_0",
        RoomDescription("living room", ("sofa", "low table"), 0.88),
        evidence_count=3,
        ordinary_evidence_count=2,
        evidence_tokens=("peek_1", "view_1", "view_2"),
        stable_min_ordinary_views=2,
        stable_confidence=0.75,
    )
    assert state.semantic_description == "living room"
    assert state.semantic_status == "stable"
    assert state.semantic_revision == 2
    assert state.semantic_evidence_tokens == ("peek_1", "view_1", "view_2")


def test_room_description_vlm_is_enqueued_without_blocking_navigation():
    config, _, graph, _ = _components()
    memory = SceneMemory(graph)
    entered = threading.Event()
    release = threading.Event()

    class VLM:
        def describe_room_multiview(self, *unused_args, **unused_kwargs):
            entered.set()
            assert release.wait(timeout=2.0)
            return RoomDescription("living room", ("sofa",), 0.9)

    runner = object.__new__(SysNavSearchRunner)
    runner.components = SimpleNamespace(
        vlm=VLM(),
        scene_memory=memory,
        config=config,
        search_policy=SimpleNamespace(reasoner=None),
    )
    runner.enable_room_semantics = True
    runner.room_image_history = {
        "room_0": [
            RoomImageEvidence(
                viewed_room_id="room_0",
                metric_room_id="room_0",
                token="ordinary_1",
                image_path="ordinary_1.png",
                image_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                pose_world=(1.0, 1.0, 0.0),
                coverage=0.1,
                detection_labels=(),
                observation_kind="ordinary",
            )
        ]
    }
    runner._room_semantic_last_counts = {}
    runner._room_vlm_manager = AsyncRoomVLMManager(max_workers=1)
    runner._verification_manager = None
    runner._verified_target = None
    runner._record_event = lambda *unused, **unused_kwargs: None
    runner._submit_object_records = lambda unused: None

    try:
        assert runner._describe_room("room_0", None, None, "") is None
        assert entered.wait(timeout=1.0)
        assert memory.rooms["room_0"].semantic_revision == 0
        release.set()
        for result in runner._room_vlm_manager.wait():
            runner._apply_room_vlm_result(result)
    finally:
        release.set()
        runner._room_vlm_manager.close()

    assert memory.rooms["room_0"].semantic_description == "living room"
    assert memory.rooms["room_0"].semantic_revision == 1


def test_next_room_preselection_falls_back_without_waiting_for_vlm():
    config, _, graph, _ = _components()
    memory = SceneMemory(graph)
    memory.update_room_description(
        "room_1",
        RoomDescription("bedroom", ("bed",), 0.9),
        evidence_count=2,
        ordinary_evidence_count=2,
    )
    entered = threading.Event()
    release = threading.Event()

    class Reasoner:
        def select_room(self, instruction, candidates, room_images=None):
            entered.set()
            assert release.wait(timeout=2.0)
            return RoomSelection("room_1", "semantic choice", used_vlm=True)

    runner = object.__new__(SysNavSearchRunner)
    runner.components = SimpleNamespace(
        room_graph=graph,
        scene_memory=memory,
        search_policy=SimpleNamespace(reasoner=Reasoner()),
    )
    runner.room_images = {"room_1": np.zeros((4, 4, 3), dtype=np.uint8)}
    runner.room_image_history = {}
    runner._instruction = SearchInstruction("find cube", "cube")
    runner._room_vlm_manager = AsyncRoomVLMManager(max_workers=1)
    runner._room_selection_cache = {}
    runner._room_selection_consumed = set()
    runner._verification_manager = None
    runner._verified_target = None
    runner._record_event = lambda *unused, **unused_kwargs: None

    try:
        runner._ensure_room_preselection("room_0")
        assert entered.wait(timeout=1.0)
        selection = runner._take_room_preselection("room_0")
        assert selection is not None
        assert not selection.used_vlm
        assert "not ready" in selection.reason
        release.set()
        for result in runner._room_vlm_manager.wait():
            runner._apply_room_vlm_result(result)
    finally:
        release.set()
        runner._room_vlm_manager.close()

    assert "room_0" not in runner._room_selection_cache


def test_runtime_persists_provisional_object_before_vlm_verification():
    config, _, graph, _ = _components()
    memory = SceneMemory(graph, merge_distance_m=config.search.object_merge_distance_m)
    localized = _localized([2.0, 2.0, 0.8], confidence=0.9)
    frame = RGBDFrame(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(4, 4, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )
    components = object.__new__(SysNavComponents)
    components.perception = SimpleNamespace(
        detect=lambda unused_rgb: [localized.detection]
    )
    components.localizer = SimpleNamespace(
        localize_all=lambda unused_frame, unused_detections: [localized]
    )
    components.scene_memory = memory
    components.search_policy = SimpleNamespace()

    update = components.process_observation(
        frame,
        (1.0, 1.0, 0.0),
        "room_0",
        image_path="view.png",
    )

    assert update.localized_detections == (localized,)
    assert len(memory.viewpoints) == 1
    assert len(memory.objects) == 1
    record = next(iter(memory.objects.values()))
    assert record.label == "cube"
    assert memory.rooms["room_0"].observed_labels == ("cube",)
    assert update.viewpoint.object_ids == [record.object_id]
    assert update.object_records == (record,)
    assert not record.type_verified


class _EarlyStopReasoner:
    def should_switch_room(
        self,
        instruction,
        current_room_id,
        current_images_rgb,
        nearby_room_id,
        nearby_images_rgb,
        *,
        current_room_semantics,
        nearby_room_semantics,
    ):
        assert len(current_images_rgb) == 2
        assert len(nearby_images_rgb) == 1
        assert current_room_semantics["status"] == "unknown"
        assert nearby_room_semantics["status"] == "unknown"
        return EarlyStopDecision(True, nearby_room_id, "nearby view contains stronger evidence")


def test_search_policy_exposes_early_room_switch():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    policy = SysNavSearchPolicy(
        graph, planner, memory, config.search, reasoner=_EarlyStopReasoner()
    )
    policy.start(SearchInstruction("find cube", "cube"))
    memory.update_room_progress("room_0", 0.6)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    decision = policy.consider_early_stop(
        "room_0", "room_1", (image, image), (image,)
    )
    assert decision.switch_room
    assert decision.target_room_id == "room_1"


def test_class_only_low_confidence_target_can_skip_extra_vlm_call():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    policy = SysNavSearchPolicy(graph, planner, memory, config.search)
    policy.start(SearchInstruction("find cube", "cube"))
    answer = policy.verify_candidate(
        _localized([2.0, 2.0, 0.8], confidence=0.15),
        np.zeros((4, 4, 3), dtype=np.uint8),
    )
    assert answer.is_target


def test_search_policy_sends_an_unannotated_candidate_crop_to_vlm():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)

    class CaptureReasoner:
        def verify_target(self, *args, candidate_crop_rgb=None, **kwargs):
            self.full_scene = np.asarray(args[2])
            self.crop = np.asarray(candidate_crop_rgb)
            self.clipped = kwargs["candidate_clipped_by_image_edge"]
            return TargetVerification(True, "test", verdict="confirmed")

    reasoner = CaptureReasoner()
    policy = SysNavSearchPolicy(
        graph, planner, memory, config.search, reasoner=reasoner
    )
    policy.start(SearchInstruction("find cube", "cube"))
    frame = np.full((4, 4, 3), (17, 23, 31), dtype=np.uint8)
    answer = policy.verify_candidate(
        _localized([2.0, 2.0, 0.8], confidence=0.9), frame
    )

    assert answer.is_target
    assert np.any(np.all(reasoner.full_scene == (0, 255, 0), axis=2))
    assert not np.any(np.all(reasoner.crop == (0, 255, 0), axis=2))
    assert np.all(reasoner.crop == (17, 23, 31))
    assert reasoner.clipped


def test_search_policy_combines_visual_relation_with_stable_room_node():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    memory.update_room_description(
        "room_0",
        RoomDescription("living room", ("sofa", "low table"), 0.9),
        evidence_count=3,
        ordinary_evidence_count=2,
        stable_min_ordinary_views=2,
        stable_confidence=0.75,
    )

    class ConfirmVisualReasoner:
        def verify_target(self, instruction, *args, **kwargs):
            assert instruction.room_condition == ""
            return TargetVerification(
                True,
                "identity and coffee-table relation confirmed",
                verdict="confirmed",
                visual_verdict="confirmed",
            )

    policy = SysNavSearchPolicy(
        graph, planner, memory, config.search, reasoner=ConfirmVisualReasoner()
    )
    policy.start(
        SearchInstruction(
            "去客厅茶几上拿 cube",
            "cube",
            room_condition="客厅",
            spatial_condition="上",
            anchor_object="茶几",
        )
    )
    answer = policy.verify_candidate(
        _localized([2.0, 2.0, 0.8], confidence=0.9),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert answer.is_target
    assert answer.visual_verdict == "confirmed"
    assert answer.room_verdict == "confirmed"
    assert "stable Room node" in answer.reason


def test_search_policy_keeps_visually_confirmed_target_pending_for_tentative_room():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    memory.update_room_description(
        "room_0",
        RoomDescription("kitchen", ("cabinet-like wall",), 0.6),
        evidence_count=1,
        ordinary_evidence_count=0,
    )

    class ConfirmVisualReasoner:
        def verify_target(self, instruction, *args, **kwargs):
            return TargetVerification(True, "visual relation confirmed", verdict="confirmed")

    policy = SysNavSearchPolicy(
        graph, planner, memory, config.search, reasoner=ConfirmVisualReasoner()
    )
    policy.start(
        SearchInstruction(
            "去客厅茶几上拿 cube",
            "cube",
            room_condition="客厅",
            spatial_condition="上",
            anchor_object="茶几",
        )
    )
    answer = policy.verify_candidate(
        _localized([2.0, 2.0, 0.8], confidence=0.9),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert not answer.is_target
    assert answer.verdict == "uncertain"
    assert answer.visual_verdict == "confirmed"
    assert answer.room_verdict == "uncertain"
    assert "semantics are tentative" in answer.reason


def test_room_label_change_reopens_object_review_but_confidence_change_does_not():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    policy = SysNavSearchPolicy(graph, planner, memory, config.search)
    policy.start(
        SearchInstruction(
            "去客厅茶几上拿 cube",
            "cube",
            room_condition="客厅",
            spatial_condition="上",
            anchor_object="茶几",
        )
    )
    target = _localized([2.0, 2.0, 0.8], confidence=0.9)
    viewpoint = memory.add_viewpoint(
        "room_0", (1.0, 1.0, 0.0), [target], image_path="view.png"
    )
    record = memory.objects[viewpoint.object_ids[0]]
    record.is_considered = True
    memory.update_room_description(
        "room_0",
        RoomDescription("kitchen", ("cabinet",), 0.6),
        evidence_count=1,
        ordinary_evidence_count=1,
    )
    assert not record.is_considered
    record.is_considered = True
    memory.update_room_description(
        "room_0",
        RoomDescription("kitchen", ("cabinet",), 0.9),
        evidence_count=2,
        ordinary_evidence_count=2,
    )
    assert record.is_considered
    memory.update_room_description(
        "room_0",
        RoomDescription("living room", ("sofa",), 0.9),
        evidence_count=3,
        ordinary_evidence_count=3,
    )
    assert not record.is_considered


def test_doorway_peek_keeps_pose_room_but_describes_viewed_room(tmp_path):
    frame = RGBDFrame(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.float32),
        intrinsics=CameraIntrinsics.from_vertical_fov(4, 4, 90.0),
        camera_position_world=np.zeros(3),
        camera_rotation_world=np.eye(3),
        camera_name="test",
        timestamp_s=1.0,
    )

    class Components:
        def __init__(self):
            self.metric_room = None

        def process_observation(
            self,
            frame,
            pose,
            room_id,
            *,
            planned_viewpoint_index,
            image_path,
            **unused_kwargs,
        ):
            self.metric_room = room_id
            return SimpleNamespace(
                viewpoint=SimpleNamespace(viewpoint_id="viewpoint_0"),
                detections=(),
                localized_detections=(),
                object_records=(),
                coverage=None,
            )

    runner = object.__new__(SysNavSearchRunner)
    runner.components = Components()
    runner.navigator = SimpleNamespace(
        current_pose=lambda: np.array([1.0, 1.0, 0.0])
    )
    runner.capture_frame = lambda: frame
    runner.room_images = {}
    runner.room_image_history = {}
    runner._save_image = lambda unused_frame, token: str(tmp_path / f"{token}.png")
    described = []
    runner._describe_room = lambda room_id, *unused: described.append(room_id)
    runner._verification_manager = None
    runner._verified_target = None
    runner._submit_object_records = lambda unused: None
    runner._poll_object_verification = lambda: None
    runner._record_event = lambda *unused, **unused_kwargs: None

    runner._observe(
        "room_0",
        token="peek",
        planned_viewpoint_index=None,
        semantic_room_id="room_1",
    )

    assert runner.components.metric_room == "room_0"
    assert described == ["room_1"]
    assert set(runner.room_images) == {"room_1"}
    assert len(runner.room_image_history["room_1"]) == 1
    assert runner.room_image_history["room_1"][0].observation_kind == "peek"


def test_peek_requires_three_ordinary_views_and_only_checks_one_door():
    config = load_sysnav_config()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    moves = []
    decisions = []
    events = []
    runner = object.__new__(SysNavSearchRunner)
    runner.enable_early_stop = True
    runner.max_viewpoints = 20
    runner._viewpoint_count = 3
    runner._ordinary_viewpoint_counts = {"room_0": 2}
    runner._peeked_edges = set()
    runner.room_images = {"room_0": np.full_like(image, 3)}
    runner.room_image_history = {
        "room_0": [
            RoomImageEvidence(
                viewed_room_id="room_0",
                metric_room_id="room_2",
                token="stale_doorway_peek",
                image_path="stale.png",
                image_rgb=np.full_like(image, 9),
                pose_world=(0.0, 0.0, 0.0),
                coverage=None,
                detection_labels=(),
                observation_kind="peek",
            ),
            *[
                RoomImageEvidence(
                    viewed_room_id="room_0",
                    metric_room_id="room_0",
                    token=f"ordinary_{index}",
                    image_path=f"ordinary_{index}.png",
                    image_rgb=np.full_like(image, index),
                    pose_world=(float(index), 0.0, 0.0),
                    coverage=0.2 * index,
                    detection_labels=(),
                    observation_kind="ordinary",
                )
                for index in range(1, 4)
            ],
        ]
    }
    runner.navigator = SimpleNamespace(
        move_to=lambda pose, name: moves.append((pose, name)) or {},
        current_pose=lambda: np.array([1.0, 1.0, 0.0]),
    )
    def room_state(coverage=0.0):
        return SimpleNamespace(
            coverage=coverage,
            semantic_description="",
            semantic_confidence=0.0,
            semantic_status="unknown",
            semantic_revision=0,
            semantic_cues=(),
        )

    reasoner = SimpleNamespace()
    runner.components = SimpleNamespace(
        vlm=object(),
        config=SimpleNamespace(search=config.search),
        scene_memory=SimpleNamespace(
            rooms={"room_0": room_state(0.8), "room_1": room_state()}
        ),
        room_graph=SimpleNamespace(
            nodes={
                "room_0": SimpleNamespace(neighbors=("room_1", "room_2"))
            },
            room_at=lambda x, y: "room_0",
        ),
        search_policy=SimpleNamespace(reasoner=reasoner),
    )
    def consider_early_stop(
        instruction,
        current,
        current_images,
        nearby,
        nearby_images,
        **unused_kwargs,
    ):
        decisions.append((current, nearby, current_images, nearby_images))
        return EarlyStopDecision(False, None, "keep searching current room")

    reasoner.should_switch_room = consider_early_stop
    runner._room_vlm_manager = AsyncRoomVLMManager(max_workers=1)
    runner._instruction = SearchInstruction("find cube", "cube")
    runner._verification_manager = None
    runner._verified_target = None
    runner._pending_room_switch = None
    runner._portal_peek_pose = lambda current, nearby: (1.0, 2.0, 0.0)
    runner._ensure_room_preselection = lambda unused_room: None
    runner._observe = lambda *unused, **unused_kwargs: (
        SimpleNamespace(rgb=image),
        None,
        None,
    )
    runner._record_event = lambda event, **payload: events.append((event, payload))

    assert runner._consider_adjacent_rooms("room_0") == (None, None)
    assert moves == []

    runner._ordinary_viewpoint_counts["room_0"] = 3
    try:
        assert runner._consider_adjacent_rooms("room_0") == (None, None)
        for result in runner._room_vlm_manager.wait():
            runner._apply_room_vlm_result(result)
    finally:
        runner._room_vlm_manager.close()
    assert len(moves) == 1
    assert runner._peeked_edges == {("room_0", "room_1")}
    assert len(decisions) == 1
    assert [int(value[0, 0, 0]) for value in decisions[0][2]] == [1, 2, 3]
    assert len(decisions[0][3]) == 1
    vlm_input = next(payload for event, payload in events if event == "early_stop_vlm_inputs")
    assert [item["token"] for item in vlm_input["current_images"]] == [
        "ordinary_1",
        "ordinary_2",
        "ordinary_3",
    ]
    assert vlm_input["nearby_images"][0]["observation_kind"] == "peek"
