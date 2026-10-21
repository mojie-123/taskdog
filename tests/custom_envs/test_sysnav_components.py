from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from custom_envs.scripts.N2M.data_generation.camera_geometry import CameraIntrinsics
from custom_envs.scripts.map.nav2_map_io import load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap
from custom_envs.scripts.sysnav.config import load_sysnav_config
from custom_envs.scripts.sysnav.coverage import CoveragePlanner
from custom_envs.scripts.sysnav.rgbd import DepthLocalizer, RGBDFrame
from custom_envs.scripts.sysnav.room_graph import AnonymousRoomGraph
from custom_envs.scripts.sysnav.runtime import SysNavComponents, build_sysnav_components
from custom_envs.scripts.sysnav.scene_memory import SceneMemory
from custom_envs.scripts.sysnav.search_policy import SysNavSearchPolicy
from custom_envs.scripts.sysnav.search_runner import (
    RoomImageEvidence,
    SysNavSearchRunner,
)
from custom_envs.scripts.sysnav.transit_detection import ContinuousTransitDetector
from custom_envs.scripts.sysnav.types import (
    Detection2D,
    EarlyStopDecision,
    LocalizedDetection,
    RoomDescription,
    SearchInstruction,
    TargetVerification,
)
from custom_envs.scripts.sysnav.vlm import QwenVLMClient


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


def test_config_and_instruction_prompts():
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
    assert config.search.early_stop_max_current_images == 4
    assert config.search.room_semantic_max_images == 6
    assert config.search.room_semantic_refresh_min_new_images == 2
    assert config.search.room_semantic_stable_min_ordinary_views == 2
    assert config.search.room_semantic_stable_confidence == 0.75
    assert config.search.target_accept_confidence == 0.20
    assert config.transit_detection.enabled
    assert config.transit_detection.frequency_hz == 2.0
    assert config.transit_detection.candidate_dedup_radius_m == 0.30
    assert config.transit_detection.reobserve_min_translation_m == 0.50
    assert config.transit_detection.reobserve_min_yaw_deg == 20.0
    assert config.transit_detection.save_images


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
            target = _localized(next(positions), confidence=0.9)
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


def test_transit_vlm_rejection_is_controlled_as_not_confirmed():
    target = _localized([2.0, 2.0, 0.5], confidence=0.9)
    frame = SimpleNamespace(rgb=np.zeros((8, 8, 3), dtype=np.uint8))
    observation = SimpleNamespace(
        sequence=1,
        image_path="candidate.png",
        robot_pose_world=np.array([1.0, 1.0, 0.0]),
        frame=frame,
    )
    candidate = SimpleNamespace(observation=observation, target=target)
    confirmed = SimpleNamespace(
        first=candidate,
        confirmation=candidate,
        representative=observation,
        target=target,
        separation_m=0.0,
    )

    class Detector:
        def __init__(self):
            self.deferred = []

        def confirm(self, *unused_args, **unused_kwargs):
            return confirmed

        def defer(self, *args, **kwargs):
            self.deferred.append((args, kwargs))

    detector = Detector()
    runner = object.__new__(SysNavSearchRunner)
    runner.navigator = SimpleNamespace(
        transit_detector=detector,
        current_pose=lambda: np.array([1.0, 1.0, 0.0]),
    )
    runner.components = SimpleNamespace(
        search_policy=SimpleNamespace(
            verify_candidate=lambda *unused: TargetVerification(
                False, "visible wall panel", verdict="rejected"
            )
        )
    )
    runner._record_event = lambda *unused, **unused_kwargs: None
    runner._commit_confirmed_transit = lambda unused: None

    outcome = runner._verify_transit_candidate(candidate, navigation_name="SEARCH")

    assert outcome == "not_confirmed"
    assert len(detector.deferred) == 1
    assert detector.deferred[0][1]["verdict"] == "rejected"


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


def test_coverage_plan_marks_actual_visible_cells():
    _, _, _, planner = _components()
    plan = planner.build_plan("room_1")
    assert len(plan.target_cells) > 100
    assert len(plan.viewpoints) > 10
    viewpoint = plan.next_viewpoint((3.9, 7.0, 0.0))
    assert viewpoint is not None
    before = plan.coverage
    plan.mark_visited(viewpoint.index)
    assert plan.coverage > before
    assert 0.0 < plan.coverage <= 1.0


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


def test_runtime_persists_object_only_after_explicit_verification():
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
    assert memory.objects == {}
    assert memory.rooms["room_0"].observed_labels == ()
    assert update.viewpoint.object_ids == []

    record = components.commit_verified_detection(
        update.viewpoint.viewpoint_id, localized
    )
    assert record.label == "cube"
    assert memory.rooms["room_0"].observed_labels == ("cube",)
    assert update.viewpoint.object_ids == [record.object_id]


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


def test_class_only_target_can_skip_extra_vlm_call():
    config, _, graph, planner = _components()
    memory = SceneMemory(graph)
    policy = SysNavSearchPolicy(graph, planner, memory, config.search)
    policy.start(SearchInstruction("find cube", "cube"))
    answer = policy.verify_candidate(
        _localized([2.0, 2.0, 0.8], confidence=0.9),
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


def test_pending_visual_target_commits_after_room_semantics_become_stable():
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
    viewpoint = memory.add_viewpoint("room_0", (1.0, 1.0, 0.0), [])
    runner = object.__new__(SysNavSearchRunner)
    runner.components = SimpleNamespace(
        config=config,
        scene_memory=memory,
        search_policy=policy,
        commit_verified_detection=memory.attach_verified_detection,
    )
    runner._pending_targets = []
    runner._record_event = lambda *unused, **unused_kwargs: None
    target = _localized([2.0, 2.0, 0.8], confidence=0.9)
    runner._remember_pending_target(
        target,
        viewpoint.viewpoint_id,
        source="test",
        visual_reason="identity and coffee-table relation confirmed",
    )

    assert memory.objects == {}
    assert runner._resolve_pending_target("room_0") is None
    memory.update_room_description(
        "room_0",
        RoomDescription("living room", ("sofa", "low table"), 0.9),
        evidence_count=3,
        ordinary_evidence_count=2,
        stable_min_ordinary_views=2,
        stable_confidence=0.75,
    )
    resolved = runner._resolve_pending_target("room_0")

    assert resolved is target
    assert len(memory.objects) == 1
    record = next(iter(memory.objects.values()))
    assert record.observation_count == 1
    assert record.viewpoint_ids == [viewpoint.viewpoint_id]
    assert runner._pending_targets == []


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
            self, frame, pose, room_id, *, planned_viewpoint_index, image_path
        ):
            self.metric_room = room_id
            return SimpleNamespace(
                viewpoint=SimpleNamespace(viewpoint_id="viewpoint_0"),
                detections=(),
                localized_detections=(),
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
    runner._verify_targets = lambda *unused, **unused_kwargs: None
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
        move_to=lambda pose, name: moves.append((pose, name)) or {}
    )
    runner.components = SimpleNamespace(
        vlm=object(),
        config=SimpleNamespace(search=config.search),
        scene_memory=SimpleNamespace(
            rooms={"room_0": SimpleNamespace(coverage=0.8)}
        ),
        room_graph=SimpleNamespace(
            nodes={
                "room_0": SimpleNamespace(neighbors=("room_1", "room_2"))
            }
        ),
        search_policy=SimpleNamespace(),
    )
    def consider_early_stop(current, nearby, current_images, nearby_images):
        decisions.append((current, nearby, current_images, nearby_images))
        return EarlyStopDecision(False, None, "keep searching current room")

    runner.components.search_policy.consider_early_stop = consider_early_stop
    runner._portal_peek_pose = lambda current, nearby: (1.0, 2.0, 0.0)
    runner._observe = lambda *unused, **unused_kwargs: (
        SimpleNamespace(rgb=image),
        None,
        None,
    )
    runner._record_event = lambda event, **payload: events.append((event, payload))

    assert runner._consider_adjacent_rooms("room_0") == (None, None)
    assert moves == []

    runner._ordinary_viewpoint_counts["room_0"] = 3
    assert runner._consider_adjacent_rooms("room_0") == (None, None)
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
