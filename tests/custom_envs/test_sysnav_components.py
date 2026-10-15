from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from custom_envs.scripts.N2M.data_generation.camera_geometry import CameraIntrinsics
from custom_envs.scripts.map.nav2_map_io import load_nav2_map
from custom_envs.scripts.map.room_map_io import RoomMap
from custom_envs.scripts.sysnav.config import load_sysnav_config
from custom_envs.scripts.sysnav.coverage import CoveragePlanner
from custom_envs.scripts.sysnav.rgbd import DepthLocalizer, RGBDFrame
from custom_envs.scripts.sysnav.room_graph import AnonymousRoomGraph
from custom_envs.scripts.sysnav.runtime import build_sysnav_components
from custom_envs.scripts.sysnav.scene_memory import SceneMemory
from custom_envs.scripts.sysnav.search_policy import SysNavSearchPolicy
from custom_envs.scripts.sysnav.search_runner import (
    RoomImageEvidence,
    SysNavSearchRunner,
)
from custom_envs.scripts.sysnav.types import (
    Detection2D,
    EarlyStopDecision,
    LocalizedDetection,
    SearchInstruction,
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


class _EarlyStopReasoner:
    def should_switch_room(self, instruction, current_room_id, current_images_rgb, nearby_room_id, nearby_images_rgb):
        assert len(current_images_rgb) == 2
        assert len(nearby_images_rgb) == 1
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
                detections=(), localized_detections=(), coverage=None
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
    runner._verify_targets = lambda *unused: None
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
