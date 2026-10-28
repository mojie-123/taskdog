from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from custom_envs.scripts.FloAff.config import DataGenerationConfig
from custom_envs.scripts.FloAff.data_generation.affordance_labels import (
    kernel_affordance_labels,
    query_grid,
)
from custom_envs.scripts.FloAff.data_generation.build_splits import build_split
from custom_envs.scripts.FloAff.data_generation.generate_instances import (
    _extend_development_manifests,
    _extend_transfer_manifests,
)
from custom_envs.scripts.FloAff.geometry import (
    angular_distance_deg,
    angle_of,
    base_yaw_parallel_to_edge,
)
from custom_envs.scripts.FloAff.sampling.manipulation_sampling import (
    sample_manipulation_candidates,
)
from custom_envs.scripts.FloAff.sampling.observation_sampling import (
    observation_quotas,
    propose_observation_poses,
)
from custom_envs.scripts.FloAff.sampling.placement_sampling import (
    EDGE_OUTWARDS,
    generate_placements_and_instances,
)
from custom_envs.scripts.FloAff.manifests import read_json, read_jsonl, write_json, write_jsonl


def test_default_placement_and_instance_counts_and_regions():
    config = DataGenerationConfig()
    placements, instances = generate_placements_and_instances(config)
    assert len(placements) == 50
    assert len(instances) == 80
    for row in placements:
        if row.category == "short_edge":
            assert 0.10 <= row.edge_distances_m["left"] <= 0.30
            assert row.edge_distances_m["bottom"] >= 0.30
            assert row.edge_distances_m["top"] >= 0.30
        elif row.category == "corner":
            assert 0.10 <= row.edge_distances_m["left"] <= 0.30
            assert 0.10 <= row.edge_distances_m["top"] <= 0.30


def test_m_sampling_is_continuous_bounded_and_parallel():
    config = DataGenerationConfig()
    _, instances = generate_placements_and_instances(config)
    instance = instances[0]
    candidates = sample_manipulation_candidates(instance, config, seed=7)
    assert len(candidates) == 100
    outward = np.asarray([row.proposal["outward_m"] for row in candidates])
    tangent = np.asarray([row.proposal["tangent_m"] for row in candidates])
    assert np.all((outward >= 0.25) & (outward <= 0.85))
    assert np.all((tangent >= -0.50) & (tangent <= 0.50))
    expected_yaw = base_yaw_parallel_to_edge(instance.outward_xy_world)
    assert all(math.isclose(row.base_pose[2], expected_yaw) for row in candidates)
    assert len(np.unique(np.round(outward, 5))) > 95


def test_observation_profiles_have_exact_final_quotas_and_corner_separation():
    config = DataGenerationConfig()
    _, instances = generate_placements_and_instances(config)
    corner_instances = [row for row in instances if row.category == "corner"][:2]
    assert {row.approach_edge for row in corner_instances} == {"left", "top"}
    for instance in corner_instances:
        assert observation_quotas(instance, config) == {
            "core": 32,
            "negative_tail": 4,
            "positive_tail": 4,
        }
        proposals = propose_observation_poses(instance, config, (0.42, 0.0), seed=11)
        assert len(proposals) == 480
        other_edge = "top" if instance.approach_edge == "left" else "left"
        target = np.asarray(instance.target_pose_world[:2])
        for proposal in proposals:
            camera = np.asarray(proposal.desired_camera_pose_world[:2])
            direction_yaw = angle_of(camera - target)
            selected = angular_distance_deg(direction_yaw, angle_of(instance.outward_xy_world))
            other = angular_distance_deg(direction_yaw, angle_of(EDGE_OUTWARDS[other_edge]))
            assert selected <= 35.0 + 1e-8
            assert other >= 55.0 - 1e-8


def test_development_and_transfer_long_edges_use_separate_profiles():
    config = DataGenerationConfig()
    config.placements.long_count = 2
    config.placements.long_edges = ("top",)
    config.placements.transfer_long_count_per_edge = 1
    config.placements.transfer_long_edges = ("top",)
    _, instances = generate_placements_and_instances(config)
    development = next(row for row in instances if row.instance_id.startswith("long_top_"))
    transfer = next(
        row for row in instances if row.instance_id.startswith("transfer_long_top_")
    )
    assert development.observation_profile == "long_edge_train"
    assert transfer.observation_profile == "long_edge"
    assert config.observations.profile(development.observation_profile).bounds("core") == (
        -35.0,
        35.0,
    )
    assert config.observations.profile(transfer.observation_profile).bounds("core") == (
        -50.0,
        50.0,
    )


def test_supplemental_observation_pool_has_disjoint_deterministic_ids():
    config = DataGenerationConfig()
    _, instances = generate_placements_and_instances(config)
    instance = next(row for row in instances if row.category == "corner")
    original = propose_observation_poses(
        instance,
        config,
        (0.42, 0.0),
        seed=11,
        strata=("negative_tail",),
    )
    supplemental = propose_observation_poses(
        instance,
        config,
        (0.42, 0.0),
        seed=12,
        strata=("negative_tail",),
        batch_index=1,
    )
    repeated = propose_observation_poses(
        instance,
        config,
        (0.42, 0.0),
        seed=12,
        strata=("negative_tail",),
        batch_index=1,
    )
    assert len(original) == len(supplemental) == 48
    assert {row.proposal_id for row in original}.isdisjoint(
        row.proposal_id for row in supplemental
    )
    assert all("__o_negative_tail_b01_" in row.proposal_id for row in supplemental)
    assert supplemental == repeated


def test_query_grid_and_kernel_labels():
    config = DataGenerationConfig()
    grid = query_grid(config.affordance)
    assert grid.shape == (273, 2)
    candidates = np.array([[0.25, -0.5], [0.85, 0.5]], dtype=np.float64)
    rates = np.array([0.0, 1.0], dtype=np.float64)
    labels, support, nearest = kernel_affordance_labels(
        grid,
        candidates,
        rates,
        sigma_m=config.affordance.kernel_sigma_m,
        support_radius_m=config.affordance.support_radius_m,
    )
    assert labels.shape == support.shape == nearest.shape == (273,)
    assert 0.0 <= float(labels.min()) <= float(labels.max()) <= 1.0
    assert labels[np.argmin(np.linalg.norm(grid - candidates[0], axis=1))] < 0.01
    assert labels[np.argmin(np.linalg.norm(grid - candidates[1], axis=1))] > 0.99


def test_split_keeps_corner_siblings_and_observations_together():
    config = DataGenerationConfig()
    placements, instances = generate_placements_and_instances(config)
    observations = [
        {
            "observation_id": f"{row.instance_id}__fake",
            "instance_id": row.instance_id,
            "placement_id": row.placement_id,
        }
        for row in instances
    ]
    split = build_split(
        config,
        [row.to_dict() for row in placements],
        [row.to_dict() for row in instances],
        observations,
    )
    placement_roles = split["placement_roles"]
    assert sum(role == "axis_train" for role in placement_roles.values()) == 20
    assert sum(role == "axis_eval_floaff_train" for role in placement_roles.values()) == 30
    for placement in placements:
        sibling_instances = [row for row in instances if row.placement_id == placement.placement_id]
        roles = {
            role
            for role, ids in split["instances"].items()
            if any(row.instance_id in ids for row in sibling_instances)
        }
        assert roles == {placement_roles[placement.placement_id]}


def test_optional_transfer_regions_are_generated_and_reserved():
    config = DataGenerationConfig()
    config.placements.transfer_short_count = 1
    config.placements.transfer_corner_count_per_region = 1
    config.placements.transfer_long_count_per_edge = 1
    placements, instances = generate_placements_and_instances(config)
    transfer = [row for row in placements if row.source_region.startswith("transfer_")]
    assert len(transfer) == 1 + 3 + 2
    split = build_split(
        config,
        [row.to_dict() for row in placements],
        [row.to_dict() for row in instances],
        [],
    )
    assert all(split["placement_roles"][row.placement_id] == "transfer_test" for row in transfer)
    transfer_corners = [row for row in instances if row.placement_id.startswith("transfer_corner")]
    for row in transfer_corners:
        assert row.corner_other_edge is not None


def test_transfer_instances_use_reduced_m_and_o_counts():
    config = DataGenerationConfig()
    config.placements.transfer_short_count = 1
    config.placements.transfer_corner_count_per_region = 1
    config.placements.transfer_long_count_per_edge = 1
    _, instances = generate_placements_and_instances(config)
    training_instance = next(
        row for row in instances if not row.placement_id.startswith("transfer_")
    )
    transfer_instance = next(
        row for row in instances if row.placement_id.startswith("transfer_")
    )

    assert len(sample_manipulation_candidates(training_instance, config, seed=1)) == 100
    assert len(sample_manipulation_candidates(transfer_instance, config, seed=1)) == 40
    assert observation_quotas(training_instance, config) == {
        "core": 32,
        "negative_tail": 4,
        "positive_tail": 4,
    }
    assert observation_quotas(transfer_instance, config) == {
        "core": 16,
        "negative_tail": 2,
        "positive_tail": 2,
    }


def test_transfer_extension_preserves_existing_rows(tmp_path):
    initial = DataGenerationConfig()
    initial_placements, initial_instances = generate_placements_and_instances(initial)
    placement_rows = [row.to_dict() for row in initial_placements]
    instance_rows = [row.to_dict() for row in initial_instances]
    write_jsonl(tmp_path / "placements.jsonl", placement_rows)
    write_jsonl(tmp_path / "instances.jsonl", instance_rows)
    persisted_placement_rows = read_jsonl(tmp_path / "placements.jsonl")
    persisted_instance_rows = read_jsonl(tmp_path / "instances.jsonl")
    legacy_config = initial.to_dict()
    legacy_config["manipulation"].pop("transfer_candidates_per_instance")
    legacy_config["observations"].pop("transfer_valid_per_instance")
    write_json(
        tmp_path / "dataset_config.json",
        {
            "created_at": "test",
            "source_config": None,
            "config": legacy_config,
            "placement_count": len(placement_rows),
            "instance_count": len(instance_rows),
        },
    )

    extended = DataGenerationConfig()
    extended.placements.transfer_short_count = 4
    extended.placements.transfer_corner_count_per_region = 3
    extended.placements.transfer_long_count_per_edge = 4
    added = _extend_transfer_manifests(
        tmp_path, extended, SimpleNamespace(config=None)
    )
    assert added == (21, 30)
    all_placements = read_jsonl(tmp_path / "placements.jsonl")
    all_instances = read_jsonl(tmp_path / "instances.jsonl")
    assert all_placements[: len(placement_rows)] == persisted_placement_rows
    assert all_instances[: len(instance_rows)] == persisted_instance_rows
    assert len(all_placements) == 71
    assert len(all_instances) == 110
    snapshot = read_json(tmp_path / "dataset_config.json")
    assert snapshot["placement_count"] == 71
    assert snapshot["instance_count"] == 110
    assert _extend_transfer_manifests(
        tmp_path, extended, SimpleNamespace(config=None)
    ) == (0, 0)


def test_development_long_extension_preserves_existing_rows(tmp_path):
    initial = DataGenerationConfig()
    initial_placements, initial_instances = generate_placements_and_instances(initial)
    placement_rows = [row.to_dict() for row in initial_placements]
    instance_rows = [row.to_dict() for row in initial_instances]
    write_jsonl(tmp_path / "placements.jsonl", placement_rows)
    write_jsonl(tmp_path / "instances.jsonl", instance_rows)
    persisted_placement_rows = read_jsonl(tmp_path / "placements.jsonl")
    persisted_instance_rows = read_jsonl(tmp_path / "instances.jsonl")
    legacy_config = initial.to_dict()
    # The already-collected dataset predates the separate training-long profile.
    legacy_config["observations"].pop("long_edge_train")
    write_json(
        tmp_path / "dataset_config.json",
        {
            "created_at": "test",
            "source_config": None,
            "config": legacy_config,
            "placement_count": len(placement_rows),
            "instance_count": len(instance_rows),
        },
    )

    extended = DataGenerationConfig()
    extended.placements.long_count = 20
    extended.placements.long_edges = ("top",)
    added = _extend_development_manifests(
        tmp_path, extended, SimpleNamespace(config=None)
    )
    assert added == (20, 20)
    all_placements = read_jsonl(tmp_path / "placements.jsonl")
    all_instances = read_jsonl(tmp_path / "instances.jsonl")
    assert all_placements[: len(placement_rows)] == persisted_placement_rows
    assert all_instances[: len(instance_rows)] == persisted_instance_rows
    new_placements = all_placements[len(placement_rows) :]
    new_instances = all_instances[len(instance_rows) :]
    assert all(row["source_region"] == "long_top" for row in new_placements)
    assert all(row["observation_profile"] == "long_edge_train" for row in new_instances)
    assert _extend_development_manifests(
        tmp_path, extended, SimpleNamespace(config=None)
    ) == (0, 0)
