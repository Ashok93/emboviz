"""Tests for the stress run-config section (pydantic validation, no GPU).

Covers both world-model backends: the cosmos3 server path and the ctrlworld
local path, including the backend-conditional field rules.

Run::

    uv run python emboviz/world_models/tests/test_stress_config.py
"""

from __future__ import annotations

from emboviz.config import AnalysisCfg, SceneSwapCfg, WorldStressCfg


def test_recorded_baseline_minimal() -> None:
    c = WorldStressCfg(server_url="http://srv:8000")
    assert c.world_model == "cosmos3"            # default backend
    assert c.policy_adapter is None              # recorded-action faithfulness baseline
    assert c.domain == "droid_lerobot" and c.action_dim == 10
    assert c.n_loop_steps == 2 and c.n_actions == 16
    assert set(c.concat_cameras) == {"wrist", "exterior_left", "exterior_right"}


def test_policy_path_valid() -> None:
    c = WorldStressCfg(
        server_url="http://srv:8000", policy_adapter="pi0",
        action_convention="delta_xyz_euler_base",
        camera_map={"primary": "exterior_left", "wrist": "wrist"},
        perturbations=["rotate the cup 90 degrees", "replace the cup with a rubber duck"],
        reasoner_url="http://reasoner:8001",
    )
    assert c.action_convention == "delta_xyz_euler_base"
    assert len(c.perturbations) == 2


def test_ctrlworld_minimal() -> None:
    c = WorldStressCfg(world_model="ctrlworld", n_actions=4)
    assert c.server_url is None                  # local backend needs no server
    assert c.profile == "droid"                  # default checkpoint profile
    # Region vocabulary and the concat_cameras default come from the checkpoint
    # profile and are resolved by the dream driver
    # (emboviz_ctrlworld.profiles.check_stress_compat), not the host schema.
    assert c.concat_cameras is None


def test_ctrlworld_policy_path_valid() -> None:
    c = WorldStressCfg(
        world_model="ctrlworld", policy_adapter="pi0",
        action_convention="droid_joint_velocity", robot="franka_panda",
        camera_map={"primary": "exterior_1", "wrist_left": "wrist"},
        n_actions=4, execute_steps=4,
    )
    assert c.world_model == "ctrlworld" and c.n_actions == 4


def test_cosmos3_requires_server_url() -> None:
    try:
        WorldStressCfg()
    except Exception as e:
        assert "server_url" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_ctrlworld_forbids_cosmos_fields() -> None:
    for kwargs in (
        {"server_url": "http://x"},
        {"domain": "droid_lerobot"},
        {"action_dim": 10},
        {"concat_resolution": (352, 640)},
    ):
        try:
            WorldStressCfg(world_model="ctrlworld", n_actions=4, **kwargs)
        except Exception as e:
            assert "cosmos3 backend" in str(e), str(e)
        else:
            raise AssertionError(f"expected validation error for {kwargs}")


def test_ctrlworld_forbids_perturbations_and_insert_swap() -> None:
    try:
        WorldStressCfg(world_model="ctrlworld", n_actions=4, perturbations=["edit"])
    except Exception as e:
        assert "not available on the" in str(e)
    else:
        raise AssertionError("expected validation error")
    try:
        WorldStressCfg(
            world_model="ctrlworld", n_actions=4,
            scene_swap={"mask_query": "the cup", "replace_query": "a duck"},
        )
    except Exception as e:
        assert "replace_query" in str(e)
    else:
        raise AssertionError("expected validation error")
    # Removal (empty replace_query) is the supported ctrlworld edit.
    c = WorldStressCfg(
        world_model="ctrlworld", n_actions=4, scene_swap={"mask_query": "the cup"}
    )
    assert c.scene_swap.replace_query == ""


def test_cosmos3_forbids_profile() -> None:
    try:
        WorldStressCfg(server_url="http://x", profile="droid")
    except Exception as e:
        assert "ctrlworld checkpoint profile" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_policy_without_convention_raises() -> None:
    try:
        WorldStressCfg(server_url="http://x", policy_adapter="pi0", camera_map={"primary": "wrist"})
    except Exception as e:
        assert "action_convention is required" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_policy_without_camera_map_raises() -> None:
    try:
        WorldStressCfg(server_url="http://x", policy_adapter="pi0", action_convention="absolute_xyz_euler")
    except Exception as e:
        assert "camera_map is required" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_bad_region_raises() -> None:
    try:
        WorldStressCfg(server_url="http://x", camera_map={"primary": "front"})
    except Exception as e:
        assert "invalid" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_bad_convention_raises() -> None:
    try:
        WorldStressCfg(server_url="http://x", action_convention="joint_velocity")
    except Exception as e:
        assert "action_convention" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_concat_cameras_must_cover_all_regions() -> None:
    try:
        WorldStressCfg(server_url="http://x", concat_cameras={"wrist": "wrist"})
    except Exception as e:
        assert "concat_cameras must map exactly" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_nested_under_analysis() -> None:
    a = AnalysisCfg(stress={"server_url": "http://x", "n_loop_steps": 3})
    assert a.stress.server_url == "http://x" and a.stress.n_loop_steps == 3
    assert AnalysisCfg().stress is None    # optional, absent by default


def test_scene_swap_insert_and_remove() -> None:
    c = WorldStressCfg(
        server_url="http://x",
        scene_swap={"mask_query": "the marker", "replace_query": "a spoon"},
    )
    assert c.scene_swap.mask_query == "the marker" and c.scene_swap.replace_query == "a spoon"
    assert c.scene_swap.detector_score_threshold == 0.5
    # Empty replace_query is valid -> removal mode.
    assert SceneSwapCfg(mask_query="the marker").replace_query == ""


def test_scene_swap_requires_mask_query() -> None:
    try:
        SceneSwapCfg(mask_query="   ")
    except Exception as e:
        assert "mask_query must be a non-empty" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_scene_swap_threshold_range() -> None:
    try:
        SceneSwapCfg(mask_query="x", detector_score_threshold=1.5)
    except Exception as e:
        assert "in [0, 1]" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_scene_swap_and_perturbations_mutually_exclusive() -> None:
    try:
        WorldStressCfg(
            server_url="http://x",
            perturbations=["replace the cup with a duck"],
            scene_swap={"mask_query": "the cup"},
        )
    except Exception as e:
        assert "not both" in str(e)
    else:
        raise AssertionError("expected validation error")


def _run_all() -> None:
    test_recorded_baseline_minimal()
    test_policy_path_valid()
    test_ctrlworld_minimal()
    test_ctrlworld_policy_path_valid()
    test_cosmos3_requires_server_url()
    test_cosmos3_forbids_profile()
    test_ctrlworld_forbids_cosmos_fields()
    test_ctrlworld_forbids_perturbations_and_insert_swap()
    test_policy_without_convention_raises()
    test_policy_without_camera_map_raises()
    test_bad_region_raises()
    test_bad_convention_raises()
    test_concat_cameras_must_cover_all_regions()
    test_nested_under_analysis()
    test_scene_swap_insert_and_remove()
    test_scene_swap_requires_mask_query()
    test_scene_swap_threshold_range()
    test_scene_swap_and_perturbations_mutually_exclusive()
    print("OK: all stress config checks passed")


if __name__ == "__main__":
    _run_all()
