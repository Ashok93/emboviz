"""Tests for the cosmos_stress run-config section (pydantic validation, no GPU).

Run::

    uv run python emboviz/world_models/tests/test_cosmos_stress_config.py
"""

from __future__ import annotations

from emboviz.config import AnalysisCfg, CosmosStressCfg


def test_recorded_baseline_minimal() -> None:
    c = CosmosStressCfg(server_url="http://srv:8000")
    assert c.policy_adapter is None              # recorded-action faithfulness baseline
    assert c.domain == "droid_lerobot" and c.action_dim == 10
    assert c.n_loop_steps == 2 and c.n_actions == 16
    assert set(c.concat_cameras) == {"wrist", "exterior_left", "exterior_right"}


def test_policy_path_valid() -> None:
    c = CosmosStressCfg(
        server_url="http://srv:8000", policy_adapter="pi0",
        action_convention="delta_xyz_euler_base",
        camera_map={"primary": "exterior_left", "wrist": "wrist"},
        perturbations=["rotate the cup 90 degrees", "replace the cup with a rubber duck"],
        reasoner_url="http://reasoner:8001",
    )
    assert c.action_convention == "delta_xyz_euler_base"
    assert len(c.perturbations) == 2


def test_policy_without_convention_raises() -> None:
    try:
        CosmosStressCfg(server_url="http://x", policy_adapter="pi0", camera_map={"primary": "wrist"})
    except Exception as e:
        assert "action_convention is required" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_policy_without_camera_map_raises() -> None:
    try:
        CosmosStressCfg(server_url="http://x", policy_adapter="pi0", action_convention="absolute_xyz_euler")
    except Exception as e:
        assert "camera_map is required" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_bad_region_raises() -> None:
    try:
        CosmosStressCfg(server_url="http://x", camera_map={"primary": "front"})
    except Exception as e:
        assert "invalid" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_bad_convention_raises() -> None:
    try:
        CosmosStressCfg(server_url="http://x", action_convention="joint_velocity")
    except Exception as e:
        assert "action_convention" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_concat_cameras_must_cover_all_regions() -> None:
    try:
        CosmosStressCfg(server_url="http://x", concat_cameras={"wrist": "wrist"})
    except Exception as e:
        assert "concat_cameras must map exactly" in str(e)
    else:
        raise AssertionError("expected validation error")


def test_nested_under_analysis() -> None:
    a = AnalysisCfg(cosmos_stress={"server_url": "http://x", "n_loop_steps": 3})
    assert a.cosmos_stress.server_url == "http://x" and a.cosmos_stress.n_loop_steps == 3
    assert AnalysisCfg().cosmos_stress is None    # optional, absent by default


def _run_all() -> None:
    test_recorded_baseline_minimal()
    test_policy_path_valid()
    test_policy_without_convention_raises()
    test_policy_without_camera_map_raises()
    test_bad_region_raises()
    test_bad_convention_raises()
    test_concat_cameras_must_cover_all_regions()
    test_nested_under_analysis()
    print("OK: all cosmos_stress config checks passed")


if __name__ == "__main__":
    _run_all()
