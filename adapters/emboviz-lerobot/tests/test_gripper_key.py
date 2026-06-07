"""Tests for the separate-feature gripper (dataset.gripper.key).

Covers the three layers the change touches:
  1. GripperCfg — source XOR key validation.
  2. make_gripper_extractor — a key-gripper leaves the state whole, no slice.
  3. LeRobotEpisodeSource._build_scene — reads the gripper from its own column
     (the DROID case: cartesian state + a separate gripper feature), and the
     existing state-index path still works.

Run::

    uv run --with torch --with pillow --with numpy python \
        adapters/emboviz-lerobot/tests/test_gripper_key.py
"""

from __future__ import annotations

import numpy as np
import pytest

from emboviz.config import GripperCfg
from emboviz_wire.dataset_build import build_profile, make_gripper_extractor
from emboviz_lerobot.source import LeRobotEpisodeSource


# ── 1. config validation ─────────────────────────────────────────────────────


def test_gripper_cfg_source_only_ok() -> None:
    g = GripperCfg(source=7)
    assert g.source == 7 and g.key is None


def test_gripper_cfg_key_only_ok() -> None:
    g = GripperCfg(key="action.gripper_position")
    assert g.key == "action.gripper_position" and g.source is None


def test_gripper_cfg_both_rejected() -> None:
    try:
        GripperCfg(source=7, key="action.gripper_position")
    except ValueError as e:
        assert "EITHER" in str(e) or "both" in str(e)
    else:
        raise AssertionError("expected ValueError when both source and key are set")


# ── 2. extractor ─────────────────────────────────────────────────────────────


def test_extractor_key_leaves_state_whole() -> None:
    # A key-gripper means the reader supplies it separately, so the extractor
    # returns the full state and no gripper value.
    ex = make_gripper_extractor({"key": "action.gripper_position"}, None)
    state = np.arange(6, dtype=np.float32)
    proprio, grip = ex(state)
    assert np.array_equal(proprio, state) and grip is None


def test_extractor_source_still_slices() -> None:
    ex = make_gripper_extractor({"source": 7}, None)
    state = np.arange(8, dtype=np.float32)
    proprio, grip = ex(state)
    assert np.array_equal(proprio, state) and grip == 7.0


def test_extractor_neither_raises() -> None:
    try:
        make_gripper_extractor({"kind": "parallel_jaw"}, None)
    except ValueError as e:
        assert "source" in str(e) and "key" in str(e)
    else:
        raise AssertionError("expected ValueError when neither source nor key is given")


# ── 3. reader: separate-feature gripper end to end ───────────────────────────


def _img_tensor():
    import torch
    return torch.zeros((3, 8, 8), dtype=torch.uint8)


def _vec(values):
    import torch
    return torch.tensor(values, dtype=torch.float32)


def _source(*, gripper_cfg: dict, state_names, state_key, gripper_key):
    profile = build_profile(
        name="droid", cameras={"primary": "observation.images.exterior_1_left"},
        state_dim=len(state_names), state_names=state_names, convention="ee_pose",
        action_dim=7, action_names=None, gripper=gripper_cfg,
    )
    return LeRobotEpisodeSource(
        repo_id="test/droid", profile=profile,
        image_keys={"primary": "observation.images.exterior_1_left"},
        state_key=state_key, action_key="action.original",
        gripper_extractor=make_gripper_extractor(gripper_cfg, state_names),
        gripper_key=gripper_key,
    )


def test_reader_separate_key_gripper() -> None:
    pytest.importorskip("torch")  # _build_scene path needs torch tensors
    # DROID layout: cartesian state in its own key, gripper in a separate key.
    src = _source(
        gripper_cfg={"key": "action.gripper_position", "kind": "parallel_jaw", "units": "unit"},
        state_names=["x", "y", "z", "roll", "pitch", "yaw"],
        state_key="observation.state.cartesian_position",
        gripper_key="action.gripper_position",
    )
    sample = {
        "observation.images.exterior_1_left": _img_tensor(),
        "observation.state.cartesian_position": _vec([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        "action.gripper_position": _vec([0.42]),
        "action.original": _vec([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.42]),
        "task": "pick",
        "frame_index": 0,
    }
    scene = src._build_scene(sample, "pick", episode_idx=0, frame_offset=0, fps=15.0)
    # State is the full cartesian vector (not split), gripper read from its own column.
    assert scene.observations.state is not None
    assert np.allclose(scene.observations.state.values, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert scene.observations.gripper is not None
    assert abs(scene.observations.gripper.value - 0.42) < 1e-6


def test_reader_state_index_gripper_still_works() -> None:
    pytest.importorskip("torch")  # _build_scene path needs torch tensors
    # Regression: the packed-state path (gripper.source) is unchanged.
    src = _source(
        gripper_cfg={"source": 7, "kind": "parallel_jaw", "units": "unit"},
        state_names=["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        state_key="observation.state",
        gripper_key=None,
    )
    sample = {
        "observation.images.exterior_1_left": _img_tensor(),
        "observation.state": _vec([0, 1, 2, 3, 4, 5, 6, 0.9]),
        "action.original": _vec([0, 0, 0, 0, 0, 0, 0.9]),
        "task": "pick",
        "frame_index": 0,
    }
    scene = src._build_scene(sample, "pick", episode_idx=0, frame_offset=0, fps=15.0)
    assert scene.observations.gripper is not None
    assert abs(scene.observations.gripper.value - 0.9) < 1e-6


def _run_all() -> None:
    test_gripper_cfg_source_only_ok()
    test_gripper_cfg_key_only_ok()
    test_gripper_cfg_both_rejected()
    test_extractor_key_leaves_state_whole()
    test_extractor_source_still_slices()
    test_extractor_neither_raises()
    test_reader_separate_key_gripper()
    test_reader_state_index_gripper_still_works()
    print("OK: all gripper.key (separate-feature gripper) checks passed")


if __name__ == "__main__":
    _run_all()
