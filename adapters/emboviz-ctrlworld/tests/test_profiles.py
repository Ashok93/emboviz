"""Tests for the checkpoint-profile registry — pure, no GPU.

Covers the droid catalog entry's contract values, the strict JSON loader, the
dataclass validation, and the driver-side stress-compatibility checks that
replaced the host config's hardcoded ctrlworld region rules.

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_profiles.py
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from emboviz_ctrlworld.profiles import (
    ACTION_DIM,
    CtrlWorldProfile,
    check_stress_compat,
    get_profile,
    load_profile,
    resolve_profile,
)


def _profile_kwargs(**overrides) -> dict:
    base = dict(
        name="test",
        description="test profile",
        embodiment="so101",
        ckpt_repo="org/repo",
        ckpt_file="ckpt.pt",
        svd_repo="stabilityai/stable-video-diffusion-img2vid",
        clip_repo="openai/clip-vit-base-patch32",
        views=("top", "wrist"),
        view_hw=(192, 320),
        native_fps=6.0,
        num_frames=5,
        num_history=6,
        history_idx=(0, 0, -12, -9, -6, -3),
        svd_fps=7,
        motion_bucket_id=127,
        state_p01=tuple([-1.0] * ACTION_DIM),
        state_p99=tuple([1.0] * ACTION_DIM),
        default_region_cameras={"top": "primary", "wrist": "wrist"},
    )
    base.update(overrides)
    return base


def test_droid_profile_matches_released_checkpoint() -> None:
    p = get_profile("droid")
    # Locked to the released checkpoint (Ctrl-World config.py + the vendored
    # droid_stat.json); a drift here is a checkpoint-contract break.
    assert p.views == ("exterior_1", "exterior_2", "wrist")
    assert p.view_hw == (192, 320) and p.native_fps == 5.0
    assert p.num_frames == 5 and p.num_history == 6
    assert p.history_idx == (0, 0, -12, -9, -6, -3)
    assert p.frames_per_chunk == 4
    assert p.latent_shape == (4, 72, 40)
    assert p.stack_hw == (576, 320)
    assert len(p.state_p01) == ACTION_DIM == 7
    assert p.embodiment == "droid"
    assert get_profile("droid") is p              # memoized


def test_unknown_profile_lists_catalog() -> None:
    try:
        get_profile("so999")
    except KeyError as e:
        assert "droid" in str(e)
    else:
        raise AssertionError("expected KeyError for an unknown profile")


def test_profile_json_round_trip_and_resolution() -> None:
    p = CtrlWorldProfile(**_profile_kwargs())
    raw = dataclasses.asdict(p)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "so101.json"
        path.write_text(json.dumps(raw))
        loaded = load_profile(path)
        assert loaded == p
        assert resolve_profile(str(path)) == p     # path form
    assert resolve_profile("droid").name == "droid"  # name form


def test_profile_json_rejects_unknown_and_missing_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        raw = dataclasses.asdict(CtrlWorldProfile(**_profile_kwargs()))
        raw["viewz"] = ["typo"]
        path.write_text(json.dumps(raw))
        try:
            load_profile(path)
        except ValueError as e:
            assert "unknown field" in str(e)
        else:
            raise AssertionError("expected ValueError for an unknown field")

        raw = dataclasses.asdict(CtrlWorldProfile(**_profile_kwargs()))
        del raw["state_p01"]
        path.write_text(json.dumps(raw))
        try:
            load_profile(path)
        except ValueError as e:
            assert "missing required field" in str(e)
        else:
            raise AssertionError("expected ValueError for a missing field")


def test_profile_validation_rejects_bad_contracts() -> None:
    for overrides, fragment in (
        ({"views": ()}, "views must be non-empty"),
        ({"views": ("top", "top")}, "duplicates"),
        ({"view_hw": (190, 320)}, "multiples of 8"),
        ({"num_frames": 1}, "num_frames"),
        ({"history_idx": (0, 0)}, "history_idx"),
        ({"history_idx": (0, 1, -1, -2, -3, -4)}, "0 (the seed) or negative"),
        ({"state_p01": tuple([0.0] * 7), "state_p99": tuple([0.0] * 7)}, "degenerate"),
        ({"default_region_cameras": {"top": "primary"}}, "must equal the views"),
    ):
        try:
            CtrlWorldProfile(**_profile_kwargs(**overrides))
        except ValueError as e:
            assert fragment in str(e), str(e)
        else:
            raise AssertionError(f"expected ValueError for {overrides}")


def test_check_stress_compat_resolves_defaults_and_rejects_mismatches() -> None:
    p = CtrlWorldProfile(**_profile_kwargs())

    resolved = check_stress_compat(
        p, camera_map={"primary": "top"}, concat_cameras=None,
        n_actions=8, control_hz=30.0,
    )
    assert resolved == {"top": "primary", "wrist": "wrist"}   # profile default

    explicit = {"top": "cam_a", "wrist": "cam_b"}
    assert check_stress_compat(
        p, camera_map={"primary": "top"}, concat_cameras=explicit,
        n_actions=4, control_hz=6.0,
    ) == explicit

    for kwargs, fragment in (
        (dict(camera_map={"primary": "exterior_1"}, concat_cameras=None,
              n_actions=4, control_hz=30.0), "not views of"),
        (dict(camera_map={"primary": "top"}, concat_cameras={"top": "cam"},
              n_actions=4, control_hz=30.0), "exactly the profile's views"),
        (dict(camera_map={"primary": "top"}, concat_cameras=None,
              n_actions=6, control_hz=30.0), "multiple"),
        (dict(camera_map={"primary": "top"}, concat_cameras=None,
              n_actions=4, control_hz=15.0), "integer"),
    ):
        try:
            check_stress_compat(p, **kwargs)
        except ValueError as e:
            assert fragment in str(e), str(e)
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_check_stress_compat_requires_mapping_without_default() -> None:
    p = CtrlWorldProfile(**_profile_kwargs(default_region_cameras={}))
    try:
        check_stress_compat(
            p, camera_map={"primary": "top"}, concat_cameras=None,
            n_actions=4, control_hz=6.0,
        )
    except ValueError as e:
        assert "concat_cameras is unset" in str(e)
    else:
        raise AssertionError("expected ValueError when no default mapping exists")


def _run_all() -> None:
    test_droid_profile_matches_released_checkpoint()
    test_unknown_profile_lists_catalog()
    test_profile_json_round_trip_and_resolution()
    test_profile_json_rejects_unknown_and_missing_fields()
    test_profile_validation_rejects_bad_contracts()
    test_check_stress_compat_resolves_defaults_and_rejects_mismatches()
    test_check_stress_compat_requires_mapping_without_default()
    print("OK: all profile checks passed")


if __name__ == "__main__":
    _run_all()
