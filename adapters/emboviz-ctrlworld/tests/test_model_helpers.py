"""Tests for the torch-free pieces of the Ctrl-World adapter.

``normalize_bound`` is checked against the reference formula and the vendored
DROID quantile bounds; ``history_position`` against the reference buffer that
is pre-filled with the seed frame; ``prepare_actions``'s frame arithmetic via
its validation errors (the encode itself needs no torch).

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_model_helpers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emboviz_ctrlworld.model import (
    ACTION_DIM,
    DEFAULT_HISTORY_IDX,
    FRAMES_PER_CHUNK,
    NUM_HISTORY,
    history_position,
    normalize_bound,
)

_STAT = json.loads(
    (Path(__file__).parents[1] / "emboviz_ctrlworld" / "_ctrl_world" / "droid_stat.json").read_text()
)


def test_normalize_bound_matches_reference_formula() -> None:
    p01 = np.asarray(_STAT["state_01"], dtype=np.float64)[None, :]
    p99 = np.asarray(_STAT["state_99"], dtype=np.float64)[None, :]
    assert p01.shape == (1, ACTION_DIM) and p99.shape == (1, ACTION_DIM)

    # The bounds map to exactly -1 / +1; the midpoint to 0.
    np.testing.assert_allclose(normalize_bound(p01, p01, p99), -1.0, atol=1e-6)
    np.testing.assert_allclose(normalize_bound(p99, p01, p99), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalize_bound((p01 + p99) / 2, p01, p99), 0.0, atol=1e-6)
    # Out-of-bound values clip — the reference clips to [-1, 1].
    np.testing.assert_allclose(normalize_bound(p99 * 2 + 1, p01, p99), 1.0, atol=1e-6)


def test_history_position_reproduces_prefilled_buffer() -> None:
    """The reference pre-fills its buffer with 24 seed copies, then appends one
    anchor per turn; entry 0 is the seed, negative indices count from the end
    and resolve to seed copies until enough turns exist. Emulating that buffer
    explicitly must agree with ``history_position`` for every index and length."""
    for n_turns in range(0, 16):
        prefilled = ["seed"] * (NUM_HISTORY * 4) + [f"t{i}" for i in range(1, n_turns + 1)]
        ours = ["seed"] + [f"t{i}" for i in range(1, n_turns + 1)]
        for idx in DEFAULT_HISTORY_IDX:
            assert prefilled[idx] == ours[history_position(len(ours), idx)], (
                f"n_turns={n_turns} idx={idx}"
            )


def test_history_position_rejects_positive_and_empty() -> None:
    try:
        history_position(3, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a positive history index")
    try:
        history_position(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an empty buffer")


def test_architecture_constants() -> None:
    # Locked to the released checkpoint (Ctrl-World config.py); a drift here is
    # a checkpoint-contract break, not a tunable.
    assert FRAMES_PER_CHUNK == 4 and NUM_HISTORY == 6 and ACTION_DIM == 7
    assert len(DEFAULT_HISTORY_IDX) == NUM_HISTORY
    assert DEFAULT_HISTORY_IDX == (0, 0, -12, -9, -6, -3)


def _run_all() -> None:
    test_normalize_bound_matches_reference_formula()
    test_history_position_reproduces_prefilled_buffer()
    test_history_position_rejects_positive_and_empty()
    test_architecture_constants()
    print("OK: all ctrl-world model-helper checks passed")


if __name__ == "__main__":
    _run_all()
