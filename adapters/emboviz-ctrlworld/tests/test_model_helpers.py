"""Tests for the torch-free pieces of the Ctrl-World adapter.

``normalize_bound`` is checked against the reference formula and the droid
profile's quantile bounds; ``history_position`` against the reference buffer
that is pre-filled with the seed frame.

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_model_helpers.py
"""

from __future__ import annotations

import numpy as np

from emboviz_ctrlworld.model import history_position, normalize_bound
from emboviz_ctrlworld.profiles import ACTION_DIM, get_profile

_DROID = get_profile("droid")


def test_normalize_bound_matches_reference_formula() -> None:
    p01 = np.asarray(_DROID.state_p01, dtype=np.float64)[None, :]
    p99 = np.asarray(_DROID.state_p99, dtype=np.float64)[None, :]
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
        prefilled = ["seed"] * (_DROID.num_history * 4) + [f"t{i}" for i in range(1, n_turns + 1)]
        ours = ["seed"] + [f"t{i}" for i in range(1, n_turns + 1)]
        for idx in _DROID.history_idx:
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


def _run_all() -> None:
    test_normalize_bound_matches_reference_formula()
    test_history_position_reproduces_prefilled_buffer()
    test_history_position_rejects_positive_and_empty()
    print("OK: all ctrl-world model-helper checks passed")


if __name__ == "__main__":
    _run_all()
