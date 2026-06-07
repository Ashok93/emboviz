"""Tests for the closed-loop dream Rerun exporter — pure, no GPU/server.

Builds synthetic original/dream views + concat seeds, writes the ``.rrd``, and
checks a non-empty file is produced across the shapes the driver emits: a single
dream column, a baseline-vs-swap pair, and a not-run (black) swap column. Plus
the input-validation contract (empty/malformed frames raise rather than writing a
half-empty comparison).

Run::

    uv run python emboviz/world_models/tests/test_dream_rerun.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emboviz.world_models.dream_rerun import DreamColumn, export_dream_rerun

CAMS = ("wrist", "exterior_left", "exterior_right")


def _views(n: int, h: int, w: int, val: int) -> dict[str, list[np.ndarray]]:
    return {c: [np.full((h, w, 3), (val + i) % 256, np.uint8) for i in range(n)] for c in CAMS}


def _seed() -> np.ndarray:
    return np.full((540, 640, 3), 60, np.uint8)


def test_single_column() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dream.rrd"
        # Dream longer than the original window: exercises the "real episode ran
        # out, freeze its last frame" path without raising.
        path = export_dream_rerun(
            out,
            original_views=_views(5, 180, 320, 10),
            dream_columns=[DreamColumn("dream", _views(8, 180, 320, 100))],
            seeds=[("seed", _seed())],
            instruction="pick the marker from the cup",
            fps=10.0,
            policy_name="pi0",
        )
        assert path == out
        assert out.exists() and out.stat().st_size > 0


def test_baseline_vs_swap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dream.rrd"
        export_dream_rerun(
            out,
            original_views=_views(8, 180, 320, 0),
            dream_columns=[
                DreamColumn("baseline", _views(8, 180, 320, 100), "unedited seed"),
                DreamColumn("swap → a spoon", _views(8, 180, 320, 200), "wrist: replace 0.91"),
            ],
            seeds=[("original", _seed()), ("swap → a spoon", _seed())],
            instruction="pick the marker from the cup",
            fps=15.0,
            policy_name="pi0",
            context_note="**Swap:** wrist swapped, exteriors original",
        )
        assert out.exists() and out.stat().st_size > 0


def test_not_run_swap_column_renders_black() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dream.rrd"
        export_dream_rerun(
            out,
            original_views=_views(6, 180, 320, 0),
            dream_columns=[
                DreamColumn("baseline", _views(6, 180, 320, 100)),
                DreamColumn("swap → a spoon", None, "'the marker' not detected in any camera"),
            ],
            seeds=[("original", _seed())],
            instruction="pick the marker from the cup",
            fps=15.0,
            policy_name="pi0",
        )
        assert out.exists() and out.stat().st_size > 0


def test_rejects_empty_and_malformed() -> None:
    good = _views(3, 8, 8, 0)
    seed_ok = [("seed", np.zeros((16, 16, 3), np.uint8))]
    bad_cam = {c: [np.zeros((8, 8), np.uint8)] for c in CAMS}  # missing channel dim
    cases = [
        (dict(original_views={}, dream_columns=[DreamColumn("d", good)], seeds=seed_ok),
         "original_views must be non-empty"),
        (dict(original_views=good, dream_columns=[], seeds=seed_ok),
         "at least one dream column"),
        (dict(original_views=good, dream_columns=[DreamColumn("d", good)], seeds=[]),
         "at least one seed panel"),
        (dict(original_views=good, dream_columns=[DreamColumn("d", bad_cam)], seeds=seed_ok),
         "(H, W, 3)"),
        (dict(original_views=good,
              dream_columns=[DreamColumn("d", {"wrist": good["wrist"]})], seeds=seed_ok),
         "differ from original"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d.rrd"
        for kwargs, needle in cases:
            try:
                export_dream_rerun(
                    out, instruction="x", fps=10.0, policy_name="pi0", **kwargs,
                )
            except ValueError as e:
                assert needle in str(e), f"{needle!r} not in {e}"
            else:
                raise AssertionError(f"expected ValueError containing {needle!r}")


def _run_all() -> None:
    test_single_column()
    test_baseline_vs_swap()
    test_not_run_swap_column_renders_black()
    test_rejects_empty_and_malformed()
    print("OK: all dream Rerun exporter checks passed")


if __name__ == "__main__":
    _run_all()
