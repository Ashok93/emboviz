"""Tests for the closed-loop dream Rerun exporter — pure, no GPU/server.

Builds synthetic original/dream frames + a concat seed, writes the ``.rrd``, and
checks a non-empty file is produced; plus the input-validation contract (empty or
malformed frames raise rather than writing a half-empty comparison).

Run::

    uv run python emboviz/world_models/tests/test_dream_rerun.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emboviz.world_models.dream_rerun import export_dream_rerun


def _frames(n: int, h: int, w: int, val: int) -> list[np.ndarray]:
    return [np.full((h, w, 3), (val + i) % 256, np.uint8) for i in range(n)]


def test_export_writes_nonempty_rrd() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dream.rrd"
        # Dream longer than the original window: exercises the "real episode ran
        # out, freeze its last frame" path without raising.
        path = export_dream_rerun(
            out,
            original_frames=_frames(5, 180, 320, 10),
            dream_frames=_frames(8, 180, 320, 100),
            seed_concat=np.full((540, 640, 3), 60, np.uint8),
            instruction="pick the marker from the cup",
            perturbation="replace the marker with a spoon",
            fps=10.0,
            policy_name="pi0",
            camera="primary",
        )
        assert path == out
        assert out.exists() and out.stat().st_size > 0


def test_export_unperturbed_no_perturbation_card() -> None:
    # The unperturbed path (perturbation=None) must also produce a valid file.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dream.rrd"
        export_dream_rerun(
            out,
            original_frames=_frames(4, 180, 320, 0),
            dream_frames=_frames(4, 180, 320, 200),
            seed_concat=np.full((540, 640, 3), 30, np.uint8),
            instruction="unfold the cloth",
            perturbation=None,
            fps=15.0,
            policy_name="pi0",
            camera="primary",
        )
        assert out.exists() and out.stat().st_size > 0


def test_rejects_empty_and_malformed() -> None:
    good = _frames(3, 8, 8, 0)
    seed = np.zeros((16, 16, 3), np.uint8)
    common = dict(
        seed_concat=seed, instruction="x", perturbation=None, fps=10.0,
        policy_name="pi0", camera="primary",
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "d.rrd"
        for kwargs, needle in (
            (dict(original_frames=good, dream_frames=[]), "dream_frames is empty"),
            (dict(original_frames=[], dream_frames=good), "original_frames is empty"),
            (dict(original_frames=good, dream_frames=[np.zeros((8, 8), np.uint8)]),
             "(H, W, 3)"),
        ):
            try:
                export_dream_rerun(out, **kwargs, **common)
            except ValueError as e:
                assert needle in str(e)
            else:
                raise AssertionError(f"expected ValueError containing {needle!r}")


def _run_all() -> None:
    test_export_writes_nonempty_rrd()
    test_export_unperturbed_no_perturbation_card()
    test_rejects_empty_and_malformed()
    print("OK: all dream Rerun exporter checks passed")


if __name__ == "__main__":
    _run_all()
