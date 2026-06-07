"""SD-inpaint worker checks that need no GPU / diffusers.

Exercises the input-validation contract of ``SDInpaintModel.fill`` (which runs
before any model load) and the resolution helper + handler dispatch table. The
actual diffusion forward needs the isolated runtime venv (torch + diffusers) and
a GPU, so it is not unit-tested here.

Run::

    uv run --with pillow --with numpy python adapters/emboviz-sd-inpaint/tests/test_model.py
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from emboviz_sd_inpaint.handler import SDInpaintHandler
from emboviz_sd_inpaint.model import SDInpaintModel, _round_to_multiple


def _png(h: int, w: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_round_to_multiple() -> None:
    assert _round_to_multiple(0) == 8
    assert _round_to_multiple(360) == 360
    assert _round_to_multiple(355) == 352
    assert _round_to_multiple(13, 8) == 16


def test_fill_validation_runs_before_load() -> None:
    m = SDInpaintModel(preload=False)  # no torch / diffusers import on this path
    mask = np.ones((8, 8), bool)

    for call, needle in (
        (lambda: m.fill(b"", mask, "a spoon"), "empty image"),
        (lambda: m.fill(b"notempty", mask, "   "), "non-empty prompt"),
        (lambda: m.fill(_png(8, 8), np.ones((4, 4), bool), "a spoon"), "does not match"),
        (lambda: m.fill(_png(8, 8), np.zeros((8, 8), bool), "a spoon"), "mask is empty"),
    ):
        try:
            call()
        except ValueError as e:
            assert needle in str(e), f"{needle!r} not in {e}"
        else:
            raise AssertionError(f"expected ValueError containing {needle!r}")


def test_handler_dispatch_table() -> None:
    h = SDInpaintHandler.from_kwargs(preload=False)
    assert set(h.methods) == {"fill", "health"}
    health = h.methods["health"]({})
    assert health["model_loaded"] is False and "model_id" in health


def _run_all() -> None:
    test_round_to_multiple()
    test_fill_validation_runs_before_load()
    test_handler_dispatch_table()
    print("OK: all sd-inpaint worker checks passed")


if __name__ == "__main__":
    _run_all()
