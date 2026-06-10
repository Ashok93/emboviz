"""Tests for the Ctrl-World stack-view geometry — pure numpy/Pillow, no GPU.

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_stack_view.py
"""

from __future__ import annotations

import numpy as np

from emboviz_ctrlworld.stack_view import (
    STACK_VIEW_ORDER,
    VIEW_HW,
    build_stack_view,
    split_stack_view,
)


def _view(value: int, h: int = 192, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def test_build_and_split_roundtrip_at_native_size() -> None:
    ext1, ext2, wrist = _view(10), _view(20), _view(30)
    stack = build_stack_view(ext1, ext2, wrist)
    assert stack.shape == (VIEW_HW[0] * 3, VIEW_HW[1], 3) and stack.dtype == np.uint8

    views = split_stack_view(stack)
    assert set(views) == set(STACK_VIEW_ORDER)
    # Native-size inputs are not resampled, so the roundtrip is exact.
    assert np.array_equal(views["exterior_1"], ext1)
    assert np.array_equal(views["exterior_2"], ext2)
    assert np.array_equal(views["wrist"], wrist)


def test_stack_order_is_training_order() -> None:
    stack = build_stack_view(_view(1), _view(2), _view(3))
    # exterior_1 on top, exterior_2 in the middle, wrist at the bottom — the
    # order the checkpoint's latent stack was trained with.
    assert stack[0, 0, 0] == 1
    assert stack[VIEW_HW[0], 0, 0] == 2
    assert stack[2 * VIEW_HW[0], 0, 0] == 3


def test_build_resizes_off_size_views() -> None:
    stack = build_stack_view(_view(10, 360, 640), _view(20, 180, 320), _view(30, 480, 640))
    assert stack.shape == (VIEW_HW[0] * 3, VIEW_HW[1], 3)
    views = split_stack_view(stack)
    assert int(views["exterior_1"][0, 0, 0]) == 10  # uniform image survives resize


def test_split_rejects_bad_shapes() -> None:
    for bad in (np.zeros((577, 320, 3), np.uint8), np.zeros((576, 320), np.uint8)):
        try:
            split_stack_view(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape {bad.shape}")


def test_build_rejects_non_uint8() -> None:
    try:
        build_stack_view(_view(1).astype(np.float32), _view(2), _view(3))
    except ValueError as e:
        assert "uint8" in str(e)
    else:
        raise AssertionError("expected ValueError for non-uint8 view")


def _run_all() -> None:
    test_build_and_split_roundtrip_at_native_size()
    test_stack_order_is_training_order()
    test_build_resizes_off_size_views()
    test_split_rejects_bad_shapes()
    test_build_rejects_non_uint8()
    print("OK: all stack-view checks passed")


if __name__ == "__main__":
    _run_all()
