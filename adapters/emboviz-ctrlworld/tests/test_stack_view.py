"""Tests for the Ctrl-World stack-view geometry — pure numpy/Pillow, no GPU.

The stack layout (which views, in which order, at which size) comes from the
checkpoint profile; these tests exercise the geometry with both the droid
profile's 3-view layout and a 2-view layout.

Run::

    uv run python adapters/emboviz-ctrlworld/tests/test_stack_view.py
"""

from __future__ import annotations

import numpy as np

from emboviz_ctrlworld.profiles import get_profile
from emboviz_ctrlworld.stack_view import build_stack_view, split_stack_view

_DROID = get_profile("droid")


def _view(value: int, h: int = 192, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def test_build_and_split_roundtrip_at_native_size() -> None:
    images = {"exterior_1": _view(10), "exterior_2": _view(20), "wrist": _view(30)}
    stack = build_stack_view(images, views=_DROID.views, view_hw=_DROID.view_hw)
    assert stack.shape == (*_DROID.stack_hw, 3) and stack.dtype == np.uint8

    views = split_stack_view(stack, views=_DROID.views)
    assert set(views) == set(_DROID.views)
    # Native-size inputs are not resampled, so the roundtrip is exact.
    for name in _DROID.views:
        assert np.array_equal(views[name], images[name])


def test_stack_order_follows_views_tuple() -> None:
    stack = build_stack_view(
        {"exterior_1": _view(1), "exterior_2": _view(2), "wrist": _view(3)},
        views=_DROID.views, view_hw=_DROID.view_hw,
    )
    h = _DROID.view_hw[0]
    # exterior_1 on top, exterior_2 in the middle, wrist at the bottom — the
    # order the droid checkpoint's latent stack was trained with.
    assert stack[0, 0, 0] == 1
    assert stack[h, 0, 0] == 2
    assert stack[2 * h, 0, 0] == 3


def test_two_view_layout() -> None:
    views = ("top", "wrist")
    stack = build_stack_view(
        {"top": _view(7, 480, 640), "wrist": _view(9, 480, 640)},
        views=views, view_hw=(192, 320),
    )
    assert stack.shape == (384, 320, 3)
    out = split_stack_view(stack, views=views)
    assert int(out["top"][0, 0, 0]) == 7 and int(out["wrist"][0, 0, 0]) == 9


def test_build_resizes_off_size_views() -> None:
    stack = build_stack_view(
        {"exterior_1": _view(10, 360, 640), "exterior_2": _view(20, 180, 320),
         "wrist": _view(30, 480, 640)},
        views=_DROID.views, view_hw=_DROID.view_hw,
    )
    assert stack.shape == (*_DROID.stack_hw, 3)
    out = split_stack_view(stack, views=_DROID.views)
    assert int(out["exterior_1"][0, 0, 0]) == 10  # uniform image survives resize


def test_build_rejects_missing_extra_and_non_uint8() -> None:
    for images, fragment in (
        ({"exterior_1": _view(1), "exterior_2": _view(2)}, "missing image"),
        ({"exterior_1": _view(1), "exterior_2": _view(2), "wrist": _view(3),
          "head": _view(4)}, "unknown view"),
        ({"exterior_1": _view(1).astype(np.float32), "exterior_2": _view(2),
          "wrist": _view(3)}, "uint8"),
    ):
        try:
            build_stack_view(images, views=_DROID.views, view_hw=_DROID.view_hw)
        except ValueError as e:
            assert fragment in str(e), str(e)
        else:
            raise AssertionError(f"expected ValueError ({fragment})")


def test_split_rejects_bad_shapes() -> None:
    for bad in (np.zeros((577, 320, 3), np.uint8), np.zeros((576, 320), np.uint8)):
        try:
            split_stack_view(bad, views=_DROID.views)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape {bad.shape}")


def _run_all() -> None:
    test_build_and_split_roundtrip_at_native_size()
    test_stack_order_follows_views_tuple()
    test_two_view_layout()
    test_build_resizes_off_size_views()
    test_build_rejects_missing_extra_and_non_uint8()
    test_split_rejects_bad_shapes()
    print("OK: all stack-view checks passed")


if __name__ == "__main__":
    _run_all()
