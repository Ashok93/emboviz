"""Tests for the masked counterfactual scene swapper — pure, no workers/GPU.

Uses fake detector / inpainter / inserter so the per-camera orchestration is
tested in isolation: detected cameras are edited (object replaced or removed),
undetected cameras keep their ORIGINAL image (so the policy always has a full
seed), the per-camera record is honest, and the none-detected branch behaves.

Run::

    uv run python emboviz/world_models/tests/test_scene_swap.py
"""

from __future__ import annotations

import numpy as np

from emboviz.core.types import Observations, RGBImage, Scene
from emboviz.perturb._target_detection import TargetDetection
from emboviz.perturb.image._image_utils import to_array
from emboviz.world_models.scene_swap import SceneSwapper

H, W = 16, 20
CONCAT = {"wrist": "wrist", "exterior_left": "primary", "exterior_right": "exterior_2"}


def _frame(wrist_val: int = 100, ext_val: int = 10) -> Scene:
    """A 3-camera frame. The wrist is 'bright' (detected by FakeDetector); the
    exteriors are 'dark' (not detected) — the realistic case where only the close
    wrist resolves the small object."""
    imgs = {
        "wrist": RGBImage(data=np.full((H, W, 3), wrist_val, np.uint8), camera_id="wrist"),
        "primary": RGBImage(data=np.full((H, W, 3), ext_val, np.uint8), camera_id="primary"),
        "exterior_2": RGBImage(data=np.full((H, W, 3), ext_val, np.uint8), camera_id="exterior_2"),
    }
    return Scene(observations=Observations(images=imgs), instruction="pick the marker", scene_id="f0")


class FakeDetector:
    """Detects only on 'bright' probe images; returns a 3x3 box mask."""

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def __call__(self, scene: Scene):
        arr = to_array(scene.observations.images["primary"].data)
        if arr.mean() <= self.threshold:
            return None
        mask = np.zeros((arr.shape[0], arr.shape[1]), bool)
        mask[2:5, 2:5] = True
        return TargetDetection(bbox=(2, 2, 5, 5), mask=mask, label="the marker", confidence=0.9)


class FakeInserter:
    """Insertion stand-in: paints 255 into the masked region."""

    def insert(self, image, mask, prompt):
        out = np.asarray(image).copy()
        out[np.asarray(mask).astype(bool)] = 255
        return out


class FakeInpainter:
    """Removal stand-in: zeroes the masked region (LaMa fills with background)."""

    def inpaint(self, image, mask, *, key=None):
        out = np.asarray(image).copy()
        out[np.asarray(mask).astype(bool)] = 0
        return out


def test_insert_swaps_detected_keeps_undetected() -> None:
    swapper = SceneSwapper(
        mask_query="the marker", replace_query="a spoon",
        detector=FakeDetector(), inserter=FakeInserter(),
    )
    res = swapper.swap(_frame(), CONCAT)

    assert res.any_edited and res.edited_regions == ["wrist"]
    assert (res.images_by_region["wrist"][2:5, 2:5] == 255).all()
    assert (res.images_by_region["exterior_left"] == 10).all()
    assert (res.images_by_region["exterior_right"] == 10).all()
    by_region = {c.region: c for c in res.per_camera}
    assert by_region["wrist"].edited and by_region["wrist"].operation == "insert"
    assert not by_region["exterior_left"].edited and not by_region["exterior_left"].detected
    assert "not detected" in by_region["exterior_right"].reason
    assert "insert" in res.summary() and "a spoon" in res.summary()


def test_remove_uses_inpainter_when_replace_empty() -> None:
    swapper = SceneSwapper(
        mask_query="the marker", detector=FakeDetector(), inpainter=FakeInpainter(),
    )
    res = swapper.swap(_frame(), CONCAT)
    assert res.any_edited
    assert (res.images_by_region["wrist"][2:5, 2:5] == 0).all()
    assert {c.region: c.operation for c in res.per_camera}["wrist"] == "remove"


def test_none_detected_keeps_all_original() -> None:
    swapper = SceneSwapper(
        mask_query="the marker", replace_query="a spoon",
        detector=FakeDetector(threshold=200.0), inserter=FakeInserter(),  # nothing bright enough
    )
    res = swapper.swap(_frame(), CONCAT)
    assert not res.any_edited and res.edited_regions == []
    assert (res.images_by_region["wrist"] == 100).all()
    assert all(not c.edited for c in res.per_camera)


def test_construction_requires_matching_tool() -> None:
    det = FakeDetector()
    for kwargs, needle in (
        (dict(mask_query="x", replace_query="a spoon", detector=det), "ObjectInserter"),
        (dict(mask_query="x", detector=det), "Inpainter"),
        (dict(mask_query="  ", replace_query="a spoon", detector=det, inserter=FakeInserter()),
         "non-empty"),
    ):
        try:
            SceneSwapper(**kwargs)
        except ValueError as e:
            assert needle in str(e), f"{needle!r} not in {e}"
        else:
            raise AssertionError(f"expected ValueError containing {needle!r}")


def test_mask_resolution_mismatch_raises() -> None:
    class BadMaskDetector:
        def __call__(self, scene):
            return TargetDetection(bbox=(0, 0, 1, 1), mask=np.ones((3, 3), bool), label="x", confidence=1.0)

    swapper = SceneSwapper(
        mask_query="x", replace_query="a spoon", detector=BadMaskDetector(), inserter=FakeInserter(),
    )
    try:
        swapper.swap(_frame(), CONCAT)
    except ValueError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("expected ValueError on mask/image resolution mismatch")


def _run_all() -> None:
    test_insert_swaps_detected_keeps_undetected()
    test_remove_uses_inpainter_when_replace_empty()
    test_none_detected_keeps_all_original()
    test_construction_requires_matching_tool()
    test_mask_resolution_mismatch_raises()
    print("OK: all scene-swap checks passed")


if __name__ == "__main__":
    _run_all()
