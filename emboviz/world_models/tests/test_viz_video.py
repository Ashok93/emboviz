"""Tests for save_video / frames_to_arrays (no GPU; needs imageio[pyav]).

Run::

    uv run --with "imageio[pyav]" python emboviz/world_models/tests/test_viz_video.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz.world_models.viz import frames_to_arrays, save_video


def test_save_video_writes_and_reads_back() -> None:
    frames = [np.full((8, 8, 3), i * 30, np.uint8) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        n = save_video(frames, path, fps=10.0)
        assert n == 5 and path.exists() and path.stat().st_size > 0
        import imageio.v3 as iio
        back = np.asarray(iio.imread(path, plugin="pyav"))
        assert back.shape[0] == 5 and back.shape[1:3] == (8, 8)


def test_save_video_trims_odd_dimensions() -> None:
    frames = [np.zeros((7, 9, 3), np.uint8) for _ in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "odd.mp4"
        save_video(frames, path, fps=5.0)
        import imageio.v3 as iio
        back = np.asarray(iio.imread(path, plugin="pyav"))
        assert back.shape[1:3] == (6, 8)            # trimmed to even


def test_save_video_rejects_mixed_shapes() -> None:
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)]
    try:
        save_video(frames, Path("/tmp/x.mp4"))
    except ValueError as e:
        assert "differing shapes" in str(e)
    else:
        raise AssertionError("expected ValueError for mixed shapes")


def test_frames_to_arrays_extracts_camera() -> None:
    frames = [
        Scene(observations=Observations(images={"primary": RGBImage(data=np.full((4, 4, 3), i, np.uint8))}))
        for i in range(3)
    ]
    traj = Trajectory(frames=frames, fps=10.0, episode_id="t", source="test")
    arrs = frames_to_arrays(traj, "primary")
    assert len(arrs) == 3 and arrs[1][0, 0, 0] == 1


def _run_all() -> None:
    test_save_video_writes_and_reads_back()
    test_save_video_trims_odd_dimensions()
    test_save_video_rejects_mixed_shapes()
    test_frames_to_arrays_extracts_camera()
    print("OK: all video-output checks passed")


if __name__ == "__main__":
    _run_all()
