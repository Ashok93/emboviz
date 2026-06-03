"""Tests for world-model trust calibration (pure, no GPU / no server).

Builds synthetic predicted/real trajectories with a controlled divergence
pattern and asserts the trust curve, noise floor, trust horizon, and the
action-dependence control behave as specified.

Run::

    uv run --with pillow python emboviz/world_models/tests/test_trust.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory

from emboviz.world_models.trust import (
    action_dependence,
    compute_trust_curve,
    frame_divergence,
)


def _traj(frames: list[np.ndarray], cam: str = "primary") -> Trajectory:
    scenes = [
        Scene(observations=Observations(images={cam: RGBImage(data=f, camera_id=cam)}))
        for f in frames
    ]
    return Trajectory(
        frames=scenes, frame_indices=list(range(len(scenes))), fps=5.0,
        episode_id="t", source="test", metadata={},
    )


def _solid(level: int, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), level, dtype=np.uint8)


# ── frame_divergence ─────────────────────────────────────────────────────────


def test_frame_divergence_identical_is_zero() -> None:
    f = _solid(100)
    assert frame_divergence(f, f, "pixel_l2") == 0.0
    assert frame_divergence(f, f, "ssim") < 1e-6


def test_frame_divergence_increases_with_difference() -> None:
    base = _solid(0)
    near = _solid(20)
    far = _solid(200)
    assert frame_divergence(base, near) < frame_divergence(base, far)
    assert 0.0 <= frame_divergence(base, far) <= 1.0


def test_frame_divergence_resizes_mismatched_sizes() -> None:
    a = _solid(50, h=16, w=16)
    b = _solid(50, h=32, w=48)
    # Same solid colour, different sizes → resize then ~0 divergence.
    assert frame_divergence(a, b, "pixel_l2") < 1e-6


# ── trust curve ──────────────────────────────────────────────────────────────


def test_perfect_prediction_is_fully_trusted() -> None:
    real = _traj([_solid(i * 5) for i in range(10)])
    predicted = _traj([_solid(i * 5) for i in range(10)])  # identical
    r = compute_trust_curve(predicted, real)
    assert r.trust_horizon == 10  # never drifts
    assert r.noise_floor < 1e-6
    assert max(r.divergence) < 1e-6


def test_trust_horizon_marks_where_drift_starts() -> None:
    # Predicted tracks real for frames 0..4, then diverges hard from frame 5.
    real = [_solid(30) for _ in range(10)]
    pred = [_solid(30) for _ in range(5)] + [_solid(220) for _ in range(5)]
    r = compute_trust_curve(_traj(pred), _traj(real))
    assert r.trust_horizon == 5, r.divergence
    # Frames before the break are within the band; the break frame exceeds it.
    assert all(d <= r.trust_band for d in r.divergence[:5])
    assert r.divergence[5] > r.trust_band


def test_noise_floor_anchors_the_band() -> None:
    # A small constant offset everywhere => floor>0, and (since uniform) the
    # whole rollout stays within multiplier*floor, i.e. fully trusted.
    real = [_solid(100) for _ in range(8)]
    pred = [_solid(108) for _ in range(8)]  # constant small offset
    r = compute_trust_curve(_traj(pred), _traj(real), trust_multiplier=2.0)
    assert r.noise_floor > 0.0
    assert r.trust_horizon == 8


def test_overlap_is_min_length() -> None:
    r = compute_trust_curve(_traj([_solid(0)] * 3), _traj([_solid(0)] * 7))
    assert len(r.horizons) == 3


# ── action-dependence control ────────────────────────────────────────────────


def test_action_dependence_detects_sensitivity() -> None:
    real = _traj([_solid(40) for _ in range(6)])
    # Real-action rollout tracks reality; shuffled-action rollout is far off.
    real_curve = compute_trust_curve(_traj([_solid(42) for _ in range(6)]), real)
    shuffled_curve = compute_trust_curve(_traj([_solid(200) for _ in range(6)]), real)
    verdict = action_dependence(real_curve, shuffled_curve)
    assert verdict["action_sensitive"] is True
    assert verdict["separation"] > 0


def test_action_dependence_flags_static_prior() -> None:
    real = _traj([_solid(40) for _ in range(6)])
    # Both rollouts equally (un)related to reality => no action sensitivity.
    same = compute_trust_curve(_traj([_solid(120) for _ in range(6)]), real)
    verdict = action_dependence(same, same)
    assert verdict["action_sensitive"] is False
    assert abs(verdict["separation"]) < 1e-9


def _run_all() -> None:
    test_frame_divergence_identical_is_zero()
    test_frame_divergence_increases_with_difference()
    test_frame_divergence_resizes_mismatched_sizes()
    test_perfect_prediction_is_fully_trusted()
    test_trust_horizon_marks_where_drift_starts()
    test_noise_floor_anchors_the_band()
    test_overlap_is_min_length()
    test_action_dependence_detects_sensitivity()
    test_action_dependence_flags_static_prior()
    print("OK: all world-model trust checks passed")


if __name__ == "__main__":
    _run_all()
