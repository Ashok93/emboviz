"""Tests for Stage-2 rollout orchestration (pure, no GPU / no server).

Uses a mock action-sensitive world model: each action row's first element is the
solid colour of the frame it produces, with each episode action set to the NEXT
frame's colour. Real-order actions therefore reproduce the real future exactly
(fully trusted); shuffled actions produce the wrong colours (action-dependent).

Run::

    uv run --with pillow python emboviz/world_models/tests/test_rollout.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability

from emboviz.world_models.rollout import (
    analyze_trust,
    reanchored_rollout,
    rollout_episode,
    summarize,
    trust_report,
)
from emboviz.world_models.viz import save_frame_comparison


def _solid(level: int, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w, 3), int(np.clip(level, 0, 255)), dtype=np.uint8)


def _episode(n: int = 8, cam: str = "primary") -> Trajectory:
    """Frames coloured 0,10,20,…; each frame's expert_action is the NEXT frame's
    colour, so a model that renders 'action colour' reproduces the real future."""
    scenes = []
    for k in range(n):
        nxt = min((k + 1) * 10, 255)  # next frame's colour
        scenes.append(Scene(
            observations=Observations(images={cam: RGBImage(data=_solid(k * 10), camera_id=cam)}),
            metadata={"expert_action": [float(nxt)]},
        ))
    return Trajectory(
        frames=scenes, frame_indices=list(range(n)), fps=5.0,
        episode_id="ep0", source="test", metadata={},
    )


class _MockColorWM(WorldModel):
    """Renders each action row as a solid frame of colour ``action[0]``."""

    @property
    def model_id(self) -> str:
        return "mock-color-wm"

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability.FORWARD_DYNAMICS

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def supported_domains(self) -> frozenset:
        return frozenset({"mock"})

    def rollout(self, init, actions, *, num_frames=None) -> Trajectory:
        reason = self.validate_rollout(init, np.asarray(actions))
        if reason:
            raise ValueError(reason)
        frames = [
            Scene(observations=Observations(
                images={"primary": RGBImage(data=_solid(int(a[0])), camera_id="primary")}))
            for a in np.asarray(actions)
        ]
        return Trajectory(
            frames=frames, frame_indices=list(range(len(frames))), fps=5.0,
            episode_id="pred", source="mock", metadata={"world_model": "mock-color-wm"},
        )


# ── prepare_actions default (logged actions) ─────────────────────────────────


def test_prepare_actions_default_extracts_expert_action() -> None:
    # _MockColorWM does not override prepare_actions, so it uses the WorldModel
    # ABC default: the per-frame expert_action.
    a = _MockColorWM().prepare_actions(_episode(5))
    assert a.shape == (5, 1)
    assert a[0, 0] == 10.0 and a[3, 0] == 40.0
    # frame_start + n_actions slice the result.
    assert _MockColorWM().prepare_actions(_episode(5), frame_start=1, n_actions=2).shape == (2, 1)


def test_prepare_actions_default_missing_raises() -> None:
    cam = "primary"
    bad = Trajectory(
        frames=[Scene(observations=Observations(images={cam: RGBImage(data=_solid(0), camera_id=cam)}))],
        frame_indices=[0], fps=5.0, episode_id="x", source="t", metadata={},
    )
    try:
        _MockColorWM().prepare_actions(bad)
    except ValueError as e:
        assert "expert_action" in str(e)
    else:
        raise AssertionError("expected ValueError for missing expert_action")


# ── alignment ────────────────────────────────────────────────────────────────


def test_rollout_alignment_lengths_match() -> None:
    ep = _episode(8)
    actions = _MockColorWM().prepare_actions(ep, n_actions=7)  # window from frame 0
    predicted, aligned_real = rollout_episode(_MockColorWM(), ep, actions, frame_start=0)
    assert len(predicted.frames) == len(aligned_real.frames) == 7
    # aligned_real starts at frame_start + 1 = frame index 1
    assert aligned_real.frame_indices[0] == 1


# ── full trust report ────────────────────────────────────────────────────────


def test_trust_report_real_actions_fully_trusted() -> None:
    ep = _episode(8)
    report = trust_report(_MockColorWM(), ep, frame_start=0, n_actions=7)
    # Real-order actions reproduce the real future exactly.
    assert report["trust_horizon"] == report["n_actions"] == 7
    assert max(report["divergence"]) < 1e-6
    # And the shuffled control proves the model is action-sensitive.
    assert report["action_dependence"]["action_sensitive"] is True
    assert report["action_dependence"]["separation"] > 0
    text = summarize(report)
    assert "TRUSTED across all" in text


def test_summarize_static_prior_is_refused() -> None:
    # A world model that ignores actions (always renders the conditioning colour)
    # → real and shuffled rollouts identical → action-dependence fails → refused.
    class _StaticWM(_MockColorWM):
        def rollout(self, init, actions, *, num_frames=None) -> Trajectory:
            n = len(np.asarray(actions))
            frames = [Scene(observations=Observations(
                images={"primary": RGBImage(data=_solid(200), camera_id="primary")}))
                for _ in range(n)]
            return Trajectory(frames=frames, frame_indices=list(range(n)), fps=5.0,
                              episode_id="pred", source="mock", metadata={})

    report = trust_report(_StaticWM(), _episode(8), frame_start=0, n_actions=7)
    assert report["action_dependence"]["action_sensitive"] is False
    assert "REFUSED" in summarize(report)


def test_analyze_trust_exposes_frames_and_renders() -> None:
    import tempfile
    from pathlib import Path

    analysis = analyze_trust(_MockColorWM(), _episode(8), frame_start=0, n_actions=7)
    # The rollout trajectories are exposed, same length, ready to render.
    assert len(analysis.predicted.frames) == len(analysis.aligned_real.frames) == 7
    assert analysis.report["trust_horizon"] == 7
    out = Path(tempfile.mkdtemp())
    n = save_frame_comparison(
        analysis.predicted, analysis.aligned_real, analysis.report["divergence"], out,
        camera=analysis.report["camera"], trust_band=analysis.report["trust_band"],
    )
    pngs = sorted(out.glob("compare_*.png"))
    assert n == 7 and len(pngs) == 7


# ── re-anchored (closed-loop) rollout ────────────────────────────────────────


def test_reanchored_rollout_drops_tiny_tail_and_emits_segments() -> None:
    # 12 frames -> 11 real actions. Re-anchor every 3, generate 8 per chunk.
    # offsets 0,3,6 each leave >= min_chunk actions and keep 3 frames; the final
    # offset 9 has only 2 actions left (< min_chunk=4) and is dropped, not sent.
    segments: list[tuple[int, int, int]] = []

    def on_segment(out_start, predicted, real_frames) -> None:
        segments.append((out_start, len(predicted), len(real_frames)))

    predicted, aligned_real = reanchored_rollout(
        _MockColorWM(), _episode(12),
        frame_start=0, n_actions=11, reanchor_every=3,
        gen_chunk=8, min_chunk=4, on_segment=on_segment,
    )
    # 3 kept segments of 3 frames; the 2-frame tail at offset 9 is dropped.
    assert [s[0] for s in segments] == [0, 3, 6]
    assert all(n_pred == n_real == 3 for _, n_pred, n_real in segments)
    assert len(predicted.frames) == len(aligned_real.frames) == 9
    assert predicted.metadata["reanchor_every"] == 3


def test_reanchored_rollout_reproduces_real_after_each_anchor() -> None:
    # The mock renders 'action colour'; real-order actions reproduce the real
    # future exactly, so a re-anchored rollout tracks reality with zero drift.
    predicted, aligned_real = reanchored_rollout(
        _MockColorWM(), _episode(12),
        frame_start=0, n_actions=9, reanchor_every=3, gen_chunk=8, min_chunk=4,
    )
    for p, r in zip(predicted.frames, aligned_real.frames):
        pi = np.asarray(p.observations.images["primary"].data)
        ri = np.asarray(r.observations.images["primary"].data)
        assert int(pi[0, 0, 0]) == int(ri[0, 0, 0])


def _run_all() -> None:
    test_prepare_actions_default_extracts_expert_action()
    test_prepare_actions_default_missing_raises()
    test_rollout_alignment_lengths_match()
    test_trust_report_real_actions_fully_trusted()
    test_summarize_static_prior_is_refused()
    test_analyze_trust_exposes_frames_and_renders()
    test_reanchored_rollout_drops_tiny_tail_and_emits_segments()
    test_reanchored_rollout_reproduces_real_after_each_anchor()
    print("OK: all world-model rollout checks passed")


if __name__ == "__main__":
    _run_all()
