"""Tests for the critical-moment stress test — pure, no GPU.

A mock world model renders each action row as a solid colour; a synthetic
episode with hand-placed grasp/settle events drives keyframe detection. Checks
that clips are produced at the keyframes (and only where there is room), that
predicted/real are aligned, that divergence is per-frame, and that ``on_clip``
fires incrementally.

Run::

    uv run python emboviz/world_models/tests/test_stress.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.observations.gripper import GripperState
from emboviz_wire.observations.state import Proprioception
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability

from emboviz.world_models.keyframes import detect_keyframes
from emboviz.world_models.stress import recorded_action_source, stress_test


def _solid(level: int, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w, 3), int(np.clip(level, 0, 255)), dtype=np.uint8)


_XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] + [0.5] * 6 + [0.6, 0.7, 0.8, 0.9] + [0.9] * 4
_GRIPPERS = [0.0] * 8 + [1.0] * 12


def _episode(fps: float = 10.0) -> Trajectory:
    frames = []
    for i, (x, g) in enumerate(zip(_XS, _GRIPPERS)):
        frames.append(
            Scene(
                observations=Observations(
                    images={"primary": RGBImage(data=_solid(i * 5), camera_id="primary")},
                    state=Proprioception(
                        values=np.array([x, 0, 0, 0, 0, 0], dtype=np.float32), convention="ee_pose"
                    ),
                    gripper=GripperState(value=float(g)),
                )
            )
        )
    return Trajectory(frames=frames, fps=fps, episode_id="stress", source="test")


class _MockWM(WorldModel):
    """Renders each action row as a solid frame of colour ``action[0]``."""

    @property
    def model_id(self) -> str:
        return "mock-wm"

    @property
    def capabilities(self) -> WorldModelCapability:
        return WorldModelCapability.FORWARD_DYNAMICS

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def supported_domains(self) -> frozenset:
        return frozenset({"mock"})

    def prepare_actions(self, episode, *, frame_start=0, n_actions=None):
        # Deterministic, distinct per step so alignment is observable.
        n = n_actions if n_actions is not None else len(episode.frames) - frame_start - 1
        return (np.arange(n, dtype=np.float32) + frame_start).reshape(-1, 1)

    def rollout(self, init, actions, *, num_frames=None) -> Trajectory:
        actions = np.asarray(actions)
        frames = [
            Scene(observations=Observations(images={"primary": RGBImage(data=_solid(int(a[0])), camera_id="primary")}))
            for a in actions
        ]
        return Trajectory(frames=frames, frame_indices=list(range(len(frames))), fps=10.0, episode_id="pred", source="mock")


def test_clips_at_keyframes_with_room_only() -> None:
    traj = _episode()
    kfs = detect_keyframes(traj)  # at 6, 8, 16
    seen: list[int] = []

    def on_clip(clip):
        seen.append(clip.keyframe.index)

    clips = stress_test(
        _MockWM(), traj,
        action_source=lambda t, s, n: np.zeros((n, 1), dtype=np.float32),
        n_actions=16, lead_s=0.5, on_clip=on_clip,
    )
    # lead=5 frames: kf@6 -> seed 1 (room), kf@8 -> seed 3 (room), kf@16 -> seed 11
    # needs 11+17=28 > 20 frames -> skipped. So 2 of 3 keyframes produced.
    assert len(kfs) == 3
    assert [c.keyframe.index for c in clips] == [6, 8]
    assert seen == [6, 8]  # on_clip fired incrementally, in order


def test_predicted_and_real_aligned() -> None:
    traj = _episode()
    clips = stress_test(
        _MockWM(), traj,
        action_source=lambda t, s, n: np.zeros((n, 1), dtype=np.float32),
        n_actions=16, lead_s=0.5,
    )
    c = clips[0]  # kf@6, seed 1
    assert c.seed_index == 1
    assert len(c.predicted.frames) == 16
    assert len(c.aligned_real.frames) == 16          # traj[2:18]
    assert len(c.divergence) == 16
    # aligned_real starts at seed + conditioning_offset = 2
    assert c.aligned_real.frame_indices[0] == 2


def test_recorded_action_source_passes_through_prepare_actions() -> None:
    traj = _episode()
    src = recorded_action_source(_MockWM())
    actions = src(traj, 3, 5)
    # _MockWM.prepare_actions returns arange + frame_start, shaped (n, 1).
    assert actions.shape == (5, 1)
    assert actions[0, 0] == 3.0 and actions[4, 0] == 7.0


def test_recorded_source_drives_a_faithful_rollout() -> None:
    # With recorded actions the mock renders colour == action value; divergence
    # is computed against the real frames (here arbitrary), but the path runs
    # end-to-end and produces one clip per in-range keyframe.
    traj = _episode()
    clips = stress_test(
        _MockWM(), traj,
        action_source=recorded_action_source(_MockWM()),
        n_actions=12, lead_s=0.3,
    )
    assert clips, "expected at least one stress clip"
    for c in clips:
        assert len(c.predicted.frames) == 12
        assert len(c.divergence) == len(c.aligned_real.frames)


def _run_all() -> None:
    test_clips_at_keyframes_with_room_only()
    test_predicted_and_real_aligned()
    test_recorded_action_source_passes_through_prepare_actions()
    test_recorded_source_drives_a_faithful_rollout()
    print("OK: all stress-test checks passed")


if __name__ == "__main__":
    _run_all()
