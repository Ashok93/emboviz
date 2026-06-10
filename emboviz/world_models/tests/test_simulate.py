"""Tests for the closed-loop simulator — pure, no GPU.

A mock world model renders each action row as a solid frame; a mock step_fn
returns fixed actions and records the conditioning image it was handed, so the
test verifies the loop carries the last dreamed frame forward into the next turn,
accumulates all frames, and fires on_step incrementally.

Run::

    uv run python emboviz/world_models/tests/test_simulate.py
"""

from __future__ import annotations

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel, WorldModelCapability

from emboviz.world_models.simulate import closed_loop_rollout


class _MockWM(WorldModel):
    """Renders ``len(actions)`` frames; the last frame's colour encodes the step
    via a counter, so the next conditioning image is observably different."""

    def __init__(self) -> None:
        self._step = 0

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

    def rollout(self, init, actions, *, history=None, num_frames=None) -> Trajectory:
        self._step += 1
        color = min(self._step * 40, 240)
        n = len(np.asarray(actions))
        frames = [
            Scene(observations=Observations(
                images={"primary": RGBImage(data=np.full((6, 6, 3), color, np.uint8), camera_id="primary")}))
            for _ in range(n)
        ]
        return Trajectory(frames=frames, frame_indices=list(range(n)), fps=10.0, episode_id="p", source="mock")


def test_loop_carries_frame_forward_and_accumulates() -> None:
    seen_inputs: list[int] = []
    on_step_calls: list[int] = []

    def step_fn(image: np.ndarray) -> np.ndarray:
        seen_inputs.append(int(image[0, 0, 0]))  # colour of the conditioning frame
        return np.zeros((4, 1), dtype=np.float32)

    def on_step(i, traj):
        on_step_calls.append(i)

    seed = np.zeros((6, 6, 3), np.uint8)  # seed colour 0
    out = closed_loop_rollout(_MockWM(), seed, step_fn, n_steps=3, on_step=on_step)

    assert out.n_steps == 3 and len(out.steps) == 3
    assert len(out.trajectory.frames) == 12                 # 3 turns x 4 frames
    # Turn 0 conditioned on the seed (0); turns 1,2 on the previous dream (40, 80).
    assert seen_inputs == [0, 40, 80]
    assert on_step_calls == [0, 1, 2]                       # fired incrementally, in order
    assert np.array_equal(out.seed_image, seed)


class _IndexedWM(_MockWM):
    """Like ``_MockWM`` but stamps each frame's *index within the turn* into the
    green channel, so a test can tell which dreamed frame the loop committed and
    carried forward."""

    def rollout(self, init, actions, *, history=None, num_frames=None) -> Trajectory:
        self._step += 1
        color = min(self._step * 40, 240)
        n = len(np.asarray(actions))
        frames = []
        for j in range(n):
            px = np.zeros((6, 6, 3), np.uint8)
            px[:, :, 0] = color   # red: turn
            px[:, :, 1] = j       # green: frame index within the turn
            frames.append(Scene(observations=Observations(
                images={"primary": RGBImage(data=px, camera_id="primary")})))
        return Trajectory(frames=frames, frame_indices=list(range(n)), fps=10.0, episode_id="p", source="mock")


def test_execute_steps_commits_prefix_and_carries_committed_frame() -> None:
    carried_green: list[int] = []
    committed_lengths: list[int] = []

    def step_fn(image: np.ndarray) -> np.ndarray:
        carried_green.append(int(image[0, 0, 1]))   # frame-in-turn index carried forward
        return np.zeros((4, 1), dtype=np.float32)    # dream 4 frames per turn

    def on_step(i, traj):
        committed_lengths.append(len(traj.frames))

    seed = np.zeros((6, 6, 3), np.uint8)
    out = closed_loop_rollout(
        _IndexedWM(), seed, step_fn, n_steps=3, execute_steps=2, on_step=on_step
    )

    # Dream 4 per turn, commit 2 -> 3 turns x 2 = 6 frames in the evaluated rollout.
    assert len(out.trajectory.frames) == 6
    assert committed_lengths == [2, 2, 2]
    # Each committed turn keeps the prefix frames[:2] -> green indices 0,1.
    for turn in out.steps:
        assert [int(f.observations.images["primary"].data[0, 0, 1]) for f in turn.frames] == [0, 1]
        assert turn.metadata["committed"] == 2 and turn.metadata["dreamed"] == 4
    # Turn 0 sees the seed (green 0); turns 1,2 are conditioned on the *committed*
    # frame frames[1] of the previous turn (green index 1), not the last dreamed frame.
    assert carried_green == [0, 1, 1]


def test_execute_steps_cannot_exceed_dreamed_frames() -> None:
    try:
        closed_loop_rollout(
            _MockWM(), np.zeros((6, 6, 3), np.uint8),
            lambda im: np.zeros((2, 1), np.float32),   # only 2 frames dreamed
            n_steps=1, execute_steps=3,                # but asked to commit 3
        )
    except RuntimeError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("expected RuntimeError when execute_steps exceeds dreamed frames")


class _HistoryWM(_MockWM):
    """Declares ``conditions_on_history`` and records the history each call sees,
    so a test can verify the loop accumulates [seed, committed-frame-per-turn]."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_history_lengths: list[int] = []
        self.seen_init_states: list = []

    @property
    def conditions_on_history(self) -> bool:
        return True

    def rollout(self, init, actions, *, history=None, num_frames=None) -> Trajectory:
        assert history is not None, "loop must pass history to a history-conditioned model"
        self.seen_history_lengths.append(len(history.frames))
        self.seen_init_states.append(init.observations.state)
        return super().rollout(init, actions, history=history, num_frames=num_frames)


def test_history_conditioned_loop_accumulates_anchors() -> None:
    wm = _HistoryWM()
    seed = np.zeros((6, 6, 3), np.uint8)
    closed_loop_rollout(
        wm, seed, lambda im: np.zeros((4, 1), np.float32), n_steps=3,
        seed_state=np.arange(6, dtype=np.float32), seed_gripper=0.5,
    )
    # Turn t sees the seed plus one committed anchor per earlier turn.
    assert wm.seen_history_lengths == [1, 2, 3]
    # The seed conditioning scene carries the pose the caller supplied.
    assert wm.seen_init_states[0] is not None
    assert wm.seen_init_states[0].convention == "ee_pose"


def test_history_conditioned_loop_requires_seed_pose() -> None:
    try:
        closed_loop_rollout(
            _HistoryWM(), np.zeros((6, 6, 3), np.uint8),
            lambda im: np.zeros((4, 1), np.float32), n_steps=1,
        )
    except ValueError as e:
        assert "seed_state" in str(e)
    else:
        raise AssertionError("expected ValueError when seed pose is missing")


def test_rejects_empty_actions_and_bad_seed() -> None:
    try:
        closed_loop_rollout(_MockWM(), np.zeros((4, 4, 3), np.uint8),
                            lambda im: np.zeros((0, 1), np.float32), n_steps=1)
    except ValueError as e:
        assert "no usable actions" in str(e)
    else:
        raise AssertionError("expected ValueError for empty actions")

    try:
        closed_loop_rollout(_MockWM(), np.zeros((4, 4), np.uint8),
                            lambda im: np.zeros((2, 1), np.float32), n_steps=1)
    except ValueError as e:
        assert "(H, W, 3)" in str(e)
    else:
        raise AssertionError("expected ValueError for bad seed shape")


def _run_all() -> None:
    test_loop_carries_frame_forward_and_accumulates()
    test_execute_steps_commits_prefix_and_carries_committed_frame()
    test_execute_steps_cannot_exceed_dreamed_frames()
    test_history_conditioned_loop_accumulates_anchors()
    test_history_conditioned_loop_requires_seed_pose()
    test_rejects_empty_actions_and_bad_seed()
    print("OK: all closed-loop simulator checks passed")


if __name__ == "__main__":
    _run_all()
