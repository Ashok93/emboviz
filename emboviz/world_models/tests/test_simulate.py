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

    def rollout(self, init, actions, *, num_frames=None) -> Trajectory:
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
    test_rejects_empty_actions_and_bad_seed()
    print("OK: all closed-loop simulator checks passed")


if __name__ == "__main__":
    _run_all()
