"""Closed-loop simulation — run a policy inside the world model, step by step.

This is the simulator at the heart of the stress test: the world model *is* the
environment, the policy *is* the thing under test. Starting from a (perturbed)
seed frame, each turn a ``step_fn`` produces the conditioning actions for the
current dreamed frame, the world model renders the next frames, and the last one
becomes the conditioning for the turn after. The policy reacting to what the
simulator showed it — a flight simulator whose simulator is the world model.

The loop is deliberately generic: it knows only the ``WorldModel`` wire contract
and a ``step_fn(conditioning_image) -> actions`` callable. Everything specific to
a world model or a policy (splitting cameras, tracking state, encoding actions)
lives behind ``step_fn`` in the adapter — never here.

Horizon. Each turn conditions on the *previous turn's dream*, so error compounds
across turns; the rollout is faithful for roughly the first turn or two and drifts
after. ``n_steps`` is small by intent — this tests the decisive moment, not a whole
task. Each turn is handed to ``on_step`` as it completes for incremental save.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from emboviz_wire.observations import RGBImage
from emboviz_wire.types import Observations, Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel

#: ``conditioning_image (H, W, 3) uint8 -> actions (n, action_dim)`` — the policy
#: step, opaque to the loop.
StepFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DreamRollout:
    """The result of a closed-loop simulation.

    ``trajectory`` is every generated frame, stitched in order; ``steps`` is the
    per-turn rollout (one :class:`Trajectory` each), so a caller can render each
    turn separately. ``seed_image`` is the (perturbed) frame the loop started from.
    """

    trajectory: Trajectory
    steps: list[Trajectory]
    seed_image: np.ndarray
    n_steps: int


def _conditioning_image_of(scene: Scene, camera: str) -> np.ndarray:
    return np.asarray(scene.observations.images[camera].data, dtype=np.uint8)


def _commit_frames(predicted: Trajectory, commit: int) -> Trajectory:
    """Return the first ``commit`` frames of a dreamed turn as a new Trajectory.

    The world model dreams the whole prediction horizon, but a receding-horizon
    loop commits only the leading ``commit`` frames before the policy re-plans.
    The remaining frames are speculative and dropped from the evaluated rollout.
    """
    return Trajectory(
        frames=predicted.frames[:commit],
        frame_indices=predicted.frame_indices[:commit],
        fps=predicted.fps,
        episode_id=predicted.episode_id,
        source=predicted.source,
        metadata={**predicted.metadata, "committed": commit, "dreamed": len(predicted.frames)},
    )


def closed_loop_rollout(
    world_model: WorldModel,
    seed_image: np.ndarray,
    step_fn: StepFn,
    *,
    n_steps: int,
    conditioning_camera: str = "primary",
    instruction: Optional[str] = None,
    execute_steps: Optional[int] = None,
    on_step: Optional[Callable[[int, Trajectory], None]] = None,
) -> DreamRollout:
    """Run the policy ⇄ world-model loop for ``n_steps`` turns from ``seed_image``.

    ``seed_image`` is the conditioning frame for the world model (e.g. a DROID
    ``concat_view``), already perturbed. ``instruction`` is the task text the world
    model conditions on each turn — required by language-conditioned world models
    (Cosmos rejects an empty prompt), so pass the seed episode's instruction.

    ``execute_steps`` is the execution horizon: each turn the world model dreams
    the full chunk ``step_fn`` conditions on, but the loop commits only the leading
    ``execute_steps`` frames and re-plans from there (receding horizon). ``None``
    commits every dreamed frame. ``step_fn`` must advance its own tracked state by
    the same number (the :class:`~emboviz_cosmos3.dream_step.PolicyDreamStepper`
    does, via its ``execute_steps``), so proprioception stays aligned with the
    committed conditioning frame.

    Returns a :class:`DreamRollout`. Raises if ``step_fn`` yields no usable actions
    or the world model returns no frames — never silently truncates the loop.
    """
    seed = np.asarray(seed_image, dtype=np.uint8)
    if seed.ndim != 3 or seed.shape[-1] != 3:
        raise ValueError(f"seed_image must be (H, W, 3) uint8 RGB, got shape {seed.shape}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if execute_steps is not None and int(execute_steps) < 1:
        raise ValueError(f"execute_steps must be >= 1, got {execute_steps}.")

    image = seed
    steps: list[Trajectory] = []
    all_frames: list[Scene] = []

    for step in range(n_steps):
        actions = np.asarray(step_fn(image), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(
                f"step_fn returned no usable actions (shape {actions.shape}) at step {step}."
            )

        scene = Scene(
            observations=Observations(
                images={conditioning_camera: RGBImage(data=image, camera_id=conditioning_camera)}
            ),
            instruction=instruction,
        )
        predicted = world_model.rollout(scene, actions)
        if not predicted.frames:
            raise RuntimeError(f"world model returned no frames at step {step}.")

        commit = len(predicted.frames) if execute_steps is None else int(execute_steps)
        if commit > len(predicted.frames):
            raise RuntimeError(
                f"execute_steps={execute_steps} exceeds the {len(predicted.frames)} "
                f"frames the world model dreamed at step {step}; the policy cannot "
                "commit more frames than were dreamed."
            )
        committed = _commit_frames(predicted, commit)

        steps.append(committed)
        all_frames.extend(committed.frames)
        if on_step is not None:
            on_step(step, committed)

        # Re-feed the LAST committed frame at whatever resolution the world model
        # returned it (Cosmos rounds height to a multiple of 16; NVIDIA's robotics
        # FD cookbook lets the conditioning ride at that size rather than resizing
        # back — re-resizing each turn only adds interpolation blur).
        image = _conditioning_image_of(committed.frames[commit - 1], conditioning_camera)

    trajectory = Trajectory(
        frames=all_frames,
        frame_indices=list(range(len(all_frames))),
        fps=getattr(steps[0], "fps", 0.0),
        episode_id="cosmos-dream",
        source=f"closed_loop:{getattr(world_model, 'model_id', 'wm')}",
        metadata={"n_steps": n_steps, "conditioning_camera": conditioning_camera},
    )
    return DreamRollout(trajectory=trajectory, steps=steps, seed_image=seed, n_steps=n_steps)


__all__ = ["DreamRollout", "StepFn", "closed_loop_rollout"]
