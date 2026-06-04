"""Critical-moment stress test — roll a policy forward from each decisive instant.

Where :mod:`emboviz.world_models.rollout` rolls a whole window and scores trust,
the stress test targets the instants that decide a task. For each keyframe
(grasp / release / settle) it seeds a short world-model rollout just before the
event, driven by an *action source*, and compares the predicted frames against
what actually happened over the same span. A short rollout from a real seed is
inside the world model's faithful horizon; a whole task is not — so this is the
honest unit of "what does the policy do at the moment that matters".

The action source is a callable ``(trajectory, seed_index, n_actions) -> actions``
returning the world model's conditioning actions. Two are useful:

  * :func:`recorded_action_source` — the episode's own logged actions, encoded by
    the world model. The predicted clip should track reality (a faithfulness
    check / baseline).
  * a policy source (built in the adapter from a ``VLAModel`` + the action
    bridge) — the user's policy drives, and the clip shows where it diverges.

Generic over the world model (the ``WorldModel`` wire contract) and the action
source. Cosmos-specific encoding lives in the adapter, never here. Each finished
clip is handed to ``on_clip`` immediately so a long run persists incrementally
rather than buffering everything to the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from emboviz_wire.types import Scene, Trajectory
from emboviz_wire.world_model_protocol import WorldModel

from emboviz.world_models.keyframes import Keyframe, detect_keyframes
from emboviz.world_models.trust import FrameMetric, frame_divergence

#: ``(trajectory, seed_index, n_actions) -> (n_actions, action_dim)`` conditioning
#: actions in the world model's normalized representation.
ActionSource = Callable[[Trajectory, int, int], np.ndarray]


@dataclass(frozen=True)
class StressClip:
    """One stress-tested keyframe.

    ``predicted`` is the world-model rollout seeded ``lead`` frames before the
    keyframe; ``aligned_real`` is the matching slice of the real episode (the
    same length); ``divergence`` is the per-frame predicted-vs-real divergence in
    ``metric``. For a recorded action source these should agree (faithfulness);
    for a policy source, divergence localizes where the policy departs from what
    really happened.
    """

    keyframe: Keyframe
    seed_index: int
    predicted: Trajectory
    aligned_real: Trajectory
    divergence: list[float]
    camera: str
    metric: FrameMetric


def recorded_action_source(world_model: WorldModel) -> ActionSource:
    """Action source that re-encodes the episode's own logged actions.

    Delegates to ``WorldModel.prepare_actions`` (the same per-domain encoding the
    trust driver uses), so the rollout is conditioned exactly as the model was
    trained. Used as the faithfulness baseline.
    """

    def source(traj: Trajectory, seed_index: int, n_actions: int) -> np.ndarray:
        return np.asarray(
            world_model.prepare_actions(traj, frame_start=seed_index, n_actions=n_actions),
            dtype=np.float32,
        )

    return source


def _frame_image(scene: Scene, camera: str) -> np.ndarray:
    return np.asarray(scene.observations.images[camera].data, dtype=np.uint8)


def stress_test(
    world_model: WorldModel,
    traj: Trajectory,
    *,
    action_source: ActionSource,
    n_actions: int = 16,
    lead_s: float = 0.5,
    conditioning_offset: int = 1,
    camera: str = "primary",
    metric: FrameMetric = "pixel_l2",
    keyframe_overrides: Optional[dict] = None,
    on_clip: Optional[Callable[[StressClip], None]] = None,
) -> list[StressClip]:
    """Stress-test every keyframe of ``traj`` and return the produced clips.

    For each keyframe the rollout is seeded ``round(lead_s * fps)`` frames before
    it (the pre-event approach, where a perturbation actually changes what the
    policy should do) and run for ``n_actions`` steps. A keyframe without room for
    the full rollout + its real comparison span (``n_actions + conditioning_offset``
    frames after the seed) is skipped; the count of produced vs total keyframes is
    visible to the caller (``len(result)`` vs ``len(detect_keyframes(...))``) so a
    short episode never silently drops events without it being observable.

    ``on_clip`` is called with each clip as it completes, for incremental save.
    """
    if traj.fps <= 0:
        raise ValueError(f"stress_test needs a positive fps, got {traj.fps}.")
    if n_actions < 1:
        raise ValueError(f"n_actions must be >= 1, got {n_actions}.")

    keyframes = detect_keyframes(traj, **(keyframe_overrides or {}))
    lead = int(round(lead_s * traj.fps))
    n_frames = len(traj.frames)
    needed = n_actions + conditioning_offset

    clips: list[StressClip] = []
    for kf in keyframes:
        seed_index = max(0, kf.index - lead)
        if seed_index + needed > n_frames:
            continue  # not enough real frames to roll + compare from this seed

        actions = np.asarray(action_source(traj, seed_index, n_actions), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(
                f"action source returned no usable actions (shape {actions.shape}) "
                f"for seed {seed_index}, n_actions={n_actions}."
            )

        predicted = world_model.rollout(traj.frames[seed_index], actions)
        start = seed_index + conditioning_offset
        aligned_real = traj[start : start + len(predicted.frames)]

        divergence = [
            frame_divergence(_frame_image(p, camera), _frame_image(r, camera), metric)
            for p, r in zip(predicted.frames, aligned_real.frames)
        ]

        clip = StressClip(
            keyframe=kf,
            seed_index=seed_index,
            predicted=predicted,
            aligned_real=aligned_real,
            divergence=divergence,
            camera=camera,
            metric=metric,
        )
        if on_clip is not None:
            on_clip(clip)
        clips.append(clip)

    return clips


__all__ = [
    "ActionSource",
    "StressClip",
    "recorded_action_source",
    "stress_test",
]
