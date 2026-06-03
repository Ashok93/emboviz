"""Stage-2 orchestration: roll a recorded episode forward in a world model and
score how far the prediction can be trusted.

Given a real episode (a :class:`Trajectory` from any reader) and a
:class:`WorldModel`, this:

1. extracts the episode's real logged actions (``scene.metadata["expert_action"]``,
   which the readers populate from the dataset's action key),
2. conditions the world model on the episode's frame at ``frame_start`` and
   rolls those real actions forward,
3. aligns the predicted rollout to the corresponding real frames and computes
   the :mod:`emboviz.world_models.trust` curve,
4. runs the action-dependence control (a second rollout under shuffled actions)
   so the curve is only trusted when the world model actually responds to
   actions.

Alignment. The conditioning frame is real frame ``frame_start``; the first
generated frame is the next step, so predicted frame ``j`` is compared to real
frame ``frame_start + conditioning_offset + j`` (default offset 1). The
offset/cadence is embodiment-specific and is the one thing to confirm against a
known-good rollout before reading the numbers as ground truth — it is exposed as
a parameter rather than hard-assumed.

This module depends only on the two wire contracts (:class:`WorldModel`,
``Trajectory``), so it is testable with a mock world model and runs against any
real adapter unchanged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz_wire.types import Trajectory
from emboviz_wire.world_model_protocol import WorldModel

from emboviz.world_models.trust import (
    FrameMetric,
    action_dependence,
    compute_trust_curve,
)


def episode_actions(real: Trajectory) -> np.ndarray:
    """Return the episode's ``(T, action_dim)`` logged actions.

    Reads ``scene.metadata["expert_action"]`` from each frame — the per-step
    action the readers populate from the dataset's action key. Raises if any
    frame lacks it (the dataset's ``action`` key must be configured), rather
    than silently fabricating actions.
    """
    rows: list[list[float]] = []
    for i, scene in enumerate(real.frames):
        a = scene.metadata.get("expert_action")
        if a is None:
            raise ValueError(
                f"frame {i} has no 'expert_action' in metadata — the reader did "
                "not expose logged actions. Configure the dataset's `action` key "
                "so the rollout conditions on real actions, never invented ones."
            )
        rows.append(list(a))
    actions = np.asarray(rows, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"episode actions are not 2-D (T, action_dim): {actions.shape}")
    return actions


def _subtrajectory(traj: Trajectory, lo: int, hi: int) -> Trajectory:
    """A view of ``traj`` over frame indices ``[lo, hi)`` (clamped)."""
    lo = max(0, lo)
    hi = min(len(traj.frames), hi)
    frames = traj.frames[lo:hi]
    return Trajectory(
        frames=frames,
        frame_indices=list(traj.frame_indices[lo:hi]),
        fps=traj.fps,
        episode_id=traj.episode_id,
        source=traj.source,
        metadata=dict(traj.metadata),
    )


def rollout_episode(
    world_model: WorldModel,
    real: Trajectory,
    actions: np.ndarray,
    *,
    frame_start: int = 0,
    conditioning_offset: int = 1,
) -> tuple[Trajectory, Trajectory]:
    """Roll ``actions`` forward from ``real.frames[frame_start]`` and return
    ``(predicted, aligned_real)`` — the predicted rollout and the matching slice
    of the real episode, the same length, ready for :func:`compute_trust_curve`.
    """
    if not 0 <= frame_start < len(real.frames):
        raise IndexError(
            f"frame_start {frame_start} out of range for episode of "
            f"{len(real.frames)} frames"
        )
    conditioning = real.frames[frame_start]
    predicted = world_model.rollout(conditioning, actions)
    start = frame_start + conditioning_offset
    aligned_real = _subtrajectory(real, start, start + len(predicted.frames))
    return predicted, aligned_real


def trust_report(
    world_model: WorldModel,
    real: Trajectory,
    *,
    frame_start: int = 0,
    n_actions: Optional[int] = None,
    camera: str = "primary",
    metric: FrameMetric = "pixel_l2",
    conditioning_offset: int = 1,
    shuffle_seed: int = 0,
) -> dict:
    """Full Stage-2 trust analysis for one episode.

    Rolls the real actions forward (the trust curve), then rolls a shuffled-action
    control forward (the action-dependence check), both compared against the same
    real frames. Returns the trust curve, the control verdict, and the headline
    numbers a report/Rerun view would surface.
    """
    actions = episode_actions(real)[frame_start:]
    if n_actions is not None:
        actions = actions[:n_actions]
    if len(actions) == 0:
        raise ValueError("no actions to roll out from frame_start")

    predicted, aligned_real = rollout_episode(
        world_model, real, actions,
        frame_start=frame_start, conditioning_offset=conditioning_offset,
    )
    real_curve = compute_trust_curve(predicted, aligned_real, camera=camera, metric=metric)

    # Action-dependence control: same conditioning + frames, shuffled action order.
    rng = np.random.RandomState(shuffle_seed)
    shuffled = actions[rng.permutation(len(actions))]
    predicted_sh, _ = rollout_episode(
        world_model, real, shuffled,
        frame_start=frame_start, conditioning_offset=conditioning_offset,
    )
    shuffled_curve = compute_trust_curve(
        predicted_sh, aligned_real, camera=camera, metric=metric,
    )
    dependence = action_dependence(real_curve, shuffled_curve)

    return {
        "episode_id": real.episode_id,
        "frame_start": frame_start,
        "n_actions": int(len(actions)),
        "camera": camera,
        "metric": metric,
        "trust_horizon": real_curve.trust_horizon,
        "noise_floor": real_curve.noise_floor,
        "trust_band": real_curve.trust_band,
        "divergence": real_curve.divergence,
        "horizons": real_curve.horizons,
        "action_dependence": dependence,
        "world_model": getattr(world_model, "model_id", "unknown"),
        "rollout_metadata": dict(predicted.metadata),
    }


def summarize(report: dict) -> str:
    """A one-paragraph plain-text verdict from a :func:`trust_report` result."""
    th = report["trust_horizon"]
    n = report["n_actions"]
    dep = report["action_dependence"]
    wm = report["world_model"]
    lines = [
        f"World model '{wm}' rolled out {n} real actions from frame "
        f"{report['frame_start']} of episode {report['episode_id']}.",
    ]
    if not dep["action_sensitive"]:
        lines.append(
            "REFUSED: the shuffled-action control shows the model is not "
            f"responding to actions (separation {dep['separation']:.4f} < "
            f"margin {dep['margin']}). The rollout reflects a static prior, not "
            "physics — it cannot be trusted to evaluate a policy here."
        )
    elif th >= n:
        lines.append(
            f"TRUSTED across all {n} frames: prediction stays within the noise "
            f"floor band (floor {report['noise_floor']:.4f}). Action-dependence "
            f"confirmed (separation {dep['separation']:.4f})."
        )
    else:
        lines.append(
            f"TRUSTED to horizon {th}/{n}: the prediction tracks reality for "
            f"{th} frames, then drift exceeds the noise floor band "
            f"({report['trust_band']:.4f}). Beyond frame {th}, a verdict computed "
            "from this rollout would be reading the world model's hallucination. "
            f"Action-dependence confirmed (separation {dep['separation']:.4f})."
        )
    return " ".join(lines)
