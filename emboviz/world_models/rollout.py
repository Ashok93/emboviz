"""Stage-2 orchestration: roll a recorded episode forward in a world model and
score how far the prediction can be trusted.

Given a real episode (a :class:`Trajectory` from any reader) and a
:class:`WorldModel`, this:

1. asks the world model to encode the episode into its conditioning actions
   (``WorldModel.prepare_actions`` — raw logged actions by default, or a
   per-embodiment encoding such as Cosmos's normalized pose deltas),
2. conditions the world model on the episode's frame at ``frame_start`` and
   rolls those actions forward,
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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from emboviz_wire.types import Trajectory
from emboviz_wire.world_model_protocol import WorldModel

from emboviz.world_models.trust import (
    FrameMetric,
    action_dependence,
    compute_trust_curve,
)


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


def reanchored_rollout(
    world_model: WorldModel,
    real: Trajectory,
    *,
    frame_start: int = 0,
    n_actions: int,
    reanchor_every: int,
    gen_chunk: int = 16,
    min_chunk: int = 4,
    conditioning_offset: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
    on_segment: Optional[Callable[[int, list, list], None]] = None,
) -> tuple[Trajectory, Trajectory]:
    """Roll the real actions forward, re-anchoring to the REAL frame every
    ``reanchor_every`` steps so prediction error never compounds past that
    horizon (the closed-loop / re-anchoring scheme: predict K, snap back to the
    real frame, predict K more, …).

    Each segment conditions on the real frame at its start and generates a
    ``gen_chunk``-frame chunk (the world model's stable granularity), of which
    the first ``reanchor_every`` frames are kept before re-anchoring. Returns
    ``(predicted, aligned_real)`` — the stitched re-anchored prediction and the
    matching real frames, the same length, ready for :func:`compute_trust_curve`.

    Cheap on the wire: actions are encoded once for the whole window, then sliced
    per segment; only one conditioning frame + its actions cross per generation.

    ``on_segment(out_start, predicted_frames, real_frames)`` is called after each
    segment with its newly generated frames and their real counterparts, so a
    caller can persist results incrementally (a long run must never buffer
    everything and lose it on a late failure). A final remainder shorter than
    ``min_chunk`` actions is skipped — the model collapses tiny chunks to no
    generated frames — dropping at most ``reanchor_every - 1`` trailing frames.
    """
    if reanchor_every < 1:
        raise ValueError(f"reanchor_every must be >= 1, got {reanchor_every}")
    # One encode for the whole window (actions are framewise; slice per segment).
    all_actions = np.asarray(
        world_model.prepare_actions(real, frame_start=frame_start, n_actions=None),
        dtype=np.float32,
    )
    if len(all_actions) < n_actions:
        n_actions = len(all_actions)

    cam = world_model.conditioning_camera
    n_segments = -(-n_actions // reanchor_every)  # ceil
    predicted_frames: list = []
    offset = 0
    seg_idx = 0
    while offset < n_actions:
        keep = min(reanchor_every, n_actions - offset)
        chunk = all_actions[offset: offset + gen_chunk]
        if len(chunk) < min_chunk:
            # The world model's temporal compression collapses a tiny chunk (a
            # 1-2 action chunk returns no generated frames), so a short final
            # remainder is dropped rather than sent. At most reanchor_every-1
            # trailing frames are skipped; nothing already produced is affected.
            break
        seg = world_model.rollout(real.frames[frame_start + offset], chunk)
        kept = seg.frames[:keep]
        out_start = len(predicted_frames)
        predicted_frames.extend(kept)
        seg_idx += 1
        # Hand the new frames (+ their real counterparts) to the caller so it can
        # persist them NOW — a long re-anchored run must never buffer everything
        # and lose it all on a late failure.
        if on_segment is not None:
            lo = frame_start + conditioning_offset + out_start
            on_segment(out_start, kept, real.frames[lo: lo + len(kept)])
        if progress is not None:
            progress(seg_idx, n_segments)
        offset += keep

    predicted = Trajectory(
        frames=predicted_frames,
        frame_indices=list(range(len(predicted_frames))),
        fps=real.fps, episode_id="cosmos-reanchored",
        source=f"reanchored:{getattr(world_model, 'model_id', 'wm')}",
        metadata={"reanchor_every": reanchor_every, "camera": cam},
    )
    start = frame_start + conditioning_offset
    aligned_real = _subtrajectory(real, start, start + len(predicted_frames))
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
    """Full Stage-2 trust analysis for one episode — the serializable report.

    Thin wrapper over :func:`analyze_trust` returning only its JSON-able report
    (the curve + verdict). Use :func:`analyze_trust` directly when you also need
    the predicted/real frames (e.g. to render the side-by-side comparison).
    """
    return analyze_trust(
        world_model, real,
        frame_start=frame_start, n_actions=n_actions, camera=camera,
        metric=metric, conditioning_offset=conditioning_offset,
        shuffle_seed=shuffle_seed,
    ).report


@dataclass(frozen=True)
class TrustAnalysis:
    """The full result of a Stage-2 trust run.

    ``report`` is the JSON-able summary (curve + verdict). ``predicted`` and
    ``aligned_real`` are the real-action rollout and the matching real frames —
    the same length, ready to render side by side. ``predicted_shuffled`` is the
    action-dependence control rollout.
    """

    report: dict
    predicted: Trajectory
    aligned_real: Trajectory
    predicted_shuffled: Trajectory


def analyze_trust(
    world_model: WorldModel,
    real: Trajectory,
    *,
    frame_start: int = 0,
    n_actions: Optional[int] = None,
    camera: str = "primary",
    metric: FrameMetric = "pixel_l2",
    conditioning_offset: int = 1,
    shuffle_seed: int = 0,
) -> TrustAnalysis:
    """Roll the real actions forward (the trust curve), then a shuffled-action
    control (the action-dependence check), both compared against the same real
    frames — and return the report plus the rollout trajectories for rendering.
    """
    # The world model owns how an episode maps to conditioning actions (raw
    # logged actions by default; a per-domain encoding for Cosmos).
    actions = np.asarray(
        world_model.prepare_actions(real, frame_start=frame_start, n_actions=n_actions),
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(
            f"world model prepared no usable actions (shape {actions.shape}) for "
            f"frame_start={frame_start}, n_actions={n_actions}"
        )

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

    report = {
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
    return TrustAnalysis(
        report=report, predicted=predicted, aligned_real=aligned_real,
        predicted_shuffled=predicted_sh,
    )


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
