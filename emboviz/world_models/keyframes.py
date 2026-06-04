"""Critical-moment (keyframe) detection for manipulation trajectories.

A manipulation episode is decided at a handful of instants — the grasp, the
release, the moment the arm settles onto a target. The stress test seeds short
world-model rollouts at exactly those instants (a quarter-second of approach +
grasp is inside the world model's faithful horizon, a whole task is not), so it
needs to find them in a recorded episode first.

Method — the standard keyframe heuristic shared by keyframe-based manipulation
policies (PerAct, RVT; James & Davison, "Q-attention"): a frame is a keyframe
when either

  (a) the gripper changes state (open↔closed) — a grasp or release, or
  (b) the end-effector comes to rest — a local minimum of motion below a small
      velocity threshold, held for a few frames (a "settle": contact/alignment).

Thresholds follow NVIDIA's cosmos-framework idle detector
(``compute_idle_frames``, ``data/vfm/action/pose_utils.py``): velocity-based and
fps-scaled, with a min-streak filter on the settle test to reject instantaneous
slowdowns. See LITERATURE.md for citations.

This module is world-model-agnostic and depends only on the wire ``Trajectory``
and numpy — it reads the recorded end-effector pose and gripper straight off the
episode. The Cosmos-specific action encoding lives in the adapter, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from emboviz_wire.types import Trajectory

KeyframeKind = Literal["gripper_change", "settle"]

# Defaults mirror cosmos-framework's DROID idle-frame thresholds, expressed per
# second so they are independent of the episode's fps:
#   translation 5 mm/s, rotation 1.5°/s, gripper 1% command change, streak 3.
_DEFAULT_EPS_TRANSLATION_PER_S = 5e-3
_DEFAULT_EPS_ROTATION_DEG_PER_S = 1.5
_DEFAULT_EPS_GRIPPER = 1e-2
_DEFAULT_MIN_SETTLE_STREAK = 3


@dataclass(frozen=True)
class Keyframe:
    """One decisive instant in an episode.

    ``index`` is the position in ``trajectory.frames``. ``kind`` is the signal
    that fired. ``speed`` is the per-second end-effector translation speed at the
    frame (m/s); ``gripper_delta`` is the signed per-frame gripper change (0 for a
    settle). The grasp/release meaning of a ``gripper_change`` depends on the
    dataset's gripper convention, so it is reported as a signed delta rather than
    labelled here.
    """

    index: int
    kind: KeyframeKind
    speed: float
    gripper_delta: float


@dataclass(frozen=True)
class CriticalWindow:
    """A frame span ``[start, stop)`` around a keyframe, for seeding a rollout.

    ``start`` is the seed frame (the pre-event approach); ``stop`` is exclusive.
    Both are clamped to the episode bounds.
    """

    keyframe: Keyframe
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def _wrap_to_pi(delta: np.ndarray) -> np.ndarray:
    """Wrap an angle difference (rad) to ``[-pi, pi]`` so Euler wraparound near
    ±pi does not register as a huge rotation."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _read_pose_and_gripper(traj: Trajectory) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Extract per-frame ``(positions (T,3), euler (T,3) | None, gripper (T,))``.

    Raises with a clear, actionable message if the proprioceptive state or the
    gripper is missing — the detector never silently degrades to a partial
    signal (a manipulation episode with no gripper mapped would otherwise drop
    every grasp keyframe without telling anyone).
    """
    n = len(traj.frames)
    if n < 2:
        raise ValueError(f"keyframe detection needs >=2 frames, got {n}.")

    positions = np.empty((n, 3), dtype=np.float32)
    eulers = np.empty((n, 3), dtype=np.float32)
    grippers = np.empty(n, dtype=np.float32)
    have_euler = True

    for i, frame in enumerate(traj.frames):
        state = frame.observations.state
        if state is None:
            raise ValueError(
                f"keyframe detection needs end-effector state at frame {i}, but "
                "observations.state is None. Map the dataset's cartesian state "
                "(e.g. observation.state.cartesian_position) in the config."
            )
        values = np.asarray(state.values, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] < 3:
            raise ValueError(
                f"keyframe detection needs a >=3-D end-effector position at frame "
                f"{i}; got state shape {values.shape}."
            )
        positions[i] = values[:3]
        if values.shape[0] >= 6:
            eulers[i] = values[3:6]
        else:
            have_euler = False

        gripper = frame.observations.gripper
        if gripper is None:
            raise ValueError(
                f"keyframe detection needs the gripper at frame {i}, but "
                "observations.gripper is None. Map the dataset's gripper in the "
                "config — grasp/release are the primary keyframe signal."
            )
        grippers[i] = float(gripper.value)

    return positions, (eulers if have_euler else None), grippers


def _settle_mask(
    positions: np.ndarray,
    eulers: np.ndarray | None,
    fps: float,
    eps_translation_per_s: float,
    eps_rotation_deg_per_s: float,
    min_streak: int,
) -> np.ndarray:
    """Per-frame boolean "the arm is at rest entering this frame" mask.

    Frame ``i`` (for ``i >= 1``) is at-rest when the motion from frame ``i-1`` to
    ``i`` is below both the translation and rotation velocity thresholds; the
    mask is then filtered to runs of at least ``min_streak`` consecutive
    at-rest frames. Frame 0 is never at-rest (no preceding motion to settle from).
    """
    n = positions.shape[0]
    trans_speed = np.zeros(n, dtype=np.float64)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps

    rot_speed = np.zeros(n, dtype=np.float64)
    if eulers is not None:
        rot_speed[1:] = np.linalg.norm(_wrap_to_pi(np.diff(eulers, axis=0)), axis=1) * fps

    eps_rotation_per_s = np.deg2rad(eps_rotation_deg_per_s)
    at_rest = (trans_speed < eps_translation_per_s) & (rot_speed < eps_rotation_per_s)
    at_rest[0] = False

    return _consecutive_streaks(at_rest, min_streak)


def _consecutive_streaks(mask: np.ndarray, min_streak: int) -> np.ndarray:
    """Zero out True bits not part of a run of ``>= min_streak`` consecutive Trues."""
    if min_streak <= 1:
        return mask
    out = np.zeros_like(mask)
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_streak:
            out[i:j] = True
        i = j
    return out


def detect_keyframes(
    traj: Trajectory,
    *,
    eps_translation_per_s: float = _DEFAULT_EPS_TRANSLATION_PER_S,
    eps_rotation_deg_per_s: float = _DEFAULT_EPS_ROTATION_DEG_PER_S,
    eps_gripper: float = _DEFAULT_EPS_GRIPPER,
    min_settle_streak: int = _DEFAULT_MIN_SETTLE_STREAK,
) -> list[Keyframe]:
    """Detect the critical instants in a recorded episode.

    Returns keyframes ordered by frame index. A gripper change and a settle that
    land on the same frame are reported once, as a ``gripper_change`` (the
    grasp/release dominates). ``traj.fps`` must be set (the thresholds are
    velocity-based); a non-positive fps raises.

    Thresholds default to cosmos-framework's DROID idle values; pass overrides
    for other embodiments. See module docstring / LITERATURE.md for the method.
    """
    if traj.fps <= 0:
        raise ValueError(
            f"keyframe detection needs a positive fps (got {traj.fps}); the "
            "velocity thresholds are per-second. Set the dataset's fps."
        )

    positions, eulers, grippers = _read_pose_and_gripper(traj)
    n = positions.shape[0]

    trans_speed = np.zeros(n, dtype=np.float64)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) * traj.fps

    gripper_delta = np.zeros(n, dtype=np.float64)
    gripper_delta[1:] = np.diff(grippers)
    is_gripper_change = np.abs(gripper_delta) >= eps_gripper

    at_rest = _settle_mask(
        positions, eulers, traj.fps, eps_translation_per_s, eps_rotation_deg_per_s, min_settle_streak
    )
    # A settle fires on the rising edge — the frame the arm first comes to rest.
    is_settle = at_rest & ~np.concatenate(([False], at_rest[:-1]))

    keyframes: list[Keyframe] = []
    for i in range(n):
        if is_gripper_change[i]:
            keyframes.append(
                Keyframe(i, "gripper_change", float(trans_speed[i]), float(gripper_delta[i]))
            )
        elif is_settle[i]:
            keyframes.append(Keyframe(i, "settle", float(trans_speed[i]), 0.0))

    return keyframes


def critical_windows(
    traj: Trajectory,
    keyframes: list[Keyframe],
    *,
    before_s: float = 5.0,
    after_s: float = 5.0,
) -> list[CriticalWindow]:
    """Build a ``[keyframe - before_s, keyframe + after_s]`` frame window per
    keyframe, fps-scaled and clamped to the episode.

    The buffer captures the full approach before the event and the consequence
    after it. Windows are returned in keyframe order; overlapping windows are NOT
    merged (each keyframe is stress-tested from its own seed).
    """
    if traj.fps <= 0:
        raise ValueError(f"critical_windows needs a positive fps, got {traj.fps}.")
    if before_s < 0 or after_s < 0:
        raise ValueError(f"before_s/after_s must be >= 0, got {before_s}/{after_s}.")

    n = len(traj.frames)
    before = int(round(before_s * traj.fps))
    after = int(round(after_s * traj.fps))

    windows: list[CriticalWindow] = []
    for kf in keyframes:
        start = max(0, kf.index - before)
        stop = min(n, kf.index + after + 1)
        windows.append(CriticalWindow(keyframe=kf, start=start, stop=stop))
    return windows


__all__ = [
    "CriticalWindow",
    "Keyframe",
    "KeyframeKind",
    "critical_windows",
    "detect_keyframes",
]
