"""World-model trust calibration.

A world model is a learned, hallucination-prone approximation of physics — its
own model card states it lacks an explicit physics simulator and that outputs
"should not be treated as physically accurate simulation." Before a predicted
rollout can be used to judge a policy, emboviz measures **how far that rollout
can be trusted**: it conditions the world model on a real episode's first frame
and its real logged actions, then compares the predicted future frame-by-frame
against what actually happened in the recording.

Methodology
-----------
The output is a **trust curve**: prediction-vs-reality divergence as a function
of rollout horizon. It carries three honest anchors, mirroring the
world-model-evaluation literature:

- **Noise floor.** Even at horizon 1 the prediction is not pixel-identical to
  reality — the model's VAE encode/decode and one-step stochasticity impose an
  irreducible divergence. The floor is the mean divergence over the first few
  frames; trust is measured *relative to* it, never against zero.

- **Trust horizon.** The largest horizon at which divergence stays within a band
  above the noise floor. Past it, the rollout has drifted enough that a verdict
  computed from it would be reporting the world model's hallucination, not the
  policy's behaviour. dWorldEval (arXiv:2604.22152) and the WoVR analysis
  (arXiv:2602.13977) both find usable horizons are short and must be measured,
  not assumed.

- **Action dependence (control).** A separate check that the metric is
  *meaningful*: roll the same frames forward under the real actions and under
  shuffled actions. If the real-action rollout tracks reality far better than
  the shuffled one, the world model is genuinely action-conditioned and the
  curve reflects physics, not a static prior. This is dWorldEval's Δ-LPIPS idea;
  without it a trust curve can look good for the wrong reason.

This module computes the curve from two already-aligned :class:`Trajectory`
objects — a predicted rollout and the real episode. Frame alignment (which real
frame each predicted frame corresponds to) is the caller's responsibility; the
driver that builds the rollout owns that, because it knows the conditioning
offset and the action/frame cadence.

Metrics
-------
Two dependency-light frame divergences ship here: normalized pixel RMSE and a
global SSIM, both pure NumPy so core stays torch-free. A perceptual metric
(LPIPS) is the natural upgrade but requires torch, so it is intentionally left
to a future worker rather than pulled into core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from emboviz_wire.types import Trajectory


FrameMetric = Literal["pixel_l2", "ssim"]

#: Default number of leading frames averaged to estimate the noise floor.
_DEFAULT_NOISE_FLOOR_FRAMES = 2
#: Divergence above ``noise_floor * trust_multiplier`` is considered drift.
_DEFAULT_TRUST_MULTIPLIER = 2.0


# ── frame extraction + resizing ─────────────────────────────────────────────


def _frame_array(traj: Trajectory, idx: int, camera: str) -> np.ndarray:
    """Return the ``(H, W, 3)`` uint8 image for one frame/camera of a Trajectory."""
    scene = traj.frames[idx]
    if camera not in scene.observations.images:
        raise KeyError(
            f"camera '{camera}' missing from frame {idx} "
            f"(have: {sorted(scene.observations.images)})"
        )
    arr = np.asarray(scene.observations.images[camera].data)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"frame {idx} camera '{camera}' is not (H, W, 3): {arr.shape}")
    return arr.astype(np.uint8, copy=False)


def _match_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize ``a`` to ``b``'s height/width if they differ (predicted and real
    rollouts may be generated at different resolutions)."""
    if a.shape[:2] == b.shape[:2]:
        return a, b
    from PIL import Image

    resized = Image.fromarray(a, mode="RGB").resize(
        (b.shape[1], b.shape[0]), Image.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8), b


# ── frame divergence metrics ────────────────────────────────────────────────


def _pixel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized RMSE in [0, 1] over RGB pixels."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)) / 255.0)


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """1 − global SSIM in [0, 1] on the grayscale frames (higher = more different).

    Global (whole-frame) SSIM with the standard luminance/contrast/structure
    constants. Less localized than windowed SSIM but pure-NumPy and adequate as
    a structural complement to pixel RMSE.
    """
    x = a.astype(np.float64).mean(axis=2)
    y = b.astype(np.float64).mean(axis=2)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return float(np.clip(1.0 - ssim, 0.0, 1.0))


_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pixel_l2": _pixel_l2,
    "ssim": _global_ssim,
}


def frame_divergence(a: np.ndarray, b: np.ndarray, metric: FrameMetric = "pixel_l2") -> float:
    """Divergence in [0, 1] between two ``(H, W, 3)`` uint8 frames.

    ``a`` is resized to ``b`` if their resolutions differ. ``metric`` is
    ``"pixel_l2"`` (normalized RMSE) or ``"ssim"`` (1 − global SSIM).
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown frame metric {metric!r}; choose from {sorted(_METRICS)}")
    a, b = _match_size(np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8))
    return _METRICS[metric](a, b)


# ── trust curve ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrustResult:
    """The trust curve for one predicted rollout against its real episode.

    Attributes
    ----------
    horizons
        Frame index of each step in the curve (0-based rollout horizon).
    divergence
        Prediction-vs-reality divergence at each horizon, in [0, 1].
    noise_floor
        Mean divergence over the first ``noise_floor_frames`` — the irreducible
        floor trust is measured against.
    trust_band
        ``noise_floor * trust_multiplier`` — divergence above this is drift.
    trust_horizon
        Largest horizon at which divergence stays within ``trust_band`` for
        every frame up to and including it. ``len(horizons)`` if the rollout
        never drifts; 0 if it drifts immediately.
    metric
        The frame metric used.
    metadata
        Free-form: domain, camera, frame counts, generation settings.
    """

    horizons: list[int]
    divergence: list[float]
    noise_floor: float
    trust_band: float
    trust_horizon: int
    metric: FrameMetric
    metadata: dict = field(default_factory=dict)


def compute_trust_curve(
    predicted: Trajectory,
    real: Trajectory,
    *,
    camera: str = "primary",
    metric: FrameMetric = "pixel_l2",
    noise_floor_frames: int = _DEFAULT_NOISE_FLOOR_FRAMES,
    trust_multiplier: float = _DEFAULT_TRUST_MULTIPLIER,
) -> TrustResult:
    """Compare an aligned predicted rollout against the real episode.

    ``predicted.frames[i]`` is compared to ``real.frames[i]`` — the caller is
    responsible for having aligned them (the rollout driver knows the
    conditioning offset). The comparison runs over ``min(len(predicted),
    len(real))`` frames.

    The noise floor is the mean divergence over the first ``noise_floor_frames``;
    the trust horizon is where divergence first exceeds ``noise_floor *
    trust_multiplier``. Thresholds are heuristic anchors, disclosed in the
    result — a deployment calibrates them against a null set, never tunes them
    to make a demo pass.
    """
    n = min(len(predicted.frames), len(real.frames))
    if n == 0:
        raise ValueError("compute_trust_curve: predicted and real have no overlapping frames")
    if noise_floor_frames < 1:
        raise ValueError("noise_floor_frames must be >= 1")

    divergence = [
        frame_divergence(
            _frame_array(predicted, i, camera), _frame_array(real, i, camera), metric
        )
        for i in range(n)
    ]
    horizons = list(range(n))

    floor_n = min(noise_floor_frames, n)
    noise_floor = float(np.mean(divergence[:floor_n]))
    trust_band = noise_floor * float(trust_multiplier)

    # Trust horizon: the run of frames from the start that all stay within the
    # band. The first frame to break it ends the trusted run.
    trust_horizon = n
    for i, d in enumerate(divergence):
        if d > trust_band:
            trust_horizon = i
            break

    return TrustResult(
        horizons=horizons,
        divergence=divergence,
        noise_floor=noise_floor,
        trust_band=trust_band,
        trust_horizon=trust_horizon,
        metric=metric,
        metadata={
            "n_frames": n,
            "camera": camera,
            "noise_floor_frames": floor_n,
            "trust_multiplier": float(trust_multiplier),
        },
    )


# ── action-dependence control ───────────────────────────────────────────────


def action_dependence(
    real_action_curve: TrustResult,
    shuffled_action_curve: TrustResult,
    *,
    margin: float = 0.02,
) -> dict:
    """Validate the trust curve is *meaningful* — that the world model responds
    to actions, not just replays a static prior.

    Compares the prediction-vs-reality divergence of a rollout driven by the
    REAL actions against one driven by SHUFFLED actions (same frames, scrambled
    action order). If the real-action rollout tracks reality meaningfully better
    (mean divergence lower by at least ``margin``), the model is genuinely
    action-conditioned and the trust curve reflects physics. If not, the curve
    is not trustworthy — the model would render the same future regardless of
    the actions, so it cannot be used to evaluate a policy. (dWorldEval's
    Δ-LPIPS control, arXiv:2604.22152.)

    Returns the two mean divergences, their separation, and an
    ``action_sensitive`` verdict.
    """
    real_mean = float(np.mean(real_action_curve.divergence))
    shuffled_mean = float(np.mean(shuffled_action_curve.divergence))
    separation = shuffled_mean - real_mean
    return {
        "real_action_mean_divergence": real_mean,
        "shuffled_action_mean_divergence": shuffled_mean,
        "separation": separation,
        "margin": float(margin),
        "action_sensitive": separation >= float(margin),
    }
