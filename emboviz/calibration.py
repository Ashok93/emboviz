"""Per-trajectory model calibration.

Open-source interpretability tools cannot tell a user what a "perfect" model
score looks like — we don't know their use case, hardware, or performance bar.
What we CAN do is anchor every metric to two model-specific reference values:

  • ``noise_floor``  — the model's intrinsic prediction noise from running
    ``predict()`` twice on identical input. Δaction below this value is
    stochastic decoding jitter, not a real intervention effect.

  • ``typical_action_magnitude``  — the median ``‖predicted_action‖`` over the
    trajectory's baseline predictions. Used as the denominator that converts a
    raw L2 distance into a dimensionless "fraction of a typical action" score.

After calibration, every diagnostic that previously reported a raw L2 distance
in the model's opaque action units reports a normalized score on a 0-1 scale
with anchored meaning:

  score = max(0, raw_delta − noise_floor) / typical_action_magnitude

  0.0  → perturbation moved the action by less than noise floor (no signal)
  0.05 → moved by 5% of a typical action above noise — small but real
  1.0  → moved by a full typical action above noise — fully sensitive

Verdict thresholds (``noise_floor``, ``grounded_threshold``, etc.) now have
consistent meaning across models because they're applied in normalized space.

Multi-sample averaging for stochastic models
--------------------------------------------
Some models (notably π0 with flow-matching, diffusion policies, any model
that samples from an action distribution) have a stochastic decoder. Single
``predict()`` calls on the same scene return different actions each time.
The noise floor for such models can be 10-15 % of the typical action
magnitude — large enough to mask most intervention effects.

The fix is standard: average N samples. If single-call noise is σ, the mean
of N samples has noise σ/√N. We use ``n_samples`` calls per "logical" predict
in both the calibration (so the noise floor we measure is the AVERAGED noise)
and in diagnostics (so every Δaction is computed between averaged actions).

For deterministic models (OpenVLA, OFT, ACT) leave ``n_samples=1`` — there's
no noise to average out and it's strictly slower.

Auto-detection: ``calibrate_model`` runs a quick single-sample probe first to
estimate the model's raw noise floor. If that noise floor is meaningfully
above zero (> 0.05 × typical magnitude), it re-runs with ``n_samples``
averaging to bring the noise floor down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from emboviz.core.types import ActionResult, Scene, Trajectory, average_action_results
from emboviz.models.protocol import VLAModel


@dataclass(frozen=True)
class ModelCalibration:
    """Per-trajectory anchors used to normalize diagnostic scores.

    Build via :func:`calibrate_model`. Pass to any diagnostic that takes a
    ``calibration`` argument — the diagnostic divides raw L2 distances by
    the typical action magnitude (after subtracting the noise floor) so its
    scalar score lives on a model-agnostic 0-1 scale.

    ``n_samples`` is the averaging factor used both during calibration AND
    intended for use by every diagnostic that takes this calibration. When
    ``n_samples > 1``, diagnostics should call :func:`averaged_predict`
    instead of ``model.predict`` directly so the same averaging applies to
    every action.
    """

    noise_floor: float
    typical_action_magnitude: float
    n_noise_probes: int
    n_baseline_frames: int
    n_samples: int = 1
    raw_baseline_magnitudes: list[float] = field(default_factory=list)
    raw_noise_deltas: list[float] = field(default_factory=list)
    single_sample_noise_floor: Optional[float] = None

    def normalize(self, raw_delta: float) -> float:
        """Convert a raw L2 Δaction into an anchored 0-1 score.

        Strict: a calibration with near-zero ``typical_action_magnitude``
        cannot produce meaningful normalized scores (every intervention
        would collapse to 0.0 regardless of magnitude). We raise rather
        than silently return 0 — the caller needs to know calibration is
        degenerate so they can refuse a verdict.
        """
        if self.typical_action_magnitude < 1e-9:
            raise ValueError(
                f"ModelCalibration.normalize: typical_action_magnitude "
                f"is {self.typical_action_magnitude} (effectively zero). "
                "The model produced near-zero actions on every baseline "
                "frame, so there is no scale to normalize against. The "
                "trajectory probably has no real motion, or the model "
                "is misconfigured. Diagnostic must refuse to emit a "
                "verdict — do not silently return 0."
            )
        return max(0.0, raw_delta - self.noise_floor) / self.typical_action_magnitude

    def signal_threshold_normalized(self, k_samples: int = 1) -> float:
        """Minimum normalized response statistically distinguishable from
        the model's sampling noise, at 2-sigma confidence, after
        averaging ``k_samples`` independent substitutions.

        Derivation: ``noise_floor`` is the empirical std of identical-
        input calls (i.e. the per-call sampling noise). For the mean of
        K independent samples, SE = noise_floor / sqrt(K). A response
        whose mean is below 2*SE is statistically indistinguishable
        from zero true response → the diagnostic cannot reject the null
        hypothesis of "no effect from this intervention."

        Returned in NORMALIZED units (divided by typical_action_magnitude)
        so it can be compared directly to ``normalize(raw_delta)``.

        Notes:
          - If ``noise_floor`` is zero (deterministic model), the
            threshold is zero too: any positive normalized response is
            real signal.
          - If ``typical_action_magnitude`` is degenerate, ``normalize``
            already refuses; this method returns ``inf`` to make the
            threshold infinite (no signal will ever pass) for symmetry.
        """
        if self.typical_action_magnitude < 1e-9:
            return float("inf")
        k = max(1, int(k_samples))
        sigma_normalized = self.noise_floor / self.typical_action_magnitude
        return 2.0 * sigma_normalized / (k ** 0.5)

    def to_summary(self) -> dict:
        return {
            "noise_floor":                self.noise_floor,
            "typical_action_magnitude":   self.typical_action_magnitude,
            "n_noise_probes":             self.n_noise_probes,
            "n_baseline_frames":          self.n_baseline_frames,
            "n_samples":                  self.n_samples,
            "raw_baseline_magnitudes":    list(self.raw_baseline_magnitudes),
            "raw_noise_deltas":           list(self.raw_noise_deltas),
            "single_sample_noise_floor":  self.single_sample_noise_floor,
        }


def averaged_predict(
    model: VLAModel, scene: Scene, n_samples: int = 1,
) -> ActionResult:
    """Average ``n_samples`` predictions; return an ActionResult with the
    mean action. For ``n_samples=1`` this is a single ``predict`` call.

    For stochastic models this reduces decoding noise by sqrt(n_samples).
    For deterministic models the average equals the single prediction.

    The averaging itself (mean action + mean chunk, with the strict
    inconsistent-chunk-shape check) lives in
    ``emboviz_wire.average_action_results`` so the host and the in-worker
    ``predict_batch`` n-sample path reduce noise identically.
    """
    if n_samples <= 1:
        return model.predict(scene)
    return average_action_results([model.predict(scene) for _ in range(n_samples)])


def averaged_predict_batch(
    model: VLAModel, scenes: list[Scene], n_samples: int = 1,
) -> list[ActionResult]:
    """Batched sibling of :func:`averaged_predict` — the diagnostics' parallel
    hot path.

    Hands the whole scene list to ``model.predict_batch`` in one call, so a
    batch-capable adapter runs a single GPU forward (worker-chunked) instead
    of one round-trip per scene. The per-scene ``n_samples`` averaging is
    applied identically to :func:`averaged_predict` (same wire helper). For
    adapters without a true batched override, ``predict_batch`` falls back to
    the sequential loop — same numbers, just not parallel.
    """
    if not scenes:
        return []
    return model.predict_batch(scenes, n_samples)


def calibrate_model(
    model: VLAModel,
    trajectory: Trajectory,
    n_noise_probes: int = 5,
    precision_target: float = 0.05,
    max_n_samples: int = 64,
) -> ModelCalibration:
    """Probe a model on a trajectory; compute n_samples FROM the math.

    For a stochastic model with single-sample decoding noise σ and typical
    action magnitude m, the standard error of the N-sample averaged action
    is σ/√N. To bound that error below ``precision_target × m`` we need:

        n_samples = ceil( (σ / (precision_target × m))² )

    Deterministic models (σ ≈ 0) get n_samples = 1.
    Highly stochastic models (e.g. π0 flow-matching) get whatever the math
    says. No arbitrary defaults; we compute the exact value each model
    requires to give the user honest, precise numbers.

    Args:
        model, trajectory: what to calibrate against.
        n_noise_probes: how many pairs of averaged predictions used in the
            final noise-floor estimate.
        precision_target: bound the averaged noise floor to ≤ this fraction
            of the typical action magnitude. Default 0.05 (5 %) — small
            enough that an intervention moving the action by 5–10 % of a
            typical action is reliably distinguishable from decoding noise.
        max_n_samples: upper bound on the per-call averaging count. Default
            64 (extreme noise models won't blow up compute infinitely).
    """
    import math
    if not trajectory.frames:
        raise ValueError("calibrate_model: trajectory has no frames")

    # Step 1: characterise the model — single-sample noise + typical magnitude.
    #
    # We measure BOTH magnitude probes (single-sample for the n_samples math,
    # and averaged-equivalent for the final typical scale) on the SAME first
    # frames, with a frame-count linked to ``n_noise_probes``. That way the
    # n_samples estimate is computed in the same scale we will later use in
    # ``normalize()`` — preventing the bias where single-sample magnitude
    # over-estimates the averaged magnitude (Jensen's inequality:
    # ``E[||a||] >= ||E[a]||``).
    first = trajectory.frames[0]
    deltas_1 = []
    for _ in range(min(3, n_noise_probes)):
        a1 = model.predict(first).action
        a2 = model.predict(first).action
        deltas_1.append(float(np.linalg.norm(a1 - a2)))
    single_sample_noise = float(np.mean(deltas_1)) if deltas_1 else 0.0

    # ``mag_probe`` is the magnitude of SINGLE-SAMPLE predictions on the
    # first few frames. This is the right reference for the n_samples
    # formula derivation: the formula relates SINGLE-sample noise to
    # SINGLE-sample magnitude. The downstream ``typical`` scale used in
    # ``normalize()`` is intentionally computed on AVERAGED predictions
    # (Step 3) — that's the right denominator for averaged-Δaction
    # comparisons by the diagnostics. The math derivation:
    #     averaged_noise ≈ single_sample_noise / sqrt(N)
    # and we want averaged_noise ≤ precision_target * single_mag
    # ⟹ N >= (single_noise / (precision_target * single_mag))².
    # The factor ``E[||avg||] / E[||single||]`` (Jensen's inequality) is
    # absorbed into the precision_target slack: in practice the averaged
    # magnitude is at most ~1.2× smaller for typical action distributions,
    # so the achieved precision is at most 1.2× the target. We document
    # this; we don't try to predict the Jensen factor exactly.
    n_baseline_probe = min(max(3, n_noise_probes), len(trajectory.frames))
    mag_probe = float(np.mean([
        float(np.linalg.norm(model.predict(s).action))
        for s in trajectory.frames[:n_baseline_probe]
    ]))

    # Step 2: solve for n_samples — exactly enough to bound averaged
    # noise floor below precision_target × typical magnitude.
    if mag_probe < 1e-9 or single_sample_noise < 1e-9:
        n_samples = 1
    else:
        ratio = single_sample_noise / (precision_target * mag_probe)
        n_samples = int(min(max_n_samples, max(1, math.ceil(ratio * ratio))))

    # Step 3: full calibration with the chosen n_samples.
    baseline_magnitudes: list[float] = []
    for scene in trajectory.frames:
        a = averaged_predict(model, scene, n_samples).action
        baseline_magnitudes.append(float(np.linalg.norm(a)))
    typical = float(np.median(baseline_magnitudes))

    deltas: list[float] = []
    for _ in range(n_noise_probes):
        a1 = averaged_predict(model, first, n_samples).action
        a2 = averaged_predict(model, first, n_samples).action
        deltas.append(float(np.linalg.norm(a1 - a2)))
    noise = float(np.mean(deltas))

    return ModelCalibration(
        noise_floor=noise,
        typical_action_magnitude=typical,
        n_noise_probes=n_noise_probes,
        n_baseline_frames=len(baseline_magnitudes),
        n_samples=n_samples,
        raw_baseline_magnitudes=baseline_magnitudes,
        raw_noise_deltas=deltas,
        single_sample_noise_floor=single_sample_noise,
    )
