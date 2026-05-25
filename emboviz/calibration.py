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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emboviz.core.types import Trajectory
from emboviz.models.protocol import VLAModel


@dataclass(frozen=True)
class ModelCalibration:
    """Per-trajectory anchors used to normalize diagnostic scores.

    Build via :func:`calibrate_model`. Pass to any diagnostic that takes a
    ``calibration`` argument — the diagnostic divides raw L2 distances by
    the typical action magnitude (after subtracting the noise floor) so its
    scalar score lives on a model-agnostic 0-1 scale.
    """

    noise_floor: float
    typical_action_magnitude: float
    n_noise_probes: int
    n_baseline_frames: int
    raw_baseline_magnitudes: list[float] = field(default_factory=list)
    raw_noise_deltas: list[float] = field(default_factory=list)

    def normalize(self, raw_delta: float) -> float:
        """Convert a raw L2 Δaction into an anchored 0-1 score.

        Subtracts the noise floor (below = no signal → 0.0) then divides by
        the typical action magnitude. We do NOT clip the upper bound — a
        score of 1.3 means "perturbation moved action by 1.3 × typical
        magnitude," which is a real and meaningful overshoot.
        """
        if self.typical_action_magnitude < 1e-9:
            return 0.0
        return max(0.0, raw_delta - self.noise_floor) / self.typical_action_magnitude

    def to_summary(self) -> dict:
        return {
            "noise_floor":              self.noise_floor,
            "typical_action_magnitude": self.typical_action_magnitude,
            "n_noise_probes":           self.n_noise_probes,
            "n_baseline_frames":        self.n_baseline_frames,
            "raw_baseline_magnitudes":  list(self.raw_baseline_magnitudes),
            "raw_noise_deltas":         list(self.raw_noise_deltas),
        }


def calibrate_model(
    model: VLAModel,
    trajectory: Trajectory,
    n_noise_probes: int = 5,
) -> ModelCalibration:
    """Probe a model on a trajectory to estimate noise floor + typical magnitude.

    Cheap: one baseline ``predict()`` per frame (already computed by most
    diagnostics) plus ``n_noise_probes`` extra calls on the first frame.

    Noise floor: pairs of ``predict()`` calls on the same scene. The mean
    pairwise L2 distance is the model's intrinsic decoding noise — actions
    that move less than this under an intervention are not really being
    moved by the intervention.

    Typical action magnitude: median ``‖action‖`` across the trajectory's
    baseline predictions. Median rather than mean so that a single
    occasional outlier action doesn't inflate the denominator.
    """
    if not trajectory.frames:
        raise ValueError("calibrate_model: trajectory has no frames")

    baseline_magnitudes: list[float] = []
    for scene in trajectory.frames:
        a = model.predict(scene).action
        baseline_magnitudes.append(float(np.linalg.norm(a)))
    typical = float(np.median(baseline_magnitudes))

    # Noise floor — pairs of identical-input predictions on the first frame.
    first = trajectory.frames[0]
    deltas: list[float] = []
    for _ in range(n_noise_probes):
        a1 = model.predict(first).action
        a2 = model.predict(first).action
        deltas.append(float(np.linalg.norm(a1 - a2)))
    noise = float(np.mean(deltas))

    return ModelCalibration(
        noise_floor=noise,
        typical_action_magnitude=typical,
        n_noise_probes=n_noise_probes,
        n_baseline_frames=len(baseline_magnitudes),
        raw_baseline_magnitudes=baseline_magnitudes,
        raw_noise_deltas=deltas,
    )
