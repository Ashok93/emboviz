"""Chunk consistency — does the model's predicted future stay coherent
between adjacent frames?

For models that predict an action *chunk* (multiple future timesteps at
once — π0, ACT, Diffusion Policy, RDT, OpenVLA-OFT, GR00T) the right
question to ask is:

    At frame t the model predicts chunk = [a_t, a_{t+1}, a_{t+2}, ...].
    At frame t+1 the model predicts chunk' = [a'_{t+1}, a'_{t+2}, ...].
    Does ``a_{t+1}`` (the prediction for t+1 made at time t) match
    ``a'_{t+1}`` (the prediction for t+1 made at time t+1)?

If yes → the model has stable lookahead; chunks are meaningful planning.
If no → the model is effectively resampling each frame and the chunk
        beyond the first step is noise. Running a multi-step controller
        on those chunks will hurt — you may as well replan every step.

Capability gate: this requires ``ActionResult.action_chunk`` to be
populated (chunk shape ≥ 2 along the time axis). For models that only
expose a single immediate action, the diagnostic skips with
Severity.UNKNOWN and a clear reason — we don't fall back to single-step
adjacent-frame deltas under the wrong name (that was a previous
implementation mistake; raw single-step deltas measure policy dynamics,
not chunk coherence — they're high on dynamic manipulation tasks even
for perfectly coherent policies).

Calibration (recommended):
    When a ``ModelCalibration`` is passed, the cross-frame delta is
    normalized to a 0-1 anchored scale (``raw_delta / typical_action``).
    A score of 1.0 means the model's chunk[t][1] and chunk[t+1][0]
    disagree by ONE typical action — substantial disagreement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.calibration import ModelCalibration, averaged_predict
from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, Trajectory
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel


class ChunkConsistencyDiagnostic(Diagnostic):
    """Compare ``chunk[t][k]`` vs ``chunk[t+k][0]`` for k=1, summarized
    across the trajectory.

    Run on a Trajectory (not a single Scene) — we need adjacent frames.
    ``run()`` is provided only to satisfy the Diagnostic ABC; the real
    API is ``run_trajectory()``.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        noise_floor: float = 0.10,
        grounded_threshold: float = 0.50,
        calibration: Optional[ModelCalibration] = None,
        compare_lookahead: int = 1,
    ):
        """Args:
            noise_floor: normalized score below which the model's chunk
                lookahead is treated as "consistent" (PASS).
            grounded_threshold: normalized score above which the chunk
                lookahead is "incoherent" (CRITICAL).
            calibration: per-model anchors. Without it, scores are raw L2
                and the thresholds become model-specific magic numbers.
            compare_lookahead: which chunk index to test
                (chunk[t][lookahead] vs chunk[t+lookahead][0]). Default 1
                tests the immediate next-step prediction.
        """
        self.noise_floor = noise_floor
        self.grounded_threshold = grounded_threshold
        self.calibration = calibration
        self.compare_lookahead = compare_lookahead
        self.name = "chunk_consistency"
        self.axis = "internal.chunk_consistency"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        return self._not_applicable(
            model, scene,
            "chunk consistency requires a Trajectory — use "
            "run_trajectory(model, traj) instead",
        )

    def run_trajectory(self, model: VLAModel, trajectory: Trajectory) -> DiagnosticResult:
        if not self.applicable_to(model):
            return self._not_applicable(
                model, trajectory.frames[0] if trajectory.frames else None,
                "model lacks INFERENCE capability",
            )
        if len(trajectory.frames) < 2:
            return self._not_applicable(
                model, trajectory.frames[0] if trajectory.frames else None,
                "need ≥2 frames for chunk consistency",
            )

        # Collect chunks from every frame.
        chunks: list[np.ndarray] = []
        for scene in trajectory.frames:
            n_samples = self.calibration.n_samples if self.calibration else 1
            ar = averaged_predict(model, scene, n_samples)
            if ar.action_chunk is None:
                return self._not_applicable(
                    model, scene,
                    f"model '{model.model_id}' does not expose action chunks "
                    "(ActionResult.action_chunk is None). Chunk consistency "
                    "needs multi-step lookahead — single-step models are not "
                    "applicable to this diagnostic. To measure adjacent-frame "
                    "action smoothness instead, use a different diagnostic; "
                    "frame-to-frame single-step delta is NOT chunk consistency.",
                )
            chunks.append(np.asarray(ar.action_chunk, dtype=np.float32))

        chunk_lens = [c.shape[0] for c in chunks]
        min_len = min(chunk_lens)
        if min_len <= self.compare_lookahead:
            return self._not_applicable(
                model, trajectory.frames[0],
                f"all chunks have length {min_len} <= lookahead "
                f"{self.compare_lookahead}; can't compare chunk[t][{self.compare_lookahead}] "
                "to chunk[t+1][0]",
            )

        # For each adjacent pair, compare chunk[t][k] vs chunk[t+k][0]
        # where k = self.compare_lookahead. Most informative with k=1.
        k = self.compare_lookahead
        raw_deltas: list[float] = []
        for t in range(len(chunks) - k):
            pred_for_t_plus_k = chunks[t][k]
            actual_at_t_plus_k = chunks[t + k][0]
            raw_deltas.append(
                float(np.linalg.norm(pred_for_t_plus_k - actual_at_t_plus_k))
            )

        raw_arr = np.asarray(raw_deltas, dtype=np.float32)
        raw_mean = float(raw_arr.mean())
        raw_max = float(raw_arr.max())

        if self.calibration is not None:
            normalized = np.array(
                [self.calibration.normalize(float(d)) for d in raw_arr],
                dtype=np.float32,
            )
        else:
            normalized = raw_arr
        mean_score = float(normalized.mean())
        max_score = float(normalized.max())

        if mean_score < self.noise_floor:
            sev = Severity.PASS
            verdict = (
                f"Chunk lookahead is consistent: chunk[t][{k}] vs chunk[t+{k}][0] "
                f"agree within noise floor ({self.noise_floor}). Normalized "
                f"mean disagreement = {mean_score:.3f}. Model has stable "
                f"multi-step planning."
            )
        elif mean_score < self.grounded_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"Chunk lookahead is partially consistent: normalized mean "
                f"disagreement {mean_score:.3f} between noise floor "
                f"({self.noise_floor}) and grounded threshold "
                f"({self.grounded_threshold}). Multi-step rollouts will drift "
                f"but the first step is reliable."
            )
        else:
            sev = Severity.CRITICAL
            verdict = (
                f"Chunk lookahead is INCONSISTENT: normalized mean "
                f"disagreement {mean_score:.3f} ≥ grounded threshold "
                f"({self.grounded_threshold}, max {max_score:.3f}). The "
                f"model's chunk-position-{k} prediction made at frame t "
                f"materially disagrees with its chunk-position-0 prediction "
                f"made at frame t+{k}. Running this model's chunks beyond "
                f"the first step is not safe — replan every step."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=trajectory.episode_id or trajectory.source or "trajectory",
            scalar_score=mean_score,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict,
            per_variant={
                f"frame_{t}_to_{t+k}": float(d)
                for t, d in enumerate(normalized)
            },
            raw={
                "compare_lookahead":     k,
                "chunk_lengths":         chunk_lens,
                "raw_deltas":            raw_arr.tolist(),
                "normalized_deltas":     normalized.tolist(),
                "raw_mean_delta":        raw_mean,
                "raw_max_delta":         raw_max,
                "noise_floor":           self.noise_floor,
                "grounded_threshold":    self.grounded_threshold,
                "calibration_used":      self.calibration.to_summary() if self.calibration else None,
            },
        )
