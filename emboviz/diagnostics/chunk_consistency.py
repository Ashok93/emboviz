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
from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import ActionResult, Scene, Trajectory
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

    def run_trajectory(
        self, model: VLAModel, trajectory: Trajectory,
        *, baselines: Optional[list[ActionResult]] = None,
    ) -> DiagnosticResult:
        """Compute chunk-consistency across the trajectory.

        ``baselines`` is an optional list of pre-computed unperturbed
        predictions, one per frame in trajectory order. When supplied,
        the diagnostic uses their ``action_chunk`` directly instead of
        re-running ``averaged_predict`` per frame — saving a full
        ``n_samples`` model forwards per frame on stochastic models.
        """
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
        if baselines is not None and len(baselines) != len(trajectory.frames):
            raise ValueError(
                f"chunk_consistency: baselines length {len(baselines)} "
                f"does not match trajectory frame count "
                f"{len(trajectory.frames)}."
            )

        # Collect chunks from every frame (re-use precomputed baselines
        # if the runner supplied them).
        chunks: list[np.ndarray] = []
        for i, scene in enumerate(trajectory.frames):
            if baselines is not None:
                ar = baselines[i]
            else:
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

        # ``compare_lookahead`` (k) counts how many ANALYZED frames ahead
        # to look. But the trajectory may be SUBSAMPLED (frame_stride > 1),
        # so analyzed frame t and t+k are
        # ``frame_indices[t+k] - frame_indices[t]`` DATASET frames apart —
        # and the model's action chunk is indexed at the dataset's control
        # frequency. Comparing ``chunk[t][k]`` (k control-steps ahead) to
        # ``chunk[t+k][0]`` would compare DIFFERENT timesteps whenever the
        # stride isn't 1. We therefore index the chunk by the true
        # dataset-frame gap, not by k. (With stride 1, gap == k and this is
        # identical to the naive version.)
        k = self.compare_lookahead
        frame_indices = list(trajectory.frame_indices)
        raw_deltas: list[float] = []
        frame_gaps: list[int] = []
        uncomparable: list[tuple[int, int]] = []   # (t, gap) pairs the chunk couldn't reach
        for t in range(len(chunks) - k):
            gap = int(frame_indices[t + k] - frame_indices[t])
            if gap < 1:
                # Non-increasing dataset indices — should not happen for a
                # well-formed trajectory; skip rather than compare garbage.
                uncomparable.append((t, gap))
                continue
            if gap >= chunk_lens[t]:
                # The model's chunk at frame t doesn't extend far enough to
                # cover the dataset-frame gap to the next analyzed frame, so
                # there is no chunk[t][gap] to compare. Record and skip —
                # never silently fold in a wrong-timestep comparison.
                uncomparable.append((t, gap))
                continue
            pred_for_next = chunks[t][gap]
            actual_at_next = chunks[t + k][0]
            raw_deltas.append(
                float(np.linalg.norm(pred_for_next - actual_at_next))
            )
            frame_gaps.append(gap)

        if not raw_deltas:
            return self._not_applicable(
                model, trajectory.frames[0],
                f"no comparable chunk pairs: with lookahead={k} analyzed "
                f"frame(s) and the trajectory's dataset-frame gaps "
                f"({sorted({g for _, g in uncomparable})}), the model's "
                f"chunk (lengths {sorted(set(chunk_lens))}) never extends "
                f"far enough to reach the next analyzed frame. Reduce "
                f"frame_stride, use a model with a longer action chunk, or "
                f"lower the analysis lookahead.",
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

        raw_numbers = {
            "lookahead_k":          k,
            "chunk_frame_gaps":     sorted(set(frame_gaps)),
            "n_uncomparable_pairs": len(uncomparable),
            "mean_disagreement_normalized": mean_score,
            "max_disagreement_normalized":  max_score,
            "mean_disagreement_raw":        raw_mean,
            "max_disagreement_raw":         raw_max,
            "noise_floor":          self.noise_floor,
            "grounded_threshold":   self.grounded_threshold,
            "n_frame_pairs":        len(raw_arr),
        }

        if mean_score < self.noise_floor:
            sev = Severity.PASS
            finding = Finding(
                observed=(
                    f"For each analyzed frame t, we compared the action the "
                    f"model's chunk (made at t) predicted for the next "
                    f"analyzed frame against the action it actually emitted "
                    f"there. The two agree closely — disagreement "
                    f"{mean_score:.3f} is below noise floor."
                ),
                meaning=(
                    "The model's multi-step planning is internally "
                    "consistent. You can trust the chunk it emits beyond "
                    "step 0."
                ),
                next_step=(
                    "No action needed — consistent chunks let you "
                    "lower the policy re-query rate at deployment time."
                ),
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Chunk lookahead is consistent: the chunk's prediction for "
                f"the next analyzed frame agrees with what the model emits "
                f"there, within noise floor ({self.noise_floor}). Normalized "
                f"mean disagreement = {mean_score:.3f}. Model has stable "
                f"multi-step planning."
            )
        elif mean_score < self.grounded_threshold:
            sev = Severity.MODERATE
            finding = Finding(
                observed=(
                    f"The model's chunk prediction (made at frame t) for "
                    f"the next analyzed frame disagrees with the action it "
                    f"actually emits there by {mean_score:.3f} of typical "
                    f"action magnitude on average — above noise but below "
                    f"the strong-disagreement threshold "
                    f"({self.grounded_threshold:.3f})."
                ),
                meaning=(
                    "Multi-step rollouts will drift — re-querying every "
                    "few steps is needed for accuracy. The first chunk "
                    "step is reliable; the tail isn't."
                ),
                next_step=(
                    "If you're running the model open-loop with long "
                    "chunks, consider shortening the horizon or "
                    "re-querying more often."
                ),
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Chunk lookahead is partially consistent: normalized mean "
                f"disagreement {mean_score:.3f} between noise floor "
                f"({self.noise_floor}) and grounded threshold "
                f"({self.grounded_threshold}). Multi-step rollouts will drift "
                f"but the first step is reliable."
            )
        else:
            sev = Severity.CRITICAL
            finding = Finding(
                observed=(
                    f"The model's chunk prediction (made at frame t) for "
                    f"the next analyzed frame disagrees with the action it "
                    f"actually emits there by {mean_score:.3f} of typical "
                    f"magnitude on average (max {max_score:.3f}). That's "
                    f"above the strong-disagreement threshold "
                    f"({self.grounded_threshold:.3f})."
                ),
                meaning=(
                    "Running this model's action chunks beyond the very "
                    "first step is unsafe. The downstream steps are "
                    "best-guess and do not match what the model later "
                    "decides to do."
                ),
                next_step=(
                    "Re-plan every step at deployment time (don't trust "
                    "chunk[1:]). If your deployment recording shows "
                    "failures that started with a chunk-rollout phase, "
                    "this may be why."
                ),
                raw_numbers=raw_numbers,
            )
            verdict = (
                f"Chunk lookahead is INCONSISTENT: normalized mean "
                f"disagreement {mean_score:.3f} ≥ grounded threshold "
                f"({self.grounded_threshold}, max {max_score:.3f}). The "
                f"model's chunk prediction (made at frame t) for the next "
                f"analyzed frame materially disagrees with the action it "
                f"actually emits there. Running this model's chunks beyond "
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
            finding=finding,
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
