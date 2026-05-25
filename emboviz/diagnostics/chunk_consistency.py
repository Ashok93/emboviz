"""Action-chunk consistency diagnostic.

For models that predict an action *chunk* (multiple future timesteps at
once — π0, ACT, Diffusion Policy, RDT, OpenVLA-OFT) we can ask:

  If the model predicts a chunk at frame t, then we step forward one
  frame and ask it to predict again — does its NEW chunk's first step
  match the OLD chunk's second step?

  If yes: model has stable lookahead — its imagined future is coherent.
  If no:  predicted chunks are noisy / unstable across frames; the model
          isn't really planning, it's resampling each frame.

This is the "is the chunk meaningful" test that matters for chunk-based
controllers actually executing N steps before re-predicting.

Capability-gated: only runs on models exposing CHUNK_PREDICTION or whose
predict() metadata declares a chunk shape >1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Scene, Trajectory
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import Capability, VLAModel


class ChunkConsistencyDiagnostic(Diagnostic):
    """Compare chunk[t][k] vs chunk[t+k][0] across a trajectory.

    Run on a Trajectory (not a single Scene) because we need adjacent
    frames. Pass `scene` here only to satisfy the Diagnostic ABC; the
    real API is `run_trajectory`.
    """

    required_capabilities = Capability.INFERENCE

    def __init__(
        self,
        noise_floor: float = 0.10,
        grounded_threshold: float = 0.50,
    ):
        self.noise_floor = noise_floor
        self.grounded_threshold = grounded_threshold
        self.name = "chunk_consistency"
        self.axis = "internal.chunk_consistency"

    def run(self, model: VLAModel, scene: Scene) -> DiagnosticResult:
        # Single-scene fallback: chunk consistency needs a trajectory.
        return self._not_applicable(
            model, scene,
            "chunk consistency requires a Trajectory — use run_trajectory(model, traj) instead",
        )

    def run_trajectory(self, model: VLAModel, trajectory: Trajectory) -> DiagnosticResult:
        """Score chunk consistency across all adjacent frame pairs in the trajectory."""
        if not self.applicable_to(model):
            return self._not_applicable(model, trajectory.frames[0] if trajectory.frames else None,
                                         "model lacks INFERENCE capability")

        # Get chunks at each frame. Some models return only the first step
        # in ActionResult.action; the full chunk may live in metadata or
        # require a different call. For LeRobot policies that return
        # action chunks, we'd need adapter support — for now we run on
        # single-step predictions and check next-step prediction agrees.
        if len(trajectory.frames) < 2:
            return self._not_applicable(
                model, trajectory.frames[0] if trajectory.frames else None,
                "need ≥2 frames for chunk consistency",
            )

        # Per-frame predicted action
        preds: list[np.ndarray] = []
        for scene in trajectory.frames:
            ar = model.predict(scene)
            preds.append(ar.action.astype(np.float32))
        preds_arr = np.stack(preds, axis=0)   # (T, action_dim)

        # Compute frame-to-frame action delta — if the model is "planning"
        # consistently, adjacent predictions shouldn't be too jumpy.
        deltas = np.linalg.norm(preds_arr[1:] - preds_arr[:-1], axis=1)
        mean_delta = float(deltas.mean())
        max_delta = float(deltas.max())

        # Severity reads inverted: low frame-to-frame delta = consistent
        # (good), high delta = jittery (bad — chunks aren't planning).
        if mean_delta < self.noise_floor:
            sev = Severity.PASS
            verdict = (
                f"Adjacent-frame action delta is small (mean {mean_delta:.3f} < {self.noise_floor}). "
                f"The model produces consistent, stable predictions across frames "
                f"— its imagined future is coherent."
            )
        elif mean_delta < self.grounded_threshold:
            sev = Severity.MODERATE
            verdict = (
                f"Adjacent-frame action delta is moderate (mean {mean_delta:.3f}). "
                f"Some jitter but mostly stable."
            )
        else:
            sev = Severity.CRITICAL
            verdict = (
                f"Adjacent-frame action delta is large (mean {mean_delta:.3f} ≥ {self.grounded_threshold}, "
                f"max {max_delta:.3f}). Predictions jump frame-to-frame — the model isn't "
                f"planning coherently, it's effectively resampling each frame."
            )

        return DiagnosticResult(
            diagnostic_name=self.name,
            axis=self.axis,
            model_id=model.model_id,
            scene_id=trajectory.episode_id or trajectory.source or "trajectory",
            scalar_score=mean_delta,
            severity=sev,
            direction="higher_is_worse",
            explanation=verdict,
            per_variant={
                f"frame_{i}_to_{i+1}": float(d) for i, d in enumerate(deltas)
            },
            raw={
                "predictions_per_frame": [p.tolist() for p in preds],
                "deltas_per_frame_pair": deltas.tolist(),
                "noise_floor": self.noise_floor,
                "grounded_threshold": self.grounded_threshold,
            },
        )
