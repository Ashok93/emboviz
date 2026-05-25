"""Run a Diagnostic across every frame of a Trajectory.

A `TrajectoryDiagnostic` wraps any existing single-scene `Diagnostic` and
applies it frame-by-frame. The result is a `TrajectoryDiagnosticResult`
with per-frame scores, severities, and auto-detected failure moments.

Pattern: composition. We don't subclass each diagnostic; we wrap it. This
keeps every single-scene diagnostic trivially trajectory-able.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from tqdm import tqdm

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.core.types import Trajectory
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import VLAModel


@dataclass
class TrajectoryDiagnosticResult:
    """Per-frame DiagnosticResults plus aggregate stats and failure moments."""

    diagnostic_name: str
    axis: str
    model_id: str
    trajectory_source: str                     # e.g. "bridge:0"
    frame_indices: list[int]                   # dataset-frame indices
    per_frame: list[DiagnosticResult]
    direction: str                             # "lower_is_worse" / "higher_is_worse"

    @property
    def scores(self) -> np.ndarray:
        return np.array([r.scalar_score for r in self.per_frame], dtype=np.float32)

    @property
    def severities(self) -> list[Severity]:
        return [r.severity for r in self.per_frame]

    @property
    def mean_score(self) -> float:
        valid = self.scores[~np.isnan(self.scores)]
        return float(valid.mean()) if valid.size else float("nan")

    @property
    def median_score(self) -> float:
        valid = self.scores[~np.isnan(self.scores)]
        return float(np.median(valid)) if valid.size else float("nan")

    @property
    def worst_frame_idx(self) -> int:
        """Dataset-frame index of the worst (most problematic) frame.

        'Worst' depends on the diagnostic direction:
          • lower_is_worse → frame with the smallest score
          • higher_is_worse → frame with the largest score
        """
        s = self.scores.copy()
        if not np.isfinite(s).any():
            return -1
        if self.direction == "lower_is_worse":
            s[np.isnan(s)] = np.inf
            i = int(np.argmin(s))
        else:
            s[np.isnan(s)] = -np.inf
            i = int(np.argmax(s))
        return self.frame_indices[i]

    def failure_moments(
        self, severity_at_least: Severity = Severity.CRITICAL,
    ) -> list[int]:
        """Dataset-frame indices whose severity ≥ `severity_at_least`.

        Ordering of severities used: pass < info < moderate < critical.
        """
        rank = {Severity.PASS: 0, Severity.INFO: 1, Severity.MODERATE: 2,
                Severity.CRITICAL: 3, Severity.UNKNOWN: -1}
        threshold = rank[severity_at_least]
        return [
            self.frame_indices[i]
            for i, r in enumerate(self.per_frame)
            if rank.get(r.severity, -1) >= threshold
        ]

    def bootstrap_ci(
        self,
        n_resamples: int = 1000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval for ``mean_score``.

        Resamples the per-frame scores with replacement and reports the
        percentile interval. Returns ``(NaN, NaN)`` if fewer than 2 valid
        scores are available. The interval communicates "how much would
        this number wobble if we picked a different set of frames from
        the same distribution?" — a critical sanity check when each
        trajectory only has 8 frames.
        """
        s = self.scores[~np.isnan(self.scores)]
        if s.size < 2:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(seed)
        means = np.array([
            rng.choice(s, size=s.size, replace=True).mean()
            for _ in range(n_resamples)
        ], dtype=np.float64)
        lo = float(np.percentile(means, 100 * alpha / 2))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return (lo, hi)

    def to_summary(self) -> dict:
        ci_lo, ci_hi = self.bootstrap_ci()
        return {
            "diagnostic_name":   self.diagnostic_name,
            "axis":              self.axis,
            "model_id":          self.model_id,
            "trajectory_source": self.trajectory_source,
            "n_frames":          len(self.per_frame),
            "mean_score":        self.mean_score,
            "mean_score_ci95":   [ci_lo, ci_hi],
            "median_score":      self.median_score,
            "worst_frame_idx":   self.worst_frame_idx,
            "failure_moments":   self.failure_moments(),
            "direction":         self.direction,
            "frame_indices":     self.frame_indices,
            "scores":            self.scores.tolist(),
            "severities":        [s.value for s in self.severities],
        }


class TrajectoryDiagnostic:
    """Wrapper: apply any single-scene Diagnostic across all frames of a Trajectory."""

    def __init__(self, diagnostic: Diagnostic, progress: bool = True):
        self.diagnostic = diagnostic
        self.progress = progress
        self.name = f"trajectory.{diagnostic.name}"
        self.axis = diagnostic.axis

    def run(self, model: VLAModel, trajectory: Trajectory) -> TrajectoryDiagnosticResult:
        # Probe direction from the first result (cheap; consistent across frames).
        iterator = trajectory.frames
        if self.progress:
            iterator = tqdm(
                trajectory.frames,
                desc=self.diagnostic.name,
                unit="frame",
                leave=False,
            )
        per_frame: list[DiagnosticResult] = []
        direction = "lower_is_worse"
        for scene in iterator:
            r = self.diagnostic.run(model, scene)
            per_frame.append(r)
            direction = r.direction
        return TrajectoryDiagnosticResult(
            diagnostic_name=self.diagnostic.name,
            axis=self.diagnostic.axis,
            model_id=model.model_id,
            trajectory_source=trajectory.source,
            frame_indices=list(trajectory.frame_indices),
            per_frame=per_frame,
            direction=direction,
        )
