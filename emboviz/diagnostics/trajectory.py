"""Run a Diagnostic across every frame of a Trajectory.

A `TrajectoryDiagnostic` wraps any existing single-scene `Diagnostic` and
applies it frame-by-frame. The result is a `TrajectoryDiagnosticResult`
with per-frame scores, severities, and auto-detected failure moments.

Pattern: composition. We don't subclass each diagnostic; we wrap it. This
keeps every single-scene diagnostic trivially trajectory-able.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from emboviz.core.results import DiagnosticResult, Finding, Severity
from emboviz.core.types import ActionResult, Trajectory
from emboviz.diagnostics.base import Diagnostic
from emboviz.models.protocol import VLAModel


# Axis-verdict cutoffs, applied over the TESTABLE frames only (UNKNOWN
# excluded — a coverage gap is not a verdict). The axis verdict is a
# PROPORTION of the frames we could actually test, never the single worst one:
# a couple of flagged frames among many clean ones is a note, not a headline.
_AXIS_CRITICAL_FRAC = 0.50   # ≥ half the testable frames flagged → CRITICAL
_AXIS_CONCERN_FRAC  = 0.20   # ≥ this fraction flagged-or-partial → MODERATE


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

    def trajectory_severity(self) -> Severity:
        """Axis-level verdict for the whole trajectory — a PROPORTION of the
        TESTABLE frames, not the single worst one.

        UNKNOWN ('couldn't test') is a coverage gap, not a verdict, so it is
        excluded from the denominator: the verdict is computed only over the
        frames the diagnostic could actually test. A handful of flagged frames
        among many clean ones is therefore a note (PASS / MODERATE), never a
        CRITICAL headline. If no frame was testable, the axis is UNKNOWN.
        :meth:`trajectory_finding` quotes a frame matching this severity, so the
        label and the text always agree.
        """
        counts = {s: 0 for s in Severity}
        for r in self.per_frame:
            counts[r.severity] += 1
        n_test = len(self.per_frame) - counts[Severity.UNKNOWN]
        if n_test <= 0:
            return Severity.UNKNOWN
        frac_critical = counts[Severity.CRITICAL] / n_test
        frac_concern = (counts[Severity.CRITICAL] + counts[Severity.MODERATE]) / n_test
        if frac_critical >= _AXIS_CRITICAL_FRAC:
            return Severity.CRITICAL
        if frac_concern >= _AXIS_CONCERN_FRAC:
            return Severity.MODERATE
        # Mostly clean. Report INFO only if every testable frame was at most
        # 'noteworthy' (no clean PASS frames to represent it); else PASS.
        if counts[Severity.PASS] == 0 and counts[Severity.INFO] > 0:
            return Severity.INFO
        return Severity.PASS

    def _representative_finding(self, axis_sev: Severity) -> Optional[Finding]:
        """A per-frame Finding that illustrates the axis verdict: the first
        frame whose severity matches ``axis_sev``, else the worst testable
        frame's Finding — so the quoted text always agrees with the headline
        label and is never empty when any frame was testable."""
        for r in self.per_frame:
            if r.severity == axis_sev and r.finding is not None:
                return r.finding
        for sev in (Severity.CRITICAL, Severity.MODERATE,
                    Severity.INFO, Severity.PASS):
            for r in self.per_frame:
                if r.severity == sev and r.finding is not None:
                    return r.finding
        return None

    def trajectory_finding(self) -> Finding:
        """Aggregate per-frame Findings into a single trajectory-level Finding.

        Counts the per-frame severity distribution, picks the dominant
        verdict, and renders three plain-English sentences describing
        what we saw across the whole episode (or window). Examples:

          • "On 7 of 8 frames the model produced near-identical actions
             when the target was masked. The remaining 1/8 showed real
             visual response." → memorized signature is dominant.
          • "On 5 of 8 frames the wrist camera was IGNORED; on the other
             3 it was USED. Looks phase-dependent."
          • "On all 8 frames the diagnostic was inconclusive — the
             frames were quiescent. Try a more dynamic episode."

        The trajectory-level Finding is what users see at the top of the
        per-episode report; the per-frame Findings are available for
        drill-down.
        """
        n = len(self.per_frame)
        if n == 0:
            return Finding(
                observed="No frames were analyzed.",
                meaning="The trajectory window was empty.",
                next_step="Pick an episode with frames and re-run.",
                raw_numbers={"n_frames": 0},
            )

        counts: dict[Severity, int] = {s: 0 for s in Severity}
        for r in self.per_frame:
            counts[r.severity] += 1
        n_pass = counts[Severity.PASS]
        n_info = counts[Severity.INFO]
        n_mod  = counts[Severity.MODERATE]
        n_crit = counts[Severity.CRITICAL]
        n_unk  = counts[Severity.UNKNOWN]
        n_test = n - n_unk

        # The quoted Finding comes from a frame matching the axis verdict (see
        # _representative_finding), so the headline label and the text agree.
        axis_sev = self.trajectory_severity()
        rep_finding = self._representative_finding(axis_sev)

        if n_test == 0:
            # Nothing was testable — report that honestly, never as a verdict.
            observed = (
                f"All {n} frame(s) were inconclusive — the diagnostic could "
                f"not test this axis on any of them."
            )
            meaning = (
                rep_finding.meaning if rep_finding is not None
                else "No frame produced a testable result on this axis."
            )
            next_step = (
                rep_finding.next_step if rep_finding is not None
                else "Re-run on a more dynamic episode or a more varied dataset."
            )
        else:
            # Distribution over the frames we could TEST; inconclusive frames
            # are reported separately, never counted toward the verdict.
            parts: list[str] = []
            if n_crit > 0: parts.append(f"{n_crit} flagged")
            if n_mod  > 0: parts.append(f"{n_mod} partial")
            if n_info > 0: parts.append(f"{n_info} noteworthy")
            if n_pass > 0: parts.append(f"{n_pass} clean")
            dist_str = ", ".join(parts)
            untested = f" ({n_unk} couldn't be tested)" if n_unk else ""
            lead = f"Of {n_test} testable frame(s){untested}: {dist_str}."
            if rep_finding is not None:
                observed  = f"{lead} Representative frame: {rep_finding.observed}"
                meaning   = rep_finding.meaning
                next_step = rep_finding.next_step
            elif axis_sev == Severity.PASS or axis_sev == Severity.INFO:
                observed  = lead
                meaning   = "This axis is healthy across the frames we could test."
                next_step = "No action needed for this axis."
            else:
                observed  = lead
                meaning   = "Mixed across the frames we could test."
                next_step = (
                    "Inspect the flagged frames in Rerun; cross-reference with "
                    "the other axes' findings."
                )

        return Finding(
            observed=observed,
            meaning=meaning,
            next_step=next_step,
            raw_numbers={
                "n_frames":        n,
                "n_testable":      n_test,
                "n_pass":          n_pass,
                "n_info":          n_info,
                "n_moderate":      n_mod,
                "n_critical":      n_crit,
                "n_unknown":       n_unk,
                "mean_score":      self.mean_score,
                "median_score":    self.median_score,
                "worst_frame_idx": self.worst_frame_idx,
            },
        )

    def to_summary(self) -> dict:
        ci_lo, ci_hi = self.bootstrap_ci()
        finding = self.trajectory_finding()
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
            "severity":          self.trajectory_severity().value,
            "severities":        [s.value for s in self.severities],
            "finding":           finding.to_dict(),
            "per_frame_findings": [
                r.finding.to_dict() if r.finding is not None else None
                for r in self.per_frame
            ],
        }


class TrajectoryDiagnostic:
    """Wrapper: apply any single-scene Diagnostic across all frames of a Trajectory.

    When ``baselines`` is supplied (one per frame, in trajectory order),
    each per-frame ``Diagnostic.run`` is called with ``baseline=...`` so
    the wrapped diagnostic skips recomputing the unperturbed prediction.
    The runner uses this to share a single per-frame baseline across
    every diagnostic — saving ``n_samples × num_diagnostics`` model
    forward passes per frame on stochastic models.

    Diagnostics whose ``run`` does NOT accept a ``baseline`` kwarg
    (legacy / no-baseline diagnostics) are called without it; we
    introspect each diagnostic's signature once at wrap time.
    """

    def __init__(self, diagnostic: Diagnostic, progress: bool = True):
        self.diagnostic = diagnostic
        self.progress = progress
        self.name = f"trajectory.{diagnostic.name}"
        self.axis = diagnostic.axis
        # Cache whether the wrapped diagnostic accepts a ``baseline``
        # kwarg so we don't pay reflection cost per frame.
        try:
            sig = inspect.signature(diagnostic.run)
            self._supports_baseline = "baseline" in sig.parameters
        except (TypeError, ValueError):
            self._supports_baseline = False

    def run(
        self, model: VLAModel, trajectory: Trajectory,
        *, baselines: Optional[list[ActionResult]] = None,
    ) -> TrajectoryDiagnosticResult:
        if baselines is not None and len(baselines) != len(trajectory.frames):
            raise ValueError(
                f"TrajectoryDiagnostic.run: baselines length "
                f"{len(baselines)} does not match trajectory frame count "
                f"{len(trajectory.frames)}. Pass exactly one baseline per "
                "frame in trajectory order, or pass None to let the "
                "diagnostic recompute its own."
            )
        # Probe direction from the first result (cheap; consistent across frames).
        iterator = enumerate(trajectory.frames)
        if self.progress:
            iterator = tqdm(
                list(iterator),
                desc=self.diagnostic.name,
                unit="frame",
                leave=False,
            )
        per_frame: list[DiagnosticResult] = []
        direction = "lower_is_worse"
        for i, scene in iterator:
            if self._supports_baseline and baselines is not None:
                r = self.diagnostic.run(model, scene, baseline=baselines[i])
            else:
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
