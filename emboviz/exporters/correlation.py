"""Cross-axis correlation analysis for trajectory diagnostics.

When multiple per-frame diagnostics fire CRITICAL at the same frame
(or within a small window), that's the failure moment most worth
investigating. This module surfaces that — instead of forcing the
user to eyeball raw per-axis severity arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from emboviz.core.results import Severity
from emboviz.diagnostics.trajectory import TrajectoryDiagnosticResult


@dataclass(frozen=True)
class FailureMoment:
    """A frame at which multiple diagnostics agree something is wrong."""

    frame_idx: int                    # dataset-frame index
    n_critical_axes: int
    critical_axes: list[str]
    expert_delta: Optional[float] = None   # if available, ‖predicted − expert‖
    notes: str = ""


def find_failure_moments(
    per_axis_results: dict[str, TrajectoryDiagnosticResult],
    expert_delta_per_frame: Optional[list[float]] = None,
    *,
    min_critical_axes: int = 2,
) -> list[FailureMoment]:
    """Find frames where ≥`min_critical_axes` diagnostics fire CRITICAL.

    Returns sorted (highest agreement first). When `expert_delta_per_frame`
    is provided, frames where it spikes above its mean+1σ are also flagged
    and their critical-count noted.
    """
    if not per_axis_results:
        return []

    # All trajectories share the same frame_indices — take from the first.
    first = next(iter(per_axis_results.values()))
    n_frames = len(first.per_frame)
    frame_indices = list(first.frame_indices)

    per_frame_critical: list[list[str]] = [[] for _ in range(n_frames)]
    for axis, tr in per_axis_results.items():
        for i, r in enumerate(tr.per_frame):
            if r.severity == Severity.CRITICAL:
                per_frame_critical[i].append(axis)

    # If expert delta is given, compute its outlier threshold (mean + 1σ).
    expert_spike_threshold: Optional[float] = None
    if expert_delta_per_frame:
        arr = np.asarray(expert_delta_per_frame, dtype=np.float32)
        valid = arr[~np.isnan(arr)]
        if valid.size:
            expert_spike_threshold = float(valid.mean() + valid.std())

    moments: list[FailureMoment] = []
    for i, crits in enumerate(per_frame_critical):
        if len(crits) >= min_critical_axes:
            ed = (
                float(expert_delta_per_frame[i])
                if expert_delta_per_frame and i < len(expert_delta_per_frame)
                else None
            )
            notes = []
            if expert_spike_threshold is not None and ed is not None and ed >= expert_spike_threshold:
                notes.append("expert-Δ also spikes here")
            moments.append(FailureMoment(
                frame_idx=frame_indices[i],
                n_critical_axes=len(crits),
                critical_axes=sorted(crits),
                expert_delta=ed,
                notes=", ".join(notes),
            ))

    moments.sort(key=lambda m: (-m.n_critical_axes, m.frame_idx))
    return moments


def format_failure_moments(
    moments: list[FailureMoment],
    *,
    max_show: int = 10,
) -> str:
    """Render failure moments as a human-readable bulleted summary."""
    if not moments:
        return "  (no frames with ≥2 critical signals)"
    lines = []
    for m in moments[:max_show]:
        line = (
            f"  frame {m.frame_idx:>4}  "
            f"{m.n_critical_axes} critical axes: "
            f"{', '.join(m.critical_axes)}"
        )
        if m.expert_delta is not None:
            line += f"   expert-Δ={m.expert_delta:.3f}"
        if m.notes:
            line += f"   [{m.notes}]"
        lines.append(line)
    if len(moments) > max_show:
        lines.append(f"  ... ({len(moments) - max_show} more frames with ≥2 critical signals)")
    return "\n".join(lines)
