"""Trajectory-level reports: per-axis timelines + multi-axis heatmap.

Two views:
  1. **Per-axis timeline grid** — one mini-plot per diagnostic axis;
     score-vs-frame line, severity-colour-coded markers, failure-moment
     vertical bars. Best for inspecting one axis closely.
  2. **Multi-axis heatmap** — rows = axes, cols = frames, color = score
     (normalized per row so axes with different scales are comparable).
     The 'failure tape' view — at-a-glance where in the rollout problems
     happen across all axes.

Both consume `TrajectorySuiteResult` only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from emboviz.core.results import Severity
from emboviz.suites.base import TrajectorySuiteResult


_SEV_COLOR = {
    Severity.CRITICAL: "#c92a2a",
    Severity.MODERATE: "#fab005",
    Severity.PASS: "#2b8a3e",
    Severity.INFO: "#1971c2",
    Severity.UNKNOWN: "#888888",
}


def render_trajectory_timelines(
    suite_result: TrajectorySuiteResult, out_path: Path,
    max_axes_per_col: int = 4,
) -> Path:
    """N-axis grid of small score-vs-frame plots."""
    results = [r for r in suite_result.results.values()
               if any(np.isfinite(r.scores))]
    if not results:
        # Fall back to a 'no data' card
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No trajectory diagnostics produced numeric scores.",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    n = len(results)
    n_cols = max(1, (n + max_axes_per_col - 1) // max_axes_per_col)
    n_rows = min(n, max_axes_per_col)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.4 * n_rows),
                             squeeze=False)

    for i, tr in enumerate(results):
        r, c = i % n_rows, i // n_rows
        ax = axes[r][c]
        x = np.array(tr.frame_indices)
        y = tr.scores
        finite = np.isfinite(y)
        ax.plot(x[finite], y[finite], "-", color="#444", lw=1.0)
        # Per-frame coloured markers by severity
        for j, (xv, yv, sev) in enumerate(zip(x, y, tr.severities)):
            if np.isfinite(yv):
                ax.plot(xv, yv, "o", color=_SEV_COLOR[sev], markersize=5)
        # Mark failure moments with vertical lines
        for fmoment in tr.failure_moments():
            ax.axvline(fmoment, color="#c92a2a", linestyle=":", lw=0.8, alpha=0.7)
        ax.set_title(f"{tr.axis}\n(worst frame: t={tr.worst_frame_idx}, "
                     f"mean={tr.mean_score:.3f})", fontsize=9, loc="left")
        ax.set_xlabel("frame", fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for k in range(n, n_rows * n_cols):
        r, c = k % n_rows, k // n_rows
        axes[r][c].axis("off")

    fig.suptitle(
        f"Emboviz Trajectory — {suite_result.model_id} on {suite_result.trajectory_source}",
        fontsize=12, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_failure_tape(
    suite_result: TrajectorySuiteResult, out_path: Path,
) -> Path:
    """Multi-axis heatmap: rows = axes, cols = frames, color = severity."""
    results = [r for r in suite_result.results.values()
               if any(np.isfinite(r.scores))]
    if not results:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path

    # Build a (n_axes, n_frames) severity-rank matrix.
    sev_rank = {Severity.PASS: 0, Severity.INFO: 1, Severity.MODERATE: 2,
                Severity.CRITICAL: 3, Severity.UNKNOWN: -1}
    all_frames = sorted({fi for r in results for fi in r.frame_indices})
    frame_to_col = {fi: i for i, fi in enumerate(all_frames)}

    mat = np.full((len(results), len(all_frames)), -1.0, dtype=np.float32)
    for ri, r in enumerate(results):
        for fi, sev in zip(r.frame_indices, r.severities):
            mat[ri, frame_to_col[fi]] = sev_rank[sev]

    fig, ax = plt.subplots(figsize=(max(8, len(all_frames) * 0.35),
                                     max(2.5, 0.5 * len(results) + 1)))
    cmap = plt.get_cmap("RdYlGn_r", 5)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=3,
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(results)))
    ax.set_yticklabels([r.axis for r in results], fontsize=9)
    ax.set_xticks(np.arange(len(all_frames))[::max(1, len(all_frames) // 10)])
    ax.set_xticklabels([f"t={all_frames[i]}" for i in
                        range(0, len(all_frames), max(1, len(all_frames) // 10))],
                       rotation=45, fontsize=8)
    ax.set_title(f"Failure tape — {suite_result.model_id} on {suite_result.trajectory_source}",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(["unknown", "pass", "info", "moderate", "critical"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
