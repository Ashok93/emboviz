"""Render a Suite's results as a per-axis bar chart with severity colouring.

This is the canonical "Failure Signature" view: every diagnostic axis on
the y-axis, scalar_score on the x-axis, colour = severity. Reading down
the page is reading down a 10-axis failure profile.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.suites.base import SuiteResult


_SEV_COLOR = {
    Severity.CRITICAL: "#c92a2a",
    Severity.MODERATE: "#fab005",
    Severity.PASS: "#2b8a3e",
    Severity.INFO: "#1971c2",
    Severity.UNKNOWN: "#888888",
}


def render_failure_matrix(suite_result: SuiteResult, out_path: Path,
                          title_extra: str = "") -> Path:
    results = list(suite_result.results.values())
    # Skip unknowns at the top for readability.
    runnable = [r for r in results if r.severity != Severity.UNKNOWN]
    skipped = [r for r in results if r.severity == Severity.UNKNOWN]

    n = len(runnable)
    fig_h = max(2.0, 0.5 * n + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    if runnable:
        runnable.sort(key=lambda r: r.scalar_score)
        names = [r.axis for r in runnable]
        scores = [r.scalar_score for r in runnable]
        colors = [_SEV_COLOR[r.severity] for r in runnable]
        y = np.arange(len(names))[::-1]
        ax.barh(y, scores, color=colors)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("scalar score (interpretation depends on diagnostic)", fontsize=9)
        # Render the scalar score as a label next to each bar; severity
        # is conveyed only by colour (see _SEV_COLOR), not by a text label
        # — we keep severity words out of user-facing rendered output.
        for i, r in enumerate(runnable):
            ax.text(r.scalar_score, y[i], f"  {r.scalar_score:.3f}",
                    va="center", fontsize=9, color=_SEV_COLOR[r.severity], fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "(no diagnostics produced runnable results)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    title = f"Emboviz Failure Matrix — {suite_result.model_id} on {suite_result.scene_id}"
    if title_extra:
        title += f"\n{title_extra}"
    if skipped:
        title += f"\n({len(skipped)} diagnostic(s) skipped: missing capability or not applicable)"
    ax.set_title(title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
