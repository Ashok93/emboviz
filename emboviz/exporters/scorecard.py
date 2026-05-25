"""Scorecard — the at-a-glance axis-by-axis severity grid.

Strict no-prose policy: this is data presented visually, not someone's
opinion of what to do about it. One row per diagnostic axis. Each row
shows:
  • axis name
  • severity badge (color-coded)
  • scalar score
  • brief direction-aware label ("lower is worse" / "higher is worse")

Users scan it in 3 seconds, decide which axes warrant drilling into,
open the corresponding detail page or Rerun playback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from emboviz.core.results import DiagnosticResult, Severity
from emboviz.suites.base import SuiteResult


SEVERITY_COLORS = {
    Severity.CRITICAL: "#c92a2a",   # red
    Severity.MODERATE: "#e67700",   # orange
    Severity.INFO:     "#1971c2",   # blue
    Severity.PASS:     "#2f9e44",   # green
    Severity.UNKNOWN:  "#868e96",   # grey
}

SEVERITY_LABELS = {
    Severity.CRITICAL: "CRITICAL",
    Severity.MODERATE: "MODERATE",
    Severity.INFO:     "INFO",
    Severity.PASS:     "PASS",
    Severity.UNKNOWN:  "N/A",
}


def render_scorecard(
    suite_result: SuiteResult,
    out_path: Path,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Path:
    """Render the suite result as a one-page axis-by-axis severity grid.

    Pure data: no prose synthesis, no recommendations. Users see the
    failure profile and decide which axis to investigate.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = list(suite_result.results.values())
    n_rows = max(len(results), 1)
    row_h = 0.45
    fig_h = max(2.0, 0.6 + n_rows * row_h + 1.0)
    fig: Figure = plt.figure(figsize=(11, fig_h), facecolor="white")
    ax = fig.add_axes((0.04, 0.05, 0.92, 0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")

    # Header
    head_y = n_rows + 0.6
    head = title or f"Emboviz — {suite_result.suite_name}"
    sub = subtitle or f"model: {suite_result.model_id}"
    ax.text(0.0, head_y, head, fontsize=18, fontweight="bold",
            transform=ax.transData, verticalalignment="center")
    ax.text(0.0, head_y - 0.38, sub, fontsize=10, color="#666",
            transform=ax.transData, verticalalignment="center")

    # Severity rollup pill row
    pill_y = head_y - 0.7
    counts: dict[Severity, int] = {s: 0 for s in Severity}
    for r in results:
        counts[r.severity] += 1
    cx = 0.0
    for sev in (Severity.CRITICAL, Severity.MODERATE, Severity.INFO,
                Severity.PASS, Severity.UNKNOWN):
        if counts[sev] == 0:
            continue
        label = f"{counts[sev]}  {SEVERITY_LABELS[sev]}"
        _draw_pill(ax, cx, pill_y, label, SEVERITY_COLORS[sev])
        cx += 0.10 + 0.0095 * len(label)

    # Header bar above rows
    bar_y = n_rows + 0.05
    ax.text(0.03, bar_y, "AXIS", fontsize=9, color="#666", weight="bold")
    ax.text(0.42, bar_y, "SEVERITY", fontsize=9, color="#666", weight="bold")
    ax.text(0.62, bar_y, "SCORE", fontsize=9, color="#666", weight="bold")
    ax.text(0.78, bar_y, "DIRECTION", fontsize=9, color="#666", weight="bold")
    ax.plot([0.0, 1.0], [bar_y - 0.10, bar_y - 0.10],
            color="#ddd", linewidth=0.6, transform=ax.transData)

    # Rows
    severity_order = {
        Severity.CRITICAL: 0, Severity.MODERATE: 1, Severity.INFO: 2,
        Severity.PASS: 3, Severity.UNKNOWN: 4,
    }
    results_sorted = sorted(
        results,
        key=lambda r: (severity_order.get(r.severity, 5), r.axis),
    )
    for i, r in enumerate(results_sorted):
        y = n_rows - i - 0.5
        ax.text(0.03, y, r.axis, fontsize=11, color="#222",
                verticalalignment="center")
        _draw_pill(
            ax, 0.42, y, SEVERITY_LABELS[r.severity],
            SEVERITY_COLORS[r.severity],
        )
        score_str = "—" if r.scalar_score != r.scalar_score else f"{r.scalar_score:.3f}"
        ax.text(0.62, y, score_str, fontsize=10, family="monospace",
                color="#222", verticalalignment="center")
        ax.text(0.78, y, r.direction.replace("_", " "), fontsize=9,
                color="#666", verticalalignment="center")

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _draw_pill(ax, x: float, y: float, text: str, color: str) -> None:
    """Draw a colored rounded-rectangle pill with centered text."""
    w = 0.018 * (len(text) + 2)
    h = 0.32
    pill = mpatches.FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor=color,
        transform=ax.transData,
    )
    ax.add_patch(pill)
    ax.text(x + w / 2, y, text, fontsize=9, color="white",
            ha="center", va="center", weight="bold")
