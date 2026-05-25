"""Model-vs-model diff reporter.

Run the same Suite on two models / two checkpoints; render side-by-side
deltas. Answers "did fine-tuning fix the noun-blindness?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from policylens.core.results import Severity
from policylens.suites.base import SuiteResult


def render_comparison(a: SuiteResult, b: SuiteResult, out_path: Path) -> Path:
    """Side-by-side axes for a vs b; arrow showing delta per axis."""
    keys = sorted(set(a.results) & set(b.results))
    if not keys:
        raise ValueError("No overlapping diagnostics to compare.")

    axes_labels = [a.results[k].axis for k in keys]
    a_scores = [a.results[k].scalar_score for k in keys]
    b_scores = [b.results[k].scalar_score for k in keys]
    deltas = [b - a for a, b in zip(a_scores, b_scores)]

    n = len(keys)
    fig, ax = plt.subplots(figsize=(14, max(2.0, 0.5 * n + 1)))
    y = np.arange(n)[::-1]
    width = 0.4
    ax.barh(y + width / 2, a_scores, height=width, color="#888", label=a.model_id)
    ax.barh(y - width / 2, b_scores, height=width, color="#1971c2", label=b.model_id)
    ax.set_yticks(y); ax.set_yticklabels(axes_labels, fontsize=9)
    ax.set_xlabel("scalar score")
    for i, (av, bv, d) in enumerate(zip(a_scores, b_scores, deltas)):
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
        color = "#2b8a3e" if d > 0 else ("#c92a2a" if d < 0 else "#666")
        ax.text(max(av, bv) + 0.005, y[i], f"  {arrow} {d:+.3f}",
                va="center", fontsize=9, color=color, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"PolicyLens Comparison — {a.model_id} vs {b.model_id} ({a.suite_name})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
