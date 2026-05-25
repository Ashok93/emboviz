"""Top-level verdict card — combines failure matrix + a recommended-action
text block + the scene image into one shareable PNG.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from policylens.core.results import Severity
from policylens.core.types import Scene
from policylens.reports.failure_matrix import render_failure_matrix
from policylens.suites.base import SuiteResult
from policylens.viz.stitch import stitch_vertical


def render_verdict_card(
    suite_result: SuiteResult, scene: Scene, out_path: Path,
) -> Path:
    """Stitch: header (scene + title) → failure matrix → recommendation block."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        header = _render_header(suite_result, scene, tmp / "01_header.png")
        matrix = render_failure_matrix(suite_result, tmp / "02_matrix.png")
        rec = _render_recommendation(suite_result, tmp / "03_rec.png")
        stitch_vertical([header, matrix, rec], out_path)
    return out_path


def _render_header(suite_result: SuiteResult, scene: Scene, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    axes[0].imshow(np.array(scene.image))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("SCENE", fontsize=10, loc="left", color="#666")

    ax = axes[1]
    ax.axis("off")
    ax.text(0.0, 1.0, f"PolicyLens — {suite_result.suite_name}",
            fontsize=20, fontweight="bold", transform=ax.transAxes,
            verticalalignment="top")
    ax.text(0.0, 0.85, f"Model: {suite_result.model_id}", fontsize=11,
            transform=ax.transAxes, color="#444")
    ax.text(0.0, 0.78, f'Instruction: "{scene.instruction}"', fontsize=11,
            transform=ax.transAxes, color="#444")

    # Quick aggregate severity summary
    sev_counts = {s: 0 for s in Severity}
    for r in suite_result.results.values():
        sev_counts[r.severity] += 1
    line = "  ·  ".join(
        f"{count} {sev.value}"
        for sev, count in sev_counts.items() if count > 0
    )
    ax.text(0.0, 0.62, "Diagnostic counts:", fontsize=10, color="#666",
            transform=ax.transAxes)
    ax.text(0.0, 0.55, line, fontsize=12, transform=ax.transAxes)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _render_recommendation(suite_result: SuiteResult, out: Path) -> Path:
    # Pick the worst-severity diagnostic with a recommendation; if none, the
    # critical/moderate one with the lowest score.
    items = list(suite_result.results.values())
    severity_rank = {
        Severity.CRITICAL: 0, Severity.MODERATE: 1,
        Severity.INFO: 2, Severity.PASS: 3, Severity.UNKNOWN: 4,
    }
    items.sort(key=lambda r: (severity_rank[r.severity], r.scalar_score))
    top = items[0] if items else None

    fig, ax = plt.subplots(figsize=(14, 4.0))
    ax.axis("off")
    ax.text(0.0, 0.95, "PRIORITY FINDING", fontsize=12, fontweight="bold",
            color="#1971c2", transform=ax.transAxes)
    if top is None:
        ax.text(0.0, 0.6, "(no diagnostics produced results)",
                fontsize=11, transform=ax.transAxes)
    else:
        ax.text(0.0, 0.82, top.axis, fontsize=14, fontweight="bold",
                transform=ax.transAxes, color="#c92a2a")
        ax.text(0.0, 0.72, top.explanation, fontsize=10,
                transform=ax.transAxes, color="#222")
        if top.recommendation:
            ax.text(0.0, 0.45, "RECOMMENDED ACTION:", fontsize=10,
                    fontweight="bold", color="#1971c2", transform=ax.transAxes)
            y = 0.36
            for ln in top.recommendation.splitlines()[:8]:
                ax.text(0.0, y, ln, fontsize=10, color="#222", transform=ax.transAxes)
                y -= 0.06

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
