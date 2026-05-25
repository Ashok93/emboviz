"""Reporters consume DiagnosticResults and produce shareable artifacts."""

from policylens.reports.failure_matrix import render_failure_matrix
from policylens.reports.json_export import export_json
from policylens.reports.markdown import render_markdown_report
from policylens.reports.trajectory_timeline import (
    render_failure_tape,
    render_trajectory_timelines,
)
from policylens.reports.verdict_card import render_verdict_card

__all__ = [
    "render_failure_matrix",
    "render_verdict_card",
    "render_markdown_report",
    "render_trajectory_timelines",
    "render_failure_tape",
    "export_json",
]
