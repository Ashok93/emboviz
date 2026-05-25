"""Reporters consume DiagnosticResults and produce shareable artifacts."""

from emboviz.reports.failure_matrix import render_failure_matrix
from emboviz.reports.json_export import export_json
from emboviz.reports.markdown import render_markdown_report
from emboviz.reports.trajectory_timeline import (
    render_failure_tape,
    render_trajectory_timelines,
)
from emboviz.reports.verdict_card import render_verdict_card

__all__ = [
    "render_failure_matrix",
    "render_verdict_card",
    "render_markdown_report",
    "render_trajectory_timelines",
    "render_failure_tape",
    "export_json",
]
