"""Exporters — emit diagnostic outputs into the tools roboticists already use.

The OSS engine emits:
  • Scorecard PNG          — axis-by-axis severity grid for at-a-glance triage
  • Rerun `.rrd`           — playback in rerun.io with per-frame diagnostic tracks
  • Foxglove `.mcap`       — playback in Foxglove Studio with diagnostic topics
  • Per-diagnostic detail  — Markdown drill-down per axis (no prose synthesis)
  • JSON dump              — full structured results for downstream tooling

Strict principle: **no blind prose synthesis**. The OSS gives evidence;
users (and later the Cloud's interactive AI) form the narrative.
"""

from emboviz.exporters.scorecard import render_scorecard

__all__ = ["render_scorecard"]


def __getattr__(name):
    # Lazy access to optional-dep exporters.
    if name == "export_rerun":
        from emboviz.exporters.rerun import export_rerun
        return export_rerun
    if name == "export_foxglove":
        from emboviz.exporters.foxglove import export_foxglove
        return export_foxglove
    if name == "render_detail_pages":
        from emboviz.exporters.details import render_detail_pages
        return render_detail_pages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
