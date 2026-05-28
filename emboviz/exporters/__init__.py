"""Exporters — emit diagnostic outputs into the tools roboticists already use.

Available exporters:
  • Scorecard PNG          — axis-by-axis severity grid for at-a-glance
                             triage (extra: ``viz``)
  • Rerun ``.rrd``         — playback in rerun.io with per-frame diagnostic
                             tracks (extra: ``rerun``)
  • Foxglove ``.mcap``     — playback in Foxglove Studio with diagnostic
                             topics (extra: ``mcap``)
  • Per-diagnostic detail  — Markdown drill-down per axis (core)
  • JSON dump              — full structured results for downstream
                             tooling (core)

Every exporter is lazy-imported via PEP 562 ``__getattr__`` so the
exporters package itself loads in a core-only install. Calling an
exporter without its extra produces a clean ImportError with the
required ``pip install emboviz[<extra>]``.

Strict principle: **no blind prose synthesis**. The OSS gives evidence;
users (and later the Cloud's interactive AI) form the narrative.
"""

from __future__ import annotations

__all__ = [
    "render_scorecard",     # extra: viz
    "export_rerun",         # extra: rerun
    "export_foxglove",      # extra: mcap
    "render_detail_pages",  # core
]


_LAZY: dict[str, tuple[str, str, str]] = {
    "render_scorecard":    ("emboviz.exporters.scorecard", "render_scorecard",    "viz"),
    "export_rerun":        ("emboviz.exporters.rerun",     "export_rerun",        "rerun"),
    "export_foxglove":     ("emboviz.exporters.foxglove",  "export_foxglove",     "mcap"),
    "render_detail_pages": ("emboviz.exporters.details",   "render_detail_pages", ""),  # core
}


def __getattr__(name: str):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name, extra = entry
    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError as e:
        if extra:
            raise ImportError(
                f"emboviz.exporters.{name} requires the '{extra}' extra. "
                f"Install with: pip install 'emboviz[{extra}]'.  "
                f"Underlying ImportError: {e}"
            ) from e
        raise
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
