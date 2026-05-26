"""Reporters — consume DiagnosticResults and produce shareable artifacts.

Two categories of reporters:

  • Text outputs (core, always available): ``export_json``,
    ``render_markdown_report``.

  • Visual outputs (require ``pip install emboviz[viz]`` — matplotlib +
    jinja2): ``render_failure_matrix``, ``render_verdict_card``,
    ``render_trajectory_timelines``, ``render_failure_tape``.

Every reporter is lazy-imported. Accessing a viz reporter without the
``viz`` extra raises a clean ImportError telling the user which pip
extra to add.
"""

from __future__ import annotations

__all__ = [
    # text (core)
    "export_json",
    "render_markdown_report",
    # visual (extra: viz)
    "render_failure_matrix",
    "render_verdict_card",
    "render_trajectory_timelines",
    "render_failure_tape",
]


_LAZY: dict[str, tuple[str, str, str]] = {
    "export_json":              ("emboviz.reports.json_export",         "export_json",                  ""),
    "render_markdown_report":   ("emboviz.reports.markdown",            "render_markdown_report",       ""),
    "render_failure_matrix":    ("emboviz.reports.failure_matrix",      "render_failure_matrix",        "viz"),
    "render_verdict_card":      ("emboviz.reports.verdict_card",        "render_verdict_card",          "viz"),
    "render_trajectory_timelines": ("emboviz.reports.trajectory_timeline", "render_trajectory_timelines", "viz"),
    "render_failure_tape":      ("emboviz.reports.trajectory_timeline", "render_failure_tape",          "viz"),
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
                f"emboviz.reports.{name} requires the '{extra}' extra. "
                f"Install with: pip install 'emboviz[{extra}]'.  "
                f"Underlying ImportError: {e}"
            ) from e
        raise
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
