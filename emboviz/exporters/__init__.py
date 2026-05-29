"""Exporters — emit diagnostic outputs into tools roboticists already use.

The shipped exporter is the **Rerun ``.rrd``** writer (``export_rerun`` in
:mod:`emboviz.exporters.rerun`): per-frame camera streams + diagnostic
overlays, verdict ribbons, and metric time-series, laid out by a blueprint
for playback in rerun.io. The runner imports it directly; it's lazy-exposed
here too so ``import emboviz.exporters`` stays cheap.

(:mod:`emboviz.exporters.correlation` provides the failure-moment helpers
the runner uses alongside it — imported directly, not via the registry.)

Strict principle: **no blind prose synthesis**. The OSS gives evidence;
the user forms the narrative.
"""

from __future__ import annotations

__all__ = ["export_rerun"]


_LAZY: dict[str, tuple[str, str, str]] = {
    "export_rerun": ("emboviz.exporters.rerun", "export_rerun", "rerun"),
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
