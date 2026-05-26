"""Emboviz — the X-ray for deployed VLA policies.

Plain-English diagnostics on failing real-robot episodes: when your VLA
picks the wrong cup, freezes, or reaches the wrong side of the table,
emboviz tells you which inputs the policy was consuming, which it was
ignoring, and where in the trajectory things went off the rails.

Public API entry points (lazy imports — heavy modules don't load on
``import emboviz``):

    from emboviz.core import Scene, ActionResult, DiagnosticResult
    from emboviz.models import VLAModel, Capability
    from emboviz.diagnostics import Diagnostic
    from emboviz.suites import Suite

See README.md for the user-facing flow and ARCHITECTURE.md for the design.
"""

# Single source of truth for version: pyproject.toml. Read via package
# metadata so we can never drift out of sync with the wheel.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("emboviz")
except PackageNotFoundError:  # editable install before metadata is materialized
    __version__ = "0+unknown"
