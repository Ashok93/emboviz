"""Emboviz — a diagnostic and interpretability framework for VLA robot policies.

Public API entry points (lazy imports — the heavy modules don't load on
`import emboviz`):

    from emboviz.core import Scene, ActionResult, DiagnosticResult
    from emboviz.models import VLAModel, Capability
    from emboviz.diagnostics import Diagnostic
    from emboviz.suites import Suite

See ARCHITECTURE.md for the design.
"""

__version__ = "0.1.0"
