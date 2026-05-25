"""PolicyLens — a diagnostic and interpretability framework for VLA robot policies.

Public API entry points (lazy imports — the heavy modules don't load on
`import policylens`):

    from policylens.core import Scene, ActionResult, DiagnosticResult
    from policylens.models import VLAModel, Capability
    from policylens.diagnostics import Diagnostic
    from policylens.suites import Suite

See ARCHITECTURE.md for the design.
"""

__version__ = "0.1.0"
