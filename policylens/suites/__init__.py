"""Preset suites — bundles of diagnostics."""

from policylens.suites.base import Suite, SuiteResult, TrajectorySuiteResult
from policylens.suites.full_profile import build_full_profile
from policylens.suites.language_grounding import build_language_grounding_suite
from policylens.suites.quick_smoke import build_quick_smoke
from policylens.suites.visual_robustness import build_visual_robustness_suite

__all__ = [
    "Suite",
    "SuiteResult",
    "TrajectorySuiteResult",
    "build_full_profile",
    "build_language_grounding_suite",
    "build_quick_smoke",
    "build_visual_robustness_suite",
]
