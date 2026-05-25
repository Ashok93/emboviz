"""Preset suites — bundles of diagnostics."""

from emboviz.suites.base import Suite, SuiteResult, TrajectorySuiteResult
from emboviz.suites.full_profile import build_full_profile
from emboviz.suites.language_grounding import build_language_grounding_suite
from emboviz.suites.quick_smoke import build_quick_smoke
from emboviz.suites.visual_robustness import build_visual_robustness_suite

__all__ = [
    "Suite",
    "SuiteResult",
    "TrajectorySuiteResult",
    "build_full_profile",
    "build_language_grounding_suite",
    "build_quick_smoke",
    "build_visual_robustness_suite",
]
