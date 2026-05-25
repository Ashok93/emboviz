"""The complete diagnostic battery — every supported diagnostic."""

from __future__ import annotations

from emboviz.suites.base import Suite
from emboviz.suites.language_grounding import build_language_grounding_suite
from emboviz.suites.visual_robustness import build_visual_robustness_suite


def build_full_profile() -> Suite:
    lang = build_language_grounding_suite()
    vis = build_visual_robustness_suite()
    return Suite(
        name="full_profile",
        description="All language + vision diagnostics. The complete failure profile.",
        diagnostics=lang.diagnostics + vis.diagnostics,
    )
