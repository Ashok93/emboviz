"""Instruction Sensitivity Score — mean action divergence across variants.

The wrapper metric used by counterfactual diagnostics: take a list of
(baseline, perturbed) ActionResult pairs and return mean divergence.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from emboviz.core.types import ActionResult
from emboviz.metrics.action_divergence import ActionDivergenceMetric
from emboviz.metrics.base import Metric


class InstructionSensitivityMetric(Metric):
    """Mean action divergence across N perturbed variants of one scene."""

    name = "instruction_sensitivity"

    def __init__(self, model=None, aggregator: str = "mean"):
        self.divergence = ActionDivergenceMetric(model=model)
        if aggregator not in ("mean", "median", "max", "min"):
            raise ValueError(f"Unknown aggregator {aggregator}")
        self.aggregator = aggregator

    def compute(self, pairs: Iterable[tuple[ActionResult, ActionResult]]) -> float:
        scores = [self.divergence.compute(b, p) for b, p in pairs]
        if not scores:
            return 0.0
        if self.aggregator == "mean":
            return float(np.mean(scores))
        if self.aggregator == "median":
            return float(np.median(scores))
        if self.aggregator == "max":
            return float(np.max(scores))
        return float(np.min(scores))
