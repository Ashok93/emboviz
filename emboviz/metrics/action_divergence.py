"""ActionDivergenceMetric — the basic 'how different are two actions?'

Delegates to the model's `compare_actions` so that model-specific scaling
(e.g., OpenVLA's bridge_orig q01/q99 normalization) is used when available.
"""

from __future__ import annotations

from emboviz.core.types import ActionResult
from emboviz.metrics.base import Metric


class ActionDivergenceMetric(Metric):
    name = "action_divergence"

    def __init__(self, model=None):
        """If `model` is provided, uses model.compare_actions for the distance;
        otherwise falls back to unscaled L2."""
        self.model = model

    def compute(self, baseline: ActionResult, perturbed: ActionResult) -> float:
        if self.model is not None and hasattr(self.model, "compare_actions"):
            return float(self.model.compare_actions(baseline, perturbed))
        from emboviz.core.distances import l2_distance
        return l2_distance(baseline, perturbed)
